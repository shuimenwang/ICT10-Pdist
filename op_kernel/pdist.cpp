// pdist.cpp —— Ascend C Kernel（第二轮优化版：tile 化 pair 距离）
// 语义：y[q] = (Σ_k |x_i,k - x_j,k|^p)^(1/p)，q 按 (i<j) 行主序展开。
//
// 两条执行路径：
//   v2（M ≤ 1024，主路径）：迁移 PyTorch CUDA pdist / 社区 tile 思路 ——
//     工作切分为单元 (jt, b)：pair (i, j) ∈ [b*8, b*8+8) × [jt*8, jt*8+8) 且 i<j，
//     单元按 jt 外层、b 内层连续编号，核间均摊。
//       1) j-tile 外层常驻 UB：一组 j 行只搬一次，服务组内全部 i-block ——
//          x 读流量从 v1 的 ~totalPairs*M 降为 ~N^2*M/16（约 8 倍降低）。
//       2) i-block（8 行）经 TQue 双缓冲预取，与 V 计算重叠。
//       3) 对角单元 (b == jt) 直接复用 j-tile 数据作 i 源，零搬运。
//       4) 单 chunk 场景 ReduceSum 直接落入 sums 槽位，省去逐 pair 的 Add 与
//          额外屏障（v1 每 pair 3 次 PipeBarrier → v2 每 pair 1 次）。
//       5) 单元粒度 64 pair，V_S/S_MTE3 同步与标量压缩频次降为 v1 的 1/8。
//   v1 回退（M > 1024）：行分段 + tile×chunk 流水（第一轮实现，保场景稳定）。
//
// 数值：FP16 输入 Cast 到 FP32，差值/幂/归约全程 FP32，最后 Cast 回原 dtype。
#include "kernel_operator.h"
#include "pdist_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t K_BLOCK = 1024;   // M 方向分块长度（元素），控制 UB 占用
constexpr uint32_t TJ = 8;           // j-tile 行数（v2 同时为 i-block 行数 TI）
constexpr uint32_t TI = 8;           // i-block 行数
constexpr uint32_t SUM_SLOT = 8;     // 32B 对齐的归约结果槽位
constexpr uint32_t OUT_STRIDE = 16;  // outBuf 行距（32B 对齐，MTE3 src 对齐要求）
}  // namespace

template <typename T, uint32_t P_MODE>
class KernelPdist {
public:
    __aicore__ inline KernelPdist() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PdistTilingData* td, TPipe* pipe)
    {
        rowNum_ = td->rowNum;
        colNum_ = td->colNum;
        totalPairs_ = td->totalPairs;
        pValue_ = td->pValue;
        pReciprocal_ = td->pReciprocal;
        pairPerCore_ = td->pairPerCore;
        pairTailCore_ = td->pairTailCore;
        unitPerCore_ = td->unitPerCore;
        unitTailCore_ = td->unitTailCore;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint64_t>(rowNum_) * colNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalPairs_);

        // ---- v2 tile 路径缓冲 ----
        pipe->InitBuffer(jQue_, 2, TJ * K_BLOCK * sizeof(T));   // j-tile 双缓冲
        pipe->InitBuffer(iQue_, 2, TJ * K_BLOCK * sizeof(T));   // i-block 双缓冲
        if constexpr (sizeof(T) == 2) {
            // 半精度输入的 FP32 化缓存（整块一次 Cast）
            pipe->InitBuffer(jFBuf_, TJ * K_BLOCK * sizeof(float));
            pipe->InitBuffer(iFBuf_, TJ * K_BLOCK * sizeof(float));
        }
        pipe->InitBuffer(diffBuf_, K_BLOCK * sizeof(float));
        pipe->InitBuffer(tmpBuf_, (K_BLOCK + 64) * sizeof(float));
        pipe->InitBuffer(sumBuf_, TJ * TJ * SUM_SLOT * sizeof(float));
        pipe->InitBuffer(chunkBuf_, SUM_SLOT * sizeof(float));
        pipe->InitBuffer(resBuf_, TJ * TJ * SUM_SLOT * sizeof(float));
        pipe->InitBuffer(outBuf_, TI * OUT_STRIDE * sizeof(T));
        if constexpr (P_MODE == PDIST_PMODE_ZERO) {
            pipe->InitBuffer(oneIBuf_, K_BLOCK * sizeof(int32_t));
            LocalTensor<int32_t> oneI = oneIBuf_.Get<int32_t>();
            Duplicate(oneI, 1, K_BLOCK);
        }
        // ---- v1 回退路径缓冲（仅 M > 1024 时使用） ----
        if (colNum_ > K_BLOCK) {
            pipe->InitBuffer(xiQue_, 1, K_BLOCK * sizeof(T));
            pipe->InitBuffer(xiBuf_, K_BLOCK * sizeof(float));
            pipe->InitBuffer(xjBuf_, K_BLOCK * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (totalPairs_ == 0 || rowNum_ < 2) {
            return;
        }
        if (colNum_ <= K_BLOCK) {
            ProcessTiled();
        } else {
            ProcessRowwise();
        }
    }

private:
    // ==================== v2：tile 化 pair 距离 ====================

    // 行 i 的首个输出下标：Σ_{r<i}(N-1-r)（i=0 时 i*(i-1) 因子为 0，安全）
    __aicore__ inline uint64_t RowStart(uint32_t i) const
    {
        return static_cast<uint64_t>(i) * (rowNum_ - 1) -
               static_cast<uint64_t>(i) * (i - 1) / 2;
    }

    __aicore__ inline uint32_t TileRowsJ(uint32_t jt) const
    {
        const uint32_t remain = rowNum_ - jt * TJ;
        return remain < TJ ? remain : TJ;
    }

    // UB 侧行距：DataCopyPad 的 dstStride 不生效（UB 必须连续），故整块搬运时行距 = len；
    // len 非 32B 整倍时逐行搬运，blockLen 按 32B 对齐 —— DataCopyPad 的 blockLen
    // 非 32B 整数倍时硬件会把上一行尾块重复进下一行（2D 与逐行均如此），
    // 故行距按元素对齐到 32B：fp16 = 16 元素，fp32 = 8 元素。
    __aicore__ inline uint32_t RowStride(uint32_t len) const
    {
        constexpr uint32_t ALIGN_ELE = 32 / sizeof(T);
        return (len % ALIGN_ELE == 0) ? len : ((len + ALIGN_ELE - 1) / ALIGN_ELE) * ALIGN_ELE;
    }

    // 搬 tileRows 行 × len 元素：一次 2D DMA（行距紧凑）或逐行 DMA（需补齐对齐）
    __aicore__ inline void LoadJUnit(const LocalTensor<T>& dst, uint32_t jStart, uint32_t t,
                                     uint32_t tileRows, uint32_t k, uint32_t len)
    {
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        const uint32_t stride = RowStride(len);
        const uint64_t base = static_cast<uint64_t>(jStart + t * TJ) * colNum_ + k * K_BLOCK;
        if (stride == len) {
            // 紧凑布局，单条 2D DMA：行间在 GM 上跳过 (M-len) 个元素
            const uint32_t srcGap = static_cast<uint32_t>((colNum_ - len) * sizeof(T));
            DataCopyExtParams params(static_cast<uint16_t>(tileRows), static_cast<uint32_t>(len * sizeof(T)),
                                     srcGap, 0, 0);
            DataCopyPad(dst, xGm_[base], params, padParams);
        } else {
            // 非 8 元素(32B)对齐长度：blockLen 向上补齐后整块搬运，规避
            // DataCopyPad 非对齐 blockLen 在尾部 repeat 的数据错位；多搬尾部
            // 落在 [len, stride) 区间，不参与长度为 len 的计算。
            for (uint32_t b = 0; b < tileRows; ++b) {
                DataCopyExtParams params(1, static_cast<uint32_t>(stride * sizeof(T)), 0, 0, 0);
                DataCopyPad(dst[b * stride], xGm_[base + static_cast<uint64_t>(b) * colNum_], params, padParams);
            }
        }
    }

    __aicore__ inline void PrefetchJTile(uint32_t jt, bool doIt)
    {
        if (!doIt) {
            return;
        }
        LocalTensor<T> t = jQue_.AllocTensor<T>();
        LoadJUnit(t, 0, jt, TileRowsJ(jt), 0, colNum_);
        jQue_.EnQue(t);
    }

    __aicore__ inline void PrefetchIBlock(uint32_t b, uint32_t bHi, uint32_t len, bool doIt)
    {
        if (!doIt || b >= bHi) {
            return;
        }
        LocalTensor<T> t = iQue_.AllocTensor<T>();
        LoadJUnit(t, 0, b, TI, 0, len);
        iQue_.EnQue(t);
    }

    // 一个 pair 的 |xi - xj|^p 累加主体（复用 Accumulate），结果直写 32B 槽位
    __aicore__ inline void PairKernel(const LocalTensor<float>& diff, const LocalTensor<float>& tmp,
                                      const LocalTensor<float>& sums, uint32_t slotIdx,
                                      const LocalTensor<float>& xi, const LocalTensor<float>& xj,
                                      uint32_t len)
    {
        Sub(diff, xi, xj, len);
        Accumulate(diff, len);
        // 单 chunk 场景无需跨块累加：ReduceSum 直接落入槽位，省去逐 pair 的 Add 与额外屏障
        ReduceSum(sums[slotIdx * SUM_SLOT], diff, tmp, len);
    }

    // 离对角单元：i-block 全部 8 行 × j-tile 前 tileRowsJ 行
    __aicore__ inline void TiledUnit(const LocalTensor<float>& jF, const LocalTensor<float>& iF,
                                     uint32_t tileRowsJ, uint32_t len, uint32_t stride,
                                     const LocalTensor<float>& sums)
    {
        LocalTensor<float> diff = diffBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        for (uint32_t r = 0; r < TI; ++r) {
            const LocalTensor<float> xi = iF[r * stride];
            for (uint32_t bj = 0; bj < tileRowsJ; ++bj) {
                PairKernel(diff, tmp, sums, r * TJ + bj, xi, jF[bj * stride], len);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    // 对角单元：tile 内部上三角 pair（i 源直接用 j-tile 数据，零搬运）
    __aicore__ inline void TiledDiagUnit(const LocalTensor<float>& jF, uint32_t tileRowsJ,
                                         uint32_t len, uint32_t stride,
                                         const LocalTensor<float>& sums)
    {
        LocalTensor<float> diff = diffBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        for (uint32_t r = 0; r + 1 < tileRowsJ; ++r) {
            const LocalTensor<float> xi = jF[r * stride];
            for (uint32_t bj = r + 1; bj < tileRowsJ; ++bj) {
                PairKernel(diff, tmp, sums, r * TJ + bj, xi, jF[bj * stride], len);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    // 单元结果落盘：64（对角单元为上三角）个 32B 槽位压缩 → 每行一次连续 DataCopyPad
    __aicore__ inline void TiledFinalize(const LocalTensor<float>& sums, bool diag, uint32_t b,
                                         uint32_t jt, uint32_t tileRowsJ)
    {
        constexpr uint32_t SLOTS = TJ * TJ * SUM_SLOT;
        LocalTensor<float> res = resBuf_.Get<float>();
        if constexpr (P_MODE == PDIST_PMODE_TWO) {
            Sqrt(res, sums, SLOTS);
        } else if constexpr (P_MODE == PDIST_PMODE_GENERIC) {
            Ln(res, sums, SLOTS);
            Muls(res, res, pReciprocal_, SLOTS);
            Exp(res, res, SLOTS);
        } else {
            Adds(res, sums, 0.0F, SLOTS);
        }
        PipeBarrier<PIPE_V>();

        const TEventID vToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        LocalTensor<T> out = outBuf_.Get<T>();
        const uint32_t outRows = diag ? (tileRowsJ - 1) : TI;
        for (uint32_t r = 0; r < outRows; ++r) {
            if (diag) {
                for (uint32_t bj = r + 1; bj < tileRowsJ; ++bj) {
                    const float v = res.GetValue((r * TJ + bj) * SUM_SLOT);
                    out.SetValue(r * OUT_STRIDE + (bj - r - 1), static_cast<T>(v));
                }
            } else {
                for (uint32_t bj = 0; bj < tileRowsJ; ++bj) {
                    const float v = res.GetValue((r * TJ + bj) * SUM_SLOT);
                    out.SetValue(r * OUT_STRIDE + bj, static_cast<T>(v));
                }
            }
        }
        const TEventID sToMte3 = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(sToMte3);
        WaitFlag<HardEvent::S_MTE3>(sToMte3);
        for (uint32_t r = 0; r < outRows; ++r) {
            const uint32_t i = diag ? (jt * TJ + r) : (b * TJ + r);
            const uint64_t q = diag ? RowStart(i)
                                    : RowStart(i) + static_cast<uint64_t>(jt) * TJ - i - 1;
            const uint32_t cnt = diag ? (tileRowsJ - 1 - r) : tileRowsJ;
            DataCopyExtParams params(1, cnt * sizeof(T), 0, 0, 0);
            DataCopyPad(yGm_[q], out[r * OUT_STRIDE], params);
        }
    }

    __aicore__ inline void ProcessTiled()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t uBegin = static_cast<uint64_t>(blockIdx) * unitPerCore_ +
                                (blockIdx < unitTailCore_ ? blockIdx : unitTailCore_);
        const uint64_t uCount = unitPerCore_ + (blockIdx < unitTailCore_ ? 1 : 0);
        if (uCount == 0) {
            return;
        }
        const uint64_t uEnd = uBegin + uCount;

        // 由单元号反解 (jt, b)：顺序推进，无需开方
        uint64_t off = uBegin;
        uint32_t jt0 = 0;
        while (off >= static_cast<uint64_t>(jt0 + 1)) {
            off -= static_cast<uint64_t>(jt0 + 1);
            ++jt0;
        }
        uint64_t offLast = uEnd - 1;
        uint32_t jtLast = 0;
        while (offLast >= static_cast<uint64_t>(jtLast + 1)) {
            offLast -= static_cast<uint64_t>(jtLast + 1);
            ++jtLast;
        }

        const uint32_t len = colNum_;
        const uint32_t stride = RowStride(len);

        // j-tile 双缓冲预取：当前组 + 下一组在途
        PrefetchJTile(jt0, true);
        PrefetchJTile(jt0 + 1, jt0 + 1 <= jtLast);

        LocalTensor<float> sums = sumBuf_.Get<float>();
        uint64_t remaining = uCount;
        uint32_t jt = jt0;
        bool first = true;
        while (remaining > 0) {
            const uint32_t tileRowsJ = TileRowsJ(jt);
            const uint32_t bLo = first ? static_cast<uint32_t>(off) : 0;
            const uint32_t avail = jt + 1 - bLo;
            const uint32_t take = remaining < avail ? static_cast<uint32_t>(remaining) : avail;
            const bool hasDiag = (bLo + take == jt + 1);
            const uint32_t bHi = bLo + take - (hasDiag ? 1u : 0u);

            LocalTensor<T> jCur = jQue_.DeQue<T>();
            LocalTensor<float> jF;
            if constexpr (sizeof(T) == sizeof(float)) {
                jF = jCur.template ReinterpretCast<float>();
            } else {
                Cast(jFBuf_.Get<float>(), jCur, RoundMode::CAST_NONE, TJ * stride);
                PipeBarrier<PIPE_V>();
                jF = jFBuf_.Get<float>();
            }

            // i-block 双缓冲预取（组内复用同一 j-tile）
            PrefetchIBlock(bLo, bHi, len, true);
            PrefetchIBlock(bLo + 1, bHi, len, true);

            for (uint32_t b = bLo; b < bHi; ++b) {
                LocalTensor<T> iBlk = iQue_.DeQue<T>();
                LocalTensor<float> iF;
                if constexpr (sizeof(T) == sizeof(float)) {
                    iF = iBlk.template ReinterpretCast<float>();
                } else {
                    Cast(iFBuf_.Get<float>(), iBlk, RoundMode::CAST_NONE, TI * stride);
                    PipeBarrier<PIPE_V>();
                    iF = iFBuf_.Get<float>();
                }
                TiledUnit(jF, iF, tileRowsJ, len, stride, sums);
                iQue_.FreeTensor(iBlk);
                PrefetchIBlock(b + 2, bHi, len, true);
                TiledFinalize(sums, false, b, jt, tileRowsJ);
            }
            if (hasDiag) {
                TiledDiagUnit(jF, tileRowsJ, len, stride, sums);
                TiledFinalize(sums, true, jt, jt, tileRowsJ);
            }

            jQue_.FreeTensor(jCur);
            PrefetchJTile(jt + 2, jt + 2 <= jtLast);

            remaining -= take;
            first = false;
            ++jt;
        }
    }

    // ==================== v1：行分段路径（M > 1024 回退） ====================

    __aicore__ inline void ProcessRowwise()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t qBegin = static_cast<uint64_t>(blockIdx) * pairPerCore_ +
                                (blockIdx < pairTailCore_ ? blockIdx : pairTailCore_);
        const uint64_t qCount = pairPerCore_ + (blockIdx < pairTailCore_ ? 1 : 0);
        const uint64_t qEnd = qBegin + qCount;
        if (qCount == 0) {
            return;
        }

        // 由全局 pair 序号反解 (i, offsetInRow)：顺序推进，无需开方
        uint64_t i = 0;
        uint64_t offset = qBegin;
        while (offset >= static_cast<uint64_t>(rowNum_ - 1 - i)) {
            offset -= static_cast<uint64_t>(rowNum_ - 1 - i);
            ++i;
        }

        uint64_t q = qBegin;
        while (q < qEnd) {
            const uint64_t rowRemain = static_cast<uint64_t>(rowNum_ - 1 - i) - offset;
            const uint64_t left = qEnd - q;
            const uint32_t cnt = static_cast<uint32_t>(rowRemain < left ? rowRemain : left);
            ProcessRowSegment(static_cast<uint32_t>(i), static_cast<uint32_t>(i + 1 + offset), cnt, q);
            q += cnt;
            ++i;
            offset = 0;
        }
    }

    __aicore__ inline uint32_t TileRows(uint32_t cnt, uint32_t t) const
    {
        const uint32_t remain = cnt - t * TJ;
        return remain < TJ ? remain : TJ;
    }

    __aicore__ inline uint32_t ChunkLen(uint32_t k) const
    {
        const uint32_t remain = colNum_ - k * K_BLOCK;
        return remain < K_BLOCK ? remain : K_BLOCK;
    }

    __aicore__ inline uint32_t NumChunks() const { return (colNum_ + K_BLOCK - 1) / K_BLOCK; }

    __aicore__ inline void LoadXiChunk(uint32_t i, uint32_t k, uint32_t len, const LocalTensor<float>& xiF)
    {
        LocalTensor<T> xiT = xiQue_.AllocTensor<T>();
        const uint32_t lenPad = (len + 7u) & ~7u;   // blockLen 对齐 32B
        DataCopyExtParams params(1, static_cast<uint32_t>(lenPad * sizeof(T)), 0, 0, 0);
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        DataCopyPad(xiT, xGm_[static_cast<uint64_t>(i) * colNum_ + k * K_BLOCK], params, padParams);
        xiQue_.EnQue(xiT);
        LocalTensor<T> xi = xiQue_.DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(xiF, xi, 0.0F, len);
        } else {
            Cast(xiF, xi, RoundMode::CAST_NONE, len);
        }
        xiQue_.FreeTensor(xi);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessRowSegment(uint32_t i, uint32_t jStart, uint32_t cnt, uint64_t qBase)
    {
        const bool cacheRow = (colNum_ <= K_BLOCK);
        if (cacheRow) {
            LoadXiChunk(i, 0, colNum_, xiBuf_.Get<float>());
        }

        const uint32_t nchunks = NumChunks();
        const uint32_t ntiles = (cnt + TJ - 1) / TJ;
        const uint32_t totalUnits = ntiles * nchunks;

        // 流水线预热
        uint32_t unit = 0;
        LocalTensor<T> first = jQue_.AllocTensor<T>();
        LoadJUnit(first, jStart, 0, TileRows(cnt, 0), 0, ChunkLen(0));
        jQue_.EnQue(first);

        LocalTensor<float> sums = sumBuf_.Get<float>();
        for (uint32_t t = 0; t < ntiles; ++t) {
            const uint32_t tileRows = TileRows(cnt, t);
            Duplicate(sums, 0.0F, TJ * SUM_SLOT);
            PipeBarrier<PIPE_V>();

            for (uint32_t k = 0; k < nchunks; ++k) {
                LocalTensor<T> jTile = jQue_.DeQue<T>();
                // 先发下一块搬运，再算当前块 —— MTE2 与 V 重叠
                if (unit + 1 < totalUnits) {
                    const uint32_t nu = unit + 1;
                    const uint32_t nt = nu / nchunks;
                    const uint32_t nk = nu % nchunks;
                    LocalTensor<T> nxt = jQue_.AllocTensor<T>();
                    LoadJUnit(nxt, jStart, nt, TileRows(cnt, nt), nk, ChunkLen(nk));
                    jQue_.EnQue(nxt);
                }
                if (!cacheRow && k == 0) {
                    LoadXiChunk(i, 0, ChunkLen(0), xiBuf_.Get<float>());
                }
                ComputeUnit(jTile, tileRows, k, cacheRow, i);
                jQue_.FreeTensor(jTile);
                ++unit;
            }
            FinalizeTile(sums, tileRows, qBase + static_cast<uint64_t>(t) * TJ);
        }
    }

    __aicore__ inline void ComputeUnit(const LocalTensor<T>& jTile, uint32_t tileRows, uint32_t k,
                                       bool cacheRow, uint32_t rowI)
    {
        const uint32_t len = ChunkLen(k);
        LocalTensor<float> xiF = xiBuf_.Get<float>();
        LocalTensor<float> xjF = xjBuf_.Get<float>();
        LocalTensor<float> diff = diffBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> chunk = chunkBuf_.Get<float>();
        LocalTensor<float> sums = sumBuf_.Get<float>();

        if (!cacheRow) {
            // 场景②：行 i 不常驻，按块现搬（同一 K 块跨 tile 命中 L2）
            LoadXiChunk(rowI, k, len, xiF);
        }

        const uint32_t stride = RowStride(len);
        for (uint32_t b = 0; b < tileRows; ++b) {
            LocalTensor<float> rowF;
            if constexpr (sizeof(T) == sizeof(float)) {
                rowF = jTile[b * stride].template ReinterpretCast<float>();
            } else {
                Cast(xjF, jTile[b * stride], RoundMode::CAST_NONE, len);
                PipeBarrier<PIPE_V>();
                rowF = xjF;
            }
            Sub(diff, xiF, rowF, len);
            Accumulate(diff, len);
            ReduceSum(chunk, diff, tmp, len);
            PipeBarrier<PIPE_V>();
            Add(sums[b * SUM_SLOT], sums[b * SUM_SLOT], chunk, SUM_SLOT);
            PipeBarrier<PIPE_V>();
        }
    }

    // 把 |diff|^p 就地算进 diff
    __aicore__ inline void Accumulate(const LocalTensor<float>& diff, uint32_t len)
    {
        if constexpr (P_MODE == PDIST_PMODE_ZERO) {
            // p=0：非零差计数。整数域位技巧构造 0/1 indicator，不依赖比较/掩码指令：
            //   t = bitcast(|d|)   |d| 符号位 0，t 为非负 int32；d==0 ⟺ t==0
            //   t = (t-1) >> 31    t==0 → -1，t>0 → 0（算术右移）
            //   t = t + 1          0→0（零差），非零→1
            //   Cast 回 float 得 0.0/1.0
            Abs(diff, diff, len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> t = diff.ReinterpretCast<int32_t>();
            LocalTensor<int32_t> oneI = oneIBuf_.Get<int32_t>();
            Sub(t, t, oneI, len);
            ShiftRight(t, t, static_cast<int32_t>(31), len);
            Add(t, t, oneI, len);
            Cast(diff, t, RoundMode::CAST_NONE, len);
        } else if constexpr (P_MODE == PDIST_PMODE_ONE) {
            Abs(diff, diff, len);
        } else if constexpr (P_MODE == PDIST_PMODE_TWO) {
            Mul(diff, diff, diff, len);
        } else {
            Abs(diff, diff, len);
            Ln(diff, diff, len);
            Muls(diff, diff, pValue_, len);
            Exp(diff, diff, len);
        }
        PipeBarrier<PIPE_V>();
    }

    // 开 p 次方 + 批量写回 tileRows 个输出
    __aicore__ inline void FinalizeTile(const LocalTensor<float>& sums, uint32_t tileRows, uint64_t qBase)
    {
        LocalTensor<float> res = resBuf_.Get<float>();
        if constexpr (P_MODE == PDIST_PMODE_TWO) {
            Sqrt(res, sums, TJ * SUM_SLOT);
        } else if constexpr (P_MODE == PDIST_PMODE_GENERIC) {
            Ln(res, sums, TJ * SUM_SLOT);
            Muls(res, res, pReciprocal_, TJ * SUM_SLOT);
            Exp(res, res, TJ * SUM_SLOT);
        } else {
            Adds(res, sums, 0.0F, TJ * SUM_SLOT);
        }
        PipeBarrier<PIPE_V>();

        // 归约结果按 32B 槽位落位，需标量压缩为连续输出
        const TEventID vToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        LocalTensor<T> out = outBuf_.Get<T>();
        for (uint32_t b = 0; b < tileRows; ++b) {
            const float v = res.GetValue(b * SUM_SLOT);
            if constexpr (sizeof(T) == sizeof(float)) {
                out.SetValue(b, v);
            } else {
                out.SetValue(b, static_cast<T>(v));
            }
        }
        const TEventID sToMte3 = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(sToMte3);
        WaitFlag<HardEvent::S_MTE3>(sToMte3);
        DataCopyExtParams params(1, static_cast<uint32_t>(tileRows * sizeof(T)), 0, 0, 0);
        DataCopyPad(yGm_[qBase], out, params);
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, 2> jQue_;
    TQue<QuePosition::VECIN, 2> iQue_;
    TBuf<TPosition::VECCALC> jFBuf_;
    TBuf<TPosition::VECCALC> iFBuf_;
    TBuf<TPosition::VECCALC> diffBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
    TBuf<TPosition::VECCALC> resBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> oneIBuf_;
    // v1 回退路径（M > 1024）独占
    TQue<QuePosition::VECIN, 1> xiQue_;
    TBuf<TPosition::VECCALC> xiBuf_;
    TBuf<TPosition::VECCALC> xjBuf_;

    uint32_t rowNum_ = 0;
    uint32_t colNum_ = 0;
    uint32_t totalPairs_ = 0;
    uint32_t pairPerCore_ = 0;
    uint32_t pairTailCore_ = 0;
    uint32_t unitPerCore_ = 0;
    uint32_t unitTailCore_ = 0;
    float pValue_ = 2.0F;
    float pReciprocal_ = 0.5F;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PdistTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    const uint32_t pMode = tilingData.pMode;
    const uint32_t dtypeFlag = tilingData.dtypeFlag;
    if (dtypeFlag == 0) {
        if (pMode == PDIST_PMODE_ZERO) {
            KernelPdist<half, PDIST_PMODE_ZERO> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else if (pMode == PDIST_PMODE_ONE) {
            KernelPdist<half, PDIST_PMODE_ONE> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else if (pMode == PDIST_PMODE_TWO) {
            KernelPdist<half, PDIST_PMODE_TWO> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else {
            KernelPdist<half, PDIST_PMODE_GENERIC> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        }
    } else {
        if (pMode == PDIST_PMODE_ZERO) {
            KernelPdist<float, PDIST_PMODE_ZERO> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else if (pMode == PDIST_PMODE_ONE) {
            KernelPdist<float, PDIST_PMODE_ONE> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else if (pMode == PDIST_PMODE_TWO) {
            KernelPdist<float, PDIST_PMODE_TWO> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        } else {
            KernelPdist<float, PDIST_PMODE_GENERIC> op; op.Init(x, y, &tilingData, &pipe); op.Process();
        }
    }
}
