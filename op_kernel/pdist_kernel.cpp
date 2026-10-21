// pdist_kernel.cpp —— Ascend C Kernel（骨架，Phase 1 起逐场景填肉）
// 计算流程（pds.md 口径）：
//   pdist_process      ：差值 → 绝对值 → 幂(ln+muls+exp) → 归约累加 → 中间结果
//   pdist_sum_process  ：ln + muls(1/p) + exp → 最终距离
// 纪律（每条对应一类高频错误，提交前逐条自查）：
//   1. 中间计算一律 FP32（防结果错误 52.4%）
//   2. AllocTensor/FreeTensor 严格配对（防 507899 / 流同步超时）
//   3. UB 偏移地址逐项验算 32B 对齐（防 507015 核挂死）
//   4. repeat_times/count_num 打印自检，不超 UB 容量
//   5. 写回地址按 pair 索引唯一，禁止多核写同一地址
#include "kernel_operator.h"
#include "pdist_tiling.h"   // 单源：TilingKey 位定义 + PdistTilingData

using namespace AscendC;
using namespace optiling;

constexpr int32_t BUFFER_NUM = 2;  // Double Buffering 乒乓（Phase 3 启用深度调整）

class KernelPdist {
public:
    __aicore__ inline KernelPdist() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipe, const PdistTilingData* td)
    {
        tiling_ = *td;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));      // FP16 视图；FP32 用模板区分
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
        // TODO(Phase1): 按本核行区间 [rowBegin, rowEnd) 计算 pair 基址
        //   rowBegin = blockIdx * rowPerCore + min(blockIdx, rowTailCore)
        //   rowCnt   = rowPerCore + (blockIdx < rowTailCore ? 1 : 0)
        // TODO(Phase1): pipe_->InitBuffer 按 tiling_ 的 blockAlignedLen 申请，32B 对齐
    }

    __aicore__ inline void Process()
    {
        // 场景分发：TilingKey 编译期分支优先（模板参数化，防 I-Cache miss）
        switch (tiling_.sceneIdx) {
            case 0: ProcessScene1(); break;  // 场景① 整行单轮
            case 1: ProcessScene2(); break;  // 场景② 按M切块多轮累加
            case 2: ProcessScene3(); break;  // 场景③ 多核行块
            default: ProcessScene4(); break; // 场景④ 非对齐 Pad+Mask
        }
    }

private:
    // 场景①：整行放得下 UB，行对 (i,j) 直接单轮算完
    __aicore__ inline void ProcessScene1()
    {
        // TODO(Phase1) 骨架填肉顺序：
        //   a. DataCopy 行 j、行 i → UB（大块搬运，512B 对齐优先）
        //   b. Sub  → 差值 ; Abs → |d|
        //   c. p=2: Mul + 归约 Add → 路径 A（Sqrt 由 sum_process 承担）
        //      泛p: Ln → Muls(p) → Exp → 归约 Add（路径 B，Phase 2）
        //      p=0: Ne(0)→Cast(1/0)→归约 Add ; p=1: 直接 Abs 后归约（Phase 2）
        //   d. 中间结果 → sumProcess_ 再开 p 次方 → yGm_ 按 PairIndex 唯一写回
        // 注意：中间计算一律 FP32（FP16 输入先 Cast）
    }

    // 场景②：一行放不下 UB，按 blockLen 切块多轮核内累加
    __aicore__ inline void ProcessScene2()
    {
        // TODO(Phase2)：
        //   a. 累加寄存器每轮清零（易错点 #1：累加变量未清零）
        //   b. 轮内同场景① a~c，但归约结果跨轮累加
        //   c. blockLoop 边界：最后一块长度 = M - (blockLoop-1)*blockLen
        //   d. 全部轮次完成后统一开 p 次方
    }

    // 场景③：数据量超大，多核按行块切分 + 尾均摊
    __aicore__ inline void ProcessScene3()
    {
        // TODO(Phase2)：
        //   a. 复用场景①/②的行处理，行区间由 rowPerCore/rowTailCore 决定
        //   b. 严禁单核扛尾巴；各核写回 PairIndex 唯一
        //   c. 核间不通信、不同步（核内完成累加与写回）
    }

    // 场景④：非 32B 对齐，Pad 补齐 + Mask 过滤
    __aicore__ inline void ProcessScene4()
    {
        // TODO(Phase2)：
        //   a. 搬运长度 = 对齐后长度 blockAlignedLen，源侧用 DataCopyPad
        //   b. 计算侧 Mask 过滤 Pad 区脏数据（易错点：Mask 漏加 → 结果混入垃圾值）
        //   c. Pad 长度验算 UB 容量（易错点：Pad 算错 → UB 越界 507015）
    }

    // 开 p 次方：d = exp( (1/p) · ln(S) )；p=0/1/2 走专用出口
    __aicore__ inline void SumProcess()
    {
        // TODO(Phase1)：p=2 → Sqrt；p=1 → 原值；p=0 → 原值（整数计数）
        // TODO(Phase2)：泛 p → Ln → Muls(pReciprocal_) → Exp（全程 FP32）
    }

private:
    GlobalTensor<half> xGm_;   // TODO(Phase1): 模板化 dtype（FP16/FP32 两实例化）
    GlobalTensor<half> yGm_;
    TPipe* pipe_ = nullptr;
    PdistTilingData tiling_;
    float pReciprocal_ = 0.5F;
};

// Kernel 入口：函数名与算子名/生成工程 Json 三处一致（防命名错误 #5）
extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelPdist op;
    TPipe pipe;
    op.Init(x, y, &pipe, &tilingData);
    op.Process();
}
