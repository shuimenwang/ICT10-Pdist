// pdist_tiling.cpp —— Host 侧 Tiling 计算（骨架，Phase 1 起逐场景填肉）
// 对应 WBS：docs/02 §4 Phase 1（场景①）→ Phase 2（场景②③④）
// 集成方式：msopgen 生成的工程中，将本文件的逻辑合入生成的 *_tiling.cpp
//           （保持生成工程的注册宏与命名空间，避免双重注册）。
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

static const uint64_t UB_CAPACITY_BYTES  = 192 * 1024;   // 910B3 UB 容量，编译期常量，实测后校准
static const uint64_t DTYPE_SIZE_FP32    = 4;
static const uint64_t ALIGN_BYTES        = 32;           // 硬件 32B 对齐硬约束

static uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// 输出对索引 (i<j) 行主序：pair = i*(2N-i-1)/2 + (j-i-1)，写回地址唯一（防多核覆盖）
static inline uint64_t PairIndex(uint64_t n, uint64_t i, uint64_t j)
{
    return i * (2 * n - i - 1) / 2 + (j - i - 1);
}

static ge::graphStatus PdistTilingFunc(gert::TilingContext* context)
{
    // ---------- 0) 防御：可选属性 p 空值判断（防 Host 段错误，错误 #11） ----------
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_LOG(context, attrs, return ge::GRAPH_FAILED);
    const float p = attrs->GetAttrPointer<float>(0) ? *(attrs->GetAttrPointer<float>(0)) : 2.0F;

    auto xShape = context->GetInputShape(0);
    auto yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_LOG(context, xShape, return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_LOG(context, yShape, return ge::GRAPH_FAILED);

    const uint64_t n = xShape->GetStorageShape().GetDim(0);
    const uint64_t m = xShape->GetStorageShape().GetDim(1);

    // ---------- 1) shape 推断：Y = (N*(N-1)/2,) ----------
    yShape->SetDimNum(1);
    yShape->SetDim(0, n * (n - 1) / 2);

    // ---------- 2) 场景决策（docs/02 §3.1 决策树，逐项填肉） ----------
    PdistTilingData tiling;
    tiling.set_rowNum(n);
    tiling.set_colNum(m);
    tiling.set_totalPairs(n * (n - 1) / 2);
    tiling.set_pValue(p);
    tiling.set_pReciprocal(p > 0.0F ? 1.0F / p : 0.0F);  // p=0 走专用分支，reciprocal 无效

    const uint64_t dtypeSize = DTYPE_SIZE_FP32;  // TODO(Phase2): 依输入 dtype 选择 4/2
    const bool rowFitUb   = m * dtypeSize <= UB_CAPACITY_BYTES / 4;   // 留双缓冲与中间量余量
    const bool misaligned = (m * dtypeSize) % ALIGN_BYTES != 0;
    // TODO(Phase2): FP16 输入 dtypeSize=2，dtypeIsFp32 由 dtype 判定
    const uint64_t dtypeIsFp32 = 1;

    uint32_t sceneIdx;
    if (rowFitUb) {
        sceneIdx = misaligned ? 3 : 0;                    // 场景① / ④
    } else {
        sceneIdx = misaligned ? 3 : 1;                    // 场景② / ④ 叠加；场景③按输出对多核切分
        // TODO(Phase2): 数据量超大时叠加场景③：按行块多核 + 尾均摊
    }

    // ---------- 3) 多核行均摊（场景①④基量；②③复用同一均摊框架） ----------
    auto platformInfo = context->GetPlatformInfo();
    uint32_t coreNum = platformInfo ? platformInfo->GetCoreNumAiv() : 0;
    if (coreNum == 0) { coreNum = 1; }
    const uint32_t usableCore = std::min<uint64_t>(coreNum, std::max<uint64_t>(n, 1));
    tiling.set_usedCoreNum(usableCore);
    tiling.set_rowPerCore(n / usableCore);
    tiling.set_rowTailCore(n % usableCore);

    // 场景②参数：按 M 切块，多轮核内累加（累加变量必须每轮清零）
    const uint64_t blockElems = UB_CAPACITY_BYTES / 4 / dtypeSize;    // TODO: 精确余量计算
    tiling.set_blockLen(static_cast<uint32_t>(blockElems));
    tiling.set_blockAlignedLen(static_cast<uint32_t>(AlignUp(blockElems * dtypeSize, ALIGN_BYTES) / dtypeSize));
    tiling.set_blockLoop(static_cast<uint32_t>((m + blockElems - 1) / blockElems));

    tiling.set_sceneIdx(sceneIdx);

    // ---------- 4) TilingKey 下发（单源自 pdist_tiling.h） ----------
    context->SetTilingKey(MakePdistTilingKey(dtypeIsFp32, sceneIdx, misaligned ? 1 : 0));

    // ---------- 5) 结构体落盘 ----------
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usableCore);

    // 调试自检口令：提交前打开一次，核对与手算一致（错误 #1/#6 首查项）
    // OP_LOGD(context->GetNodeName(),
    //     "Pdist tiling: N=%lu M=%lu scene=%u core=%u rowPerCore=%u tail=%u blk=%u loop=%u p=%f",
    //     n, m, sceneIdx, usableCore, tiling.get_rowPerCore(), tiling.get_rowTailCore(),
    //     tiling.get_blockLen(), tiling.get_blockLoop(), p);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// 注册（msopgen 生成工程若已含注册宏，则保留生成版本，删除此处避免重复注册）
namespace ge {
static ge::graphStatus PdistInferShape(gert::InferShapeContext* context)
{
    const auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_LOG(context, xShape, return ge::GRAPH_FAILED);
    const auto n = xShape->GetStorageShape().GetDim(0);
    auto yShape = context->GetOutputShape(0);
    yShape->SetDimNum(1);
    yShape->SetDim(0, n * (n - 1) / 2);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Pdist : public OpDef {
 public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0F);
        this->AICore().SetTiling(optiling::PdistTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->SetInferShape(ge::PdistInferShape);  // TODO: 若生成工程已有 infer shape 合并之
    }
};

REG_OP(Pdist);
}  // namespace ops
