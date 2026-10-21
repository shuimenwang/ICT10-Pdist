// pdist.cpp —— Host 侧：算子原型 / Shape 推断 / Tiling
// 语义严格对齐 torch.nn.functional.pdist：输入 (N, M)，输出 (N*(N-1)/2,)，行主序 (i<j) 展开。
// 设计要点：
//   1) 输出元素（pair）在核间均摊，而非按行均摊 —— 行 i 的 pair 数为 N-1-i，按行切必然负载倾斜。
//   2) p 模式（0/1/2/泛 p）与 dtype 共同编码进 TilingKey，Kernel 侧编译期分支，避免运行期判断。
//   3) 1/p 与 p 均由 Host 预计算下发，核内不做除法。
#include "../op_kernel/pdist_tiling.h"

#include <cmath>
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
constexpr uint32_t PDIST_DTYPE_FP16 = 0;
constexpr uint32_t PDIST_DTYPE_FP32 = 1;
constexpr float    PDIST_P_EPS = 1e-6F;

uint32_t ResolvePMode(float p)
{
    if (std::fabs(p) < PDIST_P_EPS) {
        return PDIST_PMODE_ZERO;
    }
    if (std::fabs(p - 1.0F) < PDIST_P_EPS) {
        return PDIST_PMODE_ONE;
    }
    if (std::fabs(p - 2.0F) < PDIST_P_EPS) {
        return PDIST_PMODE_TWO;
    }
    return PDIST_PMODE_GENERIC;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float* pAttr = attrs->GetAttrPointer<float>(0);
    const float p = (pAttr == nullptr) ? 2.0F : *pAttr;

    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetStorageShape().GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t n = static_cast<uint64_t>(xShape->GetStorageShape().GetDim(0));
    const uint64_t m = static_cast<uint64_t>(xShape->GetStorageShape().GetDim(1));
    const uint64_t totalPairs = n * (n - 1) / 2;

    PdistTilingData* tiling = context->GetTilingData<PdistTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t pMode = ResolvePMode(p);
    tiling->rowNum = static_cast<uint32_t>(n);
    tiling->colNum = static_cast<uint32_t>(m);
    tiling->totalPairs = static_cast<uint32_t>(totalPairs);
    tiling->pValue = p;
    tiling->pReciprocal = p > 0.0F ? 1.0F / p : 0.0F;
    tiling->pMode = pMode;

    // 核数：AIV 向量核；pair 数不足时按 pair 数收缩，避免空核
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }
    uint32_t coreNum = static_cast<uint32_t>(std::min<uint64_t>(aivNum, std::max<uint64_t>(totalPairs, 1)));
    tiling->usedCoreNum = coreNum;
    tiling->pairPerCore = static_cast<uint32_t>(totalPairs / coreNum);
    tiling->pairTailCore = static_cast<uint32_t>(totalPairs % coreNum);
    // v2 tile 路径：单元 (jt, b) 均摊，单元总数 = NT*(NT+1)/2，NT = ceil(N/8)
    const uint64_t tileNum = (n + 7) / 8;
    const uint64_t unitTotal = tileNum * (tileNum + 1) / 2;
    tiling->unitPerCore = static_cast<uint32_t>(unitTotal / coreNum);
    tiling->unitTailCore = static_cast<uint32_t>(unitTotal % coreNum);

    const auto xDesc = context->GetInputDesc(0);
    if (xDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtypeFlag = (xDesc->GetDataType() == ge::DT_FLOAT16) ? PDIST_DTYPE_FP16 : PDIST_DTYPE_FP32;
    tiling->dtypeFlag = dtypeFlag;

    context->SetTilingKey(0);  // 单 TilingKey：工程仅编译 key=0 变体，p/dtype 在 kernel 内运行时分支
    context->SetBlockDim(coreNum);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t n = xShape->GetDim(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(1);
    yShape->SetDim(0, n * (n - 1) / 2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Pdist);
}  // namespace ops
