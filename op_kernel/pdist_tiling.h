// pdist_tiling.h —— Tiling 结构体与常量（Host / Kernel 共用，POD 布局）
// 纪律：
//   1) 本文件禁止 include "register/tilingdata_base.h" 等 Host 头 —— 会与 Kernel 侧
//      kernel_type.h 的 DT_* 宏冲突（实测编译报错：expected identifier DT_QINT8）。
//   2) 字段顺序/类型即 Host 与 Kernel 的二进制契约，两端必须完全一致。
#ifndef PDIST_TILING_H
#define PDIST_TILING_H

#include <cstdint>

// p 模式：编译期分支依据
constexpr uint32_t PDIST_PMODE_ZERO    = 0;  // p=0 非零差计数
constexpr uint32_t PDIST_PMODE_ONE     = 1;  // p=1 曼哈顿
constexpr uint32_t PDIST_PMODE_TWO     = 2;  // p=2 欧氏
constexpr uint32_t PDIST_PMODE_GENERIC = 3;  // 泛 p（ln+exp 通路）

struct PdistTilingData {
    uint32_t rowNum;        // N：输入行数
    uint32_t colNum;        // M：特征维
    uint32_t totalPairs;    // N*(N-1)/2
    float    pValue;        // 原始 p
    float    pReciprocal;   // 1/p，Host 预计算（核内不做除法）
    uint32_t pMode;         // 0/1/2/3
    uint32_t dtypeFlag;     // 0=FP16, 1=FP32
    uint32_t usedCoreNum;   // 实际启用 AIV 核数
    uint32_t pairPerCore;   // v1 回退路径（M>1024）：每核基础 pair 数
    uint32_t pairTailCore;  // v1 回退路径：前 pairTailCore 个核各多 1 个 pair
    uint32_t unitPerCore;   // v2 tile 路径：每核基础单元数（单元=(j-tile,i-block)）
    uint32_t unitTailCore;  // v2 tile 路径：前 unitTailCore 个核各多 1 个单元
};

#endif  // PDIST_TILING_H
