// pdist_tiling.h —— Tiling 结构体与 TilingKey 单源定义
// 规则：op_host 与 op_kernel 只允许 include 本文件获取结构体与 Key 位定义，
//       严禁两侧各自重复定义（防错误码 31361001「找不到 TilingKey 对应 Kernel」）。
// 说明：本文件随 scripts/init_project.sh 生成的工程合入；若 msopgen 生成模板带同名
//       结构体，以生成模板的宏风格为准做字段合并，不新增第二份定义。
#ifndef PDIST_TILING_H
#define PDIST_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// ---------------- TilingKey 位定义 ----------------
// bit[3]   dtype    : 0=FP16, 1=FP32
// bit[2:1] sceneIdx : 0=场景①整行单轮  1=场景②按M切块多轮
//                    2=场景③多核行块  3=场景④非对齐Pad+Mask（可叠加触发）
// bit[0]   misalign : 0=对齐, 1=非对齐
constexpr uint64_t PDIST_KEY_DTYPE_SHIFT  = 3;
constexpr uint64_t PDIST_KEY_SCENE_SHIFT  = 1;
constexpr uint64_t PDIST_KEY_ALIGNED_MASK = 0x1ULL;

inline uint64_t MakePdistTilingKey(uint64_t dtypeIsFp32, uint64_t sceneIdx, uint64_t misaligned)
{
    return (dtypeIsFp32 << PDIST_KEY_DTYPE_SHIFT) |
           ((sceneIdx & 0x3ULL) << PDIST_KEY_SCENE_SHIFT) |
           (misaligned & PDIST_KEY_ALIGNED_MASK);
}

// ---------------- Tiling 结构体 ----------------
BEGIN_TILING_DATA_DEF(PdistTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, rowNum)           // N：输入行数
  TILING_DATA_FIELD_DEF(uint64_t, colNum)           // M：特征维
  TILING_DATA_FIELD_DEF(uint64_t, totalPairs)       // N*(N-1)/2 输出元素总数
  TILING_DATA_FIELD_DEF(float,    pValue)           // p
  TILING_DATA_FIELD_DEF(float,    pReciprocal)      // 1/p，Host 预计算下发（防核内除法）
  TILING_DATA_FIELD_DEF(uint32_t, sceneIdx)         // 0~3，对应场景①~④
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)      // 实际启用核数
  TILING_DATA_FIELD_DEF(uint32_t, rowPerCore)       // 场景①④：每核基量行数
  TILING_DATA_FIELD_DEF(uint32_t, rowTailCore)      // 前 rowTailCore 个核各多分 1 行（尾均摊）
  TILING_DATA_FIELD_DEF(uint32_t, blockLen)         // 场景②：M 切块长度（元素数，未对齐）
  TILING_DATA_FIELD_DEF(uint32_t, blockAlignedLen)  // 场景②④：对齐后块长（元素数）
  TILING_DATA_FIELD_DEF(uint32_t, blockLoop)        // 场景②：块循环轮数
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)

}  // namespace optiling

#endif  // PDIST_TILING_H
