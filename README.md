# ICT 挑战赛 · Pdist 算子攻坚项目

## 定位

在昇腾 910B3 上用 Ascend C 实现并深度优化对标 PyTorch `pdist` 的算子，**以第十届初赛真实赛题（方案提交 + 性能排名）为交付标准**：产出一套可上板的高性能算子工程、一份达标初赛评审的技术方案书、一组带归因链的性能数据。

## 赛题事实

- 第十届挑战赛初赛为**线上方案提交制**：公布 4 道赛题，队伍提交优化方案答卷，评委评分定晋级。第 2 题即 Pdist 算子优化（昇腾 NPU，达到任务书精度，按性能排名）。
- 2026-2027 赛季（第十一届）挑战赛：报名 2026-08-31~10-31，**初赛 2027-02**，中国总决赛 2027-04。
- 本机实测：torch_npu 原生 pdist (4096,512) FP32 单次 133ms、各 p 值耗时相同（通用路径未优化）、**p=0 慢 15.6 倍**、FP16 精度差（p=1 差 0.125）——**原生实现的四条明确短板，就是本赛题的攻坚点与排名空间**。

## 目录导览

| 目录 | 职责 | 状态 |
|---|---|---|
| `docs/` | 赛题解读 / 架构与 WBS / 排查手册 / 技术方案骨架 | 已完成 |
| `op_host/` | Host 侧：算子原型 json、shape 推断、Tiling、TilingKey（单源） | 骨架已就位 |
| `op_kernel/` | Kernel 侧：Ascend C 设备代码（四场景 / 双缓冲 / ln+exp） | 骨架已就位 |
| `tests/` | oracle 生成、精度对拍、边界矩阵、性能基线（**当前即可运行**） | 可运行 |
| `scripts/` | 工程初始化（msopgen）、编译、性能采集 | 可运行 |
| `results/` | 基线数据、PMU 归档、版本对比 | 持续追加 |
| `notes/` | 训练营笔记、复盘、答辩材料 | 持续追加 |

## 快速开始

```bash
# 0) 环境：CANN 9.1.0 + torch_npu 2.7.1.post2，4×910B3（本机已具备）

# 1) 生成 golden 数据 + 跑通原生算子基线（今天就能出第一张基线表）
python tests/oracle.py --small                    # CPU 侧 oracle，几分钟内完成
python tests/test_pdist_perf.py                   # 原生 pdist NPU 基线 → results/

# 2) 初始化可编译算子工程（msopgen 生成 CMake/工程文件，产物放 /cache 不占 work 磁盘）
bash scripts/init_project.sh

# 3) 编译部署（自定义算子包安装到个人 OPP 路径）
bash scripts/build.sh

# 4) 精度对拍（PyTorch oracle）→ 边界回归 → 性能采集
python tests/test_pdist_precision.py
python tests/test_pdist_boundary.py
bash scripts/prof_op.sh
```

> 磁盘纪律：`/home/ma-user/work` 仅余约 4.6G，**所有 build/缓存一律放 `/cache`**（脚本已内置），源码与文档放本项目。



