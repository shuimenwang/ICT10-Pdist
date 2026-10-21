# tests/ —— 测试与对拍

## 用例矩阵

| 脚本 | 覆盖 | 断言 |
|---|---|---|
| `test_pdist_precision.py` | p∈{0,1,2,3.5} × FP16/FP32 × 小数据 | 逐元素误差 ≤1e-4 |
| `test_pdist_boundary.py` | shift=133177、N=1、M=1、tail、非对齐、多核 | 无挂死、无越界、结果正确 |
| `test_pdist_perf.py` | 基线场景 ×3 取均值 | 对比 `results/` 基线 |

## 原则

- oracle 只认 PyTorch 原生 pdist 输出，独立于本实现
- 新场景先入回归再优化（防优化打崩老场景）
- 失败用例保留最小复现数据
