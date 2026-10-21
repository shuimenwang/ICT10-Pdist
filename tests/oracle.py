# -*- coding: utf-8 -*-
"""oracle.py —— PyTorch 原生 pdist 作为唯一对拍基准（CPU 侧生成）。

原则（docs/02 §5）：oracle 只认 PyTorch 原生输出；数据用固定种子即时生成，
不落盘 golden 文件（可复现 + 不占 work 磁盘）。

用法（作为库）：
    from oracle import gen_input, oracle_pdist, CASES_SMALL
    x   = gen_input(64, 128, torch.float32)
    ref = oracle_pdist(x, p=2)
"""
import torch

# 语义依据：docs.pytorch.org torch.nn.functional.pdist
# 输出按行主序展开上三角 (i<j)：pair = i*(2N-i-1)/2 + (j-i-1)

# 精度用例矩阵（shape 覆盖：最小 / 边界 / 常规 / 偏大）
CASES_SMALL = [
    (1, 8), (2, 1), (2, 131), (64, 128), (512, 256),
]
P_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# 性能用例矩阵（对齐 docs/04 基线表）
CASES_PERF = [(512, 256), (1024, 512), (4096, 512)]


def gen_input(n: int, m: int, dtype: torch.dtype, seed: int = 42) -> torch.Tensor:
    """固定种子生成 (N, M) 输入，保证任何机器可复现同一数据。"""
    g = torch.Generator().manual_seed(seed + n * 1_000_003 + m)
    x = torch.randn(n, m, generator=g)
    if dtype == torch.float16:
        # 用有限幅值避免 fp16 溢出引入的假误差
        x = (x * 2.0).clamp(-8.0, 8.0)
    return x.to(dtype)


def oracle_pdist(x_cpu: torch.Tensor, p: float) -> torch.Tensor:
    """CPU 侧 PyTorch 原生输出（FP32 口径），全项目唯一 oracle。"""
    return torch.nn.functional.pdist(x_cpu.float(), p=p)


def pair_index(n: int, i: int, j: int) -> int:
    """上三角行主序输出索引，Host Tiling 与 Kernel 写回共用同一公式。"""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


if __name__ == "__main__":
    # 冒烟：oracle 自身可用性 + 索引公式自检
    for n, m in CASES_SMALL:
        x = gen_input(n, m, torch.float32)
        for p in P_VALUES:
            y = oracle_pdist(x, p)
            assert y.shape[0] == n * (n - 1) // 2, (n, m, p, y.shape)
    n = 5
    flat = {(i, j) for i in range(n) for j in range(i + 1, n)}
    assert len({pair_index(n, i, j) for i, j in flat}) == n * (n - 1) // 2
    print(f"oracle OK: {len(CASES_SMALL)} shapes x {len(P_VALUES)} p-values, "
          f"pair_index formula verified")
