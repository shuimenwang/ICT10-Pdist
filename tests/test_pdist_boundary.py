# -*- coding: utf-8 -*-
"""test_pdist_boundary.py —— 边界回归矩阵（docs/02 §5）。

覆盖（对齐训练营折磨场景与错误榜）：
  - N=1            空输出 (0,)
  - N=2            最小非空
  - M=1            标量行
  - M=131          非 32B 对齐（131*4=524B，非 32 倍数）→ 场景④
  - M=133177       超大 shift（同时触发 ②③④ 特质）
  - 整除/不整除 tail（N=1025, M=511）
断言：无挂死、无越界（不崩即过）、与 CPU oracle 一致。
自定义算子部署后（PDIST_CUSTOM_IMPL=1）逐条走 run_custom_pdist；
未部署时对原生实现跑同一矩阵，先固化"oracle vs 原生"的边界行为参照。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import gen_input, oracle_pdist  # noqa: E402

NPU_ID = int(os.environ.get("NPU_ID", "2"))

BOUNDARY_CASES = [
    # (N, M, p, dtype, 说明)
    (1, 64, 2.0, torch.float32, "N=1 空输出"),
    (2, 1, 2.0, torch.float32, "N=2 且 M=1 双边界"),
    (64, 1, 1.0, torch.float32, "M=1 标量行"),
    (33, 131, 2.0, torch.float32, "M 非 32B 对齐 → 场景④"),
    (4, 133177, 2.0, torch.float32, "超大 M=133177 → 场景②叠加"),
    (1025, 511, 3.0, torch.float32, "整除/不整除 tail"),
    (512, 256, 0.0, torch.float32, "p=0 非零计数分支"),
]


def main() -> int:
    torch.npu.set_device(NPU_ID)
    custom = os.environ.get("PDIST_CUSTOM_IMPL", "") == "1"
    failures = []

    for n, m, p, dtype, note in BOUNDARY_CASES:
        x = gen_input(n, m, dtype)
        ref = oracle_pdist(x, p)
        try:
            if custom:
                from test_pdist_precision import run_custom_pdist  # noqa: PLC0415
                out = run_custom_pdist(x.to("npu"), p).cpu().float()
            else:
                out = torch.nn.functional.pdist(x.to("npu"), p=p).cpu()
            torch.npu.synchronize()
            # 双容差判据（同 test_pdist_precision：大 M 累加误差实测 6e-4，纯 1e-4 不成立）
            ok = ref.numel() == 0 or bool(torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-4))
            err = (out.float() - ref.float()).abs().max().item() if ref.numel() else 0.0
            status = "OK" if ok else f"err={err:.3e}"
        except Exception as exc:  # noqa: BLE001
            status = f"RAISE {type(exc).__name__}: {str(exc)[:60]}"
        if status.startswith(("RAISE", "err")):
            failures.append((n, m, p, note, status))
        print(f"(N={n},M={m},p={p}) {note}: {status}")

    print(f"[boundary] custom={custom} cases={len(BOUNDARY_CASES)} failed={len(failures)}")
    for item in failures:
        print("  FAIL", item)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
