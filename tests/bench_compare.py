#!/usr/bin/env python3
"""bench_compare.py —— 三方性能对比：自研算子 vs torch_npu 原生 vs PyTorch CPU

用法：
  python tests/bench_compare.py --tag v0_naive
  python tests/bench_compare.py --tag v1_generic --cases 512,256 1024,512

产出：results/perf_<tag>.csv + results/perf_<tag>.md
口径：每项 >=3 次取均值；自研算子由 aclnn 驱动计时（含 Host 侧组装），
      原生/CPU 由 torch 计时；三者均为单次前向端到端。
"""
import argparse
import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import gen_input  # noqa: E402
from run_custom import run_case  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
NPU_ID = int(os.environ.get("NPU_ID", "0"))

P_VALUES = [0.0, 1.0, 2.0, 3.7]
DEFAULT_CASES = [(512, 256), (1024, 512), (4096, 512)]


def bench_torch(fn, repeat: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    if torch.npu.is_available():
        torch.npu.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    if torch.npu.is_available():
        torch.npu.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="版本标签，例如 v0_naive / v1_generic")
    ap.add_argument("--cases", nargs="*", default=None, help="形如 512,256 1024,512")
    ap.add_argument("--p", nargs="*", type=float, default=P_VALUES)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    cases = DEFAULT_CASES
    if args.cases:
        cases = [tuple(int(v) for v in c.split(",")) for c in args.cases]

    torch.npu.set_device(NPU_ID)
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for n, m in cases:
        tdtype = torch.float16 if args.dtype == "fp16" else torch.float32
        x_cpu = gen_input(n, m, tdtype)
        x_npu = x_cpu.to("npu")
        for p in args.p:
            try:
                r = run_case(n, m, p, args.dtype, args.iters)
                custom_ms, ok = r["mean_ms"], r["ok"]
            except Exception as exc:  # noqa: BLE001
                custom_ms, ok = None, False
                print(f"custom FAIL {n}x{m} p={p}: {exc}")

            native_ms = bench_torch(lambda: torch.nn.functional.pdist(x_npu, p=p))
            cpu_ms = None
            if not args.skip_cpu:
                cpu_ms = bench_torch(lambda: torch.nn.functional.pdist(x_cpu, p=p), repeat=1, warmup=0)

            rows.append([n, m, args.dtype, p, custom_ms, native_ms, cpu_ms, ok])
            cpu_s = "—" if cpu_ms is None else f"{cpu_ms:.3f}"
            print(f"({n},{m}) {args.dtype} p={p}: custom={custom_ms} native={native_ms:.3f} "
                  f"cpu={cpu_s} ok={ok}")
        del x_npu
        torch.npu.empty_cache()

    tag = f"{args.tag}_{date.today().strftime('%Y%m%d')}"
    csv_path = RESULTS / f"perf_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "M", "dtype", "p", "custom_ms", "native_npu_ms", "cpu_ms", "correct"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3],
                        "" if r[4] is None else f"{r[4]:.3f}",
                        f"{r[5]:.3f}",
                        "" if r[6] is None else f"{r[6]:.3f}",
                        r[7]])

    md_path = RESULTS / f"perf_{tag}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 性能对比：{args.tag}（{date.today()}，910B3 x1）\n\n")
        f.write("- 口径：custom = aclnn 驱动 end-to-end（含 Host 组装）均值；"
                "native = torch_npu 原生 pdist；cpu = PyTorch CPU\n")
        f.write(f"- 复跑：`python tests/bench_compare.py --tag {args.tag}`\n\n")
        f.write("| N | M | dtype | p | 自研 ms | torch_npu ms | PyTorch CPU ms | 加速比(vs native) | 正确 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            speed = "—" if not r[4] else f"{r[5] / r[4]:.2f}x"
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | "
                    f"{'—' if r[4] is None else f'{r[4]:.3f}'} | {r[5]:.3f} | "
                    f"{'—' if r[6] is None else f'{r[6]:.3f}'} | {speed} | {'PASS' if r[7] else 'FAIL'} |\n")
    print(f"已写入 {csv_path} / {md_path}")


if __name__ == "__main__":
    main()
