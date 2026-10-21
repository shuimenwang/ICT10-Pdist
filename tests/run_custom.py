#!/usr/bin/env python3
"""run_custom.py —— 自定义 Pdist 算子的正确性对拍 + 性能采集入口

用法：
  python tests/run_custom.py --shape 64,128 --p 2
  python tests/run_custom.py --shape 4096,512 --p 2 --dtype fp32 --iters 20

对拍基准：torch.nn.functional.pdist（CPU，FP32 计算）。FP16 输入先转 FP32 再算基准，
          与被测算子"FP16 输入 / FP32 计算 / 原 dtype 输出"的口径一致。
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
WORKROOT = Path(os.environ.get("WORKROOT", "/cache/ict10-pdist"))
DRIVER = WORKROOT / "bin" / "pdist_driver"
OPP_INSTALL = WORKROOT / "opp_install"

RESULT_RE = re.compile(r"RESULT shape=(\d+)x(\d+) p=(\S+) dtype=(\S+) iters=(\d+) mean_ms=(\S+) best_ms=(\S+)")


def driver_env() -> dict:
    vendor = OPP_INSTALL / "vendors" / "customize"
    env = dict(os.environ)
    env["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor}:{env.get('ASCEND_CUSTOM_OPP_PATH', '')}"
    lib = str(vendor / "op_api" / "lib")
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_case(n: int, m: int, p: float, dtype: str, iters: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(n, m, dtype=torch.float32)
    if dtype == "fp16":
        x = x.half()

    with tempfile.TemporaryDirectory() as td:
        x_path = Path(td) / "x.bin"
        y_path = Path(td) / "y.bin"
        x.numpy().tofile(x_path) if dtype == "fp32" else x.numpy().tofile(x_path)
        cmd = [
            str(DRIVER), "--x", str(x_path), "--y", str(y_path),
            "--n", str(n), "--m", str(m), "--p", str(p),
            "--dtype", dtype, "--iters", str(iters), "--warmup", "3",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=driver_env())
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"driver 失败（returncode={proc.returncode}）")
        out = proc.stdout.strip()
        m_res = RESULT_RE.search(out)
        np_dtype = np.float16 if dtype == "fp16" else np.float32
        y_custom = torch.from_numpy(np.fromfile(y_path, dtype=np_dtype))
        if dtype == "fp16":
            y_custom = y_custom.half().float()

    # 基准：FP32 精确值。FP16 输出与"半精度量化后的基准"比对（ulp 口径）：
    # fp16 量化 1 ulp（|y|∈[4,8) 时 = 7.8e-3）必然超出绝对判据 1e-4，
    # 故按 torch.allclose(y, y_ref.half(), atol=1e-3, rtol=2e-3) 判定
    # （fp16 相对 ulp ≤ 2^-10 ≈ 9.8e-4 < rtol=2e-3，可容纳 ±1 ulp）
    y_ref = F.pdist(x.float(), p=float(p))
    if y_ref.numel() == 0:
        return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "mean_ms": None, "best_ms": None, "n": n, "m": m}

    y_ref_h = y_ref.half().float()
    diff = (y_custom - y_ref).abs()
    max_abs = diff.max().item()
    denom = y_ref.abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()
    if dtype == "fp16":
        ok = bool(torch.allclose(y_custom, y_ref_h, atol=1e-3, rtol=2e-3))
    else:
        ok = bool(torch.allclose(y_custom, y_ref, atol=1e-4, rtol=1e-4))
    mean_ms = float(m_res.group(6)) if m_res else None
    best_ms = float(m_res.group(7)) if m_res else None
    return {
        "ok": ok, "max_abs": max_abs, "max_rel": max_rel,
        "mean_ms": mean_ms, "best_ms": best_ms, "n": n, "m": m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="64,128", help="N,M")
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    n, m = (int(v) for v in args.shape.split(","))
    if not DRIVER.exists():
        raise SystemExit("驱动不存在，先执行 bash scripts/build_driver.sh")

    r = run_case(n, m, args.p, args.dtype, args.iters)
    status = "PASS" if r["ok"] else "FAIL"
    perf = f"mean={r['mean_ms']:.3f}ms best={r['best_ms']:.3f}ms" if r["mean_ms"] else "无计时"
    print(f"[{status}] N={n} M={m} p={args.p} dtype={args.dtype} "
          f"max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e} {perf}")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
