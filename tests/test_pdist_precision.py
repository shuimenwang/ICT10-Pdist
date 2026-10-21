# -*- coding: utf-8 -*-
"""test_pdist_precision.py —— 精度对拍（PyTorch oracle）。

两级模式（自动探测）：
  1) 自定义算子已部署（环境变量 PDIST_CUSTOM_IMPL 指向加载模块 / aclnn 就绪）
     → 全矩阵对拍：p x dtype x shape（赛题正式口径）
  2) 未部署 → 原生 torch_npu pdist 冒烟（仅 FP32），验证环境与 oracle 链路
     并明确提示：FP16 分支正是赛题缺口，Phase 1 后接入自定义算子。

验收线（docs/02 §3.3）：逐元素 |out-ref| <= atol + rtol*|ref|（默认 atol=1e-4, rtol=1e-4）。
实测依据（2026-10-20 冒烟）：NPU 累加误差随 M 增长（M=256 → 1.5e-4，M=133177 → 6.1e-4），
纯绝对误差线在大 M 不成立；任务书下发后按任务书口径替换 ATOL/RTOL。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import CASES_SMALL, P_VALUES, gen_input, oracle_pdist  # noqa: E402

ATOL = float(os.environ.get("PDIST_ATOL", "1e-4"))
RTOL = float(os.environ.get("PDIST_RTOL", "1e-4"))

NPU_ID = int(os.environ.get("NPU_ID", "2"))


def run_custom_pdist(x_npu: torch.Tensor, p: float) -> torch.Tensor:
    """自定义 Ascend C 算子调用入口（Phase 1 接入后启用）。

    接入方式（按 init_project.sh 生成的 aclnn 工程为准）：
      import paddle_npu? no —— 使用生成的 aclnn Python 封装或 torch 自定义算子注册：
      torch.ops.load_library("libcustom_pdist.so") 后调用 torch.ops.custom.pdist(x, p)
    未接入前调用本函数应抛 RuntimeError，测试自动降级为原生冒烟。
    """
    raise RuntimeError("custom op not deployed yet")


def custom_ready() -> bool:
    return os.environ.get("PDIST_CUSTOM_IMPL", "") == "1"


def main() -> int:
    torch.npu.set_device(NPU_ID)
    failures, total = [], 0

    if custom_ready():
        mode = "custom-op"
        for n, m in CASES_SMALL:
            for p in P_VALUES:
                for dtype in (torch.float32, torch.float16):
                    total += 1
                    x = gen_input(n, m, dtype)
                    ref = oracle_pdist(x, p)
                    try:
                        out = run_custom_pdist(x.to("npu"), p).cpu().float()
                    except Exception as exc:  # noqa: BLE001
                        failures.append(((n, m, p, str(dtype)), f"raise: {exc}"))
                        continue
                    ref = ref.float()
                    if ref.numel() == 0:  # N=1 空输出：只校验 shape
                        err, ok = 0.0, out.numel() == 0
                    else:
                        err = (out - ref).abs().max().item()
                        ok = bool(torch.allclose(out, ref, atol=ATOL, rtol=RTOL))
                    if not ok:
                        failures.append(((n, m, p, str(dtype)), f"abs_err={err:.3e}"))
    else:
        mode = "native-smoke (fp32 only)"
        for n, m in CASES_SMALL:
            x = gen_input(n, m, torch.float32)
            for p in (2.0, 1.0):
                total += 1
                ref = oracle_pdist(x, p)
                out = torch.nn.functional.pdist(x.to("npu"), p=p).cpu()
                err = (out - ref).abs().max().item() if ref.numel() else 0.0
                ok = ref.numel() == 0 or bool(torch.allclose(out.float(), ref, atol=ATOL, rtol=RTOL))
                if not ok:
                    failures.append(((n, m, p, "fp32"), f"abs_err={err:.3e}"))

    print(f"[precision] mode={mode} total={total} failed={len(failures)}")
    for case, msg in failures:
        print(f"  FAIL {case}: {msg}")
    if not custom_ready():
        print("[precision] 提示：自定义算子未部署（PDIST_CUSTOM_IMPL=1 启用全矩阵对拍）。"
              "原生 FP16 可用但精度差（见 results/baseline_*.md），自研 FP32 中间计算版可反超。")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
