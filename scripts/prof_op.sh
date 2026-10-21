#!/usr/bin/env bash
# prof_op.sh —— 算子级性能采集（msprof / msopprof，PMU 真实上板数据）
# 用法：bash scripts/prof_op.sh <被测脚本.py> [输出目录]
# 默认被测脚本：tests/test_pdist_perf.py
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$REPO/tests/test_pdist_perf.py}"
OUTDIR="${2:-/cache/ict10-pdist/prof/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

echo "[prof] 目标：$TARGET"
echo "[prof] 输出：$OUTDIR"

# 优先算子级工具（数据更细）；不可用时退回 msprof 全量采集
if command -v msopprof >/dev/null 2>&1; then
    msopprof "$TARGET" --output "$OUTDIR"
else
    msprof --application="$TARGET" \
           --output="$OUTDIR" \
           --aic-metrics=PipeUtilization,ArithmeticUtilization,Memory,L2Cache \
           --analysis=True
fi

# 归档要点（人工摘录进 results/）：
#   aicore__pipe_utilization.csv  → UB 利用率（<80% 即不达标，Phase 3 门禁）
#   流水图 → 定位「搬运等待 / 计算阻塞 / 标量占比」
echo "[prof] 完成。请把 UB 利用率与瓶颈结论按 results/README.md 模板记入 results/"
