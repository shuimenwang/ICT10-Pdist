#!/usr/bin/env bash
# init_project.sh —— 用 msopgen 生成可编译算子工程（产物放 /cache，不占 work 磁盘）
# 用法：bash scripts/init_project.sh
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKROOT="${WORKROOT:-/cache/ict10-pdist}"
PROJ="$WORKROOT/pdist_project"

mkdir -p "$WORKROOT"
rm -rf "$PROJ"

echo "[1/2] msopgen 生成工程（json: op_host/pdist.json → ai_core-ascend910b, aclnn 框架）"
msopgen gen -i "$REPO/op_host/pdist.json" -f aclnn -c ai_core-ascend910b -out "$PROJ" -lan cpp

echo "[2/2] 生成完成：$PROJ"
cat <<'NOTE'

后续手工合入（一次性，约 10 分钟）：
  1) cp 本仓 op_host/pdist_tiling.h   → $PROJ/op_host/（或按生成工程的 include 路径放）
  2) 生成工程自带 *_tiling.cpp / 入口 cpp 模板 —— 以生成模板的注册宏为准，
     把本仓 pdist_tiling.cpp 的 TilingFunc 逻辑、pdist_kernel.cpp 的 Kernel 骨架合入对应文件；
     删除本仓独立注册块，避免双重注册。
  3) 编译：bash scripts/build.sh
  4) 若 msopgen 参数报错，按 `msopgen gen --help` 与 CANN 文档核对（json 字段名随版本微调）。

NOTE
