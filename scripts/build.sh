#!/usr/bin/env bash
# build.sh —— 把本仓源码同步进 msopgen 生成的工程并编译，产出个人算子包
# 依赖：先执行 scripts/init_project.sh
# 磁盘纪律：build 一律放 /cache（work 盘仅余 ~4.6G）
# 用法：bash scripts/build.sh [OPP_INSTALL_DIR]
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKROOT="${WORKROOT:-/cache/ict10-pdist}"
PROJ="$WORKROOT/pdist_project"
BUILD_DIR="$PROJ/build_out"
OPP_INSTALL_DIR="${1:-${OPP_INSTALL_DIR:-$WORKROOT/opp_install}}"

[[ -d "$PROJ" ]] || { echo "工程不存在，先执行 bash scripts/init_project.sh"; exit 1; }

echo "[1/4] 同步源码 → 生成工程"
cp "$REPO/op_host/pdist.cpp"        "$PROJ/op_host/pdist.cpp"
cp "$REPO/op_kernel/pdist.cpp"      "$PROJ/op_kernel/pdist.cpp"
cp "$REPO/op_kernel/pdist_tiling.h" "$PROJ/op_kernel/pdist_tiling.h"

echo "[2/4] CMake 配置"
cmake -S "$PROJ" -B "$BUILD_DIR" --preset=default -DASCEND_COMPUTE_UNIT=ascend910b >/dev/null

echo "[3/4] 编译 + 打 run 包"
cmake --build "$BUILD_DIR" --target binary package -j "$(nproc)"

echo "[4/4] 解包安装到 $OPP_INSTALL_DIR"
mkdir -p "$OPP_INSTALL_DIR"
RUN_PKG="$(ls -t "$BUILD_DIR"/*.run 2>/dev/null | head -1)"
if [[ -z "$RUN_PKG" ]]; then
  echo "未找到 .run 产物，检查编译输出"; exit 1
fi
bash "$RUN_PKG" --quiet --install-path="$OPP_INSTALL_DIR" 2>&1 | tail -5 || true

cat <<NOTE

安装完成。测试驱动加载个人算子包：
  source scripts/env.sh
  python tests/run_custom.py --shape 64,128 --p 2
NOTE
