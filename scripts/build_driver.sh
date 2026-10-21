#!/usr/bin/env bash
# build_driver.sh —— 编译 aclnn 调用驱动（产物与可执行文件都放 /cache）
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKROOT="${WORKROOT:-/cache/ict10-pdist}"
OPP_INSTALL_DIR="${OPP_INSTALL_DIR:-$WORKROOT/opp_install}"
VENDOR="$OPP_INSTALL_DIR/vendors/customize"
AUTOGEN="$WORKROOT/pdist_project/build_out/autogen"
BIN="$WORKROOT/bin"

[[ -d "$VENDOR" ]] || { echo "未找到个人算子包，先执行 bash scripts/build.sh"; exit 1; }
mkdir -p "$BIN"

g++ -std=c++17 -O2 "$REPO/tests/driver/pdist_driver.cpp" -o "$BIN/pdist_driver" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$AUTOGEN" \
    -I"$VENDOR/op_api/include" \
    -L"$VENDOR/op_api/lib" -lcust_opapi \
    -L"$ASCEND_HOME_PATH/lib64" -lascendcl \
    -Wl,-rpath,"$VENDOR/op_api/lib" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"

echo "驱动已编译：$BIN/pdist_driver"
