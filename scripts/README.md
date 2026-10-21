# scripts/ —— 工具链脚本（占位，待环境确定后落地）

| 计划脚本 | 用途 |
|---|---|
| `build.sh` | 编译 op_host + op_kernel，产出算子二进制 |
| `deploy.sh` | 部署到运行环境 / 刷新 OPP 路径 |
| `prof_op.sh` | msprof_op 采集 PMU 数据与流水图 |
| `log_grep.sh` | plog/tlog 关键词检索（错误码 / UB addr / repeat_times / core id） |

> 命令模板见根 README「快速开始」；落地时保持脚本名与全队口令一致。
