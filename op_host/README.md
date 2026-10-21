# op_host/ —— Host 侧（CPU，管策略）

## 职责

- shape 推断：X:(N,M) → Y:(N·(N−1)/2)
- Tiling 计算：四大场景决策（见 docs/02 §3.1 决策树）
- TilingKey 编码下发（与 Kernel 侧单源定义，防错误码 31361001）
- 标量预计算：`p_reciprocal`、`0`、`1` 随 Tiling 结构体下发
- 算子注册 / Json 配置 / 命名一致性（防错误 #5 命名错误）

## 约定

- 计划文件：`pdist_tiling.cpp/h`、`pdist.cpp`
- Tiling 结构体字段变更必须同步 Kernel 侧；提交前打印 Tiling 自检一次
- 可选参数 p 必须做空值判断（防 Host 段错误）
