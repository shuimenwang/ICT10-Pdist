# op_kernel/ —— Kernel 侧（NPU，管计算）

## 函数结构（对应 pds.md）

- `pdist_process`：差值 → 绝对值 → 幂（ln + muls + exp）→ 归约累加 → 中间结果按 index_id 写回
- `pdist_sum_process`：ln + muls(1/p) + exp → 最终距离

## 纪律（每条对应一类高频错误）

1. 中间计算一律 FP32（防结果错误，52.4%）
2. AllocTensor / FreeTensor 严格配对（防内存异常 / 流同步超时）
3. UB 偏移地址逐项验算 32B 对齐（防核挂死 507015）
4. repeat_times / count_num 打印自检，不超 UB 容量（防越界）
5. 核内完成累加与写回，写回地址按 index_id 唯一（防结果覆盖）
