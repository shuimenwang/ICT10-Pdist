# results/ —— 性能与基线归档

## 记录模板（每轮优化一条）

```markdown
## [YYYY-MM-DD] <优化项>
- 场景：<shape / p / dtype / shift>
- 基线耗时：xx ms → 本版：xx ms（Δxx%）
- PMU 关键指标：UB 利用率 xx% / 标量占比 xx%
- 归因与结论：<一句话>
- 数据文件：pmu_YYYYMMDD_*.csv
```

## 规则

- 每条记录必须附可复跑命令
- 性能结论以 3 次均值口径
- 基线冻结后只增不改，用新文件对比
