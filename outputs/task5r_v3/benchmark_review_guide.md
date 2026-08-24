# Task5R-v3 基准人工复核指南

生成时间: 2026-08-24T16:32:10+0800 | viewsig: `task5r-alpha-v3-ellipseblock` |
提议器参数(冻结): `{'k': 16, 'sin_thr': 0.35, 'lin_thr': 0.6, 'plan_thr': 0.25, 'color_thr': 30.0}`

## 你在审什么
独立提议器（normal+color 区域生长，与 LeafFit 热求解器零共享代码路径）
把每株植物的稠密 3DGS 点云切成连通组件，并提出"近接触叶片对"候选
（min_gap < 0.08 m）。这些候选标签目前只是 **PROPOSER_DIAGNOSTIC**，
不是 Gold、不是正式 benchmark。

## 复核步骤
1. 打开 `benchmark_review_queue.csv`，逐行查看 `review_crops` 列列出的截图
   （位于 `review_crops/<plant>/`，每个 case 至多 6 张多视角接触区放大图）。
2. 判断两个组件是否为**两片真实存在的不同叶片**的接触：
   - 同一片叶子被错误切成两块 → RELABEL（合并）
   - 其中一块其实是茎/土/盆/漂浮噪声 → REJECT
   - 接触关系真实但几何上不构成"上下遮挡接触" → UNCERTAIN
   - 两片真实叶片的真实近接触 → KEEP
3. 在 `reviewer_decision` 列填 `KEEP` / `REJECT` / `RELABEL` / `UNCERTAIN`，
   需要时在 `reviewer_note` 写一句话理由。
4. 全部复核完成后，把本 CSV 转成交付格式并填写
   `outputs/task5r_v3/human_verification.json`：

```json
{
  "approved": true,
  "reviewer": "<姓名>",
  "date": "<YYYY-MM-DD>",
  "queue_csv_sha256": "<对已复核 CSV 的 sha256>",
  "decisions_summary": {"KEEP": N, "REJECT": N, "RELABEL": N, "UNCERTAIN": N}
}
```

`suggestions` 列只是机器建议，**不构成决定**；请以截图为准。
未完成人工复核前，Task5R-v3 的正式判决将停在
BENCHMARK_NOT_HUMAN_VERIFIED（task6_allowed=false）。
