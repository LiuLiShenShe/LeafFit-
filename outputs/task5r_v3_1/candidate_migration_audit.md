# Task5R-v3.1 候选身份迁移审计（candidate identity migration audit）

生成：scripts 自动 diff（v3.1 candidate_benchmark.json vs v3 candidate_benchmark.json）
比较键：无序组件对 (min_id, max_id)，忽略 dev/heldout 划分（划分冻结不变：
dev=DouBanLv1, heldout=HongZhang/WangWenCao2）。

## 结论

**新旧候选身份不完全一致 → 禁止自动迁移旧人工复核决定。**
v3.1 review queue 必须作为新一轮 single-reviewer 人工复核对象。

## 身份比对

- 完全一致（2 对，可参考旧复核意见但不得自动继承决定）:
  - HongZhang_c2_c1
  - WangWenCao2_c0_c54  ← 组件54"疑茎样"不确定性保留
- v3.1 新出现（8 对，此前从未被查看/计算/解释，构成潜在一次性确认性材料）:
  - DouBanLv1_c2_c4, DouBanLv1_c3_c2, DouBanLv1_c3_c4, DouBanLv1_c3_c542,
    DouBanLv1_c69_c2520, DouBanLv1_c3_c69 （dev 植株）
  - HongZhang_c1_c1356, HongZhang_c2_c98 （held-out 植株）
- v3 中存在但 v3.1 未再提出（7 对，已随 v3 作废，不参与 v3.1 评测）:
  - DouBanLv1_c2_c11, DouBanLv1_c2_c3, DouBanLv1_c3_c520, DouBanLv1_c67_c2439,
    DouBanLv1_c3_c67, HongZhang_c1_c0, HongZhang_c2_c97

## 差异原因

RGB argmax 修复改变了 rgb_views/rgb_valid → 提议器颜色特征漂移；
MAX_RADIUS_PX 生效改变可见性计数 → visibility_fraction 微调。
提议器参数本身未变（PROPOSER_PARAMS 冻结: k=16, sin_thr=.35, lin_thr=.6,
plan_thr=.25, color_thr=30）。

## 后续纪律

1. 新 queue 的 reviewer_decision 必须由人工逐行填写；程序禁止代填 KEEP。
2. 只有经人工 KEEP 的新接触对才可进入一次性确认性评估；评估一次后转 consumed。
3. 若最终未获得人工确认，报告必须声明"不存在未消耗的确认性 held-out 集"。
