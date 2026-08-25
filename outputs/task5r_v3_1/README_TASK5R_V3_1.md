# Task5R-v3.1 — 纠偏后重生成产物说明

状态：**BENCHMARK_NOT_HUMAN_VERIFIED**（`verdict.json`，task6_allowed=false）。
10 个候选接触对等待 single-reviewer 人工复核；复核通过前所有标签仅为
PROPOSER_DIAGNOSTIC，不构成确认性 held-out 验证。

## 与 v3 的关系

v3 verdict 已程序化作废：`outputs/task5r_v3/verdict_superseded.json`
（INVALIDATED_BY_IMPLEMENTATION_AUDIT，first_failure_audit=rgb_peak_block_argmax）。
v3 的一切数值结论不得再引用。

## v3.1 修复内容

1. **RGB 峰值块 groupwise argmax**（core/observation_identity.py）：v3 取的是
   每组贡献最小行（lexsort run-last 错误）；v3.1 用 `np.maximum.at` 求真峰值，
   tie-break 冻结为组内排序首个（最小 block id）。暴力对照：单测 10 seed +
   selftest 20 trials 全部一致。
2. **MAX_RADIUS_PX 一致化（方案A）**：裁剪半径同时约束候选块枚举范围与
   Mahalanobis 接受域；meta 新增 `n_gaussians_radius_clipped`
   （6 株 5258–26254）。
3. **统计门控**：正式点估计 = contact-pair macro AUROC；pooled-edge AUROC 仅作
   descriptive 匹配质量诊断（字段 `pooled_auroc_descriptive`，永不门控）。
   CI/bootstrap 以 pair 为重采样单位。最小对数 K=205 由
   `scripts/power_analysis_min_pairs.py` 在任何 v3.1 测量之前预冻结
   （pilot = 已消耗的 v3 探索性运行，per-pair var 0.1334；
   `min_pairs_freeze.json`）。
4. **held-out 方向变换**：`signed = auc if sign>=0 else 1-auc`
   （v3 的 `auc*-1` 数学错误）。
5. **上游 manifest 链校验**：viewsig（last-row-per-plant wins）、benchmark
   manifest、matched gates、dense alignment、selftest 任一 dirty 或源码不一致
   → IMPLEMENTATION_INVALID。两阶段提交合法：ancestor 且 core/scripts/tests
   diff 为空。`git_tree_dirty` 排除 outputs/**（产物自污染修正）。
6. **held-out 消耗标记**：旧 v3 held-out 对标 consumed/exploratory；v3.1 候选
   经身份 diff 后禁止自动迁移旧复核决定（见 candidate_migration_audit.md）。

## 候选身份 diff 结论（candidate_migration_audit.md）

- 完全一致 2 对（可参考旧意见，决定不得自动继承）：
  HongZhang_c2_c1、WangWenCao2_c0_c54（组件54"疑茎样"不确定性保留）
- **新出现 8 对**（此前从未查看/计算/解释，构成潜在一次性确认性材料）：
  dev=DouBanLv1_c2_c4 / c3_c2 / c3_c4 / c3_c542 / c69_c2520 / c3_c69；
  heldout=HongZhang_c1_c1356 / c2_c98
- v3 独有 7 对已随 v3 作废，不再评测

→ `benchmark_review_queue.csv`（10 行全 UNCERTAIN）+ review_crops/（60 张）
需人工逐对填写 KEEP/RELABEL/REJECT；填毕生成 human_verification.json
(approved=true) 后方可运行 run_task5r_v3_separability.py。

## 溯源

- HEAD：1154fac；六株 viewsig 记录与算法溯源均为同一干净 commit 链
  （cache-hit 刷新至当前 HEAD，npz 内容寻址等价）；
- dense_alignment 6 株 PASS（reproj 中位 0.57–1.17 px，NN 中位 0.0107–0.0250 m）;
- selftest 6 checks 全过；测试套件 61 项全过
  （test_logs/unittest_full.log）；
- matched_edges_*.csv、summary、review queue SHA256 由 verdict 写入 provenance
  （当前因未过人审门尚未生成测量类 CSV，provenance 门在人审之后到达）。

## 措辞约束

本文档及后续汇报不使用："三重独立证据链"、"可信的最终否定答案"、
"非边缘失败"、"held-out 未消耗"。benchmark 统一称
single-reviewer human-confirmed benchmark。无信号结果表述为
SEPARABILITY_NOT_DEMONSTRATED（非"已证明无信号"）。

---

## 更新（2026-08-25 晚）：人审通过后正式测量结果

用户 single-reviewer 复核：9 KEEP / 1 REJECT（DouBanLv1_c3_c4）。
正式测量 4,304,894 edges（匹配后 dev 564 / heldout 1008 进分析）。

**verdict = SEPARABILITY_NOT_DEMONSTRATED**（first_failure=heldout_gate_once，
task6_allowed=false；HEAD 82364a2，链校验全 PASS）。

- 正式点估计（pair macro AUROC）：dev R4_c_mv=0.6091 [CI 0.351–0.867]；
  heldout R4_c_mv=0.4488 [CI 0.276–0.675] < 冻结阈值 0.55 → 门停。
- 描述性（不门控）：pooled dev 0.4457 / heldout 0.3465。
- 失败模式 = preset_direction_failed（CI 未整体低于 0.5，非反向信号证明）。
- 样本门：9 确认对 << 预冻结 K=205；即使方向一致也不足以宣称可复现分离。
- heldout 内部异质：WangWenCao2_c0_c54 单对 AUROC 0.806，HongZhang 三对
  0.27–0.44——heldout 方向由植株主导而非方法主导。
- 结论措辞：多视角观测身份特征在本次 single-reviewer 确认的 9 对基准上
  **未能展示**跨株分离能力；这不构成"已证明无信号"。
