# TASK5R_RESULTS_INVALIDATED_BY_IMPLEMENTATION_AUDIT

**本目录所有科学结论自 2026-08-24 起作废（task6_allowed=false）。**

v2 实现审计确认的缺陷（详见 leaf_fit 仓库 commit de053f0 的 Task5R-v3 修复）：

1. grouped alpha transmittance 用了 inclusive 前缀，首贡献者 T_before = 1/(1−α) ≠ 1；
2. 投影协方差半径公式缺 tr/2 项，系统性低估 λ_max；
3. footprint 半径只存储、未参与遮挡计算（固定 4px 中心分桶）；
4. Cliff's delta 使用前 4000 条有序样本截断且忽略 ties；
5. edge-level bootstrap 忽略植物/叶片/接触对聚类；
6. within 占比 ~99.985%，R0 AUROC 0.98–0.99 全由距离混杂驱动（未做距离匹配）；
7. 最终判决硬编码在 write_task5r_final_report.py:31-32；
8. 旧产物记录的 source commit 中不存在 Task5R 代码；缓存选择用 sorted(glob)[-1]。

替代产物：`outputs/task5r_v3/`（版本 task5r-alpha-v3-ellipseblock，
source commit de053f00dd38e16a9d82b2448bacc62c0bae1f45）。
本目录文件按用户要求保留不删除，仅作历史记录，禁止引用其结论。
