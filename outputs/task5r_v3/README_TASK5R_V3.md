# Task5R-v3 交付说明（2026-08-24）

## 当前状态
**verdict = BENCHMARK_NOT_HUMAN_VERIFIED, task6_allowed = false**
（`verdict.json`，门控链 1–4 PASS，第 5 关等待人工复核）

v2 结论已作废：`outputs/task5r/INVALIDATED_BY_TASK5R_V3.md`
（TASK5R_RESULTS_INVALIDATED_BY_IMPLEMENTATION_AUDIT）。

## source commits（实现+测试，两阶段提交之提交1）
- `de053f0` v3 核心修复 + 测试套件
- `78f2ad3` 分块切片 broadcast 修复（真实92k点视锥触发；新增分块等价回归测试）
- `cac1f49` v2 产物作废标记
- `a7efc98` D1 候选benchmark + 复核包 + 匹配 runner
- `e3da605` git_tree_dirty 忽略未跟踪文件（verdict 自指问题）
- `e9a1784` measured verdict（提交2：干净树上的轻量产物）

## 程序正确性证据
- 单元/语义/统计测试 **51/51 OK**：`test_logs/unittest_full.log`
- selftest（golden向量、ellipse-block vs 逐像素brute-force <1e-9、
  matcher确定性/分数盲、Cliff's delta≡2AUROC−1）：`selftest.json` all_passed=true

## 对齐验证（dense_alignment.json, overall_passed=true）
6 植物：COLMAP 轨道自重投影中位 0.57–1.17px（<3px）；SfM点→稠密高斯
最近邻中位 0.011–0.025m（<0.05m），0.05m 内比例 0.709–0.917（≥0.60）。
逐视图亮度掩码指标仅作诊断（欠曝视图 mean 10–25 会破坏任何固定/Otsu 掩码，
测量记录见脚本 docstring；未作为门控）。

## v3 viewsigs（version task5r-alpha-v3-ellipseblock）
| plant | cache_key | n_points | vis_frac |
|---|---|---|---|
| DouBanLv1 | dc2ccdded389a5fdbd50bf0f88f1e98c | 51745 | 0.271 |
| XianKeLai2 | 8d7e7d9c60728662ebf1d919fd846398 | 44264 | 0.308 |
| WanNianQing2 | 7cde7b55d838411b361960a53d1a40d7 | 40757 | 0.330 |
| HongZhang | aa66fbe4d52eb69202761495fdc7595f | 93043 | 0.268 |
| WangWenCao2 | 351426d9dae3f5a8e8ce4934cf22deed | 34761 | 0.406 |
| CaoMei1 | 44ed6ba336b734d935b2fb9ca0705464 | 37014 | 0.369 |

## 候选 benchmark（PROPOSER_DIAGNOSTIC，非 Gold）
冻结提议器参数不变 → 9 个候选对（DouBanLv1×5, HongZhang×3, WangWenCao2×1;
XianKeLai2/WanNianQing2/CaoMei1 叶级组件不足无候选对）。
全部建议 UNCERTAIN（无 <3cm 对）。
人工复核入口：`benchmark_review_queue.csv` + `benchmark_review_guide.md`
+ `review_crops/`（本地，gitignored）。

## 复核后的流程（本命令之外，需人工参与后执行）
```bash
PY=/home/test/biosoft/enter/envs/agri_re_py310/bin/python
# 1) 人工填写 reviewer_decision 列并生成 human_verification.json (approved=true)
# 2) 按复核决定过滤 candidate_benchmark 后运行匹配+统计：
$PY scripts/run_task5r_v3_separability.py     # 无人审时拒绝运行
# 3) 判决唯一出口：
$PY scripts/write_task5r_verdict.py
```

## 本地不入库的大文件
`outputs/task5r_v3/projection_cache/`（viewsig npz + 解码图像缓存）、
`outputs/task5r_v3/review_crops/`。历史膨胀（旧 outputs 已跟踪的 ~8127 个
npy/json 大文件）按用户要求本次不清理，仅在此记录。

## 复核范围决定（2026-08-24，用户确认）
接受数据边界：CaoMei1 / XianKeLai2 / WanNianQing2 本轮无候选对（提议器在丛生
形态融叶，非数据缺失），不参与本轮 benchmark；复核范围锁定
**DouBanLv1(5对) / HongZhang(3对) / WangWenCao2(1对，组件54疑茎样重点审)**。
若未来要覆盖丛生型植物，须开新一轮提案（修改并重新冻结提议器参数）。

---

# 最终判决（2026-08-24，全流程完成）

## verdict = SEPARABILITY_FAIL, task6_allowed = false
（first_failure: dev_gate_no_signal — 判决由 write_task5r_verdict.py 从测量产物自动产生）

## 完整门控链（verdict.json）
1. implementation_selftest PASS（51/51 测试 + selftest brute-force 对照<1e-9）
2. source_tree_clean PASS
3. viewsig_cache_consistency PASS（6 植物 v3 精确 key）
4. dense_alignment PASS
5. human_verification PASS（用户口头复核 9/9 KEEP，来源如实记录）
6. matching_gates PASS（1:1 formal: prev=0.500, 控制 AUROC=0.515, 无不可匹配；
   units dev:ALL n_cross=298 / heldout:ALL n_cross=351 ≥ 30 下限）
7. dev_gate_no_signal **FAIL** → SEPARABILITY_FAIL

## 科学结论（如实）
距离匹配协议成功去除了 v2 的距离混杂（R0 控制 AUROC：dev 0.528 /
heldout 0.512，均落入 [0.45,0.55] 零带）。去混杂后：

| 消融 | dev AUROC | heldout AUROC |
|---|---|---|
| R0_dist(控制) | 0.528 | 0.512 |
| R1_c_vis | 0.473 | 0.494 |
| R2_c_app | 0.453 | 0.540 |
| R3_c_occ | **0.626** | 0.493 |
| R4_c_mv | 0.463 | 0.505 |
| R5_surface | 0.477 | 0.592 |
| R6_mv_and_surface | 0.471 | 0.538 |

- **预设方向门（dev 上 R4/R6 |AUROC−0.5|≥0.05）未通过**：R4=0.463、R6=0.471，
  多视角复合特征在匹配后的边集上无信号；
- 唯一 dev 有信号的是 R3_c_occ（0.626，Cliff's δ=0.25），但 held-out 反转
  （0.493）——不构成可靠证据；heldout 的 R5_surface 0.592 因 CI 仅
  descriptive_only（n_clusters=4）不作 PASS 依据；
- cluster bootstrap 以接触对为单位（dev 5 对 / heldout 4 对）。

**多视角身份证据（c_vis/c_app/c_occ 及其组合）在真实数据、去距离混杂的
条件下不足以分离 within/cross 接触边。** 与 Task4（合成 MV 门控 FAIL）、
Task5（真实观测 Phase-0 饱和 FAIL）形成三重独立证据链：该方向在当前
数据/方法下不成立。task6 不允许启动。

## 样本量与统计效力声明
本轮有效样本为 9 个接触对（DouBanLv1×5 + HongZhang×3 + WangWenCao2×1，
其中 WWC2 对疑茎样碎片但经人工 KEEP）。CaoMei1/XianKeLai2/WanNianQing2 因
提议器融叶无候选对。此样本量下即使存在弱信号也可能检出不足——但预设方向
门在 dev 上即无信号（非边缘失败），且 held-out 未消耗（判决在第7关短路，
第8关 held-out once 未执行，保留其一次有效性）。
