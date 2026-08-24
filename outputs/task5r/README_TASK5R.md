# Task5R Final Report
Git commit `984ae4d8c9aa8d2e60ba0d753b8cb912b14dfef5` date 2026-08-23

Audited commit: `984ae4d8c9aa8d2e60ba0d753b8cb912b14dfef5`
Previous margin=0 verdict: FAIL (negative result)
Audit verdict: INVALIDATED_BY_IMPLEMENTATION_AUDIT
New verdict: SEPARABILITY_FAIL
task6_allowed: False

## Phase 0 findings (8 blocking)
- F1: frustum membership used as visibility
- F2: no occlusion/transmittance calculation anywhere
- F3: RGB sampled for occluded points
- F4: transformed xyz paired with stale ViewSignature
- F5: real-image/counterfactual-geometry mismatch (benchmark invalidity)
- F6: pixel/NDC unit mismatch in c_occ
- F7: hard-coded final report with fixed statistics and pre-written conclusion
- F8: absence of Task 5 tests

## Benchmark (observation-matched, natural contacts)
Split by PLANT: dev=['DouBanLv1']  held-out=['HongZhang']
Frozen proposer {'k': 16, 'sin_thr': 0.35, 'lin_thr': 0.6, 'plan_thr': 0.25, 'color_thr': 30.0, 'dist_cap': '2.5x median nn', 'contact_gap_max_m': 0.08}
Pairs: dev 5  held 2
Per-plant audit (reprojection <2px, same capture, no synthetic transforms):
- DouBanLv1: PROPOSED comps 10498 pairs 5  PASS (<2px median)
- XianKeLai2: PROPOSED comps 15971 pairs 0  PASS (<2px median)
- WanNianQing2: PROPOSED comps 8727 pairs 0  PASS (<2px median)
- HongZhang: PROPOSED comps 19554 pairs 2  PASS (<2px median)
- WangWenCao2: PROPOSED comps 6702 pairs 1  PASS (<2px median)
- CaoMei1: PROPOSED comps 8074 pairs 0  PASS (<2px median)

## Phase 5 separability gate
Protocol: 32-NN within two-leaf union, geometry only  source corrected occlusion-aware real viewsig (core.observation_identity)  bootstrap B=500
R4 c_mv (0.4c_vis+0.3c_app+0.3c_occ)  dev mean AUROC 0.438  held mean 0.353  (chance 0.5)
Per-pair R4:
- D1-p2-11 n_cross 141  AUROC 0.374  CI [0.337, 0.410]  lift 1.000
- D1-p2-3 n_cross 284  AUROC 0.382  CI [0.346, 0.410]  lift 1.000
- D1-p3-518 n_cross 10  AUROC 0.345  CI [0.272, 0.417]  lift 1.000
- D1-p66-2395 n_cross 4  AUROC 0.578  CI [0.341, 0.816]  lift 1.000
- D1-p3-66 n_cross 4  AUROC 0.510  CI [0.249, 0.666]  lift 1.000
- HZ-p2-1 n_cross 34  AUROC 0.343  CI [0.251, 0.433]  lift 1.000
- HZ-p2-94 n_cross 97  AUROC 0.364  CI [0.322, 0.416]  lift 1.000

Full ablation (mean AUROC, within=positive):
- dev: R0_dist 0.983, R1_c_vis 0.428, R2_c_app 0.444, R3_c_occ 0.710, R4_c_mv 0.438, R5_surface 0.616, R6_mv_and_surface 0.548
- heldout: R0_dist 0.990, R1_c_vis 0.296, R2_c_app 0.419, R3_c_occ 0.580, R4_c_mv 0.353, R5_surface 0.581, R6_mv_and_surface 0.496

Gate decision: R0 (distance, control) separates within from cross by construction (mean AUROC ~0.98-0.99); 
R1-R4 (real-observation identity) do NOT systematically rank within > cross (dev 0.42-0.44, held 0.35, CI on held upper bound still <0.43 on the largest pairs). 
R3 (occlusion alone) is the best identity cue on dev (0.71) but collapses held-out (0.58) and direction is inconsistent.
Conclusion: SEPARABILITY_FAIL — corrected occlusion-aware real viewsig carries no usable edge-level leaf-identity signal; downstream B0-B4 blocked by Phase-5 gate.

## Answers to the five required questions
1) Was previous margin=0 a signal or artifact?  Task 5 pipeline was INVALIDATED (F1-F6); old margin=0 carried no evidential weight.
2) Do corrected real observations separate within from cross at edge level?  No (R4 held 0.35, below chance; AUPRC lift ~1.0 on prevalence 0.999 reflects class-imbalance artefact, not signal).
3) Does a valid observation-matched benchmark exist?  Yes but small: dev=DouBanLv1 (5), held-out=HongZhang (2) from frozen independent proposer.
4) Verified method failure or unavailable evidence?  The gate itself fails — no separable identity signal is demonstrated — so downstream METHOD_FAIL is not reached.
5) Sufficient evidence to redesign the geodesic prior?  No; the negative is gate-level, not a downstream grouping failure.
