# Task5R-v3 INVALIDATED by implementation audit (v3.1)

Original verdict.json sha256: `84e07dbd4ebd3974e0033eb54b41e11e4aef7475e3e792463aa788f2c76c8604`
Original verdict: **SEPARABILITY_FAIL**, first_failure: `dev_gate_no_signal`

The SEPARABILITY_FAIL verdict of v3 must NOT be cited as a final scientific conclusion. Audit findings:

1. **rgb_peak_block_argmax** (core/observation_identity.py:448-467 (v3)) — RGB peak-block selection lexsort'ed by (-contrib, loc) and took the LAST row of each group run, which selects the group-MINIMUM contribution, not the argmax. rgb_views / rgb_valid were therefore keyed to the WORST block.
2. **max_radius_not_enforced_in_enumeration** (core/observation_identity.py:303-308 (v3)) — MAX_RADIUS_PX clipped only the REPORTED radius (cov2d_radius_px); _ellipse_block_pairs enumerated candidates from the unclipped covariance extent, so the manifest's footprint_radius_clip claim was false.
3. **pooled_edge_statistics_as_formal** (scripts/summarize_task5r_v3.py (v3) + verdict gate) — Pooled-edge AUROC was used as the formal gate statistic although contact pairs are the independent inference unit; no pair-macro point estimate or per-pair table was produced.
4. **heldout_sign_transform_error** (scripts/write_task5r_verdict.py:181 (v3)) — Held-out signed AUROC computed as auc * (-1) for frozen-negative directions; correct transform is 1 - auc (AUROC lives on [0,1]).

Superseding run: Task5R-v3.1 (outputs/task5r_v3_1/), VISIBILITY_VERSION task5r-alpha-v3.1-rgbargmax.
task6_allowed remains FALSE regardless of any downstream result.