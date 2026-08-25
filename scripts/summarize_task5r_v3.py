#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3.1 Phase D3 — separability summary from MATCHED edges.

v3.1 changes (statistical governance):
  * contact pairs are the INDEPENDENT INFERENCE UNIT — every pair gets its
    own row in splits.{dev,heldout}.per_pair;
  * formal point estimate = PAIR MACRO AUROC (unweighted mean of per-pair
    AUROCs, matching core.task_stats.cluster_bootstrap_auroc's point);
  * pooled-edge AUROC is kept for comparison only, flagged
    descriptive_only and renamed pooled_auroc_descriptive;
  * cluster bootstrap resamples PAIRS (not edges), B=1000, percentile CI,
    n_clusters reported; CI with <5 clusters stays descriptive_only.

Writes edge_separability_summary_v3.json (same filename as v3; directory
differs: outputs/task5r_v3_1/).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "core")]
from core.task_stats import auroc, cliffs_delta, auprc, cluster_bootstrap_auroc  # noqa: E402

OUT = REPO / "outputs" / "task5r_v3_1"
ABLATIONS = ["R0_dist", "R1_c_vis", "R2_c_app", "R3_c_occ",
             "R4_c_mv", "R5_surface", "R6_mv_and_surface"]
B_BOOT = 1000


def main() -> int:
    scores = defaultdict(dict)      # (case,ga,gb) -> ablation -> score
    labels = {}
    with open(OUT / "edge_scores_v3.csv") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["gauss_a"], r["gauss_b"])
            labels[key] = bool(int(r["label"]))
            for a in ABLATIONS:
                scores[key][a] = float(r[a])

    matched = []
    with open(OUT / "matched_edges_1to1.csv") as f:
        for r in csv.DictReader(f):
            matched.append((r["case_id"], r["gauss_a"], r["gauss_b"]))

    by_split_pair = {"dev": defaultdict(list), "heldout": defaultdict(list)}
    DEV = {"DouBanLv1"}
    n_missing = 0
    for cid, ga, gb in matched:
        key = (cid, str(ga), str(gb))
        if key not in scores:
            n_missing += 1
            continue
        split = "dev" if any(cid.startswith(p + "_") for p in DEV) else "heldout"
        lab = labels[key]
        for a in ABLATIONS:
            by_split_pair[split][a].append((scores[key][a], lab, cid))

    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "matcher": "task5r-v3.1-distmatch-1v1",
               "statistics_version": "pair-macro-formal-v3.1",
               "n_matched_1v1": len(matched),
               "n_missing_in_scores": n_missing,
               "splits": {}}
    for split in ("dev", "heldout"):
        s_split = {"by_ablation": {}, "per_pair": {}}
        for a in ABLATIONS:
            rows = by_split_pair[split][a]
            if not rows:
                continue
            s = np.array([r[0] for r in rows])
            y = np.array([r[1] for r in rows])
            # ---- per-contact-pair table (independent inference units) ----
            pairs_map = defaultdict(list)
            for sc, lb, cid in rows:
                pairs_map[cid].append((sc, lb))
            per_pair_rows = []
            boot_rows = []
            for case, items in pairs_map.items():
                scs = np.array([i[0] for i in items])
                lbs = np.array([i[1] for i in items])
                has_both = bool(lbs.sum()) and bool((~lbs).sum())
                if has_both:
                    boot_rows.append((case, scs, lbs))
                per_pair_rows.append({
                    "case_id": case,
                    "n_edges": int(len(lbs)),
                    "n_within": int(lbs.sum()),
                    "n_cross": int((~lbs).sum()),
                    "auroc": round(float(auroc(scs, lbs)), 4) if has_both else None,
                })
            macro_point = None
            if boot_rows:
                macro_point = float(np.mean(
                    [auroc(scs, lbs) for _, scs, lbs in boot_rows]))
            entry = {
                "n_pairs": int(len(per_pair_rows)),
                "n_pairs_degenerate_excluded_from_macro":
                    int(sum(1 for p in per_pair_rows if p["auroc"] is None)),
                "macro_auroc": round(macro_point, 4) if macro_point is not None else None,
                "pooled_auroc_descriptive": round(float(auroc(s, y)), 4),
                "pooled_descriptive_only": True,
                "cliffs_delta_pooled_descriptive":
                    round(float(cliffs_delta(s[y], s[~y])), 4)
                    if y.sum() and (~y).sum() else None,
                "auprc_pooled_descriptive": round(float(auprc(s, y)), 4),
                "n_edges": int(len(y)),
                "n_within": int(y.sum()), "n_cross": int((~y).sum()),
            }
            if boot_rows:
                cb = cluster_bootstrap_auroc(
                    [(c, scs, lbs) for c, scs, lbs in boot_rows],
                    B=B_BOOT, seed=0)
                entry["cluster_bootstrap"] = {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in cb.items()}
                entry["cluster_bootstrap"]["unit"] = "contact_pair"
            entry["per_pair"] = per_pair_rows
            s_split["by_ablation"][a] = entry
        n_all_pairs = {a: s_split["by_ablation"][a]["n_pairs"]
                       for a in s_split["by_ablation"]} or {}
        s_split["pairs"] = n_all_pairs
        summary["splits"][split] = s_split
        print(f"[{split}]",
              {a: {"macro": s_split['by_ablation'][a]['macro_auroc'],
                   "pooled(desc)": s_split['by_ablation'][a]['pooled_auroc_descriptive']}
               for a in s_split['by_ablation']})

    (OUT / "edge_separability_summary_v3.json").write_text(
        json.dumps(summary, indent=2))
    print("WROTE", OUT / "edge_separability_summary_v3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
