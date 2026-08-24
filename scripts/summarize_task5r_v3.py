#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 Phase D3 — separability summary from MATCHED edges.

Reads matched_edges_1to1.csv (formal) + edge_scores_v3.csv, computes per-
split/per-ablation AUROC (midrank), Cliff's delta (=2AUROC-1), AUPRC and
CLUSTER bootstrap CIs over CONTACT PAIRS (not edges). Writes
edge_separability_summary_v3.json in the schema the verdict gate expects:

  splits.dev.by_ablation / splits.heldout.by_ablation with keys:
    auroc, cliffs_delta, auprc, n_edges, n_within, n_cross,
    cluster_bootstrap {point, lo, hi, n_clusters, B, descriptive_only}
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

OUT = REPO / "outputs" / "task5r_v3"
ABLATIONS = ["R0_dist", "R1_c_vis", "R2_c_app", "R3_c_occ",
             "R4_c_mv", "R5_surface", "R6_mv_and_surface"]
B_BOOT = 1000


def main() -> int:
    scores = defaultdict(dict)      # (case,ga,gb) -> ablation -> score
    labels = {}
    meta = {}
    with open(OUT / "edge_scores_v3.csv") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["gauss_a"], r["gauss_b"])
            labels[key] = bool(int(r["label"]))
            meta[key] = r["case_id"].rsplit("_c", 1)[0] if False else None
            for a in ABLATIONS:
                scores[key][a] = float(r[a])

    matched = []
    with open(OUT / "matched_edges_1to1.csv") as f:
        for r in csv.DictReader(f):
            matched.append((r["case_id"], r["gauss_a"], r["gauss_b"]))

    by_split = {"dev": defaultdict(list), "heldout": defaultdict(list)}
    DEV = {"DouBanLv1"}
    n_missing = 0
    for cid, ga, gb in matched:
        key = (cid, str(ga), str(gb))
        if key not in scores:
            # gauss ids in the matched csv are absolute; scores keyed the same
            n_missing += 1
            continue
        plant = cid.rsplit("_c", 1)[0].rsplit("_", 1)[0]
        plant = cid.split("_c")[0]
        split = "dev" if any(cid.startswith(p + "_") for p in DEV) else "heldout"
        lab = labels[key]
        for a in ABLATIONS:
            by_split[split][a].append((scores[key][a], lab, key))

    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "matcher": "task5r-v3-distmatch-1v1",
               "n_matched_1v1": len(matched),
               "n_missing_in_scores": n_missing,
               "splits": {}}
    for split in ("dev", "heldout"):
        s_split = {"by_ablation": {}, "pairs": {}}
        for a in ABLATIONS:
            rows = by_split[split][a]
            if not rows:
                continue
            s = np.array([r[0] for r in rows])
            y = np.array([r[1] for r in rows])
            entry = {
                "n_edges": int(len(y)),
                "n_within": int(y.sum()), "n_cross": int((~y).sum()),
                "prevalence": round(float(y.mean()), 4),
                "auroc": round(float(auroc(s, y)), 4),
                "cliffs_delta": round(float(cliffs_delta(s[y], s[~y])), 4)
                if y.sum() and (~y).sum() else None,
                "auprc": round(float(auprc(s, y)), 4),
            }
            # cluster bootstrap over contact pairs: group edges by case_id
            pairs_map = defaultdict(list)
            for sc, lb, key in rows:
                pairs_map[key[0]].append((sc, lb))
            boot_rows = []
            for case, items in pairs_map.items():
                scs = np.array([i[0] for i in items])
                lbs = np.array([i[1] for i in items])
                if lbs.sum() and (~lbs).sum():
                    boot_rows.append((case, scs, lbs))
            if boot_rows:
                cb = cluster_bootstrap_auroc(
                    [(c, scs, lbs) for c, scs, lbs in boot_rows],
                    B=B_BOOT, seed=0)
                entry["cluster_bootstrap"] = {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in cb.items()}
            s_split["by_ablation"][a] = entry
        summary["splits"][split] = s_split
        print(f"[{split}]", {a: s_split['by_ablation'][a]['auroc']
                             for a in s_split['by_ablation']})

    (OUT / "edge_separability_summary_v3.json").write_text(
        json.dumps(summary, indent=2))
    print("WROTE", OUT / "edge_separability_summary_v3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
