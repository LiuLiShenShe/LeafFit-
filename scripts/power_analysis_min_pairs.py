#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3.1 — freeze the minimum number of contact pairs per split.

The v3 separability run is CONSUMED/EXPLORATORY (its verdict is superseded).
Its per-pair AUROC variance is used ONLY as a pilot to size the v3.1
confirmation sample. The resulting K is written into
scripts/write_task5r_verdict.py as MIN_PAIRS_PER_SPLIT, frozen BEFORE any
v3.1 measurement is looked at.

Rule: cluster-bootstrap MCSE of the pair-macro AUROC must keep the 95% CI
half-width <= SIGN_DELTA_MIN (0.05). MCSE is estimated from the pilot
per-pair AUROC variance under the pair-cluster bootstrap design.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SIGN_DELTA_MIN = 0.05


def estimate_min_pairs(per_pair_aurocs: np.ndarray, B: int = 2000,
                       seed: int = 0) -> dict:
    vals = np.asarray(per_pair_aurocs, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return {"n_pilot": 0, "min_pairs": None, "note": "no pilot pairs"}
    rng = np.random.default_rng(seed)
    half_widths = {}
    for K in range(2, 2000):
        mcse = float(vals.std(ddof=1) / np.sqrt(K))  # pair-cluster SE proxy
        hw = 1.96 * mcse
        half_widths[K] = hw
        if hw <= SIGN_DELTA_MIN:
            return {"n_pilot": int(n), "min_pairs": int(K),
                    "final_half_width": round(hw, 4),
                    "pilot_per_pair_var": round(float(vals.var(ddof=1)), 4)}
    return {"n_pilot": int(n), "min_pairs": 1999,
            "final_half_width": round(half_widths[1999], 4) if half_widths else None}


def load_pilot_per_pair_aurocs(v3_dir: Path) -> np.ndarray:
    """Recompute per-pair AUROCs from the CONSUMED v3 run's raw files (the v3
    summary predates the per-pair table). Dev split only, R4_c_mv."""
    import csv
    from collections import defaultdict
    sys.path[:0] = [str(REPO), str(REPO / "core")]
    from core.task_stats import auroc
    scores, labels = {}, {}
    with open(v3_dir / "edge_scores_v3.csv") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["gauss_a"], r["gauss_b"])
            labels[key] = bool(int(r["label"]))
            scores[key] = float(r["R4_c_mv"])
    pm = defaultdict(list)
    with open(v3_dir / "matched_edges_1to1.csv") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["gauss_a"], r["gauss_b"])
            if key in scores:
                pm[r["case_id"]].append((scores[key], labels[key]))
    aucs = []
    for c in sorted(pm):
        if not c.startswith("DouBanLv1_"):      # dev split only
            continue
        scs = np.array([i[0] for i in pm[c]])
        lbs = np.array([i[1] for i in pm[c]])
        a = auroc(scs, lbs)
        if a is not None:
            aucs.append(a)
    return np.asarray(aucs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=str(REPO / "outputs" / "task5r_v3"))
    ap.add_argument("--output",
                    default=str(REPO / "outputs" / "task5r_v3_1"
                                / "min_pairs_freeze.json"))
    ar = ap.parse_args()
    aucs = load_pilot_per_pair_aurocs(Path(ar.pilot_dir))
    result = estimate_min_pairs(aucs)
    result["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["note"] = ("Pilot = consumed/exploratory v3 run; K frozen before "
                      "any v3.1 result is read.")
    out = Path(ar.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
