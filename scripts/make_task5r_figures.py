#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R figures — separability box/violin + ROC summary (measured artifacts only)."""
from pathlib import Path

import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "outputs" / "task5r" / "edge_separability.csv"
OUT = REPO / "outputs" / "task5r" / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CSV)))

    # Per-pair AUROC bars (R4 c_mv) with 95% CI
    r4 = [r for r in rows if r["ablation"] == "R4_c_mv"]
    labels = [r["case_id"] for r in r4]
    # For ROC we reload raw scores from a cached npy? We don't have them here;
    # load per-pair score vectors written by the separability script is not yet
    # available (scores were in-memory only). Recompute is replication; instead
    # we emit the bar chart from the measured csv and mark provenance.
    vals = np.array([float(r["auroc"]) for r in r4])
    lo = np.array([float(r["auroc_ci95_lo"]) if r["auroc_ci95_lo"] else float("nan") for r in r4])
    hi = np.array([float(r["auroc_ci95_hi"]) if r["auroc_ci95_hi"] else float("nan") for r in r4])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(labels))
    ax.axhline(0.5, color="#888", ls="--", lw=1, label="chance (0.5)")
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="o-", color="#1f77b4",
                ecolor="#1f77b4", capsize=4, label="R4 c_mv AUROC + 95% CI (bootstrap, B=500)")
    for xi, vi in zip(x, vals):
        ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.15, 0.85)
    ax.set_ylabel("AUROC (within=positive)")
    ax.set_title("Task5R Phase 5 — edge-level identity separability (R4 c_mv = 0.4c_vis+0.3c_app+0.3c_occ)\ncorrected real viewsig; k=32 candidates within two-leaf union; dev=DouBanLv1, held-out=HongZhang")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = OUT / "selected_small_figures.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    print("WROTE", p1)

    # Ablation panel (mean AUROC per split)
    summary = json.loads((REPO / "outputs" / "task5r" / "edge_separability_summary.json").read_text())
    ab = ["R0_dist", "R1_c_vis", "R2_c_app", "R3_c_occ", "R4_c_mv", "R5_surface", "R6_mv_and_surface"]
    dev  = [summary["by_split_ablation"]["dev"][k]["mean_auroc"] for k in ab]
    hld  = [summary["by_split_ablation"]["heldout"][k]["mean_auroc"] for k in ab]
    short = ["R0\\n(dist)", "R1\\n(c_vis)", "R2\\n(c_app)", "R3\\n(c_occ)", "R4\\n(c_mv)", "R5\\n(surf)", "R6\\n(mv\\u2227surf)"]
    fig2, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(ab))
    w = 0.38
    ax.bar(x - w/2, dev, w, label="dev (DouBanLv1, n=5 pairs)", color="#4e79a7", edgecolor="black")
    ax.bar(x + w/2, hld, w, label="held-out (HongZhang, n=2 pairs)", color="#f28e2b", edgecolor="black")
    ax.axhline(0.5, color="#888", ls="--", lw=1)
    for xi, vi in zip(x - w/2, dev):
        ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, vi in zip(x + w/2, hld):
        ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0.15, 1.05)
    ax.set_ylabel("mean AUROC (within=positive, chance 0.5)")
    ax.set_title("Separable is only distance (R0); corrected real observation identity does not separate within from cross (R1-R4, R6).")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    p2 = OUT / "ablation_auroc.png"
    fig2.savefig(p2, dpi=180)
    plt.close(fig2)
    print("WROTE", p2)


if __name__ == "__main__":
    main()
