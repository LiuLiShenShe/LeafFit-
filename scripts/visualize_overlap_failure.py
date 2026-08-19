#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize failure curves from controlled overlap benchmark (Task 2).

Creates per-pair and aggregate failure curve plots.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_FIGDIR = os.path.join(_OUTROOT, "figures")


def load_summary() -> list[dict]:
    csv_path = os.path.join(_OUTROOT, "benchmark_summary.csv")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["mIoU"] = float(row["mIoU"])
            row["PQ"] = float(row["PQ"])
            row["merge_level"] = int(row["merge_level"])
            row["apex_recall"] = float(row["apex_recall"])
            row["contact"] = float(row["contact"])
            row["overlap"] = float(row["overlap"])
            rows.append(row)
    return rows


def plot_failure_curves():
    """Create failure curve plots for horizontal and vertical modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = load_summary()
    os.makedirs(_FIGDIR, exist_ok=True)

    severity_order = {
        "horizontal": ["H0", "H1", "H2", "H3", "H4"],
        "vertical": ["V0", "V1", "V2", "V3", "V4"],
    }

    pairs = sorted(set(r["pair_key"] for r in rows))

    for mode in ["horizontal", "vertical"]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"LeafFit Figure 13 Failure Mechanisms — {mode.capitalize()} Mode",
                     fontsize=14, fontweight="bold")

        # Subplot 1: PQ vs severity
        ax = axes[0, 0]
        for pk in pairs:
            subset = [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            subset.sort(key=lambda r: severity_order[mode].index(r["severity"]))
            sevs = [r["severity"] for r in subset]
            pqs = [r["PQ"] for r in subset]
            label = pk.replace("_pair_", " ").replace("_", " ")[:25]
            ax.plot(sevs, pqs, marker="o", linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("PQ (Panoptic Quality)")
        ax.set_title("PQ vs Severity")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

        # Subplot 2: merge_level vs severity
        ax = axes[0, 1]
        for pk in pairs:
            subset = [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            subset.sort(key=lambda r: severity_order[mode].index(r["severity"]))
            sevs = [r["severity"] for r in subset]
            merges = [r["merge_level"] for r in subset]
            label = pk.replace("_pair_", " ").replace("_", " ")[:25]
            ax.plot(sevs, merges, marker="s", linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Merge Level (cross-leaf instances)")
        ax.set_title("Merge Level vs Severity")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Subplot 3: mIoU vs contact_fraction (geometric severity)
        ax = axes[1, 0]
        for pk in pairs:
            subset = [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            contacts = [r["contact"] for r in subset]
            mious = [r["mIoU"] for r in subset]
            label = pk.replace("_pair_", " ").replace("_", " ")[:25]
            ax.plot(contacts, mious, marker="^", linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Contact Fraction (geometric severity)")
        ax.set_ylabel("mIoU")
        ax.set_title("mIoU vs Contact Fraction")
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

        # Subplot 4: PQ vs contact_fraction
        ax = axes[1, 1]
        for pk in pairs:
            subset = [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            contacts = [r["contact"] for r in subset]
            pqs = [r["PQ"] for r in subset]
            label = pk.replace("_pair_", " ").replace("_", " ")[:25]
            ax.plot(contacts, pqs, marker="D", linewidth=2, markersize=8, label=label)
        ax.set_xlabel("Contact Fraction (geometric severity)")
        ax.set_ylabel("PQ (Panoptic Quality)")
        ax.set_title("PQ vs Contact Fraction")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(_FIGDIR, f"{mode}_failure_curve.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def create_diagnostic_table():
    """Create a summary table of failure boundaries."""
    rows = load_summary()
    severity_order = {
        "horizontal": ["H0", "H1", "H2", "H3", "H4"],
        "vertical": ["V0", "V1", "V2", "V3", "V4"],
    }

    pairs = sorted(set(r["pair_key"] for r in rows))

    for mode in ["horizontal", "vertical"]:
        print(f"\n{'='*80}")
        print(f"{mode.upper()} MODE — Failure Boundary Analysis")
        print(f"{'='*80}")

        for pk in pairs:
            subset = [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            subset.sort(key=lambda r: severity_order[mode].index(r["severity"]))

            first_merge = None
            for r in subset:
                if r["merge_level"] > 0:
                    first_merge = r["severity"]
                    break

            print(f"\n{pk}:")
            print(f"  First merge at: {first_merge or 'none'}")
            for r in subset:
                flag = " *** MERGE" if r["merge_level"] > 0 else ""
                print(f"  {r['severity']}: PQ={r['PQ']:.3f}  merge={r['merge_level']}  "
                      f"contact={r['contact']:.3f}  mIoU={r['mIoU']:.4f}{flag}")


def main():
    plot_failure_curves()
    create_diagnostic_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
