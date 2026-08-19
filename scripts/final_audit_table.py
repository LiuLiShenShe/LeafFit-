#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final audit table for Task 2 (the table the user requested)."""
from __future__ import annotations

import csv
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")


def load_row_metrics(pk: str, mode: str, sev: str) -> dict:
    m_path = os.path.join(_OUTROOT, "controlled", pk, mode, sev, "failure_metrics.json")
    with open(m_path) as f:
        return json.load(f)


def main() -> int:
    with open(os.path.join(_OUTROOT, "benchmark_summary.csv")) as f:
        reader = list(csv.DictReader(f))

    # Build header
    cols = ["Pair", "Mode", "Sev", "shortcut_ratio", "cross_leaf_merge",
            "cross_leaf_path", "dist_shortened", "below_lower%",
            "first_stage", "dominant_stage", "PQ"]
    widths = [45, 8, 5, 15, 16, 16, 15, 12, 12, 14, 8]
    header = "".join(f"{c:{w}s}" for c, w in zip(cols, widths))
    print("FINAL AUDIT TABLE")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for row in reader:
        pk = row["pair_key"]
        mode = row["mode"]
        sev = row["severity"]
        m = load_row_metrics(pk, mode, sev)

        sc = m.get("shortcut", {})
        ev = sc.get("shortcut_evidence", {})

        if mode == "vertical":
            sr = sc.get("shortcut_ratio")
            sr_str = f"{sr:.4f}" if sr is not None else "None"
            clm = str(ev.get("cross_leaf_merge", False))
            clp = str(ev.get("cross_leaf_path_detected", False))
            ds = str(ev.get("distance_shortened", False))
            below = f"{ev.get('upper_below_lower_ratio', 0) * 100:.1f}%" if ev else "N/A"
        else:
            sr_str = "N/A"
            clm = "N/A"
            clp = "N/A"
            ds = "N/A"
            below = "N/A"

        fs = m["first_failure_stage"]
        dom = m.get("dominant_failure_stage", fs)
        pq = f"{m['instance']['PQ']:.3f}"

        vals = [pk, mode, sev, sr_str, clm, clp, ds, below, fs, dom, pq]
        print("".join(f"{str(v):{w}s}" for v, w in zip(vals, widths)))

    print("=" * len(header))
    print("\nShortcut confirmed = (shortcut_ratio < 1.0) AND (cross_leaf_merge) AND (cross_leaf_path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
