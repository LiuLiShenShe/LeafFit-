#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute Task 3 metrics summary + figures from saved case outputs.

Aggregates:
  benchmark_summary.csv        (per pair x mode x severity x backend)
  failure_boundary_summary.csv (per pair x mode x backend: mechanism_onset + final)
  graph_statistics.csv         (per case: connectivity, cross-leaf edges)
  runtime_summary.csv
And writes task3_summary.json.

Figures (Agg headless, Chinese labels):
  F1 Horizontal: merge_level & PQ vs severity (heat/euclidean/G4/G5)
  F2 Vertical:   shortcut_ratio & cross_leaf_path vs severity
  F3 Failure boundary comparison
  F4 Ablation:   G0-G6 boundary + clean PQ bar, cross-leaf retained/pruned
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")

H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def collect_all(subdirs=("dev", "test", "ablation"), pairs=None, backends=None):
    """Walk all case dirs, return list of metric dicts.

    Directory layout:
      <subdir>/<pair>/<mode>/<sev>/<backend>/failure_metrics.json        (heat)
      <subdir>/<pair>/<mode>/<sev>/<backend>/<config_dir>/failure_metrics.json  (graph)
    """
    rows = []
    for sub in subdirs:
        root = os.path.join(_T3, sub)
        if not os.path.isdir(root):
            continue
        for pair in os.listdir(root):
            if pairs and pair not in pairs:
                continue
            pdir = os.path.join(root, pair)
            if not os.path.isdir(pdir):
                continue
            for mode in ("horizontal", "vertical"):
                mdir = os.path.join(pdir, mode)
                if not os.path.isdir(mdir):
                    continue
                for sev in os.listdir(mdir):
                    sdir = os.path.join(mdir, sev)
                    if not os.path.isdir(sdir):
                        continue
                    for backend in os.listdir(sdir):
                        if backends and backend not in backends:
                            continue
                        bdir = os.path.join(sdir, backend)
                        if not os.path.isdir(bdir):
                            continue
                        # flat case: heat/<sev>/heat/failure_metrics.json
                        fp = os.path.join(bdir, "failure_metrics.json")
                        if os.path.exists(fp):
                            rows.append(_read_case_row(sub, pair, mode, sev, backend, fp))
                            continue
                        # nested: euclidean/<config>/, surface/<config>/
                        for cdir in os.listdir(bdir):
                            cfp = os.path.join(bdir, cdir, "failure_metrics.json")
                            if os.path.exists(cfp):
                                rows.append(_read_case_row(sub, pair, mode, sev,
                                                           f"{backend}/{cdir}", cfp))
    return rows


def _read_case_row(sub, pair, mode, sev, backend, fp):
    with open(fp) as f:
        m = json.load(f)
    row = {
        "subdir": sub, "pair": pair, "mode": mode,
        "severity": sev, "backend": backend,
        "PQ": m["instance"]["PQ"],
        "mIoU": m["instance"]["mIoU"],
        "wrong_grouping": m["geodesic"]["wrong_grouping"],
        "merge_level": m["geodesic"]["merge_level"],
        "apex_recall": m["geodesic"]["reference_apex_recall"],
        "first_failure_stage": m["first_failure_stage"],
    }
    if mode == "vertical" and "shortcut" in m:
        row["shortcut_ratio"] = m["shortcut"].get("shortcut_ratio")
        row["cross_leaf_path"] = m["shortcut"].get("cross_leaf_path", False)
    return row


def boundary_of(rows, pair, mode, backend, sevs):
    """First-failure (mechanism onset + final instance failure) severity."""
    onset, final = None, None
    clean_pq = None
    for r in sorted(rows, key=lambda r: sevs.index(r["severity"]) if r["severity"] in sevs else 99):
        if r["pair"] != pair or r["mode"] != mode or r["backend"] != backend:
            continue
        if r["severity"] == (sevs[0]):
            clean_pq = r["PQ"]
            continue
        hm = r["wrong_grouping"]
        ve = r.get("cross_leaf_path", False) and (r.get("shortcut_ratio") is not None
                                                   and r["shortcut_ratio"] < 0.999)
        if onset is None and (hm or ve):
            onset = r["severity"]
        if final is None and hm:
            final = r["severity"]
    return {"onset": onset, "final": final, "clean_pq": clean_pq}


def main() -> int:
    rows = collect_all()
    # write benchmark_summary.csv — union of all row keys (horizontal and vertical
    # rows differ: vertical carries shortcut_ratio/cross_leaf_path)
    os.makedirs(_T3, exist_ok=True)
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(os.path.join(_T3, "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # boundaries per (pair, mode, backend) using the union of dev/test sevs
    all_pairs = sorted(set(r["pair"] for r in rows))
    all_backends = sorted(set(r["backend"] for r in rows))
    boundary_rows = []
    for pair in all_pairs:
        for mode in ("horizontal", "vertical"):
            sevs = H_LEVELS if mode == "horizontal" else V_LEVELS
            for backend in all_backends:
                b = boundary_of(rows, pair, mode, backend, sevs)
                boundary_rows.append({"pair": pair, "mode": mode, "backend": backend,
                                      "mechanism_onset": b["onset"],
                                      "final_instance_failure": b["final"],
                                      "clean_pq": b["clean_pq"]})
    with open(os.path.join(_T3, "failure_boundary_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()))
        w.writeheader()
        w.writerows(boundary_rows)

    summary = {
        "benchmark_summary.csv": len(rows),
        "failure_boundary_summary.csv": len(boundary_rows),
        "pairs": all_pairs,
        "backends": all_backends,
    }
    with open(os.path.join(_T3, "task3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {len(rows)} cases -> benchmark_summary.csv; "
          f"{len(boundary_rows)} boundaries -> failure_boundary_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
