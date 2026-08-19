#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all controlled-overlap cases and compute failure metrics.

Iterates over all (pair × mode × severity) combinations from the frozen
benchmark_transforms.json, runs run_overlap_case.py, then compute_failure_metrics.py.

Usage:
    python scripts/run_all_task2.py [--phase N]  # N=1: run cases, N=2: metrics, N=3: analyze
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON = "/home/test/biosoft/enter/envs/agri_re_py310/bin/python"
_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")


def load_transforms() -> dict:
    with open(os.path.join(_OUTROOT, "benchmark_transforms.json")) as f:
        return json.load(f)


def generate_cases() -> list[tuple[str, str, str]]:
    """Generate all (pair_key, mode, severity) combinations."""
    transforms = load_transforms()
    cases = []
    for pk, data in transforms.items():
        for mode in ["horizontal", "vertical"]:
            for sev in data[mode]:
                cases.append((pk, mode, sev["severity"]))
    return cases


def run_phase1():
    """Run all controlled-overlap segmentation cases."""
    cases = generate_cases()
    print(f"Phase 1: Running {len(cases)} cases")

    results = []
    for i, (pk, mode, sev) in enumerate(cases):
        case_dir = os.path.join(_OUTROOT, "controlled", pk, mode, sev)
        if os.path.exists(os.path.join(case_dir, "status.json")):
            with open(os.path.join(case_dir, "status.json")) as f:
                status = json.load(f)
            if status.get("status") == "completed":
                print(f"  [{i+1}/{len(cases)}] SKIP (already completed): {pk}/{mode}/{sev}")
                continue

        print(f"  [{i+1}/{len(cases)}] Running {pk}/{mode}/{sev} ...")
        t0 = time.time()
        result = subprocess.run(
            [_PYTHON, os.path.join(_REPO_ROOT, "scripts", "run_overlap_case.py"),
             "--pair-key", pk, "--mode", mode, "--severity", sev],
            capture_output=True, text=True, timeout=300,
        )
        runtime = time.time() - t0
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-200:]}")
        else:
            print(f"    OK ({runtime:.1f}s)")
        results.append({"pair_key": pk, "mode": mode, "severity": sev,
                        "returncode": result.returncode, "runtime": runtime})

    return results


def run_phase2():
    """Compute failure metrics for all completed cases."""
    cases = generate_cases()
    print(f"Phase 2: Computing metrics for {len(cases)} cases")

    results = []
    for pk, mode, sev in cases:
        case_dir = os.path.join(_OUTROOT, "controlled", pk, mode, sev)
        metrics_path = os.path.join(case_dir, "failure_metrics.json")
        if os.path.exists(metrics_path):
            print(f"  SKIP (exists): {pk}/{mode}/{sev}")
            continue

        if not os.path.exists(os.path.join(case_dir, "status.json")):
            print(f"  SKIP (no run): {pk}/{mode}/{sev}")
            continue

        print(f"  Computing: {pk}/{mode}/{sev} ...")
        result = subprocess.run(
            [_PYTHON, os.path.join(_REPO_ROOT, "scripts", "compute_failure_metrics.py"),
             "--case-dir", case_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-200:]}")
        else:
            print(f"    OK")
        results.append({"pair_key": pk, "mode": mode, "severity": sev,
                        "returncode": result.returncode})

    return results


def run_phase3():
    """Analyze and summarize all failure metrics."""
    cases = generate_cases()
    print(f"Phase 3: Analyzing {len(cases)} cases")

    summary = []
    shortcut_evidence = {}

    for pk, mode, sev in cases:
        case_dir = os.path.join(_OUTROOT, "controlled", pk, mode, sev)
        metrics_path = os.path.join(case_dir, "failure_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        summary.append({
            "pair_key": pk,
            "mode": mode,
            "severity": sev,
            "mIoU": m["instance"]["mIoU"],
            "PQ": m["instance"]["PQ"],
            "merge_level": m["geodesic"]["merge_level"],
            "apex_recall": m["geodesic"]["reference_apex_recall"],
            "first_failure": m["first_failure_stage"],
            "dominant_failure": m["dominant_failure_stage"],
            "contact": m["construction"]["contact_fraction"],
            "gap_ratio": m["construction"].get("projected_overlap_fraction", None),
            "shortcut_ratio": m.get("shortcut", {}).get("shortcut_ratio", None),
            "shortcut_confirmed": m.get("shortcut", {}).get("shortcut_confirmed", False),
            "cross_leaf_merge": m.get("shortcut", {}).get("shortcut_evidence", {}).get("cross_leaf_merge", False),
            "cross_leaf_path": m.get("shortcut", {}).get("shortcut_evidence", {}).get("cross_leaf_path_detected", False),
        })

        # Collect vertical shortcut evidence
        if mode == "vertical":
            if pk not in shortcut_evidence:
                shortcut_evidence[pk] = {}
            sc = m.get("shortcut", {})
            ev = sc.get("shortcut_evidence", {})
            shortcut_evidence[pk][sev] = {
                "shortcut_ratio": sc.get("shortcut_ratio"),
                "shortcut_confirmed": sc.get("shortcut_confirmed", False),
                "distance_shortened": ev.get("distance_shortened", False),
                "cross_leaf_merge": ev.get("cross_leaf_merge", False),
                "shared_instances": ev.get("shared_instances", []),
                "cross_leaf_path_detected": ev.get("cross_leaf_path_detected", False),
            }

    # Save summary CSV
    csv_path = os.path.join(_OUTROOT, "benchmark_summary.csv")
    with open(csv_path, "w") as f:
        f.write("pair_key,mode,severity,mIoU,PQ,merge_level,apex_recall,first_failure,dominant_failure,contact,overlap,shortcut_ratio,shortcut_confirmed,cross_leaf_merge,cross_leaf_path\n")
        for row in summary:
            sr = row.get("shortcut_ratio")
            sr_str = f"{sr:.4f}" if sr is not None else "None"
            sc = row.get("shortcut_confirmed", False)
            clm = row.get("cross_leaf_merge", False)
            clp = row.get("cross_leaf_path", False)
            f.write(f"{row['pair_key']},{row['mode']},{row['severity']},"
                    f"{row['mIoU']:.4f},{row['PQ']:.4f},{row['merge_level']},"
                    f"{row['apex_recall']:.4f},{row['first_failure']},{row['dominant_failure']},"
                    f"{row['contact']:.4f},{row['gap_ratio']:.4f},{sr_str},{sc},{clm},{clp}\n")

    print(f"\nSummary saved to {csv_path}")
    print(f"\n{'='*70}")
    print(f"{'pair_key':40s} {'mode':10s} {'sev':5s} {'mIoU':8s} {'PQ':8s} {'merge':6s} {'apex_r':8s} {'contact':8s}")
    print(f"{'='*70}")
    for row in summary:
        print(f"{row['pair_key']:40s} {row['mode']:10s} {row['severity']:5s} "
              f"{row['mIoU']:8.4f} {row['PQ']:8.4f} {row['merge_level']:6d} "
              f"{row['apex_recall']:8.4f} {row['contact']:8.4f}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True,
                        help="1=run cases, 2=compute metrics, 3=analyze summary")
    args = parser.parse_args()

    if args.phase == 1:
        results = run_phase1()
    elif args.phase == 2:
        results = run_phase2()
    elif args.phase == 3:
        results = run_phase3()
    else:
        print("Invalid phase")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
