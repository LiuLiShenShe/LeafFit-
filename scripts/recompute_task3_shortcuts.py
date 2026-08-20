#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute Task 3 vertical shortcut metrics with SAME-BACKEND V0 reference.

The original Stage 1 run normalized shortcut_ratio against the heat V0 field.
For graph backends the Dijkstra field differs from heat even on clean V0, so a
graph backend was penalized by field shape rather than by the actual shortcut.
This recomputes ratio/shortcut_confirmed using each case's OWN backend V0 field
(the case dir's sibling V0 dir), matching the fix in run_task3_case.

Only touches failure_metrics.json['shortcut'] (vertical cases only). Identity
of other keys is preserved.

Usage:
  python scripts/recompute_task3_shortcuts.py [--tree dev] [--pair plant1_green_pepper_pair_8_4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_failure_metrics as cfm  # noqa: E402
from run_task3_case import load_case_transforms, _task3_shortcut  # noqa: E402

_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")


def recompute_tree(root: str, pair: str | None = None) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "failure_metrics.json" not in filenames:
            continue
        fp = os.path.join(dirpath, "failure_metrics.json")
        with open(fp) as f:
            m = json.load(f)
        if m.get("mode") != "vertical" or "shortcut" not in m:
            continue
        case_dir = dirpath
        if pair and not case_dir.startswith(os.path.join(root, pair)):
            continue
        # case_dir = <root>/<pair>/<mode>/<sev>/<backend_dir>  -> sev is 2 up
        sev = os.path.basename(os.path.dirname(os.path.dirname(case_dir)))
        if sev == "V0":
            continue  # V0 is the reference itself; ratio should be 1.0 by construction
        pair_key = m["pair_key"]
        # Reuse the run_task3_case metric function with the fixed V0 lookup
        # (reads this case's paths.json + gt labels + same-backend V0).
        labels, gt_labels, apexes = cfm.load_case(case_dir)["labels"], \
            cfm.load_case(case_dir)["gt_labels"], cfm.load_case(case_dir)["apexes"]
        root_g = np.load(os.path.join(case_dir, "root_geodesic_multisource.npy"))
        pk_data = load_case_transforms(pair_key, "vertical", sev)
        upper_id = pk_data["upper_leaf_id"]
        lower_id = pk_data["lower_leaf_id"]
        sc = _task3_shortcut(labels, gt_labels, root_g, apexes,
                             upper_id, lower_id, pair_key, sev, case_dir)
        m["shortcut"] = sc
        # also update first_failure_stage / dominant if GROUPING previously from
        # a heat-referenced ratio — but shortcut doesn't drive stage, keep as-is.
        with open(fp, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="dev")
    ap.add_argument("--pair", default=None)
    ar = ap.parse_args()
    root = os.path.join(_T3, ar.tree)
    n = recompute_tree(root, ar.pair)
    print(f"[OK] recomputed shortcut metrics for {n} vertical cases under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
