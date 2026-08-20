#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3 held-out + ablation runner (Stage 3/4).

Uses the FROZEN config from outputs/task3/frozen_method_config.json on the
held-out pairs (plant2, plant7). Runs G4 (Ours), G0 (euclidean control) and
heat over the full level set (coarse H0-H4 / V0-V4 + fine HF1-4 / VF1-4).

The held-out pairs are NEVER used for parameter selection (Stage 1/2 ran on
plant1 only). This script is invoked only AFTER frozen_method_config.json exists.

Usage:
  python scripts/run_heldout_task3.py            # both held-out pairs
  python scripts/run_heldout_task3.py --pair plant2_rubber_tree_pair_3_12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_task3_case import run_case, resolve_transforms  # noqa: E402

_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")

HELDOUT_PAIRS = ["plant2_rubber_tree_pair_3_12", "plant7_black_pearl_pepper_pair_4_8"]
H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def load_frozen_config() -> dict:
    with open(os.path.join(_T3, "frozen_method_config.json")) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None)
    ap.add_argument("--force", action="store_true")
    ar = ap.parse_args()

    frozen = load_frozen_config()
    sel = frozen.get("selected", {})
    # default: use horizontal config for both modes (config freeze may pick
    # separate per-mode settings; here we use whichever exists, preferring the
    # mode-specific one).
    cfg_h = sel.get("horizontal", {}).get("cfg") or sel.get("_default", {}).get("cfg")
    cfg_v = sel.get("vertical", {}).get("cfg") or cfg_h
    if cfg_h is None:
        print("[FATAL] no frozen config — run select_frozen_config.py first")
        return 1

    # G4 frozen + G0 euclidean control (same k/mutual as frozen) + heat
    configs = [
        ("surface", cfg_h),          # G4 Ours (mode-agnostic)
        ("euclidean", {"k": cfg_h.get("k", 64), "mutual": cfg_h.get("mutual", False)}),
        ("heat", {}),
    ]
    # mode-specific G4 if different
    if cfg_v and cfg_v != cfg_h:
        configs.append(("surface", cfg_v))

    pairs = [ar.pair] if ar.pair else HELDOUT_PAIRS
    jobs = []
    for pk in pairs:
        for mode in ("horizontal", "vertical"):
            sevs = H_LEVELS if mode == "horizontal" else V_LEVELS
            c4 = cfg_v if mode == "vertical" and cfg_v != cfg_h else cfg_h
            for backend, cfg in configs:
                # heat/euclidean use generic cfg
                c_use = cfg if backend != "surface" else (c4 if mode == "vertical" else cfg_h)
                for sev in sevs:
                    jobs.append((pk, mode, sev, backend, c_use))

    n = len(jobs)
    print(f"[heldout] {n} jobs across {pairs}")
    for i, (pk, mode, sev, backend, cfg) in enumerate(jobs):
        t0 = time.time()
        try:
            r = run_case(pk, mode, sev, backend, cfg, "test",
                         skip_if_exists=not ar.force)
            m = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
            pq = m.get("instance", {}).get("PQ")
            print(f"[{i+1}/{n}] {backend} {pk} {mode} {sev}: {r['status']} PQ={pq} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[{i+1}/{n}] {backend} {pk} {mode} {sev}: ERROR {type(e).__name__}: {str(e)[:80]}")
    print("[OK] held-out runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
