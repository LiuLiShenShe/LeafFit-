#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel Task 3 held-out + ablation runner (Stage 3/4).

Uses the FROZEN config from outputs/task3/frozen_method_config.json on the
held-out pairs (plant2, plant7). Runs G4 (Ours) + G0 (euclidean control) over
the full level set (coarse H0-H4 / V0-V4 + fine HF1-4 / VF1-4). Heat is NOT
re-run — the Phase 0 heat fine baseline + Task 2 coarse heat results are the
heat reference (avoids re-running the slowest backend redundantly).

Parallelized with a thread pool across (pair, mode, severity, backend). The
GIL is released during the heavy numpy/geometry work, so threads get real
parallelism here. Uses skip_if_exists to preserve any already-computed cases.

Usage:
  python scripts/run_heldout_task3_parallel.py [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_task3_case import run_case  # noqa: E402

_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")

HELDOUT_PAIRS = ["plant2_rubber_tree_pair_3_12", "plant7_black_pearl_pepper_pair_4_8"]
H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def load_frozen_config() -> dict:
    with open(os.path.join(_T3, "frozen_method_config.json")) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--pair", default=None)
    ar = ap.parse_args()

    frozen = load_frozen_config()
    sel = frozen.get("selected", {})
    cfg_h = sel.get("horizontal", {}).get("cfg") or sel.get("_default", {}).get("cfg")
    cfg_v = sel.get("vertical", {}).get("cfg") or cfg_h
    if cfg_h is None:
        print("[FATAL] no frozen config — run select_frozen_config.py first")
        return 1

    pairs = [ar.pair] if ar.pair else HELDOUT_PAIRS
    jobs = []
    for pk in pairs:
        for mode in ("horizontal", "vertical"):
            sevs = H_LEVELS if mode == "horizontal" else V_LEVELS
            c4 = cfg_v if mode == "vertical" and cfg_v != cfg_h else cfg_h
            for sev in sevs:
                # G4 (Ours) + G0 euclidean control (same k/mutual). Heat reference
                # comes from Phase 0 (fine) + Task 2 (coarse) — not re-run here.
                jobs.append((pk, mode, sev, "surface", c4))
                jobs.append((pk, mode, sev, "euclidean",
                             {"k": cfg_h.get("k", 256), "mutual": cfg_h.get("mutual", False)}))

    n = len(jobs)
    print(f"[heldout-parallel] {n} jobs, {ar.jobs} workers", flush=True)
    t0 = time.time()
    done, fail = 0, 0
    errors = []
    with ThreadPoolExecutor(max_workers=ar.jobs) as ex:
        futs = {ex.submit(run_case, pk, mode, sev, backend, cfg, "test",
                          skip_if_exists=True): (pk, mode, sev, backend)
                for (pk, mode, sev, backend, cfg) in jobs}
        for i, fut in enumerate(as_completed(futs)):
            pk, mode, sev, backend = futs[fut]
            try:
                r = fut.result()
                status = r.get("status", "?")
                m = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
                pq = m.get("instance", {}).get("PQ")
                done += 1
                print(f"[{i+1}/{n}] {backend}/{sev} {pk} {mode}: {status} PQ={pq} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                fail += 1
                errors.append((pk, mode, sev, backend, f"{type(e).__name__}: {str(e)[:100]}"))
                print(f"[{i+1}/{n}] {backend}/{sev} {pk} {mode}: ERROR {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
    print(f"\n[heldout-parallel] {done} ok, {fail} errors in {time.time()-t0:.0f}s")
    for e in errors:
        print("  ERR:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
