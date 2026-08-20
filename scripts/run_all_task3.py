#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3 orchestration driver (Stage 1 dev, Stage 2 freeze).

Phases (mirror plan):
  --phase 0   fine boundary sweep (already frozen)          [no-op here]
  --phase 1   Stage 1: DEV parameter search on plant1 only
  --phase 2   Stage 2: select_frozen_config on plant1 dev, freeze
  --phase 3   metrics collection + figures + summary        [delegated]

Stage 1 DEV (plant1_green_pepper_pair_8_4, 18 levels = H0 + HF1-4 + H1-4 +
V0 + VF1-4 + V1-4):
  * G4 (Ours)  : gate sweep (tau_d x tau_t) x k x mutual
  * G0 (control): same k/mutual grid
  * G5 (diagnostic ablation): light lambda sweep on DEV only; record
    METRIC_INCOMPATIBLE where the soft-metric crashes the petiole base-finder.
  * heat        : baseline reference (Phase 0 already has it, but re-run in
    dev/ layout for uniform aggregation)

No held-out (plant2/7) is touched here. Frozen root basins from Task 2 are
injected for every backend.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_failure_metrics as cfm  # noqa: E402
from run_task3_case import run_case, resolve_transforms  # noqa: E402

_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")

DEV_PAIR = "plant1_green_pepper_pair_8_4"
H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def _levels_for(mode: str) -> list[str]:
    return H_LEVELS if mode == "horizontal" else V_LEVELS


def _clean_ref(mode: str) -> str:
    return "V0" if mode == "vertical" else "H0"


def load_param_grid() -> dict:
    with open(os.path.join(_T3, "parameter_grid.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1 DEV sweep
# ---------------------------------------------------------------------------
DEFS = None  # placeholder; grids below


def g4_gate_grid(pg: dict) -> list[dict]:
    """G4 operating grid: gate x k x mutual.

    k>=256 for G4 surface (k<256 fragments the graph / flattens the Dijkstra
    field enough that LeafFit's petiole base-finder intermittently hits an empty
    `argmin` on coarse levels; probed k=128 partially, k=256 robust across
    H0/H1/H3/V0/V1/V3). Stage 1 sweeps the full gate x mutual grid at k=256
    (k is the secondary knob); the frozen config's k-robustness is confirmed at
    Stage 2 with k=320.
    """
    taus = [{"tau_d": td, "tau_t": tt}
            for td in pg["tau_grid"]["tau_d"] for tt in pg["tau_grid"]["tau_t"]]
    out = []
    k_stage1 = 256  # single safe-k for Stage 1; k-robustness checked at Stage 2
    for t, m in itertools.product(taus, pg["mutual_grid"]):
        out.append({"feature_set": "G4", **t, "k": k_stage1, "mutual": m})
    return out


def g0_grid(pg: dict) -> list[dict]:
    """Euclidean graph control (G0): k=256 x mutual (matches G4 Stage-1 k for a
    fair Full-vs-Euclidean comparison; k>=256 robust on coarse levels)."""
    out = []
    for m in pg["mutual_grid"]:
        out.append({"feature_set": "G0", "k": 256, "mutual": m})
    return out


def g5_light_grid(pg: dict) -> list[dict]:
    """G5 diagnostic ablation (DEV only): light lambda subset at k=256."""
    combos = [
        {"lambda_n": 0.5, "lambda_t": 1.0},
        {"lambda_n": 1.0, "lambda_t": 2.0},
        {"lambda_n": 0.0, "lambda_t": 4.0},
    ]
    out = []
    for c in combos:
        out.append({"feature_set": "G5", **c, "p": pg["p"],
                    "tau_d": 3.0, "tau_t": 0.5, "k": 256, "mutual": False})
    return out


def run_stage1(pairs: list[str], skip_if_exists: bool = True) -> dict:
    pg = load_param_grid()
    # Per plan Stage 1 coarse screen: only H0/H1/H3 + V0/V1/V3 for the FULL grid;
    # full 18-level run comes at Stage 2 after config freeze. For G4 we sweep the
    # operating grid over a representative subset here (H0,V0 clean + H1/V1/H3/V3
    # boundaries), then Stage 2 runs the frozen config over all 18 levels.
    coarse_h = ["H0", "HF1", "HF2", "HF3", "H1", "H3"]
    coarse_v = ["V0", "VF1", "VF2", "VF3", "V1", "V3"]

    jobs = []
    for mode, lv in [("horizontal", coarse_h), ("vertical", coarse_v)]:
        # heat baseline
        for sev in lv:
            jobs.append((DEV_PAIR, mode, sev, "heat", {}))
        # G4 grid (full operating grid on the coarse subset)
        for cfg in g4_gate_grid(pg):
            for sev in lv:
                jobs.append((DEV_PAIR, mode, sev, "surface", cfg))
        # G0 control
        for cfg in g0_grid(pg):
            for sev in lv:
                jobs.append((DEV_PAIR, mode, sev, "euclidean", cfg))
    # G5 diagnostic (DEV only): small predefined light-λ sweep (3 combos) on
    # clean + first boundary (H0/H1/V0/V1) at the safe k=256. This quantifies
    # soft metric renormalization vs LeafFit's petiole base-finder compatibility.
    # All levels complete -> stable, candidate for held-out; any crash ->
    # METRIC_INCOMPATIBLE (recorded, not retried). G5 does NOT drive PASS/FAIL.
    g5_levels = {"horizontal": ["H0", "H1"], "vertical": ["V0", "V1"]}
    for mode, lv in g5_levels.items():
        for cfg in g5_light_grid(pg):
            for sev in lv:
                jobs.append((DEV_PAIR, mode, sev, "surface", cfg))

    n = len(jobs)
    print(f"[stage1] {n} jobs", flush=True)
    results = []
    for i, (pk, mode, sev, backend, cfg) in enumerate(jobs):
        t0 = time.time()
        try:
            r = run_case(pk, mode, sev, backend, cfg, "dev", skip_if_exists=skip_if_exists)
            dt = time.time() - t0
            m = r.get("metrics", {})
            pq = m.get("instance", {}).get("PQ") if isinstance(m, dict) else None
            print(f"[{i+1}/{n}] {backend}/{cfg.get('feature_set', cfg.get('k',''))} "
                  f"{sev}: {'ok' if r['status']=='completed' else r['status']} PQ={pq} ({dt:.1f}s)", flush=True)
            results.append(r)
        except Exception as e:
            print(f"[{i+1}/{n}] {backend}/{cfg.get('feature_set')} {sev}: "
                  f"ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
            results.append({"pair_key": pk, "mode": mode, "severity": sev,
                            "backend": backend, "config": cfg,
                            "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"})

    # G5 crash census: which configs are METRIC_INCOMPATIBLE
    g5_crashes = [r for r in results if r.get("backend") == "surface"
                  and r.get("config", {}).get("feature_set") == "G5"
                  and r.get("status") == "error"]
    g5_ok = [r for r in results if r.get("backend") == "surface"
             and r.get("config", {}).get("feature_set") == "G5"
             and r.get("status") == "completed"]
    print(f"\n[stage1] G5 diagnostic: {len(g5_ok)} stable configs, "
          f"{len(g5_crashes)} METRIC_INCOMPATIBLE")
    return {"jobs": n, "results": results, "g5_stable": len(g5_ok),
            "g5_incompatible": len(g5_crashes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--skip", action="store_true", default=True)
    ap.add_argument("--no-skip", action="store_true")
    ar = ap.parse_args()
    skip = not ar.no_skip

    if ar.phase == 1:
        summary = run_stage1([DEV_PAIR], skip_if_exists=skip)
        with open(os.path.join(_T3, "stage1_dev_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("[OK] stage1_dev_summary.json written")
    elif ar.phase == 2:
        print("[phase2] select_frozen_config delegated (run select_frozen_config.py)")
    elif ar.phase == 3:
        print("[phase3] metrics + figures + summary delegated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
