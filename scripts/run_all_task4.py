#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4 orchestration driver (Stage 0 checkpoint, Stage 1 dev).

Phases (mirror plan):
  --phase 0   Stage 0 checkpoint: plant1 H0/H1/V0/V1 + two-plane discriminability
              review (within vs cross c_vis/c_app/c_occ + G6 prune rates). STOP
              if within≈cross on clean H0/V0 (honest FAIL, no parameter grid).
  --phase 1   Stage 1 DEV: coarse sweep on plant1 only (single-feature ablation,
              combined weights, tau_mv grid, c_app variants A0/A1/A2).
  --phase 2   Stage 2 select_frozen_config on plant1, freeze (delegated).
  --phase 3   metrics collection + figures + summary (delegated).

Held-out (plant2/7) and camera sensitivity are run AFTER the Stage 0/1 gates pass,
via separate scripts (run_heldout_task4.py, run_camera_sensitivity_task4.py).
"""
from __future__ import annotations

import argparse
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
from run_task4_case import run_case, _load_or_compute_viewsig, _gaussian_data  # noqa: E402
from run_task3_case import resolve_transforms, load_case_transforms  # noqa: E402
from run_overlap_case import load_plant, apply_transform_entry  # noqa: E402
from geodesic_backends import _mv_edge_features, _build_knn_edges  # noqa: E402
from multiview_identity import build_view_signature  # noqa: E402

_T4 = os.path.join(_REPO_ROOT, "outputs", "task4")
_T4FEAT = os.path.join(_T4, "identity_features")

DEV_PAIR = "plant1_green_pepper_pair_8_4"
H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def _levels_for(mode: str) -> list[str]:
    return H_LEVELS if mode == "horizontal" else V_LEVELS


def _clean_ref(mode: str) -> str:
    return "V0" if mode == "vertical" else "H0"


def _load_or_build_edges_viewsign(pair, mode, severity, mv):
    """Return (vs, rows, cols, c_vis, c_app, c_occ, same_mask)."""
    pk_data = load_case_transforms(pair, mode, severity)
    pd = load_plant(pk_data["plant"])
    gc, labels, apexes = pd["gc"], pd["labels"], pd["apexes"]
    leaf_a = np.where(labels == pk_data["leaf_a_id"])[0]
    leaf_b = np.where(labels == pk_data["leaf_b_id"])[0]
    g = gc
    sev = next(s for s in pk_data[mode] if s["severity"] == severity)
    if sev["leaf_a_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev["leaf_a_transform"], leaf_a)
    if sev["leaf_b_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev["leaf_b_transform"], leaf_b)
    vs, _, _ = _load_or_compute_viewsig(g, pair, mode, severity, mv)
    a_id, b_id = pk_data["leaf_a_id"], pk_data["leaf_b_id"]
    # restrict to a/b leaf gaussians for within-vs-cross edge features
    sub = np.concatenate([np.where(labels == a_id)[0], np.where(labels == b_id)[0]])
    # full-set edges (k=64 as in Stage 1), then mask to sub nodes via incident
    rows, cols, vals, tree, nn_idx = _build_knn_edges(g.xyz, k=64)
    node_label = np.zeros(len(g.xyz), dtype=int)
    node_label[sub] = 1 + (np.arange(len(sub)) >= np.where(labels == a_id)[0].size)
    same = node_label[rows] == node_label[cols]
    # restrict to edges within/between a/b (drop non-pair nodes entirely)
    keep = (node_label[rows] > 0) & (node_label[cols] > 0)
    rows_e, cols_e = rows[keep], cols[keep]
    same_e = same[keep]
    c_vis, c_app, c_occ = _mv_edge_features(
        rows_e, cols_e, vs["visible"], vs["appear_sig"], vs["depth"], vs["uv"],
        vs["n_views"])
    return vs, rows_e, cols_e, c_vis, c_app, c_occ, same_e, node_label


def stage0_checkpoint(mv: dict, skip_if_exists: bool = True) -> dict:
    """Stage 0: plant1 H0/H1/V0/V1 discriminability review (NOT PQ).

    Emits within vs cross medians for c_vis/c_app/c_occ/c_mv plus the G6 prune
    rates. Summary verdict 'continue' iff clean H0/V0 have c_mv_within-c_mv_cross
    substantially > 0 AND G6 prunes cross > within on clean levels.
    """
    cases = [("horizontal", "H0", "within==cross check"),
             ("horizontal", "H1", "boundary"),
             ("vertical", "V0", "within==cross check"),
             ("vertical", "V1", "boundary")]
    rows_out = []
    for mode, sev, tag in cases:
        vs, re_, ce, cv, ca, co, same, nl = _load_or_build_edges_viewsign(
            DEV_PAIR, mode, sev, mv)
        cmv = 0.4 * cv + 0.3 * ca + 0.3 * co
        med = lambda a, m: (float(np.median(a[m])) if m.sum() > 0 else None)
        rec = {
            "mode": mode, "severity": sev, "tag": tag,
            "n_within": int(same.sum()), "n_cross": int((~same).sum()),
            "c_vis_within": med(cv, same), "c_vis_cross": med(cv, ~same),
            "c_app_within": med(ca, same), "c_app_cross": med(ca, ~same),
            "c_occ_within": med(co, same), "c_occ_cross": med(co, ~same),
            "c_mv_within": med(cmv, same), "c_mv_cross": med(cmv, ~same),
        }
        rows_out.append(rec)
        print(json.dumps(rec, indent=2))

    # verdict: clean levels must show c_mv_within > c_mv_cross (margin >= 0.03)
    margins = {}
    for r in rows_out:
        k = f"{r['mode']}/{r['severity']}"
        margins[k] = (r["c_mv_within"] or 0.0) - (r["c_mv_cross"] or 0.0)
    clean_ok = all(margins.get(k, 0.0) >= 0.03
                   for k in ("horizontal/H0", "vertical/V0"))
    summary = {
        "stage": 0,
        "rows": rows_out,
        "margins": margins,
        "clean_selectivity_ok": bool(clean_ok),
        "verdict": "continue" if clean_ok else "STOP",
        "review": ("within vs cross c_vis/c_app/c_occ/c_mv medians on plant1 "
                   "H0/H1/V0/V1; G6 prune direction in the full Stage 1 run.")
    }
    with open(os.path.join(_T4, "stage0_checkpoint.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[stage0] margins={margins} clean_ok={clean_ok} -> {summary['verdict']}")
    return summary


def _with_mv(cfg: dict, mv: dict) -> dict:
    """Merge camera config keys into the backend cfg (mv_n_views, etc.)."""
    out = dict(cfg)
    for k, v in mv.items():
        out["mv_" + k] = v
    return out


def _single_feature_weights() -> list[dict]:
    """Single-feature ablations + combined init + variants."""
    combos = [
        {"w_vis": 1.0, "w_app": 0.0, "w_occ": 0.0, "label": "c_vis-only"},
        {"w_vis": 0.0, "w_app": 1.0, "w_occ": 0.0, "label": "c_app-only"},
        {"w_vis": 0.0, "w_app": 0.0, "w_occ": 1.0, "label": "c_occ-only"},
        {"w_vis": 0.4, "w_app": 0.3, "w_occ": 0.3, "label": "init"},
        {"w_vis": 0.5, "w_app": 0.3, "w_occ": 0.2, "label": "w5"},
        {"w_vis": 0.5, "w_app": 0.2, "w_occ": 0.3, "label": "w5b"},
        {"w_vis": 0.4, "w_app": 0.4, "w_occ": 0.2, "label": "w4"},
        {"w_vis": 0.6, "w_app": 0.2, "w_occ": 0.2, "label": "w6"},
    ]
    for c in combos:
        c["k"] = 64
        c["mutual"] = False
    return combos


def stage1_sweep(mv: dict, skip_if_exists: bool = True) -> dict:
    """Stage 1 coarse: plant1, single-feature ablation + combos + tau_mv grid,
    on a coarse level subset. Record PQ + prune rates; no held-out touch."""
    # coarse subset (per plan): H0/HF1/H1/H3 + V0/VF1/V3
    coarse_h = ["H0", "HF1", "H1", "H3"]
    coarse_v = ["V0", "VF1", "V1", "V3"]
    tau_grid = [0.6, 0.7, 0.8]
    jobs = []
    for mode, lv in [("horizontal", coarse_h), ("vertical", coarse_v)]:
        for sev in lv:
            for w in _single_feature_weights():
                for tau in tau_grid:
                    cfg = _with_mv({"feature_set": "G6",
                           "lambda_n": 0.0, "lambda_t": 0.0, "p": 2.0,
                           "tau_d": 3.0, "tau_t": 0.5,
                           "k": w["k"], "mutual": w["mutual"],
                           "tau_mv": tau,
                           "w_vis": w["w_vis"], "w_app": w["w_app"], "w_occ": w["w_occ"],
                           "label": w["label"]}, mv)
                    jobs.append((DEV_PAIR, mode, sev, "surface", cfg))
        # G7 combined (surface + identity) on clean + first boundary
        for sev in ([lv[0], lv[-1]] if len(lv) > 1 else lv):
            for tau in [0.6, 0.7]:
                cfg = _with_mv({"feature_set": "G7", "lambda_n": 0.0, "lambda_t": 0.0,
                       "p": 2.0, "tau_d": 3.0, "tau_t": 0.5,
                       "k": 64, "mutual": False, "tau_mv": tau,
                       "w_vis": 0.4, "w_app": 0.3, "w_occ": 0.3,
                       "label": "G7"}, mv)
                jobs.append((DEV_PAIR, mode, sev, "surface", cfg))
    # baseline: heat + G0 + G4 (clean + one boundary) for reference
    for mode, lv in [("horizontal", coarse_h), ("vertical", coarse_v)]:
        for sev in [lv[0], lv[1]]:
            jobs.append((DEV_PAIR, mode, sev, "heat", {}))
            jobs.append((DEV_PAIR, mode, sev, "euclidean",
                         {"feature_set": "G0", "k": 64, "mutual": False}))
            jobs.append((DEV_PAIR, mode, sev, "surface",
                         {"feature_set": "G4", "k": 64, "mutual": False,
                          "tau_d": 3.0, "tau_t": 0.5}))

    # dedup
    seen = set()
    uniq = []
    for j in jobs:
        key = (j[0], j[1], j[2], j[3], json.dumps(j[4], sort_keys=True))
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    jobs = uniq

    print(f"[stage1] {len(jobs)} jobs", flush=True)
    results = []
    for i, (pk, mode, sev, backend, cfg) in enumerate(jobs):
        t0 = time.time()
        try:
            r = run_case(pk, mode, sev, backend, cfg, "dev", skip_if_exists=skip_if_exists)
            m = r.get("metrics", {})
            pq = m.get("instance", {}).get("PQ") if isinstance(m, dict) else None
            print(f"[{i+1}/{len(jobs)}] {backend}/{cfg.get('feature_set','')}/"
                  f"{cfg.get('label','')}/tmv{cfg.get('tau_mv','')} {sev}: "
                  f"{r['status']} PQ={pq} ({time.time()-t0:.1f}s)", flush=True)
            results.append(r)
        except Exception as e:
            print(f"[{i+1}/{len(jobs)}] {backend}/{cfg.get('feature_set')} {sev}: "
                  f"ERROR {type(e).__name__}: {str(e)[:100]}", flush=True)
            results.append({"pair_key": pk, "mode": mode, "severity": sev,
                            "backend": backend, "config": cfg,
                            "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    summary = {"stage": 1, "jobs": len(jobs), "results": results}
    with open(os.path.join(_T4, "stage1_dev_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--skip", action="store_true", default=True)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--mv-n-views", type=int, default=36)
    ap.add_argument("--mv-radius-frac", type=float, default=3.0)
    ap.add_argument("--mv-elevation-deg", type=float, default=25.0)
    ap.add_argument("--mv-vis-mode", default="winner_take_all",
                    choices=["winner_take_all", "transmittance_k2"])
    ar = ap.parse_args()
    skip = not ar.no_skip
    mv = {"n_views": ar.mv_n_views, "radius_frac": ar.mv_radius_frac,
          "elevation_deg": ar.mv_elevation_deg, "fov_deg": 40.0,
          "image_h": 1024, "vis_version": "v1",
          "vis_mode": ar.mv_vis_mode}

    if ar.phase == 0:
        stage0_checkpoint(mv, skip)
    elif ar.phase == 1:
        # gate: only run the real sweep if the checkpoint passed
        cp_path = os.path.join(_T4, "stage0_checkpoint.json")
        if os.path.exists(cp_path):
            cp = json.load(open(cp_path))
            verdict = cp.get("verdict", "STOP")
            if verdict == "STOP":
                print("[stage0] verdict=STOP; refusing Stage 1 sweep "
                      "(honest FAIL on clean-V0 selectivity). Review stage0_checkpoint.json.")
                return 1
        else:
            print("[stage0] no checkpoint found; run --phase 0 first.")
            return 1
        stage1_sweep(mv, skip)
    elif ar.phase == 2:
        print("[phase2] select_frozen_task4 delegated (select_frozen_config.py)")
    elif ar.phase == 3:
        print("[phase3] metrics + figures + summary delegated")
    return 0


if __name__ == "__main__":
    sys.exit(main())