#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-topology checkpoint for Task 3 (plan Step 3).

Runs the surface-aware graph backends on a single plant1 case (H1 horizontal,
V1 vertical) and reports TOPOLOGY evidence (NOT PQ — that is acceptance):

  * G0 (candidate)   : kNN edge counts + feature medians BEFORE any gate.
  * G4 (gate)        : which cross-leaf edges survive / are pruned (retained vs
                       pruned counts), median c_n / c_t / c_d within vs cross,
                       graph connectivity after pruning.
  * G5 (gate+penalty): ditto.
  * within-leaf geodesic distortion: backend root-distance field vs the frozen
                       heat root field, per-leaf relative deviation — to check
                       the in-leaf geodesic isn't massively warped by the
                       graph/dijkstra replacement.

Same k for all backends.  Root + root-basin FROZEN from Task 2 (byte-identical
across backends).  solver_factory injection is the ONLY difference vs baseline.

Output: outputs/task3/checkpoint/<pair>/<mode>/<severity>/topology_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.headless_segmentation as hs  # noqa: E402
from geodesic_backends import (  # noqa: E402
    EuclideanGraphBackend,
    SurfaceAwareGraphBackend,
)
from run_overlap_case import (  # noqa: E402
    load_plant, apply_transform_entry,
)

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task3", "checkpoint")
_T2TRANSFORMS = os.path.join(_REPO_ROOT, "outputs", "task2", "benchmark_transforms.json")
_T2CONT = os.path.join(_REPO_ROOT, "outputs", "task2", "controlled")


def load_transformed(plant: str, pair_key: str, mode: str, severity: str):
    pd = load_plant(plant)
    gc, labels = pd["gc"], pd["labels"]
    with open(_T2TRANSFORMS) as f:
        transforms = json.load(f)
    pk = transforms[pair_key]
    a_id, b_id = pk["leaf_a_id"], pk["leaf_b_id"]
    root_index = pk["root_index"]
    sev = next(s for s in pk[mode] if s["severity"] == severity)
    leaf_a = np.where(labels == a_id)[0]
    leaf_b = np.where(labels == b_id)[0]
    g = gc
    if sev["leaf_a_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev["leaf_a_transform"], leaf_a)
    if sev["leaf_b_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev["leaf_b_transform"], leaf_b)
    basin_path = os.path.join(_T2CONT, pair_key, mode, severity, "root_basin_indices.npy")
    frozen_basin = np.load(basin_path) if os.path.exists(basin_path) else None
    return g, labels, root_index, frozen_basin, a_id, b_id


def build_backend(feature_set: str, points: NDArray, g: hs.GaussianData, k: int, cfg: dict):
    if feature_set == "G0_euclid":
        return EuclideanGraphBackend(points, k=k)
    return SurfaceAwareGraphBackend(
        points, g, k=k, feature_set=feature_set,
        lambda_n=cfg.get("lambda_n", 1.0), lambda_t=cfg.get("lambda_t", 2.0),
        p=cfg.get("p", 2.0), tau_d=cfg.get("tau_d", 3.0), tau_t=cfg.get("tau_t", 0.5),
        mutual=cfg.get("mutual", False))


def _fmt(d: dict):
    return d["median"] if isinstance(d, dict) and d.get("median") is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-key", default="plant1_green_pepper_pair_8_4")
    ap.add_argument("--mode", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--severity", default="H1")
    ap.add_argument("--k", type=int, default=256)
    ar = ap.parse_args()

    plant = ar.pair_key.split("_pair_")[0]
    g, labels, root_index, frozen_basin, a_id, b_id = load_transformed(
        plant, ar.pair_key, ar.mode, ar.severity)
    points = np.asarray(g.xyz, dtype=np.float64)

    report = {
        "pair_key": ar.pair_key, "mode": ar.mode, "severity": ar.severity,
        "root_index": int(root_index),
        "frozen_basin_size": int(len(frozen_basin)) if frozen_basin is not None else None,
        "backend_k": ar.k,
        "gt_pair_leaf_ids": [int(a_id), int(b_id)],
        "backends": {},
    }

    # candidate (G0, no gate, has features) + pure euclid + gate/penalty variants
    cfgs = {
        "G0": {"feature_set": "G0"},
        "G0_euclid": {"feature_set": "G0_euclid"},
        "G4": {"feature_set": "G4", "tau_d": 3.0, "tau_t": 0.5},
        "G5": {"feature_set": "G5", "lambda_n": 1.0, "lambda_t": 2.0,
                            "p": 2.0, "tau_d": 3.0, "tau_t": 0.5},
    }
    name_map = {"G0": "G0_candidate", "G0_euclid": "G0_euclid",
                "G4": "G4", "G5": "G5"}

    backends = {}
    for key, cfg in cfgs.items():
        bb = build_backend(cfg["feature_set"], points, g, ar.k, cfg)
        backends[key] = bb

    # ---- pruning evidence (G4/G5 vs G0 candidate cross-leaf edges) ----
    diag0 = backends["G0"].crossleaf_diagnostics(labels)
    cand_cross = diag0["n_cross_leaf_edges"]
    cand_within = diag0["n_within_leaf_edges"]

    for key, label in [("G4", "G4"), ("G5", "G5")]:
        bb = backends[key]
        diag = bb.crossleaf_diagnostics(labels)
        kept_cross = diag["n_cross_leaf_edges"]
        report[f"pruning_{label}"] = {
            "candidate_cross_edges": cand_cross,
            "retained_cross_edges": kept_cross,
            "pruned_cross_edges": int(cand_cross - kept_cross),
            "pruned_fraction_cross": float((cand_cross - kept_cross) / cand_cross)
                if cand_cross > 0 else 0.0,
            "candidate_within_edges": cand_within,
            "retained_within_edges": diag["n_within_leaf_edges"],
        }

    # ---- per-backend graph stats + feature medians ----
    for key, label in name_map.items():
        bb = backends[key]
        stats = bb.graph_stats
        diag = bb.crossleaf_diagnostics(labels)
        report["backends"][label] = {
            "graph_stats": stats,
            "crossleaf_diagnostics": diag,
        }
        print(f"\n===== {label} (k={ar.k}) =====")
        print(f"  nodes={stats['num_nodes']} undirected_edges={stats['num_edges']} "
              f"cc={stats['connected_components']} "
              f"largest_frac={stats['largest_component_fraction']:.6f} "
              f"isolated={stats['isolated_nodes']}")
        print(f"  within={diag['n_within_leaf_edges']} cross={diag['n_cross_leaf_edges']}")
        for feat in ("c_t", "c_n", "c_d"):
            w, c = _fmt(diag[f"median_{feat}_within"]), _fmt(diag[f"median_{feat}_cross"])
            ws = f"{w:.3f}" if w is not None else "  -  "
            cs = f"{c:.3f}" if c is not None else "  -  "
            print(f"  median_{feat}: within={ws}  cross={cs}")

    # ---- within-leaf geodesic distortion vs frozen heat field ----
    # heat single-source root field from Task 2 (same backend, identity-equivalent
    # for H0; for H1/V1 we compare the SAME transformed case's heat root field).
    heat_dir = os.path.join(_T2CONT, ar.pair_key, ar.mode, ar.severity)
    heat_path = os.path.join(heat_dir, "root_geodesic_single.npy")
    distortion = {}
    if os.path.exists(heat_path):
        d_heat = np.load(heat_path).astype(np.float64)
        # root-reachable mask (finite in both)
        for key, label in name_map.items():
            bb = backends[key]
            d_back = bb.compute_distance(int(root_index))
            # restrict to non-root leaves' interior (exclude basin/garbage)
            mask = np.isfinite(d_heat) & np.isfinite(d_back) & (d_heat > 1e-9)
            # per-leaf relative deviation: median |d_back - d_heat| / median(d_heat)
            leaf_id = a_id
            lm = mask & (labels == leaf_id)
            if lm.sum() > 100:
                med_h = np.median(d_heat[lm])
                rel = float(np.median(np.abs(d_back[lm] - d_heat[lm])) / med_h)
                corr = float(np.corrcoef(d_heat[lm], d_back[lm])[0, 1])
                distortion[f"{label}_leaf_{leaf_id}"] = {
                    "median_heat": float(med_h),
                    "median_abs_dev": float(np.median(np.abs(d_back[lm] - d_heat[lm]))),
                    "rel_median_deviation": rel,
                    "correlation": corr,
                    "n_points": int(lm.sum()),
                }
        report["within_leaf_distortion"] = distortion
        print("\n===== within-leaf geodesic distortion vs heat (leaf A) =====")
        for kk, v in distortion.items():
            print(f"  {kk}: rel_dev={v['rel_median_deviation']:.3f} corr={v['correlation']:.4f} "
                  f"(n={v['n_points']})")

    out = os.path.join(_OUTROOT, ar.pair_key, ar.mode, ar.severity)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "topology_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] -> {os.path.join(out, 'topology_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
