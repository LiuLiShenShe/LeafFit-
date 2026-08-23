#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 5 — Phase-0 gate: real-observation identity cue on DENSE 3DGS cloud.

Proves the spec's Phase-2 gate on the real view signature WITHOUT invoking the
heat solver (which hangs on >~100K-point potpourri3d clouds). We build kNN
edges, compute per-edge c_vis / c_app / c_occ directly from the REAL view
signature via core.geodesic_backends._mv_edge_features, then split edges into
within-leaf vs cross-leaf using GT labels and report the median c_mv margin.

This mirrors the G6/G7 gate used inside SurfaceAwareGraphBackend, but run as a
regression check so we never touch potpourri3d for the gate.

Usage:
    python scripts/run_task5_phase0_gate.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

REPO = "/data/fj/LeafFit论文复现及修改/leaf_fit"
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.real_observation import load_dense_gaussian_plant, load_dense_observations  # noqa: E402
from core.geodesic_backends import _mv_edge_features  # noqa: E402

_BASE = os.path.join(REPO, "outputs", "task5", "dense_baseline")
_VSIG = os.path.join(REPO, "outputs", "task5", "projection_cache")
_TRANSF = os.path.join(REPO, "outputs", "task5", "benchmark_transforms.json")

W_VIS, W_APP, W_OCC = 0.4, 0.3, 0.3
K = 64


def load_real_sig(plant: str):
    z = np.load(os.path.join(_VSIG, plant, "real_viewsig_dense.npz"))
    return {
        "visible": z["visible"], "appear_sig": z["appear_sig"],
        "visibility_fraction": z["visibility_fraction"],
        "depth": z["depth"], "uv": z["uv"], "n_views": int(z["visible"].shape[0]),
    }


def knn_edges(xyz: np.ndarray, k: int = K):
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k + 1)
    rows = np.repeat(np.arange(len(xyz)), k)
    cols = idx[:, 1:].ravel()
    return rows, cols, np.linalg.norm(xyz[rows] - xyz[cols], axis=1)


def phase0_gate_for_pair(plant, labels, vsig, pair_key, mode, severity, pk, leaf_a, leaf_b):
    # apply transform entry
    from overlap_geometry import transform_leaf_gaussians
    g = load_dense_gaussian_plant(plant)
    xyz = g.xyz.astype(np.float64)
    la = np.where(labels == leaf_a)[0]
    lb = np.where(labels == leaf_b)[0]
    se = next(s for s in pk[mode] if s["severity"] == severity)
    ta = se["leaf_a_transform"]; tb = se["leaf_b_transform"]

    def apply(t, idx):
        if t.get("pivot") is None:
            return
        R = np.asarray(t["R"], float)
        piv = np.asarray(t["pivot"], float)
        xyz[idx] = (R @ (xyz[idx] - piv).T).T + piv
    apply(ta, la)
    apply(tb, lb)

    rows, cols, d = knn_edges(xyz, K)
    visible = np.asarray(vsig["visible"], np.uint8)
    appear = np.asarray(vsig["appear_sig"], np.float32)
    depth = np.asarray(vsig["depth"], np.float32)
    uv = np.asarray(vsig["uv"], np.float32)
    c_vis, c_app, c_occ = _mv_edge_features(rows, cols, visible, appear, depth, uv, vsig["n_views"])
    c_mv = W_VIS * c_vis + W_APP * c_app + W_OCC * c_occ

    same = (labels[rows] == labels[cols]) & (labels[rows] > 0)
    diff = (labels[rows] != labels[cols]) & (labels[rows] > 0) & (labels[cols] > 0)
    med_within = float(np.median(c_mv[same])) if same.sum() else float("nan")
    med_cross = float(np.median(c_mv[diff])) if diff.sum() else float("nan")
    return {
        "pair_key": pair_key, "mode": mode, "severity": severity,
        "n_edges": int(len(rows)), "n_within": int(same.sum()), "n_cross": int(diff.sum()),
        "median_cmv_within": med_within, "median_cmv_cross": med_cross,
        "margin": med_within - med_cross,
    }


def main():
    btrans = json.load(open(_TRANSF))
    rows = []
    for pair_key, pk in btrans.items():
        plant = pk["plant"]
        labels = np.load(os.path.join(_BASE, plant, "labels.npy"))
        vsig = load_real_sig(plant)
        la, lb = pk["leaf_a_id"], pk["leaf_b_id"]
        for mode in ("horizontal", "vertical"):
            for sev in (("H0", "H1") if mode == "horizontal" else ("V0", "V1")):
                r = phase0_gate_for_pair(plant, labels, vsig, pair_key, mode, sev, pk, la, lb)
                rows.append(r)
                print(f"{pair_key:24s} {mode:10s} {sev}  within={r['median_cmv_within']:.3f} "
                      f"cross={r['median_cmv_cross']:.3f}  margin={r['margin']:+.3f}  "
                      f"edges={r['n_edges']} (w={r['n_within']},c={r['n_cross']})", flush=True)
    # overall gate: all margins >= 0.03?
    margins = np.array([r["margin"] for r in rows])
    passed = bool((margins >= 0.03).all())
    summary = {
        "gate_passed": passed,
        "min_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "n_cases": len(rows),
        "threshold": 0.03,
        "cases": rows,
    }
    out = os.path.join(REPO, "outputs", "task5", "phase0_real_observation_gate.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n[PHASE-0 GATE] passed={passed}  min_margin={summary['min_margin']:.3f} "
          f"mean_margin={summary['mean_margin']:.3f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
