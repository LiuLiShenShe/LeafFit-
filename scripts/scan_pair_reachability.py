#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan candidate source pairs for both H and V reachability.

For each candidate pair (a,b):
  - Horizontal: 2-param (angle_a, angle_b) grid, step 5°, compute max projected
    overlap, max contact, min min_cross_leaf_distance_ratio.
  - Vertical: upper=higher-Z leaf, axis = arm × (apex→lower_centroid),
    scan sign × angle 0..180 step 1°, report achievable gap_ratio near targets
    {4,2,1,0.5} and the floor (min gap_ratio).

Usage:
    python scripts/scan_pair_reachability.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overlap_geometry import (  # noqa: E402
    fit_leaf_pca,
    axis_angle_to_matrix,
    compute_projected_overlap_fraction,
    compute_contact_fraction,
    compute_apex_gap,
)

_DATA = os.path.join(_REPO_ROOT, "data")
_BASELINE = os.path.join(_REPO_ROOT, "outputs", "baseline")


def load(plant: str):
    import core.headless_segmentation as hs
    g = hs.load_gaussian_data(os.path.join(_DATA, f"{plant}.ply"))
    gc = hs.center_gaussians(g)
    labels = np.load(os.path.join(_BASELINE, plant, "labels.npy"))
    apexes = json.load(open(os.path.join(_BASELINE, plant, "apexes.json")))
    xyz = np.asarray(gc.xyz, dtype=np.float64)
    return {"labels": labels, "apexes": apexes, "xyz": xyz}


def h_reach(pd, a_id, b_id, step=10):
    """2-param horizontal scan -> (max_overlap, best_angles, max_contact, min_min_cross)."""
    xyz, labels, apexes = pd["xyz"], pd["labels"], pd["apexes"]
    xa = xyz[labels == a_id]
    xb = xyz[labels == b_id]
    ca, cb = xa.mean(0), xb.mean(0)
    base_a = xyz[apexes[a_id - 1]["base_gaussian_index"]]
    base_b = xyz[apexes[b_id - 1]["base_gaussian_index"]]
    pca_a, pca_b = fit_leaf_pca(xa), fit_leaf_pca(xb)

    def leaf_axis(centroid, base, normal):
        d = centroid - base
        d = d / (norm(d) + 1e-12)
        ax = np.cross(normal, d)
        return ax / (norm(ax) + 1e-12)

    ax_a = leaf_axis(pca_a["centroid"], base_a, pca_a["normal"])
    ax_b = leaf_axis(pca_b["centroid"], base_b, pca_b["normal"])
    R1 = axis_angle_to_matrix(ax_a, np.radians(1.0))
    nc = R1 @ (ca - base_a) + base_a
    sa = 1 if np.dot(nc - ca, cb - ca) > 0 else -1
    R1 = axis_angle_to_matrix(ax_b, np.radians(1.0))
    nc = R1 @ (cb - base_b) + base_b
    sb = 1 if np.dot(nc - cb, ca - cb) > 0 else -1

    all_pts = np.vstack([xa, xb])
    tree = cKDTree(all_pts)
    d, _ = tree.query(all_pts, k=min(7, len(all_pts)))
    vox = float(np.median(d[:, -1])) / 2.0 + 1e-12
    spa = float(np.median(cKDTree(xa).query(xa, k=min(7, len(xa)))[0][:, -1]))
    spb = float(np.median(cKDTree(xb).query(xb, k=min(7, len(xb)))[0][:, -1]))
    avg_sp = (spa + spb) / 2

    max_ov, max_ov_ang = 0.0, (0, 0)
    max_cf, max_cf_ang = 0.0, (0, 0)
    min_mc, min_mc_ang = float("inf"), (0, 0)
    angles = range(0, 181, step)
    # precompute rotated clouds for each leaf at each angle
    na_cache, nb_cache = {}, {}
    for ang in angles:
        Ra = axis_angle_to_matrix(sa * ax_a, np.radians(ang))
        na_cache[ang] = (Ra @ (xa - base_a).T).T + base_a
        Rb = axis_angle_to_matrix(sb * ax_b, np.radians(ang))
        nb_cache[ang] = (Rb @ (xb - base_b).T).T + base_b

    # cross-leaf NN for min_cross: reuse one KDTree per cloud pair
    for aa in angles:
        na = na_cache[aa]
        tree_b = cKDTree(na)
        for ab in angles:
            nb = nb_cache[ab]
            ov = compute_projected_overlap_fraction(na, nb, voxel_size=vox)["overlap_fraction"]
            cf = compute_contact_fraction(na, nb, spacing=avg_sp)
            mc = cf["min_cross_leaf_distance_ratio"]
            if ov > max_ov:
                max_ov, max_ov_ang = ov, (aa, ab)
            if cf["contact_fraction"] > max_cf:
                max_cf, max_cf_ang = cf["contact_fraction"], (aa, ab)
            if mc < min_mc:
                min_mc, min_mc_ang = mc, (aa, ab)
    return {"max_overlap": max_ov, "max_overlap_ang": max_ov_ang,
            "max_contact": max_cf, "max_contact_ang": max_cf_ang,
            "min_min_cross_ratio": min_mc, "min_min_cross_ang": min_mc_ang,
            "centroid_dist": float(norm(ca - cb))}


def v_reach(pd, upper_id, lower_id):
    """Vertical reachability: achievable apex_gap_ratio near targets {4,2,1,0.5} + floor."""
    xyz, labels, apexes = pd["xyz"], pd["labels"], pd["apexes"]
    upper_xyz = xyz[labels == upper_id]
    lower_xyz = xyz[labels == lower_id]
    upper_base = xyz[apexes[upper_id - 1]["base_gaussian_index"]]
    upper_apex_gauss = apexes[upper_id - 1]["gaussian_index"]
    upper_indices = np.where(labels == upper_id)[0]
    apex_local_idx = int(np.where(upper_indices == upper_apex_gauss)[0][0])
    apex_pos = upper_xyz[apex_local_idx]
    lower_normal = fit_leaf_pca(lower_xyz)["normal"]
    lower_centroid = lower_xyz.mean(0)
    d_nn, _ = cKDTree(lower_xyz).query(lower_xyz, k=min(7, len(lower_xyz)))
    spacing = float(np.median(d_nn[:, -1])) + 1e-12

    arm = apex_pos - upper_base
    target_dir = lower_centroid - apex_pos
    axis = np.cross(arm, target_dir)
    axis_n = axis / (norm(axis) + 1e-12)

    pts = []
    for sign in (1, -1):
        for angle in range(0, 181):
            R = axis_angle_to_matrix(sign * axis_n, np.radians(angle))
            new_apex = R @ arm + upper_base
            gr = compute_apex_gap(new_apex, lower_xyz, lower_normal)["apex_euclidean_gap"] / spacing
            pts.append((sign, angle, gr))
    floor = min(pts, key=lambda p: p[2])[2]
    out = {"floor": float(floor)}
    for t in (4.0, 2.0, 1.0, 0.5):
        closest = min(pts, key=lambda p: (p[2] - t) ** 2)
        out[f"t{t}"] = {"gap": float(closest[2]), "sign": closest[0], "angle": closest[1]}
    return out


def main():
    plants = ["plant1_green_pepper", "plant2_rubber_tree", "plant7_black_pearl_pepper"]
    pds = {p: load(p) for p in plants}

    # candidate pairs (from centroid-distance scans)
    candidates = {
        "plant2_rubber_tree": [(3, 12), (9, 15), (11, 13), (1, 14), (4, 10)],
        "plant7_black_pearl_pepper": [(4, 5), (8, 9), (9, 11), (4, 8)],
        "plant1_green_pepper": [(6, 8), (8, 4), (4, 8), (5, 7)],
    }

    for plant, pairs in candidates.items():
        pd = pds[plant]
        for (a, b) in pairs:
            h = h_reach(pd, a, b)
            # vertical: upper = higher Z
            ca = pd["xyz"][pd["labels"] == a].mean(0)
            cb = pd["xyz"][pd["labels"] == b].mean(0)
            upper, lower = (a, b) if ca[2] >= cb[2] else (b, a)
            v = v_reach(pd, upper, lower)
            print(f"\n{plant} ({a},{b}) centroid_dist={h['centroid_dist']:.3f} upper={upper} lower={lower}")
            print(f"  H: max_ov={h['max_overlap']:.3f}@({h['max_overlap_ang'][0]},{h['max_overlap_ang'][1]}) "
                  f"max_cf={h['max_contact']:.3f}@({h['max_contact_ang'][0]},{h['max_contact_ang'][1]}) "
                  f"min_cross={h['min_min_cross_ratio']:.3f}@({h['min_min_cross_ang'][0]},{h['min_min_cross_ang'][1]})")
            print(f"  V: floor={v['floor']:.3f} t4={v['t4.0']['gap']:.3f}@{v['t4.0']['angle']} "
                  f"t2={v['t2.0']['gap']:.3f}@{v['t2.0']['angle']} "
                  f"t1={v['t1.0']['gap']:.3f}@{v['t1.0']['angle']} "
                  f"t0.5={v['t0.5']['gap']:.3f}@{v['t0.5']['angle']}")


if __name__ == "__main__":
    main()
