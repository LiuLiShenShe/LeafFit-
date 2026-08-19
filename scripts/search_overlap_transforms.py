#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search and freeze controlled overlap transforms (Task 2 benchmark).

Design (revised after reachability scan):
  Horizontal severity is driven by CONTACT_FRACTION (cross-leaf proximity),
  because that is the quantity that makes geodesic paths from the two apexes
  share an ambiguous region (Fig. 13a). Projected overlap is capped by
  base-anchor rotation geometry (max ~0.17 for separated leaves) and is
  REPORTED, not frozen.
  Vertical severity is driven by apex_gap_ratio = d(upper_apex, lower_surface)
  / spacing (Fig. 13b). Upper leaf rotates around its base (t=0) with axis
  = arm × (apex → lower-centroid).

For each pair × mode × severity:
  Coarse grid -> select closest to target -> fine grid (±2° step 0.5°).

Output: outputs/task2/benchmark_transforms.json (FROZEN — single write).

CRITICAL: Only geometry metrics computed here. LeafFit is NOT run until the
transforms are frozen and manually reviewed (Step-3 checkpoint).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overlap_geometry import (  # noqa: E402
    fit_leaf_pca,
    axis_angle_to_matrix,
    transform_leaf_gaussians,
    compute_projected_overlap_fraction,
    compute_contact_fraction,
    compute_apex_gap,
    compute_vertical_gap_and_spacing,
)
from gaussian_utils import GaussianData  # noqa: E402
import core.headless_segmentation as hs  # noqa: E402

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_BASELINE = os.path.join(_REPO_ROOT, "outputs", "baseline")
_DATA = os.path.join(_REPO_ROOT, "data")

# ---- Horizontal targets (primary = contact_fraction) ----
H_TARGETS = {
    "H0": {"target_contact": (0.0, 0.005)},
    "H1": {"target_contact": (0.02, 0.06)},
    "H2": {"target_contact": (0.06, 0.12)},
    "H3": {"target_contact": (0.12, 0.20)},
    "H4": {"target_contact": (0.18, 0.30)},  # widened: not all pairs reach 0.30
}

# ---- Vertical targets (apex_gap_ratio: d(upper_apex, lower_surface) / spacing) ----
V_TARGETS = {
    "V0": {"target_apex_gap_ratio": None},  # identity
    "V1": {"target_apex_gap_ratio": 4.0},
    "V2": {"target_apex_gap_ratio": 2.0},
    "V3": {"target_apex_gap_ratio": 1.0},
    "V4": {"target_apex_gap_ratio": 0.5},
}


def load_source_pairs() -> dict:
    path = os.path.join(_OUTROOT, "source_pairs.json")
    with open(path) as f:
        return json.load(f)


def load_plant(plant: str):
    g = hs.load_gaussian_data(os.path.join(_DATA, f"{plant}.ply"))
    gc = hs.center_gaussians(g)
    labels = np.load(os.path.join(_BASELINE, plant, "labels.npy"))
    apexes = json.load(open(os.path.join(_BASELINE, plant, "apexes.json")))
    return {"g": g, "gc": gc, "labels": labels, "apexes": apexes}


def get_leaf_xyz(gc: GaussianData, labels: NDArray, leaf_id: int) -> NDArray[np.float64]:
    return np.asarray(gc.xyz[labels == leaf_id], dtype=np.float64)


def compute_leaf_spacing(xyz: NDArray) -> float:
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(7, len(xyz)))
    return float(np.median(d[:, -1])) + 1e-12


def compute_leaf_axis(normal: NDArray, centroid: NDArray, base: NDArray) -> NDArray:
    """In-plane axis perpendicular to the petiole direction (centroid→base)."""
    direction = centroid - base
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    axis = np.cross(normal, direction)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-6:
        v = direction - np.dot(direction, normal) * normal
        axis = v
        ax_norm = np.linalg.norm(axis)
    return axis / (ax_norm + 1e-12)


def choose_rotation_sign(axis: NDArray, base: NDArray, leaf_xyz: NDArray,
                          other_centroid: NDArray) -> int:
    """Choose sign so that rotated leaf centroid moves toward *other_centroid*."""
    centroid = leaf_xyz.mean(0)
    R = axis_angle_to_matrix(axis, np.radians(1.0))
    new_centroid = R @ (centroid - base) + base
    return 1 if np.dot(new_centroid - centroid, other_centroid - centroid) > 0 else -1


def identity_transform_entry(severity: str, mode: str) -> dict:
    I3 = np.eye(3).tolist()
    zeros = [0.0, 0.0, 0.0]
    return {
        "severity": severity,
        "mode": mode,
        "leaf_a_transform": {"pivot": None, "R": I3, "t": zeros, "angle_deg": 0.0},
        "leaf_b_transform": {"pivot": None, "R": I3, "t": zeros, "angle_deg": 0.0},
        "case_type": "identity_control",
    }


# ---------------------------------------------------------------------------
# Horizontal search (2-parameter asymmetric rotation)
# ---------------------------------------------------------------------------

def _h_eval(na: NDArray, nb: NDArray, spacing: float, voxel_size: float) -> dict:
    ov = compute_projected_overlap_fraction(na, nb, voxel_size=voxel_size)
    cf = compute_contact_fraction(na, nb, spacing=spacing)
    tree_b = cKDTree(nb)
    d_a_to_b, _ = tree_b.query(na, k=1)
    return {
        "projected_overlap_fraction": ov["overlap_fraction"],
        "contact_fraction": cf["contact_fraction"],
        "min_cross_leaf_distance_ratio": float(d_a_to_b.min() / spacing),
    }


def _h_score(res: dict, target_contact: tuple) -> float:
    lo, hi = target_contact
    center = (lo + hi) / 2
    return abs(res["contact_fraction"] - center) / max(hi - lo, 0.01)


def search_horizontal(
    xyz_a: NDArray, xyz_b: NDArray,
    base_a: NDArray, base_b: NDArray,
    spacing: float,
) -> list[dict]:
    pca_a = fit_leaf_pca(xyz_a)
    pca_b = fit_leaf_pca(xyz_b)
    axis_a = compute_leaf_axis(pca_a["normal"], pca_a["centroid"], base_a)
    axis_b = compute_leaf_axis(pca_b["normal"], pca_b["centroid"], base_b)
    sign_a = choose_rotation_sign(axis_a, base_a, xyz_a, xyz_b.mean(0))
    sign_b = choose_rotation_sign(axis_b, base_b, xyz_b, xyz_a.mean(0))

    # H0 = identity
    h0 = identity_transform_entry("H0", "horizontal")
    h0["achieved_projected_overlap"] = 0.0
    h0["achieved_contact_fraction"] = 0.0
    h0["achieved_min_cross_leaf_distance_ratio"] = None
    results = {"H0": h0}

    # precompute voxel size (median NN spacing / 2)
    _all = np.vstack([xyz_a, xyz_b])
    _tree = cKDTree(_all)
    _d, _ = _tree.query(_all, k=min(7, len(_all)))
    voxel_size = float(np.median(_d[:, -1])) / 2.0 + 1e-12

    # coarse: both angles 0..180 step 5°  (large angles = folded leaves
    # forcing genuine interpenetration; measured peak contact at 120-180°)
    coarse = {}
    angles = range(0, 181, 5)
    # precompute rotated clouds
    na_cache = {ang: (axis_angle_to_matrix(sign_a * axis_a, np.radians(ang)) @
                      (xyz_a - base_a).T).T + base_a for ang in angles}
    nb_cache = {ang: (axis_angle_to_matrix(sign_b * axis_b, np.radians(ang)) @
                      (xyz_b - base_b).T).T + base_b for ang in angles}
    for aa in angles:
        for ab in angles:
            coarse[(aa, ab)] = _h_eval(na_cache[aa], nb_cache[ab], spacing, voxel_size)

    for level in ["H1", "H2", "H3", "H4"]:
        target = H_TARGETS[level]["target_contact"]
        # find best coarse cell
        best_key = min(coarse, key=lambda k: _h_score(coarse[k], target))
        best_aa, best_ab = best_key

        # fine: ±3° step 0.5° around best coarse cell (2-param)
        fine = {}
        for aa in np.arange(max(0, best_aa - 3.0), best_aa + 3.5, 0.5):
            Ra = axis_angle_to_matrix(sign_a * axis_a, np.radians(aa))
            na = (Ra @ (xyz_a - base_a).T).T + base_a
            for ab in np.arange(max(0, best_ab - 3.0), best_ab + 3.5, 0.5):
                Rb = axis_angle_to_matrix(sign_b * axis_b, np.radians(ab))
                nb = (Rb @ (xyz_b - base_b).T).T + base_b
                fine[(float(aa), float(ab))] = _h_eval(na, nb, spacing, voxel_size)

        best_key = min(fine, key=lambda k: _h_score(fine[k], target))
        aa, ab = best_key
        res = fine[best_key]

        entry = {
            "severity": level, "mode": "horizontal",
            "leaf_a_transform": {
                "pivot": base_a.tolist(), "axis": (sign_a * axis_a).tolist(),
                "R": axis_angle_to_matrix(sign_a * axis_a, np.radians(aa)).tolist(),
                "t": [0.0, 0.0, 0.0],
                "angle_deg": float(aa),
            },
            "leaf_b_transform": {
                "pivot": base_b.tolist(), "axis": (sign_b * axis_b).tolist(),
                "R": axis_angle_to_matrix(sign_b * axis_b, np.radians(ab)).tolist(),
                "t": [0.0, 0.0, 0.0],
                "angle_deg": float(ab),
            },
            "achieved_projected_overlap": res["projected_overlap_fraction"],
            "achieved_contact_fraction": res["contact_fraction"],
            "achieved_min_cross_leaf_distance_ratio": res["min_cross_leaf_distance_ratio"],
            "case_type": "rotation_only",
        }
        results[level] = entry

    return list(results.values())


# ---------------------------------------------------------------------------
# Vertical search (base-anchor rotation of upper leaf, t=0)
# ---------------------------------------------------------------------------

def compute_vertical_rotation_axis(apex_pos: NDArray, upper_base: NDArray,
                                   lower_centroid: NDArray) -> NDArray:
    """Axis = arm × (apex→lower_centroid) — rotates the apex toward the lower leaf."""
    arm = apex_pos - upper_base
    target_dir = lower_centroid - apex_pos
    axis = np.cross(arm, target_dir)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-6:
        # degenerate: arm parallel to target — use arbitrary in-plane axis
        ref = np.array([0.0, 0.0, 1.0])
        axis = np.cross(arm, ref)
        ax_norm = np.linalg.norm(axis)
    return axis / (ax_norm + 1e-12)


def _v_eval(
    upper_xyz: NDArray, lower_xyz: NDArray,
    upper_base: NDArray, lower_normal: NDArray,
    axis: NDArray, sign: int,
    angle_deg: float,
    upper_apex_idx: int,
    spacing: float,
) -> dict:
    R = axis_angle_to_matrix(sign * axis, np.radians(angle_deg))
    new_upper = (R @ (upper_xyz - upper_base).T).T + upper_base
    new_apex = new_upper[upper_apex_idx]

    gap_info = compute_apex_gap(new_apex, lower_xyz, lower_normal, upper_xyz=new_upper)
    vs_info = compute_vertical_gap_and_spacing(new_upper, lower_xyz, lower_normal)

    return {
        "angle_deg": angle_deg,
        "apex_euclidean_gap_ratio": gap_info["apex_euclidean_gap"] / spacing,
        "apex_normal_gap_ratio": gap_info["apex_normal_gap"] / spacing,
        "whole_leaf_median_gap_ratio": vs_info["median_euclidean_gap"] / spacing,
        "apex_euclidean_gap": gap_info["apex_euclidean_gap"],
        "apex_normal_gap": gap_info["apex_normal_gap"],
        "R": R,
        "new_upper_xyz": new_upper,
        "normal_alignment_cos": vs_info["normal_alignment_cos"],
    }


def _v_score(result: dict, target_ratio: float) -> float:
    return abs(result["apex_euclidean_gap_ratio"] - target_ratio)


def search_vertical(
    upper_xyz: NDArray, lower_xyz: NDArray,
    upper_base: NDArray,
    upper_normal: NDArray, lower_normal: NDArray,
    upper_apex_idx: int,
    spacing: float,
) -> list[dict]:
    apex_pos = upper_xyz[upper_apex_idx]
    lower_centroid = lower_xyz.mean(0)
    axis = compute_vertical_rotation_axis(apex_pos, upper_base, lower_centroid)

    # choose sign so the apex moves TOWARD the lower centroid
    R_test = axis_angle_to_matrix(axis, np.radians(1.0))
    new_apex_test = R_test @ (apex_pos - upper_base) + upper_base
    if np.linalg.norm(new_apex_test - lower_centroid) < np.linalg.norm(apex_pos - lower_centroid):
        sign = 1
    else:
        sign = -1

    v0 = identity_transform_entry("V0", "vertical")
    v0["achieved_apex_gap_ratio"] = None
    results = {"V0": v0}

    # coarse: 0–180° step 2°
    coarse = []
    for angle in np.arange(0, 182, 2.0):
        r = _v_eval(upper_xyz, lower_xyz, upper_base, lower_normal,
                    axis, sign, angle, upper_apex_idx, spacing)
        coarse.append(r)

    for level in ["V1", "V2", "V3", "V4"]:
        target = V_TARGETS[level]["target_apex_gap_ratio"]
        best_r = min(coarse, key=lambda r: _v_score(r, target))
        best_angle = best_r["angle_deg"]

        # fine: ±2° step 0.5°
        fine = []
        for angle in np.arange(max(0, best_angle - 2.0), best_angle + 2.5, 0.5):
            r = _v_eval(upper_xyz, lower_xyz, upper_base, lower_normal,
                        axis, sign, angle, upper_apex_idx, spacing)
            fine.append(r)

        chosen = min(fine, key=lambda r: _v_score(r, target))

        upper_rot = {
            "pivot": upper_base.tolist(),
            "axis": (sign * axis).tolist(),
            "R": chosen["R"].tolist(),
            "t": [0.0, 0.0, 0.0],  # t=0 for clean causal evidence
            "angle_deg": float(chosen["angle_deg"]),
        }
        entry = {
            "severity": level, "mode": "vertical",
            "upper_transform": upper_rot,
            "lower_transform": identity_transform_entry(level, "vertical")["leaf_a_transform"],
            "achieved_apex_gap_ratio": float(chosen["apex_euclidean_gap_ratio"]),
            "achieved_apex_normal_gap_ratio": float(chosen["apex_normal_gap_ratio"]),
            "achieved_whole_leaf_median_gap_ratio": float(chosen["whole_leaf_median_gap_ratio"]),
            "achieved_normal_alignment_cos": float(chosen["normal_alignment_cos"]),
            "case_type": "rotation_only",
        }
        results[level] = entry

    return list(results.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(_OUTROOT, exist_ok=True)

    sp = load_source_pairs()
    frozen_roots = json.load(open(os.path.join(_REPO_ROOT, "outputs", "frozen_roots.json")))
    transforms = {}

    for pair in sp["pairs"]:
        plant = pair["plant"]
        a_id = pair["leaf_a_id"]
        b_id = pair["leaf_b_id"]
        pair_key = f"{plant}_pair_{a_id}_{b_id}"
        pd = load_plant(plant)
        gc, labels = pd["gc"], pd["labels"]
        xyz_a = get_leaf_xyz(gc, labels, a_id)
        xyz_b = get_leaf_xyz(gc, labels, b_id)
        base_a = np.asarray(pair["leaf_a"]["base_xyz"], dtype=np.float64)
        base_b = np.asarray(pair["leaf_b"]["base_xyz"], dtype=np.float64)
        spacing_a = compute_leaf_spacing(xyz_a)
        spacing_b = compute_leaf_spacing(xyz_b)
        avg_spacing = (spacing_a + spacing_b) / 2.0

        pca_a = fit_leaf_pca(xyz_a)
        pca_b = fit_leaf_pca(xyz_b)

        # --- Horizontal ---
        print(f"[H] {pair_key}: searching horizontal ...")
        h_results = search_horizontal(xyz_a, xyz_b, base_a, base_b, avg_spacing)
        for hr in h_results:
            print(f"  {hr['severity']}: contact={hr.get('achieved_contact_fraction',0):.3f} "
                  f"overlap={hr.get('achieved_projected_overlap',0):.3f}")

        # --- Vertical: determine upper/lower ---
        centroid_a = xyz_a.mean(0)
        centroid_b = xyz_b.mean(0)
        if centroid_a[2] >= centroid_b[2]:
            upper_id, lower_id, upper_xyz, lower_xyz = a_id, b_id, xyz_a, xyz_b
            upper_base, lower_base = base_a, base_b
            upper_normal, lower_normal = pca_a["normal"], pca_b["normal"]
        else:
            upper_id, lower_id, upper_xyz, lower_xyz = b_id, a_id, xyz_b, xyz_a
            upper_base, lower_base = base_b, base_a
            upper_normal, lower_normal = pca_b["normal"], pca_a["normal"]

        # apex index in upper leaf array
        upper_apex_gauss = [p for p in [pair["leaf_a"], pair["leaf_b"]]
                           if p["leaf_id"] == upper_id][0]["apex_gaussian_index"]
        upper_mask_arr = labels == upper_id
        upper_indices = np.where(upper_mask_arr)[0]
        apex_local_idx = int(np.where(upper_indices == upper_apex_gauss)[0][0]) \
            if upper_apex_gauss in upper_indices else 0
        upper_spacing = compute_leaf_spacing(upper_xyz)

        print(f"[V] {pair_key}: upper=leaf{upper_id} lower=leaf{lower_id} apex_local_idx={apex_local_idx}")
        v_results_raw = search_vertical(
            upper_xyz, lower_xyz, upper_base,
            upper_normal, lower_normal, apex_local_idx, upper_spacing)

        # Map upper/lower transforms to leaf_a/leaf_b based on which leaf is upper
        v_results = []
        upper_is_a = (upper_id == a_id)
        for vr in v_results_raw:
            vr_mapped = dict(vr)
            if "upper_transform" in vr:
                # V1-V4: map upper/lower to leaf_a/leaf_b
                vr_mapped["leaf_a_transform"] = vr["upper_transform"] if upper_is_a else vr["lower_transform"]
                vr_mapped["leaf_b_transform"] = vr["lower_transform"] if upper_is_a else vr["upper_transform"]
                vr_mapped.pop("upper_transform", None)
                vr_mapped.pop("lower_transform", None)
            else:
                # V0: identity entry already has leaf_a/b_transform; just re-key for consistency
                pass
            v_results.append(vr_mapped)

        for vr in v_results:
            print(f"  {vr['severity']}: apex_gap_ratio={vr.get('achieved_apex_gap_ratio','id')}")

        transforms[pair_key] = {
            "plant": plant,
            "leaf_a_id": a_id,
            "leaf_b_id": b_id,
            "root_index": frozen_roots[plant]["root_index"],
            "upper_leaf_id": upper_id,
            "lower_leaf_id": lower_id,
            "spacing_a": spacing_a,
            "spacing_b": spacing_b,
            "horizontal": h_results,
            "vertical": v_results,
        }

    out_path = os.path.join(_OUTROOT, "benchmark_transforms.json")
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] frozen {len(transforms)} pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
