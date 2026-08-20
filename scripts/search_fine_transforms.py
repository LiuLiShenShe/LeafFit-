#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0: fine boundary sweep transforms (Task 3).

Freezes controlled-overlap transforms BETWEEN H0-H1 and V0-V1, so the baseline
failure boundary is localized precisely before any surface-aware backend runs:

  HF1-HF4 : contact_fraction targets {0.010, 0.020, 0.030, 0.040}
             (H0 contact ~ 0.0,  H1 achieved contact ~ 0.04)
  VF1-VF4 : apex_gap_ratio targets {6.0, 5.0, 4.5, 4.0}
             (V0 = identity, apex gap ratio 28-63 measured;  V1 target 4.0)

Only GEOMETRY is computed here. LeafFit is NOT run until the transforms are
frozen and reviewed. Output (FROZEN — single write):
  outputs/task3/fine_boundary_transforms.json

Schema mirrors outputs/task2/benchmark_transforms.json exactly (same keys:
plant / leaf_a_id / leaf_b_id / root_index / upper_leaf_id / lower_leaf_id /
spacing_a / spacing_b / horizontal[] / vertical[]), so run-task3-case and
metric code paths stay identical. Severity labels are HF1-HF4 / VF1-VF4.

Per pair: the coarse/fine grid used by Task 2 is re-run with the fine targets,
reusing search_overlap_transforms geometry helpers verbatim.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"), os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from search_overlap_transforms import (  # noqa: E402
    load_source_pairs,
    load_plant,
    get_leaf_xyz,
    compute_leaf_spacing,
    fit_leaf_pca,
    compute_leaf_axis,
    choose_rotation_sign,
    identity_transform_entry,
    axis_angle_to_matrix,
    compute_contact_fraction,
    compute_projected_overlap_fraction,
    compute_vertical_rotation_axis,
    _h_eval,
    _h_score,
    _v_eval,
    _v_score,
)

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task3")

# ---- Fine horizontal targets (single-point contact_fraction) ----
HF_TARGETS = {
    "HF1": (0.010, 0.010),
    "HF2": (0.020, 0.020),
    "HF3": (0.030, 0.030),
    "HF4": (0.040, 0.040),
}
HF_LEVELS = ["HF1", "HF2", "HF3", "HF4"]

# ---- Fine vertical targets (apex_gap_ratio: d(upper_apex, lower_surface)/spacing) ----
VF_TARGETS = {
    "VF1": 6.0,
    "VF2": 5.0,
    "VF3": 4.5,
    "VF4": 4.0,
}
VF_LEVELS = ["VF1", "VF2", "VF3", "VF4"]


def search_horizontal_fine(
    xyz_a: NDArray, xyz_b: NDArray,
    base_a: NDArray, base_b: NDArray,
    spacing: float,
) -> list[dict]:
    """Same coarse→fine scheme as search_horizontal, but only HF1-HF4 targets."""
    pca_a = fit_leaf_pca(xyz_a)
    pca_b = fit_leaf_pca(xyz_b)
    axis_a = compute_leaf_axis(pca_a["normal"], pca_a["centroid"], base_a)
    axis_b = compute_leaf_axis(pca_b["normal"], pca_b["centroid"], base_b)
    sign_a = choose_rotation_sign(axis_a, base_a, xyz_a, xyz_b.mean(0))
    sign_b = choose_rotation_sign(axis_b, base_b, xyz_b, xyz_a.mean(0))

    _all = np.vstack([xyz_a, xyz_b])
    from scipy.spatial import cKDTree
    _tree = cKDTree(_all)
    _d, _ = _tree.query(_all, k=min(7, len(_all)))
    voxel_size = float(np.median(_d[:, -1])) / 2.0 + 1e-12

    # coarse: 0..180 step 5° (same grid as Task 2)
    angles = range(0, 181, 5)
    na_cache = {ang: (axis_angle_to_matrix(sign_a * axis_a, np.radians(ang)) @
                      (xyz_a - base_a).T).T + base_a for ang in angles}
    nb_cache = {ang: (axis_angle_to_matrix(sign_b * axis_b, np.radians(ang)) @
                      (xyz_b - base_b).T).T + base_b for ang in angles}
    coarse = {}
    for aa in angles:
        for ab in angles:
            coarse[(aa, ab)] = _h_eval(na_cache[aa], nb_cache[ab], spacing, voxel_size)

    results = []
    for level in HF_LEVELS:
        target = HF_TARGETS[level]
        best_key = min(coarse, key=lambda k: _h_score(coarse[k], target))
        best_aa, best_ab = best_key

        # fine: ±3° step 0.5° around best coarse cell
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
        results.append(entry)

    return results


def search_vertical_fine(
    upper_xyz: NDArray, lower_xyz: NDArray,
    upper_base: NDArray,
    upper_normal: NDArray, lower_normal: NDArray,
    upper_apex_idx: int,
    spacing: float,
) -> list[dict]:
    """Same coarse→fine scheme as search_vertical, but only VF1-VF4 targets.

    Targets are clamped to just below the identity apex gap (rotating the apex
    toward the lower leaf can only DECREASE the gap). Identity ratios measured
    28-63 for all three pairs, so {6,5,4.5,4} are reachable without clamping.
    """
    apex_pos = upper_xyz[upper_apex_idx]
    lower_centroid = lower_xyz.mean(0)
    axis = compute_vertical_rotation_axis(apex_pos, upper_base, lower_centroid)

    # choose sign so apex moves TOWARD the lower centroid
    R_test = axis_angle_to_matrix(axis, np.radians(1.0))
    new_apex_test = R_test @ (apex_pos - upper_base) + upper_base
    sign = 1 if np.linalg.norm(new_apex_test - lower_centroid) < np.linalg.norm(apex_pos - lower_centroid) else -1

    # identity gap for clamping
    r_id = _v_eval(upper_xyz, lower_xyz, upper_base, lower_normal,
                   axis, sign, 0.0, upper_apex_idx, spacing)
    identity_ratio = float(r_id["apex_euclidean_gap_ratio"])
    max_reachable = identity_ratio - 0.05  # keep a hair below identity

    # coarse: 0..180 step 2°
    coarse = []
    for angle in np.arange(0, 182, 2.0):
        r = _v_eval(upper_xyz, lower_xyz, upper_base, lower_normal,
                    axis, sign, angle, upper_apex_idx, spacing)
        coarse.append(r)

    results = []
    for level in VF_LEVELS:
        target = min(VF_TARGETS[level], max_reachable)
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
            "t": [0.0, 0.0, 0.0],
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
            "identity_apex_gap_ratio": identity_ratio,
        }
        results.append(entry)

    return results


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

        # --- Horizontal fine ---
        print(f"[HF] {pair_key}: searching fine horizontal ...")
        h_results = search_horizontal_fine(xyz_a, xyz_b, base_a, base_b, avg_spacing)
        for hr in h_results:
            print(f"  {hr['severity']}: contact={hr['achieved_contact_fraction']:.4f} "
                  f"overlap={hr['achieved_projected_overlap']:.4f}")

        # --- Vertical fine: determine upper/lower (mirror of search_overlap main) ---
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

        upper_apex_gauss = [p for p in [pair["leaf_a"], pair["leaf_b"]]
                           if p["leaf_id"] == upper_id][0]["apex_gaussian_index"]
        upper_indices = np.where(labels == upper_id)[0]
        apex_local_idx = int(np.where(upper_indices == upper_apex_gauss)[0][0]) \
            if upper_apex_gauss in upper_indices else 0
        upper_spacing = compute_leaf_spacing(upper_xyz)

        print(f"[VF] {pair_key}: upper=leaf{upper_id} lower=leaf{lower_id} apex_local_idx={apex_local_idx}")
        v_results_raw = search_vertical_fine(
            upper_xyz, lower_xyz, upper_base,
            upper_normal, lower_normal, apex_local_idx, upper_spacing)

        # Map upper/lower transforms to leaf_a/leaf_b
        v_results = []
        upper_is_a = (upper_id == a_id)
        for vr in v_results_raw:
            vr_mapped = dict(vr)
            if "upper_transform" in vr:
                vr_mapped["leaf_a_transform"] = vr["upper_transform"] if upper_is_a else vr["lower_transform"]
                vr_mapped["leaf_b_transform"] = vr["lower_transform"] if upper_is_a else vr["upper_transform"]
                vr_mapped.pop("upper_transform", None)
                vr_mapped.pop("lower_transform", None)
            v_results.append(vr_mapped)

        for vr in v_results:
            print(f"  {vr['severity']}: apex_gap_ratio={vr['achieved_apex_gap_ratio']:.3f} "
                  f"(target {VF_TARGETS[vr['severity']]})")

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

    out_path = os.path.join(_OUTROOT, "fine_boundary_transforms.json")
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] frozen {len(transforms)} pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
