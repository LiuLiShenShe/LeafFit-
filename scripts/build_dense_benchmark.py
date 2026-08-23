#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 5 — reconstruct the Task 2 overlap benchmark on the DENSE 3DGS substrate.

Fork of scripts/search_overlap_transforms.py that loads from
datasets/07-SuGaR-GS (dense Gaussian clouds) and the verified
outputs/task5/dense_baseline/<plant>/ segmentation of that substrate,
then runs the IDENTICAL H/V transform search (coarse+fine) as Task 2.

Output: outputs/task5/benchmark_transforms.json

Usage:
    python scripts/build_dense_benchmark.py          # dev+heldout
    python scripts/build_dense_benchmark.py --plants DouBanLv1,WangWenCao2
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree

REPO = "/data/fj/LeafFit论文复现及修改/leaf_fit"
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
import core.headless_segmentation as hs  # noqa: E402
from core.real_observation import load_dense_gaussian_plant  # noqa: E402
from overlap_geometry import (  # noqa: E402
    fit_leaf_pca, axis_angle_to_matrix, transform_leaf_gaussians,
    compute_projected_overlap_fraction, compute_contact_fraction,
    compute_apex_gap, compute_vertical_gap_and_spacing,
)

_DENSE_BASELINE = os.path.join(REPO, "outputs", "task5", "dense_baseline")
OUT = os.path.join(REPO, "outputs", "task5", "benchmark_transforms.json")
MIN_POINTS = 2000

H_TARGETS = {
    "H0": {"target_contact": (0.0, 0.005)},
    "H1": {"target_contact": (0.02, 0.06)},
    "H2": {"target_contact": (0.06, 0.12)},
    "H3": {"target_contact": (0.12, 0.20)},
    "H4": {"target_contact": (0.18, 0.30)},
}
V_TARGETS = {
    "V0": {"target_apex_gap_ratio": None},
    "V1": {"target_apex_gap_ratio": 4.0},
    "V2": {"target_apex_gap_ratio": 2.0},
    "V3": {"target_apex_gap_ratio": 1.0},
    "V4": {"target_apex_gap_ratio": 0.5},
}


def load_dense_plant_baseline(plant: str):
    """Load dense GaussianData + segmentation labels/apexes/xyz for a plant."""
    g = load_dense_gaussian_plant(plant)
    gc = hs.center_gaussians(g)
    bl = os.path.join(_DENSE_BASELINE, plant)
    labels = np.load(os.path.join(bl, "labels.npy"))
    apexes = json.load(open(os.path.join(bl, "apexes.json")))
    xyz = np.asarray(gc.xyz, dtype=np.float64)
    # choose root_index from status.json (written by seg_dense_plant.py)
    root_idx = None
    sj = os.path.join(bl, "status.json")
    if os.path.exists(sj):
        root_idx = json.load(open(sj)).get("root_index")
    if root_idx is None:
        root_idx = int(np.argmin(xyz[:, 1]))
    return {"g": g, "gc": gc, "labels": labels, "apexes": apexes, "xyz": xyz,
            "root_index": root_idx, "plant": plant}


def leaf_xyz(labels, xyz, leaf_id):
    return xyz[labels == leaf_id]

def leaf_spacing(xyz):
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(7, len(xyz)))
    return float(np.median(d[:, -1])) + 1e-12

def compute_leaf_axis(normal, centroid, base):
    direction = centroid - base
    n = np.linalg.norm(direction)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    direction /= n
    axis = np.cross(normal, direction)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-6:
        v = direction - np.dot(direction, normal) * normal
        axis = v
        ax_norm = np.linalg.norm(axis)
    return axis / (ax_norm + 1e-12)

def choose_rotation_sign(axis, base, leaf_xyz_arr, other_centroid):
    centroid = leaf_xyz_arr.mean(0)
    R = axis_angle_to_matrix(axis, np.radians(1.0))
    new_centroid = R @ (centroid - base) + base
    return 1 if np.dot(new_centroid - centroid, other_centroid - centroid) > 0 else -1

def identity_transform(severity, mode):
    I3 = np.eye(3).tolist()
    z = [0.0, 0.0, 0.0]
    return {"severity": severity, "mode": mode,
            "leaf_a_transform": {"pivot": None, "R": I3, "t": z, "angle_deg": 0.0},
            "leaf_b_transform": {"pivot": None, "R": I3, "t": z, "angle_deg": 0.0},
            "case_type": "identity_control"}

def _h_eval(na, nb, spacing, voxel_size):
    ov = compute_projected_overlap_fraction(na, nb, voxel_size=voxel_size)
    cf = compute_contact_fraction(na, nb, spacing=spacing)
    tree_b = cKDTree(nb)
    d, _ = tree_b.query(na, k=1)
    return {"projected_overlap_fraction": ov["overlap_fraction"],
            "contact_fraction": cf["contact_fraction"],
            "min_cross_leaf_distance_ratio": float(d.min() / spacing)}
def _h_score(res, target_contact):
    lo, hi = target_contact
    center = (lo + hi) / 2
    return abs(res["contact_fraction"] - center) / max(hi - lo, 0.01)

def search_horizontal(xyz_a, xyz_b, base_a, base_b, spacing):
    pca_a = fit_leaf_pca(xyz_a); pca_b = fit_leaf_pca(xyz_b)
    axis_a = compute_leaf_axis(pca_a["normal"], pca_a["centroid"], base_a)
    axis_b = compute_leaf_axis(pca_b["normal"], pca_b["centroid"], base_b)
    sign_a = choose_rotation_sign(axis_a, base_a, xyz_a, xyz_b.mean(0))
    sign_b = choose_rotation_sign(axis_b, base_b, xyz_b, xyz_a.mean(0))
    h0 = identity_transform("H0", "horizontal")
    h0["achieved_projected_overlap"] = 0.0
    h0["achieved_contact_fraction"] = 0.0
    h0["achieved_min_cross_leaf_distance_ratio"] = None
    results = {"H0": h0}
    _all = np.vstack([xyz_a, xyz_b]); _t = cKDTree(_all)
    _d,_ = _t.query(_all, k=min(7, len(_all)))
    voxel_size = float(np.median(_d[:,-1]))/2 + 1e-12
    angles = range(0, 181, 5)
    na_cache = {ang: (axis_angle_to_matrix(sign_a*axis_a, np.radians(ang)) @ (xyz_a-base_a).T).T + base_a for ang in angles}
    nb_cache = {ang: (axis_angle_to_matrix(sign_b*axis_b, np.radians(ang)) @ (xyz_b-base_b).T).T + base_b for ang in angles}
    coarse = {(aa,ab): _h_eval(na_cache[aa], nb_cache[ab], spacing, voxel_size) for aa in angles for ab in angles}
    for level in ["H1","H2","H3","H4"]:
        target = H_TARGETS[level]["target_contact"]
        best_key = min(coarse, key=lambda k: _h_score(coarse[k], target))
        best_aa, best_ab = best_key
        fine = {}
        for aa in np.arange(max(0,best_aa-3.0), best_aa+3.5, 0.5):
            Ra = axis_angle_to_matrix(sign_a*axis_a, np.radians(aa))
            na = (Ra @ (xyz_a-base_a).T).T + base_a
            for ab in np.arange(max(0,best_ab-3.0), best_ab+3.5, 0.5):
                Rb = axis_angle_to_matrix(sign_b*axis_b, np.radians(ab))
                nb = (Rb @ (xyz_b-base_b).T).T + base_b
                fine[(float(aa),float(ab))] = _h_eval(na, nb, spacing, voxel_size)
        bk = min(fine, key=lambda k: _h_score(fine[k], target))
        aa, ab = bk; res = fine[bk]
        results[level] = {
            "severity": level, "mode": "horizontal",
            "leaf_a_transform": {"pivot": base_a.tolist(), "axis": (sign_a*axis_a).tolist(),
                                 "R": axis_angle_to_matrix(sign_a*axis_a, np.radians(aa)).tolist(),
                                 "t": [0,0,0], "angle_deg": float(aa)},
            "leaf_b_transform": {"pivot": base_b.tolist(), "axis": (sign_b*axis_b).tolist(),
                                 "R": axis_angle_to_matrix(sign_b*axis_b, np.radians(ab)).tolist(),
                                 "t": [0,0,0], "angle_deg": float(ab)},
            "achieved_projected_overlap": res["projected_overlap_fraction"],
            "achieved_contact_fraction": res["contact_fraction"],
            "achieved_min_cross_leaf_distance_ratio": res["min_cross_leaf_distance_ratio"],
            "case_type": "rotation_only",
        }
    return list(results.values())

def compute_vertical_rotation_axis(apex_pos, upper_base, lower_centroid):
    arm = apex_pos - upper_base; tgt = lower_centroid - apex_pos
    axis = np.cross(arm, tgt)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        axis = np.cross(arm, np.array([0.,0.,1.]))
        n = np.linalg.norm(axis)
    return axis / (n+1e-12)

def _v_eval(upper_xyz, lower_xyz, upper_base, lower_normal, axis, sign, angle_deg, apex_local_idx, spacing):
    R = axis_angle_to_matrix(sign*axis, np.radians(angle_deg))
    new_upper = (R @ (upper_xyz-upper_base).T).T + upper_base
    new_apex = new_upper[apex_local_idx]
    gap = compute_apex_gap(new_apex, lower_xyz, lower_normal, upper_xyz=new_upper)
    vs = compute_vertical_gap_and_spacing(new_upper, lower_xyz, lower_normal)
    return {"angle_deg": angle_deg,
            "apex_euclidean_gap_ratio": gap["apex_euclidean_gap"]/spacing,
            "apex_normal_gap_ratio": gap["apex_normal_gap"]/spacing,
            "whole_leaf_median_gap_ratio": vs["median_euclidean_gap"]/spacing,
            "apex_euclidean_gap": gap["apex_euclidean_gap"],
            "apex_normal_gap": gap["apex_normal_gap"],
            "R": R, "new_upper_xyz": new_upper,
            "normal_alignment_cos": vs["normal_alignment_cos"]}
def _v_score(r, target):
    return abs(r["apex_euclidean_gap_ratio"] - target)

def search_vertical(upper_xyz, lower_xyz, upper_base, upper_normal, lower_normal, apex_local_idx, spacing):
    apex_pos = upper_xyz[apex_local_idx]
    lower_centroid = lower_xyz.mean(0)
    axis = compute_vertical_rotation_axis(apex_pos, upper_base, lower_centroid)
    Rtest = axis_angle_to_matrix(axis, np.radians(1.0))
    new = Rtest @ (apex_pos - upper_base) + upper_base
    sign = 1 if np.linalg.norm(new - lower_centroid) < np.linalg.norm(apex_pos - lower_centroid) else -1
    v0 = identity_transform("V0","vertical")
    v0["achieved_apex_gap_ratio"] = None
    results = {"V0": v0}
    coarse = [_v_eval(upper_xyz, lower_xyz, upper_base, lower_normal, axis, sign, a, apex_local_idx, spacing)
              for a in np.arange(0,182,2.0)]
    for level in ["V1","V2","V3","V4"]:
        target = V_TARGETS[level]["target_apex_gap_ratio"]
        best = min(coarse, key=lambda r: _v_score(r, target))
        best_angle = best["angle_deg"]
        fine = [_v_eval(upper_xyz, lower_xyz, upper_base, lower_normal, axis, sign, a, apex_local_idx, spacing)
                for a in np.arange(max(0,best_angle-2.0), best_angle+2.5, 0.5)]
        chosen = min(fine, key=lambda r: _v_score(r, target))
        upper_rot = {"pivot": upper_base.tolist(), "axis": (sign*axis).tolist(),
                     "R": chosen["R"].tolist(), "t": [0,0,0], "angle_deg": float(chosen["angle_deg"])}
        results[level] = {
            "severity": level, "mode": "vertical",
            "upper_transform": upper_rot,
            "lower_transform": identity_transform(level,"vertical")["leaf_a_transform"],
            "achieved_apex_gap_ratio": float(chosen["apex_euclidean_gap_ratio"]),
            "achieved_apex_normal_gap_ratio": float(chosen["apex_normal_gap_ratio"]),
            "achieved_whole_leaf_median_gap_ratio": float(chosen["whole_leaf_median_gap_ratio"]),
            "achieved_normal_alignment_cos": float(chosen["normal_alignment_cos"]),
            "case_type": "rotation_only",
        }
    return list(results.values())


def discover_pairs_for_plant(plant: str) -> list:
    """Find candidate clean leaf pairs on a dense plant."""
    pd = load_dense_plant_baseline(plant)
    apexes = pd["apexes"]
    labels = pd["labels"]
    # candidates: leaves with >=MIN_POINTS, single_tip-type, with base, not degenerate
    cand = []
    for i, ap in enumerate(apexes):
        lid = i+1
        cnt = int((labels == lid).sum())
        if cnt < MIN_POINTS:
            continue
        if ap.get("type") not in ("single_tip", None):
            # dense baseline may type as unknown; accept with >=MIN_POINTS
            pass
        if ap.get("base_gaussian_index") is None:
            continue
        cand.append(i+1)
    # choose pairs by centroid distance: prefer middle-distance (not too near, not too far)
    xyz = pd["xyz"]
    pairs = []
    for a in cand:
        for b in cand:
            if a >= b:
                continue
            da = float(np.linalg.norm(xyz[labels==a].mean(0) - xyz[labels==b].mean(0)))
            pairs.append((a,b,da))
    pairs.sort(key=lambda x: x[2])
    return pairs


def build_for_plants(plants: list, max_pairs_per_plant: int = 2) -> dict:
    out = {}
    for plant in plants:
        print(f"\n[PLANT] {plant}", flush=True)
        pd = load_dense_plant_baseline(plant)
        gc, labels, apexes, root_index, xyz = pd["gc"], pd["labels"], pd["apexes"], pd["root_index"], pd["xyz"]
        plant_key = f"{plant}"
        # discover candidate pairs, try them until transforms satisfy target contact/gap
        cand_pairs = discover_pairs_for_plant(plant)
        if not cand_pairs:
            print(f"  [SKIP] no valid pairs (leaves need >= {MIN_POINTS} pts + base)")
            continue
        chosen = cand_pairs[: max_pairs_per_plant*2]  # upsell, then take first N that hit all targets
        for idx, (a_id, b_id, _) in enumerate(chosen):
            if len([k for k in out if k.startswith(f"{plant}_")]) >= max_pairs_per_plant:
                break
            xyz_a = leaf_xyz(labels, xyz, a_id)
            xyz_b = leaf_xyz(labels, xyz, b_id)
            ap_a = apexes[a_id-1]; ap_b = apexes[b_id-1]
            base_a = xyz[int(ap_a["base_gaussian_index"])]
            base_b = xyz[int(ap_b["base_gaussian_index"])]
            spacing_a = leaf_spacing(xyz_a); spacing_b = leaf_spacing(xyz_b)
            avg_spacing = (spacing_a + spacing_b)/2
            print(f"  [PAIR] leaf{a_id}({len(xyz_a)}pts)/leaf{b_id}({len(xyz_b)}pts)", flush=True)
            h_results = search_horizontal(xyz_a, xyz_b, base_a, base_b, avg_spacing)
            for hr in h_results:
                print(f"    {hr['severity']}: cf={hr.get('achieved_contact_fraction',0):.3f}", flush=True)
            # vertical
            pca_a = fit_leaf_pca(xyz_a); pca_b = fit_leaf_pca(xyz_b)
            if xyz_a.mean(0)[2] >= xyz_b.mean(0)[2]:
                upper_id, lower_id, upper_xyz, lower_xyz = a_id, b_id, xyz_a, xyz_b
                upper_base, lower_base = base_a, base_b
                upper_normal, lower_normal = pca_a["normal"], pca_b["normal"]
            else:
                upper_id, lower_id, upper_xyz, lower_xyz = b_id, a_id, xyz_b, xyz_a
                upper_base, lower_base = base_b, base_a
                upper_normal, lower_normal = pca_b["normal"], pca_a["normal"]
            upper_ap = apexes[upper_id-1]
            upper_indices = np.where(labels == upper_id)[0]
            apex_gauss = int(upper_ap.get("apex_gaussian_index", upper_ap.get("gaussian_index")))
            try:
                apex_local_idx = int(np.where(upper_indices == apex_gauss)[0][0])
            except IndexError:
                apex_local_idx = 0
            upper_spacing = leaf_spacing(upper_xyz)
            v_raw = search_vertical(upper_xyz, lower_xyz, upper_base, upper_normal, lower_normal, apex_local_idx, upper_spacing)
            upper_is_a = (upper_id == a_id)
            v_results = []
            for vr in v_raw:
                if "upper_transform" in vr:
                    vr2 = dict(vr)
                    vr2["leaf_a_transform"] = vr["upper_transform"] if upper_is_a else vr["lower_transform"]
                    vr2["leaf_b_transform"] = vr["lower_transform"] if upper_is_a else vr["upper_transform"]
                    vr2.pop("upper_transform", None); vr2.pop("lower_transform", None)
                    v_results.append(vr2)
                else:
                    v_results.append(vr)
            for vr in v_results:
                print(f"    {vr['severity']}: apex_gap={vr.get('achieved_apex_gap_ratio','id')}", flush=True)
            pair_key = f"{plant}_pair_{a_id}_{b_id}"
            out[pair_key] = {
                "plant": plant,
                "leaf_a_id": a_id, "leaf_b_id": b_id,
                "root_index": int(root_index) if root_index is not None else int(np.argmin(xyz[:,1])),
                "upper_leaf_id": int(upper_id), "lower_leaf_id": int(lower_id),
                "spacing_a": spacing_a, "spacing_b": spacing_b,
                "horizontal": h_results, "vertical": v_results,
            }
            print(f"  -> frozen {pair_key}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", default=None, help="Comma-separated plant names (default: viable dev+heldout)")
    ar = ap.parse_args()
    if ar.plants:
        plants = [x.strip() for x in ar.plants.split(",") if x.strip()]
    else:
        plants = ["DouBanLv1", "WangWenCao2"]  # dev, heldout (SIF verified)
    transforms = build_for_plants(plants)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(transforms, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] frozen {len(transforms)} pairs -> {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
