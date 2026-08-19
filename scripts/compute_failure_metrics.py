#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute failure metrics for a single controlled-overlap case.

Reads the outputs from run_overlap_case.py and computes:
  - apex_recall (identity + spatial)
  - wrong_grouping
  - first_failure_stage
  - merge_level (for horizontal)
  - shortcut_ratio (for vertical)
  - pair_instance_IoU / PQ / FP_leakage / FN_rate

Usage:
    python scripts/compute_failure_metrics.py \
        --case-dir outputs/task2/controlled/plant2_rubber_tree_pair_3_12/horizontal/H3
"""
from __future__ import annotations

import argparse
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

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_BLROOT = os.path.join(_REPO_ROOT, "outputs", "baseline")


def _to_python(x):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def load_case(case_dir: str) -> dict:
    """Load all outputs from a case directory."""
    with open(os.path.join(case_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(case_dir, "construction_metrics.json")) as f:
        construction = json.load(f)
    with open(os.path.join(case_dir, "transforms.json")) as f:
        transforms = json.load(f)
    labels = np.load(os.path.join(case_dir, "labels.npy"))
    gt_labels = np.load(os.path.join(case_dir, "construction_gt_labels.npy"))
    with open(os.path.join(case_dir, "apexes.json")) as f:
        apexes = json.load(f)
    with open(os.path.join(case_dir, "apex_grouping.json")) as f:
        apex_grouping = json.load(f)
    with open(os.path.join(case_dir, "petioles.json")) as f:
        petioles = json.load(f)
    return {
        "config": config,
        "construction": construction,
        "transforms": transforms,
        "labels": labels,
        "gt_labels": gt_labels,
        "apexes": apexes,
        "apex_grouping": apex_grouping,
        "petioles": petioles,
    }


def compute_hungarian_iou(pred_labels: NDArray, gt_labels: NDArray) -> dict:
    """Compute Hungarian-matched instance IoU and PQ."""
    from scipy.optimize import linear_sum_assignment

    n_pred = int(pred_labels.max())
    n_gt = int(gt_labels.max())
    if n_pred == 0 or n_gt == 0:
        return {"mIoU": 0.0, "PQ": 0.0, "matched_pairs": 0,
                "false_positives": n_pred, "false_negatives": n_gt}

    # Build IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for p in range(1, n_pred + 1):
        mask_p = pred_labels == p
        for g in range(1, n_gt + 1):
            mask_g = gt_labels == g
            inter = np.sum(mask_p & mask_g)
            union = np.sum(mask_p | mask_g)
            if union > 0:
                iou_matrix[p - 1, g - 1] = inter / union

    # Hungarian matching with IoU >= 0.5 threshold
    MATCH_THRESHOLD = 0.5
    cost = np.where(iou_matrix >= MATCH_THRESHOLD, -iou_matrix, 1e6)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter matches by threshold
    valid = iou_matrix[row_ind, col_ind] >= MATCH_THRESHOLD
    matched_iou = iou_matrix[row_ind[valid], col_ind[valid]]

    n_matched = len(matched_iou)
    miou = float(matched_iou.mean()) if n_matched > 0 else 0.0
    pq = float(matched_iou.sum() / (0.5 * (n_pred + n_gt))) if n_matched > 0 else 0.0

    return {
        "mIoU": miou,
        "PQ": pq,
        "matched_pairs": n_matched,
        "false_positives": n_pred - n_matched,
        "false_negatives": n_gt - n_matched,
    }


def compute_pair_metrics(
    pred_labels: NDArray,
    gt_labels: NDArray,
    leaf_a_id: int,
    leaf_b_id: int,
) -> dict:
    """Compute pair-local metrics (full instance, not ROI)."""
    # Find predicted instances that overlap with GT A or B
    gt_a = gt_labels == leaf_a_id
    gt_b = gt_labels == leaf_b_id

    # Find predicted instances that overlap with GT A or B
    matched_pred_a = set()
    matched_pred_b = set()
    n_pred = int(pred_labels.max())

    for p in range(1, n_pred + 1):
        mask_p = pred_labels == p
        if np.any(mask_p & gt_a):
            matched_pred_a.add(p)
        if np.any(mask_p & gt_b):
            matched_pred_b.add(p)

    # All predicted instances that overlap with either A or B
    matched_pred = matched_pred_a | matched_pred_b
    all_pred_mask = np.isin(pred_labels, list(matched_pred)) if matched_pred else np.zeros_like(pred_labels, dtype=bool)

    # False positive leakage: predicted points not in GT A or B
    fp_mask = all_pred_mask & ~gt_a & ~gt_b
    fp_leakage = float(fp_mask.sum() / max(all_pred_mask.sum(), 1))

    # False negative: GT A or B not covered by matched predictions
    if matched_pred:
        fn_mask = (gt_a | gt_b) & ~all_pred_mask
    else:
        fn_mask = gt_a | gt_b
    fn_rate = float(fn_mask.sum() / max((gt_a | gt_b).sum(), 1))

    return {
        "matched_pred_a": len(matched_pred_a),
        "matched_pred_b": len(matched_pred_b),
        "fp_leakage": fp_leakage,
        "fn_rate": fn_rate,
    }


def compute_geodesic_metrics(
    labels: NDArray,
    gt_labels: NDArray,
    root_geodesic: NDArray,
    apexes: list,
    gt_leaf_a_id: int,
    gt_leaf_b_id: int,
    pair_key: str,
) -> dict:
    """Compute apex recall and grouping metrics.

    reference_apex_recall = fraction of GT pair leaves (a, b) whose GT apex
    gaussian (from source_pairs.json) still has a detected apex in the SAME
    predicted instance.
    wrong_grouping = whether the two GT pair leaves end up in the same predicted instance.
    """
    # Load GT apex gaussian indices from source_pairs (ground-truth positions)
    with open(os.path.join(_OUTROOT, "benchmark_transforms.json")) as f:
        transforms_data = json.load(f)
    plant_name = transforms_data[pair_key]["plant"]
    with open(os.path.join(_OUTROOT, "source_pairs.json")) as f:
        sp = json.load(f)
    gt_apex_indices = {}
    for p in sp["pairs"]:
        if p["plant"] != plant_name:
            continue
        gt_apex_indices[p["leaf_a_id"]] = p["leaf_a"]["apex_gaussian_index"]
        gt_apex_indices[p["leaf_b_id"]] = p["leaf_b"]["apex_gaussian_index"]

    # Merge level: how many predicted instances cover GT A and B
    gt_a_mask = gt_labels == gt_leaf_a_id
    gt_b_mask = gt_labels == gt_leaf_b_id

    pred_instances_a = set(int(x) for x in labels[gt_a_mask] if x > 0)
    pred_instances_b = set(int(x) for x in labels[gt_b_mask] if x > 0)
    shared_instances = pred_instances_a & pred_instances_b

    merge_level = len(shared_instances)
    wrong_grouping = merge_level > 0

    # Reference apex recall: for each GT pair leaf, is its GT apex detected
    # in the same predicted instance?
    pair_leaves = [gt_leaf_a_id, gt_leaf_b_id]
    apex_detected_count = 0
    apex_detail = {}

    for leaf_id in pair_leaves:
        gt_apex_idx = gt_apex_indices.get(leaf_id)
        if gt_apex_idx is None or gt_apex_idx >= len(labels):
            apex_detail[f"leaf_{leaf_id}"] = {"found": False, "reason": "gt_apex_not_in_source_pairs"}
            continue
        pred_inst = labels[gt_apex_idx]
        if pred_inst <= 0:
            apex_detail[f"leaf_{leaf_id}"] = {"found": False, "reason": "apex_unassigned", "pred_instance": int(pred_inst)}
            continue
        # Check if any detected apex is in the same predicted instance
        has_detected = any(
            a.get("gaussian_index", -1) >= 0
            and a["gaussian_index"] < len(labels)
            and labels[a["gaussian_index"]] == pred_inst
            for a in apexes
        )
        if has_detected:
            apex_detected_count += 1
        apex_detail[f"leaf_{leaf_id}"] = {
            "found": has_detected,
            "gt_apex_idx": int(gt_apex_idx),
            "pred_instance": int(pred_inst),
        }

    reference_apex_recall = apex_detected_count / max(len(pair_leaves), 1)

    # New unmatched apex count: detected apexes NOT in GT pair leaves
    new_unmatched = 0
    new_unmatched_on_pair = 0
    for a in apexes:
        gi = a.get("gaussian_index", -1)
        if gi < 0 or gi >= len(gt_labels):
            continue
        gt_leaf = gt_labels[gi]
        if gt_leaf not in pair_leaves:
            new_unmatched += 1
        else:
            new_unmatched_on_pair += 1

    return {
        "reference_apex_recall": float(reference_apex_recall),
        "wrong_grouping": wrong_grouping,
        "merge_level": int(merge_level),
        "pred_instances_covering_a": int(len(pred_instances_a)),
        "pred_instances_covering_b": int(len(pred_instances_b)),
        "apex_detail": apex_detail,
        "new_unmatched_apex_count": int(new_unmatched),
    }


def compute_shortcut_metrics(
    labels: NDArray,
    gt_labels: NDArray,
    root_geodesic: NDArray,
    apexes: list,
    upper_id: int,
    lower_id: int,
    pair_key: str,
    severity: str,
) -> dict:
    """Compute vertical shortcut metrics.

    Shortcut evidence is the geodesic distance to root at the upper leaf's apex
    being SHORTER than at the V0 (identity) baseline, combined with:
      - upper/lower GT leaves sharing a predicted instance (cross-leaf merge)
      - upper apex path crossing to lower leaf

    The shortcut ratio = d_root(upper_apex_case) / d_root(upper_apex_V0).
    A ratio < 1.0 means the geodesic was shortened (shortcut occurred).
    """
    # Get the GT upper apex gaussian index from source_pairs (ground truth position)
    with open(os.path.join(_OUTROOT, "benchmark_transforms.json")) as f:
        transforms_data = json.load(f)
    plant_name = transforms_data[pair_key]["plant"]
    with open(os.path.join(_OUTROOT, "source_pairs.json")) as f:
        sp = json.load(f)
    upper_apex_idx = None
    for p in sp["pairs"]:
        if p["plant"] != plant_name:
            continue
        if p["leaf_a_id"] == upper_id:
            upper_apex_idx = p["leaf_a"]["apex_gaussian_index"]
            break
        elif p["leaf_b_id"] == upper_id:
            upper_apex_idx = p["leaf_b"]["apex_gaussian_index"]
            break

    if upper_apex_idx is None or upper_apex_idx >= len(root_geodesic):
        return {"shortcut_ratio": None, "shortcut_evidence": {}, "reason": "upper_apex_not_found"}

    # Get V0 baseline distance at the same GT upper apex position
    v0_dir = os.path.join(_OUTROOT, "controlled", pair_key, "vertical", "V0")
    v0_geodesic_path = os.path.join(v0_dir, "root_geodesic_multisource.npy")
    if os.path.exists(v0_geodesic_path):
        v0_geodesic = np.load(v0_geodesic_path)
        d_v0 = float(v0_geodesic[upper_apex_idx]) if upper_apex_idx < len(v0_geodesic) else None
    else:
        d_v0 = None

    d_case = float(root_geodesic[upper_apex_idx])

    # Shortcut ratio: case distance / V0 baseline distance
    # ratio < 1.0 => geodesic shortened (shortcut)
    shortcut_ratio = None
    shortcut_evidence = {}
    if d_v0 is not None and d_v0 > 0:
        shortcut_ratio = d_case / d_v0
        shortcut_evidence["d_case"] = d_case
        shortcut_evidence["d_v0"] = d_v0
        shortcut_evidence["distance_shortened"] = shortcut_ratio < 1.0

    # Check upper/lower GT labels share a predicted instance
    upper_mask = gt_labels == upper_id
    lower_mask = gt_labels == lower_id
    upper_pred = set(int(x) for x in labels[upper_mask] if x > 0)
    lower_pred = set(int(x) for x in labels[lower_mask] if x > 0)
    shared_instances = upper_pred & lower_pred
    shortcut_evidence["upper_pred_instances"] = sorted(upper_pred)
    shortcut_evidence["lower_pred_instances"] = sorted(lower_pred)
    shortcut_evidence["shared_instances"] = sorted(shared_instances)
    shortcut_evidence["cross_leaf_merge"] = len(shared_instances) > 0

    # Check upper apex distance < lower leaf median (shortcut makes upper apex
    # appear "closer to root" through the lower leaf)
    lower_d = root_geodesic[lower_mask]
    lower_med = float(np.median(lower_d))
    upper_d = root_geodesic[upper_mask]
    upper_below_lower_median = int((upper_d < lower_med).sum())
    shortcut_evidence["upper_points_below_lower_median"] = upper_below_lower_median
    shortcut_evidence["upper_total_points"] = int(len(upper_mask))
    shortcut_evidence["upper_below_lower_ratio"] = upper_below_lower_median / max(len(upper_mask), 1)

    # Check cross-leaf path transition
    has_cross_leaf_path = False
    paths_path = os.path.join(_OUTROOT, "controlled", pair_key, "vertical", severity, "paths.json")
    if os.path.exists(paths_path):
        with open(paths_path) as f:
            paths = json.load(f)
        for path in paths:
            path_gauss = path.get("path_gaussian_indices", [])
            if not path_gauss:
                continue
            path_arr = np.array(path_gauss)
            gt_along = gt_labels[path_arr]
            upper_count = int((gt_along == upper_id).sum())
            lower_count = int((gt_along == lower_id).sum())
            if upper_count > 0 and lower_count > 0:
                has_cross_leaf_path = True
                break
    shortcut_evidence["cross_leaf_path_detected"] = has_cross_leaf_path

    # Determine if shortcut mechanism is confirmed
    shortcut_confirmed = (
        shortcut_ratio is not None and shortcut_ratio < 1.0 - 1e-6
    )

    return {
        "shortcut_ratio": float(shortcut_ratio) if shortcut_ratio is not None else None,
        "shortcut_evidence": shortcut_evidence,
        "shortcut_confirmed": shortcut_confirmed,
        "upper_apex_idx": int(upper_apex_idx),
        "pred_instance": int(labels[upper_apex_idx]),
        "reason": "ok" if shortcut_ratio is not None else "no_v0_baseline",
    }


def compute_all_metrics(case_dir: str) -> dict:
    """Compute all failure metrics for a case."""
    data = load_case(case_dir)

    config = data["config"]
    labels = data["labels"]
    gt_labels = data["gt_labels"]
    apexes = data["apexes"]

    # Extract the actual GT leaf IDs for this pair from the config
    pair_key = config["pair_key"]
    transforms = json.load(open(os.path.join(_OUTROOT, "benchmark_transforms.json")))
    pk_data = transforms[pair_key]
    gt_leaf_a = pk_data["leaf_a_id"]
    gt_leaf_b = pk_data["leaf_b_id"]

    # Load root geodesic
    geodesic_path = os.path.join(case_dir, "root_geodesic_multisource.npy")
    root_geodesic = np.load(geodesic_path) if os.path.exists(geodesic_path) else None

    # Instance-level metrics
    instance_metrics = compute_hungarian_iou(labels, gt_labels)

    # Pair-local metrics using actual leaf IDs
    pair_metrics = compute_pair_metrics(labels, gt_labels, gt_leaf_a, gt_leaf_b)

    # Geodesic / apex metrics
    geodesic_metrics = compute_geodesic_metrics(labels, gt_labels, root_geodesic, apexes,
                                                gt_leaf_a, gt_leaf_b, str(config["pair_key"]))

    # Shortcut metrics (for vertical)
    shortcut_metrics = None
    if config["mode"] == "vertical":
        upper_id = pk_data["upper_leaf_id"]
        lower_id = pk_data["lower_leaf_id"]
        shortcut_metrics = compute_shortcut_metrics(
            labels, gt_labels, root_geodesic, apexes, upper_id, lower_id,
            str(config["pair_key"]), str(config["severity"]))

    # Determine first_failure_stage (algorithm causal order: GEODESIC → APEX → PATH → GROUPING → PETIOLE → SEGMENTATION)
    first_failure_stage = "NONE"
    if geodesic_metrics["wrong_grouping"]:
        first_failure_stage = "GROUPING"   # premature cross-leaf merge (Fig.13a)
    elif geodesic_metrics["reference_apex_recall"] < 1.0:
        first_failure_stage = "APEX"        # apex missed (Fig.13b)
    elif instance_metrics["mIoU"] < 0.9:
        first_failure_stage = "SEGMENTATION"

    # Dominant failure stage (diagnostic priority: GROUPING > APEX > SEGMENTATION)
    # This reflects the primary observed failure pattern
    dominant_failure_stage = "NONE"
    if geodesic_metrics["wrong_grouping"]:
        dominant_failure_stage = "GROUPING"
    elif geodesic_metrics["reference_apex_recall"] < 1.0:
        dominant_failure_stage = "APEX"
    elif instance_metrics["mIoU"] < 0.9:
        dominant_failure_stage = "SEGMENTATION"

    metrics = {
        "pair_key": str(config["pair_key"]),
        "mode": str(config["mode"]),
        "severity": str(config["severity"]),
        "instance": instance_metrics,
        "pair_local": pair_metrics,
        "geodesic": geodesic_metrics,
        "first_failure_stage": str(first_failure_stage),
        "dominant_failure_stage": str(dominant_failure_stage),
        "construction": {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in data["construction"].items()},
    }
    if shortcut_metrics is not None:
        metrics["shortcut"] = shortcut_metrics

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args()

    try:
        metrics = compute_all_metrics(args.case_dir)
        # Save metrics
        out_path = os.path.join(args.case_dir, "failure_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
