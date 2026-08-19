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
) -> dict:
    """Compute apex recall and grouping metrics."""
    n_pred_leaves = int(labels.max())
    n_gt_leaves = int(gt_labels.max())

    # Apex recall: for each GT apex, is there a detected apex nearby in the same predicted instance?
    apex_recall = 0.0
    detected_apex_count = 0

    for gt_apex_info in apexes:
        gt_idx = gt_apex_info.get("gaussian_index", -1)
        if gt_idx < 0 or gt_idx >= len(gt_labels):
            continue
        gt_leaf = gt_labels[gt_idx]
        if gt_leaf <= 0:
            continue
        # Find the predicted instance containing this GT apex
        pred_instance = labels[gt_idx]
        if pred_instance <= 0:
            continue
        # Check if any predicted apex is in the same instance
        has_detected = any(
            a.get("gaussian_index", -1) >= 0 and
            a["gaussian_index"] < len(labels) and
            labels[a["gaussian_index"]] == pred_instance
            for a in apexes
        )
        if has_detected:
            apex_recall += 1
        detected_apex_count += 1

    apex_recall = apex_recall / max(detected_apex_count, 1)

    # Merge level: how many predicted instances cover GT A and B
    gt_a_mask = gt_labels == gt_leaf_a_id
    gt_b_mask = gt_labels == gt_leaf_b_id

    pred_instances_a = set(int(x) for x in labels[gt_a_mask] if x > 0)
    pred_instances_b = set(int(x) for x in labels[gt_b_mask] if x > 0)

    merge_level = len(pred_instances_a & pred_instances_b)

    return {
        "apex_recall": float(apex_recall),
        "detected_apex_count": int(detected_apex_count),
        "merge_level": int(merge_level),
        "pred_instances_covering_a": int(len(pred_instances_a)),
        "pred_instances_covering_b": int(len(pred_instances_b)),
    }


def compute_shortcut_metrics(
    labels: NDArray,
    gt_labels: NDArray,
    root_geodesic: NDArray,
    apexes: list,
    upper_id: int,
    lower_id: int,
) -> dict:
    """Compute vertical shortcut metrics."""
    # Find the GT apex for upper leaf
    upper_apex_idx = None
    for a in apexes:
        if a.get("gaussian_index", -1) >= 0 and gt_labels[a["gaussian_index"]] == upper_id:
            upper_apex_idx = a["gaussian_index"]
            break

    if upper_apex_idx is None:
        return {"shortcut_ratio": None, "reason": "upper_apex_not_found"}

    # Get predicted instance for this apex
    pred_instance = labels[upper_apex_idx]
    if pred_instance <= 0:
        return {"shortcut_ratio": None, "reason": "apex_unassigned"}

    # Compute geodesic ratio: d(root, predicted_apex) / d(root, gt_upper_apex)
    # This measures how much the geodesic is shortened by the shortcut
    pred_apex_xyz = None
    gt_upper_xyz = None

    # Find the predicted apex in the same predicted instance as the GT apex
    pred_apex_idx = None
    for a in apexes:
        if a.get("gaussian_index", -1) >= 0 and labels[a["gaussian_index"]] == pred_instance:
            pred_apex_idx = a["gaussian_index"]
            break

    # Compute shortcut ratio if we have both paths
    shortcut_ratio = None
    if pred_apex_idx is not None and pred_apex_idx < len(root_geodesic) and upper_apex_idx < len(root_geodesic):
        d_pred = root_geodesic[pred_apex_idx]
        d_gt = root_geodesic[upper_apex_idx]
        if d_gt > 0:
            shortcut_ratio = d_pred / d_gt

    return {
        "shortcut_ratio": float(shortcut_ratio) if shortcut_ratio is not None else None,
        "upper_apex_idx": int(upper_apex_idx),
        "pred_instance": int(pred_instance),
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
                                                gt_leaf_a, gt_leaf_b)

    # Shortcut metrics (for vertical)
    shortcut_metrics = None
    if config["mode"] == "vertical":
        upper_id = pk_data["upper_leaf_id"]
        lower_id = pk_data["lower_leaf_id"]
        shortcut_metrics = compute_shortcut_metrics(
            labels, gt_labels, root_geodesic, apexes, upper_id, lower_id)

    # Determine first failure stage (priority: grouping > apex > segmentation)
    first_failure_stage = "NONE"
    if geodesic_metrics["merge_level"] > 0:
        first_failure_stage = "GROUPING"    # premature cross-leaf merge (Fig.13a)
    elif geodesic_metrics["apex_recall"] < 1.0:
        first_failure_stage = "APEX"        # apex missed (Fig.13b)
    elif instance_metrics["mIoU"] < 0.9:
        first_failure_stage = "SEGMENTATION"

    metrics = {
        "pair_key": str(config["pair_key"]),
        "mode": str(config["mode"]),
        "severity": str(config["severity"]),
        "instance": instance_metrics,
        "pair_local": pair_metrics,
        "geodesic": geodesic_metrics,
        "first_failure_stage": str(first_failure_stage),
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
