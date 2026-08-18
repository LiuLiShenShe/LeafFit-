#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instance-level evaluation for the LeafFit headless segmentation baseline.

The evaluation is INSTANCE-LEVEL and permutation-invariant: predicted instances are
matched to ground-truth instances via IoU + Hungarian optimal assignment (label IDs
are unordered, so a direct label-to-label comparison is meaningless).

Usage:
    python scripts/evaluate_segmentation.py \
        --pred outputs/baseline/plant1_green_pepper/labels.npy \
        --gt   /path/to/gt_labels.npy \

Metrics (background label = 0 excluded from instance metrics, used in accuracy):
  - Accuracy: (TP pixels over matched pairs + true background) / N
  - mIoU   : mean IoU over Hungarian-matched (pred,gt) instance pairs
  - mF1    : mean F1 over matched pairs
  - PQ     : Panoptic Quality = sum IoU(matched pairs) / (0.5*(|P|+|G|))

If --gt is not provided (or points to a missing file), prints "no GT available"
and exits 0 WITHOUT fabricating any metric.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _intersection_counts(pred: np.ndarray, gt: np.ndarray):
    """Per (pred_label, gt_label) intersection counts; background label is 0.

    Indexed directly by label value, so background sits in counts[0,0]; instance
    rows/cols start at 1.
    """
    pred_lbl = pred.astype(np.int64)
    gt_lbl = gt.astype(np.int64)
    n_p = int(pred_lbl.max()) + 1
    n_g = int(gt_lbl.max()) + 1
    counts = np.zeros((n_p, n_g), dtype=np.int64)
    flat_p = pred_lbl.ravel()
    flat_g = gt_lbl.ravel()
    pairs = flat_p.astype(np.int64) * n_g + flat_g
    uniq, cnt = np.unique(pairs, return_counts=True)
    counts.ravel()[uniq] = cnt
    return counts, n_p, n_g


def hungarian_match(counts: np.ndarray):
    """Maximize total IoU between pred instances (rows, excl 0) and gt instances (cols, excl 0).

    Builds a cost matrix = -IoU for every (pred_i, gt_j) with any overlap; cells with
    zero overlap are left at a large positive cost so no spurious matches are made.
    Uses scipy.optimize.linear_sum_assignment.
    """
    from scipy.optimize import linear_sum_assignment
    n_p = counts.shape[0] - 1     # exclude background row 0
    n_g = counts.shape[1] - 1     # exclude background col 0
    if n_p == 0 or n_g == 0:
        return [], []
    iou = np.zeros((n_p, n_g), dtype=np.float64)
    for i in range(1, n_p + 1):     # instance rows start at label 1
        for j in range(1, n_g + 1):
            inter = counts[i, j]
            union = (counts[i, :].sum() + counts[:, j].sum() - inter)
            if union > 0:
                iou[i - 1, j - 1] = inter / union
    BIG = 1e6
    cost = np.where(iou > 0.0, -iou, BIG)
    # allow dummy "no-match" rows/cols so unmatched instances carry 0 IoU
    rows, cols = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(rows, cols):
        iou_v = iou[r, c]
        if iou_v > 0.0:
            matched.append((r + 1, c + 1, iou_v))     # pred_label, gt_label, IoU
    return matched, iou


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    counts, n_p, n_g = _intersection_counts(pred, gt)
    matched, iou = hungarian_match(counts)
    n_pred_inst = counts.shape[0] - 1        # exclude background row 0
    n_gt_inst = counts.shape[1] - 1          # exclude background col 0

    N = int(pred.size)
    bg_pred = (pred == 0)
    bg_gt = (gt == 0)
    bg_correct = int(np.logical_and(bg_pred, bg_gt).sum())

    inst_correct = sum(int(counts[i, j]) for i, j, _ in matched)
    accuracy = (bg_correct + inst_correct) / N

    mIoU = float(np.mean([v for _, _, v in matched])) if matched else 0.0
    f1s = []
    for i, j, v in matched:
        prec = counts[i, j] / max(counts[i, :].sum(), 1)
        rec = counts[i, j] / max(counts[:, j].sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    mF1 = float(np.mean(f1s)) if f1s else 0.0

    pq = sum(v for _, _, v in matched) / max(0.5 * (n_pred_inst + n_gt_inst), 1e-12)

    # per-pair detail
    detail = []
    for i, j, v in sorted(matched, key=lambda t: -t[2]):
        detail.append({"pred_label": int(i), "gt_label": int(j), "iou": round(float(v), 4)})

    return {
        "accuracy": round(accuracy, 4),
        "mIoU": round(mIoU, 4),
        "mF1": round(mF1, 4),
        "PQ": round(float(pq), 4),
        "num_pred_instances": int(n_pred_inst),
        "num_gt_instances": int(n_gt_inst),
        "num_matched": len(matched),
        "matched_pairs": detail,
        "note": "metrics computed AFTER IoU+Hungarian optimal matching (permutation invariant)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="predicted labels.npy (0=stem/background, 1..K=leaf)")
    ap.add_argument("--gt", help="ground-truth labels.npy; omit => report no-GT and exit 0")
    ap.add_argument("--out", default=None, help="evaluation_report.json output path")
    args = ap.parse_args()

    if args.gt is None or not os.path.exists(args.gt):
        print("no GT available: skipping evaluation (no fabricated metrics)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"evaluated": False, "reason": "no GT available"}, f, indent=2)
        return 0

    pred = np.load(args.pred)
    gt = np.load(args.gt)
    rep = evaluate(pred, gt)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "matched_pairs"}, indent=2, ensure_ascii=False))
    print(f"matched pairs: {len(rep['matched_pairs'])}")
    for d in rep["matched_pairs"][:10]:
        print(f"  pred={d['pred_label']} gt={d['gt_label']} IoU={d['iou']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
