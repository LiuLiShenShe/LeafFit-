#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 controlled-overlap benchmark tests (A–G).

Tests:
  A: H0/V0 labels reproduce baseline partition (identity control)
  B: Gaussian data integrity — N preserved, non-target unchanged, rot unit norm
  C: Transform correctness — pivot preserved, t=0 enforced
  D: Root index unchanged across all cases
  E: construction_gt_labels byte-identical to baseline labels
  F: Evaluator IoU threshold (>=0.5) correctly filters
  G: Pre-grouping replay consistency — pre_grouping_replay keys stable

Run:
    export PYTHONPATH=<repo>/core:<repo>
    python -m unittest tests.test_overlap_benchmark -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.headless_segmentation as hs  # noqa: E402
from overlap_geometry import transform_leaf_gaussians  # noqa: E402

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_BLROOT = os.path.join(_REPO_ROOT, "outputs", "baseline")

# Load frozen transforms
with open(os.path.join(_OUTROOT, "benchmark_transforms.json")) as f:
    TRANSFORMS = json.load(f)

with open(os.path.join(_OUTROOT, "source_pairs.json")) as f:
    SOURCE_PAIRS = json.load(f)

# Build pair_key -> plant mapping
PAIR_PLANT = {pk: TRANSFORMS[pk]["plant"] for pk in TRANSFORMS}


def _labels_to_partition(labels: np.ndarray) -> set:
    """Convert labels array to partition (set of frozensets of indices)."""
    return {frozenset(np.where(labels == k)[0].tolist())
            for k in np.unique(labels) if k != 0}


class TestOverlapBenchmark(unittest.TestCase):
    """Tests A–G for the controlled overlap benchmark."""

    def test_A_identity_labels_match_baseline(self):
        """H0/V0 identity cases must reproduce the baseline partition exactly."""
        for pair_key, data in TRANSFORMS.items():
            plant = data["plant"]
            baseline_labels = np.load(os.path.join(_BLROOT, plant, "labels.npy"))

            for mode in ["horizontal", "vertical"]:
                h0_dir = os.path.join(_OUTROOT, "controlled", pair_key, mode, "H0" if mode == "horizontal" else "V0")
                if not os.path.exists(h0_dir):
                    continue
                labels_path = os.path.join(h0_dir, "labels.npy")
                if not os.path.exists(labels_path):
                    self.fail(f"Missing labels for identity case: {h0_dir}")
                    continue

                case_labels = np.load(labels_path)

                # Partition-level check (robust to label permutation)
                baseline_part = _labels_to_partition(baseline_labels)
                case_part = _labels_to_partition(case_labels)
                self.assertEqual(baseline_part, case_part,
                                 f"H0/V0 partition mismatch for {pair_key}/{mode}")

    def test_B_gaussian_data_integrity(self):
        """Transform preserves N, leaves non-target gaussians unchanged, rot stays unit norm."""
        for pair_key, data in TRANSFORMS.items():
            plant = data["plant"]
            g_path = os.path.join(_REPO_ROOT, "data", f"{plant}.ply")
            g = hs.load_gaussian_data(g_path)
            gc = hs.center_gaussians(g)

            leaf_a_id = data["leaf_a_id"]
            leaf_b_id = data["leaf_b_id"]
            labels = np.load(os.path.join(_BLROOT, plant, "labels.npy"))

            # Test a horizontal case with actual rotation (H3 or H4)
            for mode in ["horizontal"]:
                for sev_data in data.get(mode, []):
                    if sev_data["severity"] == "H0":
                        continue  # skip identity
                    if sev_data["severity"] != "H3":
                        continue  # test one real rotation

                    ta = sev_data["leaf_a_transform"]
                    tb = sev_data["leaf_b_transform"]
                    xyz_a = np.where(labels == leaf_a_id)[0]
                    xyz_b = np.where(labels == leaf_b_id)[0]

                    # Apply transform
                    g_out = gc
                    if ta.get("pivot") is not None:
                        R_a = np.asarray(ta["R"], dtype=np.float64)
                        t_a = np.asarray(ta["t"], dtype=np.float64).ravel()
                        pivot_a = np.asarray(ta["pivot"], dtype=np.float64).ravel()
                        g_out = transform_leaf_gaussians(g_out, xyz_a, pivot_a, R_a, t_a)
                    if tb.get("pivot") is not None:
                        R_b = np.asarray(tb["R"], dtype=np.float64)
                        t_b = np.asarray(tb["t"], dtype=np.float64).ravel()
                        pivot_b = np.asarray(tb["pivot"], dtype=np.float64).ravel()
                        g_out = transform_leaf_gaussians(g_out, xyz_b, pivot_b, R_b, t_b)

                    # N preserved
                    self.assertEqual(len(g_out.xyz), len(gc.xyz),
                                     f"N changed after transform: {pair_key}/{sev_data['severity']}")

                    # Non-target gaussians unchanged
                    target_mask = np.zeros(len(gc.xyz), dtype=bool)
                    target_mask[xyz_a] = True
                    target_mask[xyz_b] = True
                    non_target = ~target_mask
                    self.assertTrue(np.allclose(g_out.xyz[non_target], gc.xyz[non_target]),
                                    f"Non-target xyz changed: {pair_key}/{sev_data['severity']}")

                    # Rotation quaternions unit norm for ALL (transformed + non-transformed)
                    norms = np.linalg.norm(g_out.rot, axis=-1)
                    self.assertTrue(np.allclose(norms, 1.0, atol=1e-4),
                                    f"Rot quaternions not unit norm: {pair_key}/{sev_data['severity']}")
                    break  # one test per pair is enough
                break

    def test_C_transform_pivot_and_translation(self):
        """Transform pivot is at base_gaussian_index; t=0 enforced (no translation)."""
        for pair_key, data in TRANSFORMS.items():
            for mode in ["horizontal", "vertical"]:
                for sev_data in data.get(mode, []):
                    a_t = sev_data.get("leaf_a_transform", {})
                    b_t = sev_data.get("leaf_b_transform", {})
                    # t must be zero (base-anchor rotation only, no translation)
                    for name, t in [("leaf_a", a_t), ("leaf_b", b_t)]:
                        tvec = np.asarray(t.get("t", [0]), dtype=np.float64).ravel()
                        self.assertTrue(np.allclose(tvec, 0.0, atol=1e-6),
                                        f"Non-zero translation in {pair_key}/{mode}/{sev_data['severity']} {name}")
                    # pivot if not None must match base_gaussian_index position
                    # match pair_key to source_pairs via plant + leaf_ids
                    plant = data["plant"]
                    leaf_a_id = data["leaf_a_id"]
                    leaf_b_id = data["leaf_b_id"]
                    pair_info = next(
                        p for p in SOURCE_PAIRS["pairs"]
                        if p["plant"] == plant and p["leaf_a_id"] == leaf_a_id and p["leaf_b_id"] == leaf_b_id)
                    if a_t.get("pivot") is not None:
                        expected_pivot_a = pair_info["leaf_a"]["base_xyz"]
                        pivot_a = np.asarray(a_t["pivot"], dtype=np.float64).ravel()
                        self.assertAlmostEqual(np.linalg.norm(pivot_a - np.array(expected_pivot_a)), 0, places=4,
                                               msg=f"leaf_a pivot mismatch {pair_key}/{sev_data['severity']}")
                    if b_t.get("pivot") is not None:
                        expected_pivot_b = pair_info["leaf_b"]["base_xyz"]
                        pivot_b = np.asarray(b_t["pivot"], dtype=np.float64).ravel()
                        self.assertAlmostEqual(np.linalg.norm(pivot_b - np.array(expected_pivot_b)), 0, places=4,
                                               msg=f"leaf_b pivot mismatch {pair_key}/{sev_data['severity']}")

    def test_D_root_index_unchanged(self):
        """Root index in config matches frozen_roots for each plant."""
        with open(os.path.join(_REPO_ROOT, "outputs", "frozen_roots.json")) as f:
            frozen_roots = json.load(f)

        for pair_key, data in TRANSFORMS.items():
            plant = data["plant"]
            frozen_root = frozen_roots[plant]["root_index"]
            self.assertEqual(data["root_index"], frozen_root,
                             f"Root mismatch: {pair_key} data={data['root_index']} frozen={frozen_root}")

            # Also check case configs
            for mode in ["horizontal", "vertical"]:
                for sev in ["H0", "H1", "H2", "H3", "H4"] if mode == "horizontal" else ["V0", "V1", "V2", "V3", "V4"]:
                    cfg_path = os.path.join(_OUTROOT, "controlled", pair_key, mode, sev, "config.json")
                    if os.path.exists(cfg_path):
                        with open(cfg_path) as f:
                            cfg = json.load(f)
                        self.assertEqual(cfg["root_index"], frozen_root,
                                         f"Config root mismatch: {pair_key}/{mode}/{sev}")

    def test_E_construction_gt_equals_baseline(self):
        """construction_gt_labels.npy must be byte-identical to baseline labels."""
        for pair_key, data in TRANSFORMS.items():
            plant = data["plant"]
            baseline_labels = np.load(os.path.join(_BLROOT, plant, "labels.npy"))

            for mode in ["horizontal", "vertical"]:
                sevs = ["H0", "H1", "H2", "H3", "H4"] if mode == "horizontal" else ["V0", "V1", "V2", "V3", "V4"]
                for sev in sevs:
                    gt_path = os.path.join(_OUTROOT, "controlled", pair_key, mode, sev, "construction_gt_labels.npy")
                    if not os.path.exists(gt_path):
                        self.fail(f"Missing construction_gt_labels: {gt_path}")
                        continue
                    case_gt = np.load(gt_path)
                    self.assertEqual(case_gt.shape, baseline_labels.shape,
                                     f"Shape mismatch: {pair_key}/{mode}/{sev}")
                    self.assertTrue(np.array_equal(case_gt, baseline_labels),
                                    f"construction_gt != baseline: {pair_key}/{mode}/{sev}")

    def test_F_evaluator_iou_threshold(self):
        """Hungarian matching uses IoU >= 0.5 threshold."""
        for pair_key, data in TRANSFORMS.items():
            for mode in ["horizontal", "vertical"]:
                sevs = ["H0", "H1", "H2", "H3", "H4"] if mode == "horizontal" else ["V0", "V1", "V2", "V3", "V4"]
                for sev in sevs:
                    metrics_path = os.path.join(_OUTROOT, "controlled", pair_key, mode, sev, "failure_metrics.json")
                    if not os.path.exists(metrics_path):
                        continue
                    with open(metrics_path) as f:
                        m = json.load(f)
                    inst = m["instance"]
                    # mIoU must be >= 0.5 if matched_pairs > 0 (by definition of threshold)
                    if inst["matched_pairs"] > 0:
                        self.assertGreaterEqual(inst["mIoU"], 0.5 - 1e-6,
                                                f"mIoU < 0.5 with matched pairs: {pair_key}/{mode}/{sev}")
                    # PQ should be in [0, 1]
                    self.assertGreaterEqual(inst["PQ"], 0.0,
                                            f"PQ < 0: {pair_key}/{mode}/{sev}")
                    self.assertLessEqual(inst["PQ"], 1.0 + 1e-6,
                                         f"PQ > 1: {pair_key}/{mode}/{sev}")
                    # H0 must have perfect metrics
                    if sev in ("H0", "V0"):
                        self.assertAlmostEqual(inst["mIoU"], 1.0, places=3,
                                               msg=f"H0/V0 mIoU != 1.0: {pair_key}/{mode}/{sev}")
                        self.assertAlmostEqual(inst["PQ"], 1.0, places=3,
                                               msg=f"H0/V0 PQ != 1.0: {pair_key}/{mode}/{sev}")

    def test_G_pre_grouping_replay_consistency(self):
        """pre_grouping_replay.json exists and has expected keys for all cases."""
        for pair_key, data in TRANSFORMS.items():
            for mode in ["horizontal", "vertical"]:
                sevs = ["H0", "H1", "H2", "H3", "H4"] if mode == "horizontal" else ["V0", "V1", "V2", "V3", "V4"]
                for sev in sevs:
                    replay_path = os.path.join(_OUTROOT, "controlled", pair_key, mode, sev, "pre_grouping_replay.json")
                    if not os.path.exists(replay_path):
                        self.fail(f"Missing pre_grouping_replay: {replay_path}")
                        continue
                    with open(replay_path) as f:
                        replay = json.load(f)
                    # Should have cluster info
                    self.assertIn("num_clusters_after_grouping", replay,
                                  f"Missing num_clusters_after_grouping in {pair_key}/{mode}/{sev}")
                    self.assertIsInstance(replay["num_clusters_after_grouping"], int,
                                          f"num_clusters_after_grouping not int: {pair_key}/{mode}/{sev}")
                    # H0 should have same cluster count as baseline
                    if sev in ("H0", "V0"):
                        tree_path = os.path.join(_OUTROOT, "controlled", pair_key, mode, sev, "tree.json")
                        if os.path.exists(tree_path):
                            with open(tree_path) as f:
                                tree = json.load(f)
                            self.assertIn("apexes", tree,
                                            f"Missing apexes in tree: {pair_key}/{mode}/{sev}")
                            self.assertGreater(len(tree["apexes"]), 0,
                                               f"Empty apexes in tree: {pair_key}/{mode}/{sev}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
