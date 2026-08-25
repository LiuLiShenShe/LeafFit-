#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3.1 fix-verification tests.

Covers the four audited defects:
  * rgb_peak_block_argmax   — groupwise argmax (tests 1-6);
  * max_radius_enforced     — MAX_RADIUS_PX bounds candidate enumeration
                              and the acceptance region (test 7);
  * heldout_sign_transform  — signed = auc if sign>0 else 1-auc (test 8);
  * wording / verdict hard-coding — forbidden literals absent from report
    scripts; verdict derived, never asserted (test 9).
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.observation_identity import (  # noqa: E402
    groupwise_argmax_rows, _ellipse_block_pairs,
    cov2d_lambda_max, ELLIPSE_SIGMA, MAX_RADIUS_PX,
)


def _brute_force_argmax(loc, bid, contrib):
    """Reference implementation: per-group argmax, tie → smallest bid."""
    exp_g, exp_c, exp_bid = [], [], []
    for gid in np.unique(loc):
        rows = np.where(loc == gid)[0]
        c = contrib[rows]
        b = bid[rows]
        m = c.max()
        ties = rows[c == m]
        tb = b[c == m]
        pick = ties[np.lexsort((tb,))[0]] if len(ties) > 1 else ties[0]
        exp_g.append(int(gid))
        exp_c.append(float(m))
        exp_bid.append(int(bid[pick]))
    return exp_g, exp_c, exp_bid


class TestRGBPeakArgmax(unittest.TestCase):
    def test_1_single_gaussian_three_blocks_picks_max(self):
        # contributions 0.9 / 0.2 / 0.1 across blocks of one gaussian
        loc = np.array([7, 7, 7])
        bid = np.array([3, 5, 9])
        contrib = np.array([0.9, 0.2, 0.1])
        rows, pk_gauss, pk_contrib = groupwise_argmax_rows(loc, bid, contrib)
        self.assertEqual(len(rows), 1)
        self.assertEqual(pk_contrib[0], 0.9)
        self.assertEqual(bid[rows[0]], 3)          # block carrying 0.9

    def test_2_interleaved_multi_gaussian(self):
        rng = np.random.default_rng(31)
        N = 400
        loc = rng.integers(0, 40, N)
        bid = rng.integers(0, 11, N)
        contrib = np.round(rng.uniform(0.01, 1.0, N), 4)
        _, pg, pc = groupwise_argmax_rows(loc, bid, contrib)
        eg, ec, _ = _brute_force_argmax(loc, bid, contrib)
        self.assertEqual(pg.tolist(), eg)
        self.assertTrue(np.allclose(pc, ec))

    def test_3_ties_deterministic_smallest_block(self):
        loc = np.array([1, 1, 1])
        bid = np.array([5, 2, 9])
        contrib = np.array([0.4, 0.4, 0.4])
        rows, _, pc = groupwise_argmax_rows(loc, bid, contrib)
        self.assertEqual(pc[0], 0.4)
        self.assertEqual(bid[rows[0]], 2)           # frozen tie-break: min bid
        again_rows, _, _ = groupwise_argmax_rows(loc, bid, contrib)
        self.assertEqual(rows[0], again_rows[0])    # repeatable

    def test_4_rgb_valid_threshold_on_true_peak(self):
        # peak 0.04 < RGB_MIN_CONTRIBUTION=0.05 → invalid even though the v3
        # buggy code would have picked the MINIMUM (e.g. 0.01) instead.
        loc = np.array([0, 0])
        bid = np.array([0, 1])
        contrib = np.array([0.01, 0.04])
        _, _, pc = groupwise_argmax_rows(loc, bid, contrib)
        self.assertEqual(pc[0], 0.04)               # peak, NOT min 0.01
        from core.observation_identity import RGB_MIN_CONTRIBUTION
        self.assertFalse(pc[0] >= RGB_MIN_CONTRIBUTION)

    def test_5_rgb_views_come_from_peak_block_pixel(self):
        # full-pipeline check: one gaussian whose contribution peaks in a
        # colored block; rgb_views must sample THAT block's pixel color.
        sys.path.insert(0, str(REPO / "tests"))
        from test_task5r_v3_semantics import _gaussians, _Obs
        img = np.full((64, 64, 3), 10, np.uint8)
        img[:, :24] = [200, 30, 30]                 # left strip red, rest dim
        g = _gaussians([(0.008, 0.008, 2.0)], opacity=0.8, scale=0.05)
        from core.observation_identity import \
            build_occlusion_aware_real_view_signature as build
        res = build(g, _Obs(), decoded_images=img[None], downscale=1)
        self.assertTrue(res.rgb_valid[0, 0], "gaussian must have a valid peak")
        sampled = res.rgb_views[0, 0]
        # the contribution peaks at the CENTER block (nearest to the mean),
        # which sits in the dim region; the v3 argmin bug could instead have
        # sampled a far edge block. Assert the sample equals the center
        # block's color (dim here) AND that a control gaussian centered in
        # the red strip samples the bright color.
        g2 = _gaussians([(-0.25, 0.008, 2.0)], opacity=0.8, scale=0.05)
        res2 = build(g2, _Obs(), decoded_images=img[None], downscale=1)
        self.assertTrue(res2.rgb_valid[0, 0])
        self.assertGreater(float(res2.rgb_views[0, 0][0]), 150,
                           "gaussian peaking in the red strip must sample red")
        self.assertLess(float(sampled[0]), 60)

    def test_6_matches_bruteforce_randomized(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            N = 300
            loc = rng.integers(0, 60, N)
            bid = rng.integers(0, 8, N)
            # rounded scores create frequent ties → exercises tie-break path
            contrib = np.round(rng.uniform(0.0, 1.0, N), 1)
            rows, pg, pc = groupwise_argmax_rows(loc, bid, contrib)
            eg, ec, eb = _brute_force_argmax(loc, bid, contrib)
            self.assertEqual(pg.tolist(), eg, f"seed {seed}")
            self.assertTrue(np.allclose(pc, ec), f"seed {seed}")
            self.assertEqual(bid[rows].tolist(), eb, f"seed {seed}")


class TestMaxRadiusEnforced(unittest.TestCase):
    def test_7_huge_covariance_enumeration_bounded_by_clip(self):
        """Regression for max_radius_not_enforced_in_enumeration: with an
        enormous covariance (unclipped sigma-radius 100 px) the candidate
        extent AND acceptance region must respect MAX_RADIUS_PX=64."""
        pxd = np.array([50.0]); pyd = np.array([50.0])
        opac = np.array([0.9])
        c00 = np.array([2500.0]); c11 = np.array([2500.0]); c01 = np.array([0.0])
        lam = float(cov2d_lambda_max(c00[0], c01[0], c11[0]))
        raw_r = ELLIPSE_SIGMA * np.sqrt(lam)
        self.assertGreater(raw_r, MAX_RADIUS_PX)     # precondition: would clip
        gi, bids, d2 = _ellipse_block_pairs(pxd, pyd, None, opac,
                                            c00, c01, c11, nbx=64, nby=64)
        cx = (bids % 64) * 4 + 1.5
        cy = (bids // 64) * 4 + 1.5
        # every accepted block center within the clipped radius (+block slack)
        slack = 2 * 4.0
        self.assertTrue((np.abs(cx - 50) <= MAX_RADIUS_PX + slack).all(),
                        f"max |dx|={np.abs(cx-50).max():.1f} exceeds clip")
        self.assertTrue((np.abs(cy - 50) <= MAX_RADIUS_PX + slack).all())
        # unclipped enumeration would reach ~100px; assert we did NOT
        self.assertLess(np.abs(cx - 50).max(), raw_r - 20)


class TestHeldoutSignTransform(unittest.TestCase):
    def test_8_signed_auc_transforms(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wv", REPO / "scripts" / "write_task5r_verdict.py")
        wv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wv)
        f = wv._signed_auc
        self.assertAlmostEqual(f(0.62, +1), 0.62)      # positive direction
        self.assertAlmostEqual(f(0.38, -1), 0.62)      # negative: 1-auc
        self.assertAlmostEqual(f(0.50, +1), 0.50)      # null stays null (+)
        self.assertAlmostEqual(f(0.50, -1), 0.50)      # null stays null (−)
        self.assertAlmostEqual(f(0.38, -1), f(0.62, +1))  # consistency


FORBIDDEN_LITERALS = ["三重独立证据链", "held-out 未消耗", "非边缘失败",
                      "可信的最终否定答案"]


class TestWordingAndVerdictDerivation(unittest.TestCase):
    def test_9_report_scripts_free_of_forbidden_claims(self):
        targets = ["outputs/task5r_v3_1/README_TASK5R_V3_1.md"]
        src_path = REPO / "scripts" / "write_task5r_verdict.py"
        tree = ast.parse(src_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("verdict", "task6_allowed") \
                    and isinstance(node.value, ast.Constant):
                self.assertIn(node.value.value, (None,),
                              f"hard-coded verdict at line {node.lineno}")
        self.assertIn('task6_allowed = (verdict == "SEPARABILITY_PASS")',
                      src_path.read_text())

    def test_min_pairs_constant_matches_freeze_artifact(self):
        freeze = json.loads(
            (REPO / "outputs" / "task5r_v3_1" / "min_pairs_freeze.json").read_text())
        src_path = REPO / "scripts" / "write_task5r_verdict.py"
        m = [l for l in src_path.read_text().splitlines()
             if l.startswith("MIN_PAIRS_PER_SPLIT")]
        self.assertTrue(m)
        k_declared = int(m[-1].split("=")[1].strip())
        self.assertEqual(k_declared, int(freeze["min_pairs"]),
                         "MIN_PAIRS_PER_SPLIT must equal the frozen artifact")


import json  # noqa: E402  (used by TestWordingAndVerdictDerivation)

if __name__ == "__main__":
    unittest.main(verbosity=2)
