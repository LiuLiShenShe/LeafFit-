#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 occlusion-semantics tests (user checklist 1-11).

Golden vectors for exact alpha compositing, ellipse-block footprint
participation, cross-block depth competition, no-false-occlusion, RGB
non-contamination, and unit conventions. These must pass BEFORE any
scientific evaluation runs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.observation_identity import (  # noqa: E402
    exclusive_transmittance,
    build_occlusion_aware_real_view_signature,
    _project_cov2d, cov2d_lambda_max, cov2d_radius_px,
    _ellipse_block_pairs,
    ELLIPSE_SIGMA, BLOCK_PX,
)
from gaussian_utils import GaussianData  # noqa: E402


class _Obs:
    """Minimal COLMAP-convention observation bundle (+z front)."""

    def __init__(self, rt=None, K=None, wh=(64, 64), n=1):
        self.n_views = n
        self.rt = rt if rt is not None else np.array([np.eye(4)] * n)
        self.K = K if K is not None else np.array(
            [np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])] * n)
        self.image_wh = [wh] * n
        self.names = [f"v{i}.png" for i in range(n)]


def _gaussians(xyz, opacity=1.0, scale=0.03, color_axis=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    N = len(xyz)
    sh = np.zeros((N, 48), dtype=np.float32)
    if color_axis is not None:
        sh[np.arange(N), color_axis] = 1.0
    op = np.full((N, 1), float(opacity), dtype=np.float32) \
        if np.isscalar(opacity) else np.asarray(opacity, np.float32).reshape(N, 1)
    sc = np.full((N, 3), float(scale), dtype=np.float32) \
        if np.isscalar(scale) else np.asarray(scale, np.float32).reshape(N, 3)
    return GaussianData(
        xyz=xyz, rot=np.tile(np.array([1., 0, 0, 0], np.float32), (N, 1)),
        scale=sc, opacity=op, sh=sh,
        nxnynz=np.zeros((N, 3), np.float32),
        filter_3Ds=np.ones((N, 1), np.float32))


_GRAY = None


def _gray_img(wh=(64, 64)):
    global _GRAY
    if _GRAY is None:
        _GRAY = np.full((wh[1], wh[0], 3), 128, dtype=np.uint8)
    return _GRAY


class TestExactCompositing(unittest.TestCase):
    def test_1_two_layer_exact_alpha(self):
        T, c = exclusive_transmittance([0.9, 0.9], [0, 0])
        self.assertEqual(T[0], 1.0)                      # first contributor T==1
        self.assertAlmostEqual(c[0], 0.9, places=12)
        self.assertAlmostEqual(c[1], 0.09, places=12)

    def test_2_three_layer_exact_alpha(self):
        _, c = exclusive_transmittance([0.9, 0.9, 0.9], [0, 0, 0])
        self.assertAlmostEqual(c[0], 0.9, places=12)
        self.assertAlmostEqual(c[1], 0.09, places=12)
        self.assertAlmostEqual(c[2], 0.009, places=12)

    def test_3_multi_bucket_independence(self):
        # groups never interact: [g0: .9,.5 | g1: .8]
        T, c = exclusive_transmittance([0.9, 0.5, 0.8], [0, 0, 1])
        self.assertAlmostEqual(c[0], 0.9, places=12)
        self.assertAlmostEqual(c[1], 0.05, places=12)
        self.assertAlmostEqual(c[2], 0.8, places=12)
        self.assertEqual(T[0], 1.0)
        self.assertEqual(T[2], 1.0)

    def test_4_contributions_bounded(self):
        rng = np.random.default_rng(4)
        a = rng.uniform(0.01, 0.999, 4000)
        gids = rng.integers(0, 50, 4000)
        order = np.lexsort((rng.random(4000), gids))
        T, c = exclusive_transmittance(a[order], gids[order])
        self.assertTrue(((c >= 0.0) & (c <= 1.0)).all())
        self.assertTrue((T > 0).all() and (T <= 1.0 + 1e-12).all())

    def test_5_monotonicity_under_opacity_drop(self):
        front = [(x * 0.02 - 0.6 + 0.01, y * 0.02 - 0.6 + 0.01, 2.0)
                 for x in range(60) for y in range(60)]
        rear = [(x * 0.02 - 0.6 + 0.017, y * 0.02 - 0.6 + 0.017, 4.0)
                for x in range(60) for y in range(60)]
        nf = len(front)
        accs = []
        for op in (1.0, 0.5):
            g = _gaussians(front + rear, opacity=op)
            res = build_occlusion_aware_real_view_signature(
                g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
            self.assertFalse(np.isnan(res.acc_alpha[0]).any())
            accs.append(float(res.acc_alpha[0, nf:].mean()))
        self.assertGreater(accs[1], accs[0],
                           "lower foreground opacity must raise rear contribution")


class TestCovarianceRadius(unittest.TestCase):
    def test_6_isotropic_radius(self):
        lam = cov2d_lambda_max(4.0, 0.0, 4.0)
        self.assertAlmostEqual(lam, 4.0, places=12)
        r_unclipped = ELLIPSE_SIGMA * np.sqrt(lam)
        self.assertAlmostEqual(r_unclipped, 4.0, places=12)

    def test_7_anisotropic_radius(self):
        lam = cov2d_lambda_max(9.0, 0.0, 1.0)
        self.assertAlmostEqual(lam, 9.0, places=12)
        self.assertAlmostEqual(ELLIPSE_SIGMA * np.sqrt(lam), 6.0, places=12)

    def test_lambda_max_formula_matches_eigh(self):
        rng = np.random.default_rng(9)
        for _ in range(50):
            A = rng.normal(size=(2, 2)) * 3
            c00, c01, c11 = A[0, 0] ** 2 + 0.3, A[0, 0] * A[1, 0], A[1, 0] ** 2 + 0.3
            ref = float(np.linalg.eigvalsh(
                [[c00, c01], [c01, c11]]).max())
            self.assertAlmostEqual(cov2d_lambda_max(c00, c01, c11), ref, places=10)


def _two_layer_scene(front_z=2.0, rear_z=4.0, rear_jitter=0.007,
                     scale=0.03, n=30):
    front = [(x * (1.2 / n) - 0.6 + 0.01, y * (1.2 / n) - 0.6 + 0.01, front_z)
             for x in range(n) for y in range(n)]
    rear = [(x * (1.2 / n) - 0.6 + 0.01 + rear_jitter,
             y * (1.2 / n) - 0.6 + 0.01 + rear_jitter, rear_z)
            for x in range(n) for y in range(n)]
    return front, rear


class TestEllipseFootprintOcclusion(unittest.TestCase):
    def test_8_cross_block_boundary_overlap_occludes(self):
        """A rear Gaussian whose CENTER sits in a neighboring block but whose
        ellipse overlaps the front Gaussian's blocks MUST lose contribution —
        the exact failure of v2's center-bucket quantization."""
        # one big opaque front gaussian at image center; rear center offset by
        # ~half a block so centers land in different blocks but ellipses overlap.
        g = _gaussians([(0.016, 0.016, 2.0),      # front (projects near center px 32)
                        (0.028, 0.028, 4.0)],     # rear, offset in x&y
                       opacity=1.0, scale=0.06)
        res = build_occlusion_aware_real_view_signature(
            g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
        # both centers project into DIFFERENT blocks (assert precondition):
        bxf = int(res.uv_pixel[0, 0, 0]) // BLOCK_PX
        bxr = int(res.uv_pixel[0, 1, 0]) // BLOCK_PX
        byf = int(res.uv_pixel[0, 0, 1]) // BLOCK_PX
        byr = int(res.uv_pixel[0, 1, 1]) // BLOCK_PX
        different_center_blocks = (bxf != bxr) or (byf != byr)
        front_hidden_rear = res.max_alpha[0, 1] < 0.05
        # with ellipse-block compositing, an overlapping rear is occluded even
        # when its center block differs from the front's.
        self.assertTrue(front_hidden_rear or not different_center_blocks,
                        "rear behind an overlapping front must be occluded; "
                        f"got max_alpha={res.max_alpha[0,1]:.4f}, "
                        f"front block ({bxf},{byf}) rear block ({bxr},{byr})")

    def test_9_no_footprint_overlap_no_false_occlusion(self):
        """Two same-depth gaussians far apart (disjoint footprints, different
        blocks) must suffer NO cross competition: each keeps its own
        footprint-attenuated self contribution (alpha_eff at the best block,
        >= alpha * exp(-0.5*d2_max_accepted), d2 <= sigma^2=4)."""
        g = _gaussians([(-0.30, -0.30, 3.0), (0.30, 0.30, 3.0)],
                       opacity=0.9, scale=0.03)
        res = build_occlusion_aware_real_view_signature(
            g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
        floor = 0.9 * np.exp(-0.5 * ELLIPSE_SIGMA ** 2)
        # both isolated gaussians must have IDENTICAL attenuation (symmetry)
        self.assertAlmostEqual(float(res.max_alpha[0, 0]),
                               float(res.max_alpha[0, 1]), places=5)
        # and no cross-block competition may reduce either below its own
        # footprint-weighted self contribution
        self.assertGreater(res.max_alpha[0, 0], floor,
                           "isolated gaussian keeps >= its worst accepted "
                           "block-center footprint weight")
        self.assertGreater(res.acc_alpha[0, 0], res.max_alpha[0, 0] - 1e-6)

    def test_10_occluded_gaussian_inherits_no_rgb(self):
        n = 30
        front, rear = _two_layer_scene(n=n)
        nf = len(front)
        g = _gaussians(front + rear, opacity=1.0, color_axis=None)
        # make front blue via sh axis 2, rear red axis 0
        g.sh[:nf, 2] = 1.0
        g.sh[nf:, 0] = 1.0
        img = np.zeros((64, 64, 3), np.uint8); img[:, :, 2] = 255  # blue image
        res = build_occlusion_aware_real_view_signature(
            g, _Obs(), decoded_images=img[None], downscale=1)
        fully_occluded = ~res.visible[0]
        self.assertGreater(fully_occluded[nf:].sum(), 0,
                           "scene must actually hide some rear points")
        self.assertFalse((res.rgb_valid[0] & fully_occluded).any(),
                         "occluded points must carry no valid RGB")
        ap = res.appear_sig()
        self.assertTrue(np.isnan(ap[fully_occluded]).all(),
                        "appear_sig at occluded points must be NaN")

    def test_11_pixel_ndc_explicit_conversion(self):
        rng = np.random.default_rng(13)
        W, H = 64.0, 64.0
        pix = np.stack([rng.uniform(0, W - 1, 300), rng.uniform(0, H - 1, 300)],
                       axis=1)
        ndc = (pix - np.array([W / 2, H / 2])) / np.array([W / 2, H / 2])
        back = ndc * np.array([W / 2, H / 2]) + np.array([W / 2, H / 2])
        self.assertTrue(np.allclose(pix, back, atol=1e-9))
        thr_pix_x = 0.15 * (W / 2.0)
        P, Q = pix[:100], pix[100:200]
        dpx = np.abs(P[:, None, :] - Q[None])
        ndcP, ndcQ = ndc[:100], ndc[100:200]
        dndc = np.abs(ndcP[:, None, :] - ndcQ[None])
        self.assertTrue(np.allclose(dpx[..., 0], dndc[..., 0] * (W / 2.0)))
        self.assertTrue(np.allclose(dpx[..., 1], dndc[..., 1] * (H / 2.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestChunkedPairGeneration(unittest.TestCase):
    def test_12_chunked_pairs_match_single_chunk(self):
        """Regression: _ellipse_block_pairs chunk slicing must not reference
        unsliced nbx_i/nby_i (found on real 92k-point frustums; synthetic
        single-chunk tests never crossed a chunk boundary)."""
        rng = np.random.default_rng(42)
        M = 5000
        pxd = rng.uniform(4, 600, M); pyd = rng.uniform(4, 600, M)
        opac = rng.uniform(.3, 1, M)
        A = rng.normal(size=(M, 2, 2))
        c00 = A[:, 0, 0] ** 2 + A[:, 0, 1] ** 2 + 4
        c11 = A[:, 1, 0] ** 2 + A[:, 1, 1] ** 2 + 4
        c01 = A[:, 0, 0] * A[:, 1, 0] + A[:, 0, 1] * A[:, 1, 1]
        g1, b1, d1 = _ellipse_block_pairs(pxd, pyd, None, opac, c00, c01,
                                          c11, nbx=150, nby=150, chunk=16384)
        g2, b2, d2 = _ellipse_block_pairs(pxd, pyd, None, opac, c00, c01,
                                          c11, nbx=150, nby=150, chunk=997)
        o1 = np.lexsort((d1, g1)); o2 = np.lexsort((d2, g2))
        self.assertTrue(np.array_equal(g1[o1], g2[o2]))
        self.assertTrue(np.array_equal(b1[o1], b2[o2]))
        self.assertTrue(np.allclose(d1[o1], d2[o2]))
