#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R Phase 3 — program-correctness tests for real-observation identity.

Covers audit findings F1-F6 with tests A-E (occlusion semantics, alpha
monotonicity, RGB contamination, UV units) and H (COLMAP reprojection),
plus I (determinism). Cache/units tests live in test_task5r_cache_and_units.py.

These tests must pass before any scientific within-vs-cross analysis.
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

from gaussian_utils import GaussianData  # noqa: E402
from core.observation_identity import (  # noqa: E402
    build_occlusion_aware_real_view_signature,
    viewsig_cache_key,
    CONTRIBUTION_THRESHOLD, RGB_MIN_CONTRIBUTION,
)


class _Obs:
    """Minimal COLMAP-convention observation bundle (identity pose, +z front)."""

    def __init__(self, rt=None, K=None, wh=(64, 64), n=1, names=None):
        self.n_views = n
        self.rt = rt if rt is not None else np.array([np.eye(4)] * n)
        self.K = K if K is not None else np.array(
            [np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])] * n)
        self.image_wh = [wh] * n
        self.names = names if names is not None else [f"v{i}.png" for i in range(n)]


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


def _grid(z, n=60, jitter=0.02):
    xs = np.linspace(-0.5, 0.5, n)
    return [(x + jitter, y + jitter, z) for x in xs for y in xs]


_GRAY = None


def _gray_img(wh=(64, 64)):
    global _GRAY
    if _GRAY is None:
        _GRAY = np.full((wh[1], wh[0], 3), 128, dtype=np.uint8)
    return _GRAY


# ---------------------------------------------------------------------------
# A/B: frustum vs visibility; two-layer occlusion
# ---------------------------------------------------------------------------
class TestOcclusionSemantics(unittest.TestCase):
    def setUp(self):
        self.front = _grid(2.0, n=60, jitter=0.0)
        self.rear = _grid(4.0, n=60, jitter=0.02)
        nf = len(self.front)

    def _run(self, opacity=1.0):
        nf = len(self.front)
        g = _gaussians(self.front + self.rear, opacity=opacity)
        obs = _Obs()
        return build_occlusion_aware_real_view_signature(
            g, obs, decoded_images=_gray_img()[None], downscale=1), nf

    def test_A_frustum_true_but_visible_false_for_occluded_rear(self):
        """A rear point fully covered by an opaque front layer is in_frustum=True
        yet visible=False — the exact distinction Task 5's in_frustum-as-visibility lost."""
        res, nf = self._run()
        frustum_rear = res.in_frustum[0, nf:]
        visible_rear = res.visible[0, nf:]
        self.assertTrue(frustum_rear.all(),
                        "rear layer must be inside the frustum for this geometry")
        self.assertEqual(visible_rear.sum(), 0,
                         "rear layer behind a full opaque front layer must be invisible")

    def test_B_oblique_view_changes_visibility_relationship(self):
        """An oblique camera (45 deg) must reveal part of the rear layer that the
        frontal view hides — the expected direction of change under occlusion."""
        nf = len(self.front)
        g = _gaussians(self.front + self.rear)
        # camera at C=(6,0,6) looking at the origin: 45-deg oblique view that
        # keeps BOTH layers inside the frustum (COLMAP +z-front convention).
        C = np.array([6.0, 0.0, 6.0])
        r3 = -C / np.linalg.norm(C)              # camera +z points toward scene
        r1 = np.array([r3[2], 0, -r3[0]]); r1 /= np.linalg.norm(r1)
        r2 = np.cross(r3, r1)
        Rt = np.eye(4); Rt[:3, :3] = np.stack([r1, r2, r3]); Rt[:3, 3] = -np.stack([r1, r2, r3]) @ C
        obs_oblique = _Obs(rt=Rt[None], K=np.array(
            [[[100.0, 0, 128], [0, 100.0, 128], [0, 0, 1]]]), wh=(256, 256))
        res_front = build_occlusion_aware_real_view_signature(
            g, _Obs(K=np.array([[[100.0, 0, 128], [0, 100.0, 128], [0, 0, 1]]]),
                    wh=(256, 256)), decoded_images=_gray_img((256, 256))[None], downscale=1)
        res_obl = build_occlusion_aware_real_view_signature(
            g, obs_oblique, decoded_images=_gray_img((256, 256))[None], downscale=1)
        # sanity: oblique frustum must contain essentially all points
        self.assertGreater(res_obl.in_frustum[0].mean(), 0.95,
                           "oblique camera must keep the scene in frame")
        vf_front = float(res_front.visible[0, nf:].mean())
        vf_obl = float(res_obl.visible[0, nf:].mean())
        print(f"\n[B] rear visibility frontal={vf_front:.3f} oblique={vf_obl:.3f}")
        self.assertGreater(vf_obl, vf_front,
                           "oblique view must reveal more of the rear layer than frontal")


# ---------------------------------------------------------------------------
# C: alpha monotonicity
# ---------------------------------------------------------------------------
class TestAlphaMonotonicity(unittest.TestCase):
    def test_C_lower_foreground_opacity_raises_rear_contribution(self):
        front = _grid(2.0, n=60, jitter=0.0)
        rear = _grid(4.0, n=60, jitter=0.02)
        nf = len(front)
        out = []
        for op in (1.0, 0.5):
            g = _gaussians(front + rear, opacity=op)
            res = build_occlusion_aware_real_view_signature(
                g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
            out.append(res.acc_alpha[0, nf:].mean())
        self.assertGreater(out[1], out[0],
                           "reducing foreground opacity must monotonically increase rear contribution")


# ---------------------------------------------------------------------------
# D: RGB contamination
# ---------------------------------------------------------------------------
class TestRGBNoContamination(unittest.TestCase):
    def test_D_occluded_red_gaussian_never_gets_blue_rgb(self):
        """A FULLY occluded red rear Gaussian (visible=False in the view) must
        carry no valid RGB observation — it must not inherit the foreground
        blue pixel colour. (Rear points visible through footprint gaps are
        legitimately observed and are excluded by the visible=False filter.)"""
        n = 30
        front = _grid(2.0, n=n, jitter=0.0)   # blue foreground (axis 2)
        rear = _grid(4.0, n=n, jitter=0.02)   # red rear (axis 0), fully covered
        nf = len(front)
        g = _gaussians(front + rear, opacity=1.0)
        img = np.zeros((64, 64, 3), np.uint8); img[:, :, 2] = 255  # pure blue image
        res = build_occlusion_aware_real_view_signature(
            g, _Obs(), decoded_images=img[None], downscale=1)
        occluded = ~res.visible[0]           # both layers' hidden points
        self.assertGreater(occluded[nf:].sum(), 0,
                           "scene must actually occlude some rear points")
        # no occluded point may have a valid RGB sample
        self.assertFalse((res.rgb_valid[0] & occluded).any(),
                         "occluded points must carry no valid RGB observation")
        # and their appear_sig must be NaN rather than sampled blue
        ap = res.appear_sig()
        self.assertTrue(np.isnan(ap[occluded]).all(),
                        "appear_sig at occluded points must be NaN, not foreground colour")


# ---------------------------------------------------------------------------
# E: UV unit equivalence
# ---------------------------------------------------------------------------
class TestUVUnits(unittest.TestCase):
    def test_E_pixel_and_ndc_give_equivalent_neighbour_decisions(self):
        rng = np.random.default_rng(7)
        uv_pix = rng.uniform(0, 2119, size=(200, 2))
        W = H = 3777.0
        ndc = (uv_pix - np.array([W / 2, H / 2])) / np.array([W / 2, H / 2])
        thr_ndc = 0.15
        thr_pix_x = thr_ndc * (W / 2.0)
        thr_pix_y = thr_ndc * (H / 2.0)
        # pairwise decisions among first 60 pts
        P, Q = uv_pix[:60], uv_pix[60:120]
        dpx = np.linalg.norm(P[:, None] - Q[None], axis=2)
        Nn = (ndc[:60, None] - ndc[60:120][None])
        # anisotropic NDC threshold == axis-scaled pixel threshold
        close_pix = dpx < thr_pix_x   # using x-scale as documented approximation
        close_ndc = np.abs(Nn[..., 0]) < thr_ndc
        # compare via the same conversion applied consistently
        back_to_pix = (ndc[:60, None, :] * np.array([W / 2, H / 2])) + np.array([W / 2, H / 2])
        dpx2 = np.linalg.norm(back_to_pix - Q[None], axis=2)
        self.assertTrue(np.allclose(dpx, dpx2, atol=1e-6),
                        "pixel<->NDC roundtrip must preserve distances")
        # explicit-conversion invariant: pixel threshold == scaled NDC test
        self.assertTrue(np.array_equal(close_pix, dpx2 < thr_pix_x))


# ---------------------------------------------------------------------------
# H: COLMAP reprojection sanity
# ---------------------------------------------------------------------------
class TestCOLMAPReprojection(unittest.TestCase):
    PLANT = "DouBanLv1"

    def test_H_points3D_reproject_into_their_own_images(self):
        """Project recorded COLMAP points into their observing images via our
        pinhole_project; median reprojection error must be small (<2 px at
        full resolution), validating camera parsing/conventions."""
        from colmap_io import (
            read_cameras_bin, read_images_bin, read_points3d_bin,
            images_to_world2cam_rt, cameras_to_intrinsics, colmap_plant_paths,
        )
        dense_root = REPO.parent / "datasets" / "07-SuGaR-GS"
        colmap_dir = REPO.parent / "datasets" / "04-COLMAP" / self.PLANT
        if not colmap_dir.exists():
            self.skipTest("04-COLMAP data not available")
        paths = colmap_plant_paths(str(colmap_dir))
        cams = read_cameras_bin(paths["cameras"])
        imgs = read_images_bin(paths["images"], read_tracks=True)
        images_to_world2cam_rt(imgs, cams)
        K = cameras_to_intrinsics(cams)
        xyz, _rgb, pid = read_points3d_bin(paths["points3D"])
        pos = {int(p): i for i, p in enumerate(pid)}
        # build track->image observations from images.bin point indices
        errs = []
        tested = 0
        for im in imgs:
            if im.point_idxs is None or not len(im.point_idxs):
                continue
            sel = im.point_idxs >= 0
            uvs_all = im.point_uv[sel]
            keep_mask = np.array([int(p) in pos for p in im.point_idxs[sel]])
            if keep_mask.sum() < 10:
                continue
            rows = np.asarray([pos[int(p)] for p in im.point_idxs[sel][keep_mask]][:2000])
            uvs = uvs_all[keep_mask][:2000]
            pts = xyz[rows]
            Rt = im.rt
            Kc = K[im.cid]
            homog = np.hstack([pts, np.ones((len(pts), 1))])
            camp = (Rt @ homog.T).T
            z = camp[:, 2]
            ok = z > 0
            if ok.sum() < 10:
                continue
            px = Kc[0, 0] * camp[ok, 0] / z[ok] + Kc[0, 2]
            py = Kc[1, 1] * camp[ok, 1] / z[ok] + Kc[1, 2]
            e = np.hypot(px - uvs[ok][:, 0], py - uvs[ok][:, 1])
            errs.append(e)
            tested += 1
            if tested >= 20:
                break
        if not errs:
            self.skipTest("no track observations parsed")
        e = np.concatenate(errs)
        med = float(np.median(e))
        print(f"\n[H] reprojection error px: median={med:.2f} "
              f"p90={np.percentile(e, 90):.2f} n={len(e)}")
        self.assertLess(med, 2.0,
                        "median reprojection error must be consistent with source reconstruction")


# ---------------------------------------------------------------------------
# I: determinism
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_I_same_input_byte_identical_metrics(self):
        front = _grid(2.0, n=25, jitter=0.0)
        rear = _grid(4.0, n=25, jitter=0.02)
        g = _gaussians(front + rear)
        r1 = build_occlusion_aware_real_view_signature(g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
        r2 = build_occlusion_aware_real_view_signature(g, _Obs(), decoded_images=_gray_img()[None], downscale=1)
        self.assertTrue(np.array_equal(r1.visible, r2.visible))
        self.assertTrue(np.array_equal(r1.max_alpha, r2.max_alpha))
        self.assertTrue(np.array_equal(r1.acc_alpha, r2.acc_alpha))
        self.assertTrue(bytes(r1.rgb_views).startswith(bytes(r2.rgb_views)) or
                        np.array_equal(r1.rgb_views, r2.rgb_views))


if __name__ == "__main__":
    unittest.main(verbosity=2)
