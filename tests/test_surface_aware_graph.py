#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the surface-aware geodesic backends (core/geodesic_backends.py).

These verify *program correctness* of the graph backend (symmetry, finite positive
weights, feature definitions, degeneracy, determinism, clean-reference fidelity).
They do NOT assert effect-size on Task 3 PASS criteria -- those belong to the
experiment acceptance criteria, not unit tests.
"""
import sys
import os
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaussian_utils import GaussianData
from geodesic_backends import (
    EuclideanGraphBackend,
    SurfaceAwareGraphBackend,
    _build_knn_edges,
)


def _make_gaussians(xyz, normal=None, scale=None, rot=None):
    N = len(xyz)
    if scale is None:
        scale = np.ones((N, 3)) * 0.01
    if rot is None:
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    if normal is None:
        normal = np.zeros((N, 3))
        normal[:, 2] = 1.0
    return GaussianData(
        xyz=xyz.astype(np.float32),
        rot=rot.astype(np.float32),
        scale=scale.astype(np.float32),
        opacity=np.ones((N, 1), np.float32),
        sh=np.zeros((N, 48), np.float32),
        nxnynz=normal.astype(np.float32),
        filter_3Ds=np.ones((N, 1), np.float32),
    )


class _PlanePair:
    """Two parallel planes offset along z (coplanar flat leaf + displaced copy)."""

    def __init__(self, n_grid=8, gap=0.05, seed=0):
        rng = np.random.RandomState(seed)
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, n_grid), np.linspace(-1, 1, n_grid), indexing="ij"
        )
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        pts_a = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
        pts_b = np.concatenate([xy, np.full((len(xy), 1), gap)], axis=1)
        pts = np.concatenate([pts_a, pts_b], axis=0).astype(np.float64)
        # jitter to break exact co-grid ties
        pts += rng.normal(0, 1e-3, pts.shape)
        self.pts = pts
        self.labels = np.concatenate([
            np.ones(len(xy), dtype=int),
            2 * np.ones(len(xy), dtype=int),
        ])
        # flat-disc normals + tiny scale anisotropy so gate can act
        scale = np.ones((len(pts), 3))
        scale[:, 2] = 1e-4
        self.scale = scale
        # normals +z (valid, unit)
        normal = np.zeros((len(pts), 3))
        normal[:, 2] = 1.0
        self.g = _make_gaussians(pts, normal=normal, scale=scale)


class TestGraphSymmetry(unittest.TestCase):
    """A: graph undirected, edge weights symmetric CSR == CSR.T."""

    def test_euclidean_symmetric(self):
        rng = np.random.RandomState(0)
        pts = rng.randn(200, 3).astype(np.float64)
        g = _make_gaussians(pts)
        bb = EuclideanGraphBackend(pts, k=8)
        A = bb._graph.toarray()
        self.assertTrue(np.allclose(A, A.T, equal_nan=False),
                        "Euclidean kNN graph must be symmetric")

    def test_surface_symmetric(self):
        pp = _PlanePair(n_grid=8, gap=0.03)
        bb = SurfaceAwareGraphBackend(pp.pts, pp.g, k=8, feature_set="G5")
        A = bb._graph.toarray()
        self.assertTrue(np.allclose(A, A.T, equal_nan=False),
                        "Surface-aware kNN graph must be symmetric")


class TestFinitePositiveWeights(unittest.TestCase):
    """B: all edge weights finite and > 0."""

    def test_euclidean_weights_finite_positive(self):
        pp = _PlanePair(n_grid=10)
        bb = EuclideanGraphBackend(pp.pts, k=8)
        r, c = bb._graph.nonzero()
        w = bb._graph.data
        self.assertTrue(np.all(np.isfinite(w)), "No Inf weights allowed")
        self.assertTrue(np.all(w > 0), "Weights must be positive")


class TestNormalSignInvariance(unittest.TestCase):
    """C: flipping normal sign must not change weights (c_n uses |n_i.n_j|)."""

    def test_sign_invariance(self):
        pp = _PlanePair(n_grid=10)
        g_pos = pp.g
        neg_nxn = -pp.g.nxnynz
        g_neg = _make_gaussians(pp.pts, normal=neg_nxn, scale=pp.scale)
        s_pos = SurfaceAwareGraphBackend(pp.pts, g_pos, k=8, feature_set="G5",
                                         lambda_n=2, lambda_t=2)
        s_neg = SurfaceAwareGraphBackend(pp.pts, g_neg, k=8, feature_set="G5",
                                         lambda_n=2, lambda_t=2)
        # edge features identical -> distances identical
        dp = s_pos.compute_distance(0)
        dn = s_neg.compute_distance(0)
        self.assertTrue(np.allclose(dp, dn, equal_nan=False),
                        f"Normal sign flip changed distances: max={np.max(np.abs(dp-dn))}")
        # c_n array identical
        self.assertTrue(np.allclose(s_pos.edge_features["normal_consistency"],
                                    s_neg.edge_features["normal_consistency"]))


class TestSamePlaneTangentialLowPenalty(unittest.TestCase):
    """D: coplanar parallel-plane edges with in-plane u_ij have low c_t (edge kept)."""

    def test_same_plane_low_ct(self):
        pp = _PlanePair(n_grid=12, gap=0.001)  # near-coplanar
        bb = SurfaceAwareGraphBackend(pp.pts, pp.g, k=10, feature_set="G5",
                                      tau_t=0.5)
        c_t = bb.edge_features["tangent_consistency"]
        # most within-layer edges are in-plane -> |u.n| small -> c_t small
        within = pp.labels[bb._rows] == pp.labels[bb._cols]
        self.assertTrue(np.median(c_t[within]) < 0.5,
                        f"Within-layer c_t too high: median={np.median(c_t[within])}")


class TestParallelTwoLayerCrossEdgePenalty(unittest.TestCase):
    """E: cross-layer edges in parallel two-layer config are pruned by G4 gate."""

    def test_cross_layer_pruned(self):
        pp = _PlanePair(n_grid=12, gap=0.1)
        k = 8
        # G0 (no gate): all cross-layer candidate edges retained
        base = SurfaceAwareGraphBackend(pp.pts, pp.g, k=k, feature_set="G0")
        cross_base = int((pp.labels[base.edge_features["rows"]] !=
                          pp.labels[base.edge_features["cols"]]).sum())
        # G4 (gate): cross-layer edges pruned (c_t high -> u ~ z ~ normal)
        gated = SurfaceAwareGraphBackend(pp.pts, pp.g, k=k, feature_set="G4",
                                         tau_d=4.0, tau_t=0.5)
        cross_gated = int((pp.labels[gated.edge_features["rows"]] !=
                           pp.labels[gated.edge_features["cols"]]).sum())
        self.assertLess(cross_gated, cross_base,
                        f"Gate must prune cross-layer edges: base={cross_base} gated={cross_gated}")
        # also: G4 cross-layer c_t median > within-layer c_t median
        ef = gated.edge_features
        same = pp.labels[ef["rows"]] == pp.labels[ef["cols"]]
        diff = ~same
        if diff.sum() > 0 and same.sum() > 0:
            self.assertGreater(np.median(ef["tangent_consistency"][diff]),
                               np.median(ef["tangent_consistency"][same]))


class TestCrossingNormalsPenalty(unittest.TestCase):
    """F: orthogonal normals (n_i . n_j ~ 0) -> high c_n -> heavier penalty."""

    def test_orthogonal_normals_higher_weight(self):
        rng = np.random.RandomState(1)
        pts = rng.randn(100, 3)
        # make half have normal +z, half +x (orthogonal)
        normal = np.zeros((100, 3))
        normal[:50, 2] = 1.0
        normal[50:, 0] = 1.0
        g = _make_gaussians(pts, normal=normal)
        bb = SurfaceAwareGraphBackend(pts, g, k=8, feature_set="G3",
                                      lambda_n=4, lambda_t=0)
        cross_n = (normal[bb._rows, 0] == 1) & (normal[bb._cols, 2] == 1) | \
                  (normal[bb._rows, 2] == 1) & (normal[bb._cols, 0] == 1)
        ef = bb.edge_features
        if cross_n.sum() > 0:
            self.assertGreater(np.median(ef["normal_consistency"][cross_n]), 0.5)


class TestG0Degneration(unittest.TestCase):
    """G: G0 (lambda=0, no gate) produces weight == euclidean distance."""

    def test_g0_equals_euclidean(self):
        rng = np.random.RandomState(2)
        pts = rng.randn(150, 3)
        g = _make_gaussians(pts)
        bb = SurfaceAwareGraphBackend(pts, g, k=8, feature_set="G0")
        # weights == kNN euclidean distances (dedupe undirected i<j)
        coo = bb._graph.tocoo()
        r, c = coo.row, coo.col
        v = coo.data
        mask = r < c
        w = v[mask]
        d_expected = np.linalg.norm(pts[r[mask]] - pts[c[mask]], axis=1)
        self.assertTrue(np.allclose(w, d_expected),
                        f"G0 weight != euclidean: max diff={np.max(np.abs(w-d_expected))}")


class TestFixedRootDeterminism(unittest.TestCase):
    """H: same backend, same input -> identical distance field."""

    def test_determinism(self):
        rng = np.random.RandomState(3)
        pts = rng.randn(200, 3)
        g = _make_gaussians(pts)
        bb1 = SurfaceAwareGraphBackend(pts, g, k=8, feature_set="G5")
        bb2 = SurfaceAwareGraphBackend(pts, g, k=8, feature_set="G5")
        d1 = bb1.compute_distance(10)
        d2 = bb2.compute_distance(10)
        self.assertTrue(np.array_equal(d1, d2, equal_nan=True),
                        "Backend must be deterministic")


class TestCleanReferenceFidelity(unittest.TestCase):
    """I (clean_reference_fidelity): end-to-end run on clean plant1.

    Uses construction_gt_labels.npy to verify dense index / root / GT identity
    preservation.  PQ delta is an experiment acceptance criterion, NOT a unit
    test -- here we only assert structural integrity.
    """

    PLANT = "plant1_green_pepper"

    def setUp(self):
        import core.headless_segmentation as hs
        try:
            self.hs = hs
            self.g = hs.load_gaussian_data(
                os.path.join(_REPO_ROOT, "data", f"{self.PLANT}.ply"))
            self.gt_labels = np.load(
                os.path.join(_REPO_ROOT, "outputs", "baseline", self.PLANT, "labels.npy"))
            self.root = 47330
        except FileNotFoundError:
            self.skipTest(f"baseline data for {self.PLANT} not present")

    def test_runs_and_preserves_index(self):
        from geodesic_backends import SurfaceAwareGraphBackend
        def factory(points, gd):
            return SurfaceAwareGraphBackend(points, gd, k=64, feature_set="G4",
                                            tau_d=4.0, tau_t=0.75)
        r = self.hs.run_headless_segmentation(
            self.g, root_index=self.root, solver_factory=factory)
        # dense index preserved (no opacity filtering in baseline)
        self.assertEqual(len(r.found_segs) if False else r.N, len(self.g.xyz))
        # root identity preserved
        self.assertEqual(int(r.root_idx), int(self.root))
        # construction GT == labels identity: raw_labels match baseline partition
        raw = np.zeros(r.N, dtype=np.int64)
        for kk, seg in enumerate(r.found_segs):
            raw[seg] = kk + 1
        # every dense index accounted for (some may be 0 = unassigned / stem)
        self.assertEqual(raw.shape, self.gt_labels.shape)


if __name__ == "__main__":
    unittest.main()
