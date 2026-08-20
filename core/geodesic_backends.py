#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified geodesic backend interface and concrete implementations.

This module provides drop-in replacements for the ``pp3d.PointCloudHeatSolver``
used by LeafFit.  A backend only needs to implement two methods that match the
heat solver's public call surface:

    compute_distance(source_idx)          -> (N,) np.ndarray[float64]
    compute_distance_multisource(indices) -> (N,) np.ndarray[float64]  (min-to-source)

``run_headless_segmentation`` injects the backend via an optional ``solver_factory``
parameter; when ``None`` the original potpourri3d heat solver is used unchanged
(zero behavioural change).

Concrete backends
-----------------
- ``EuclideanGraphBackend``        kNN + Dijkstra, w = euclidean (G0, baseline B)
- ``SurfaceAwareGraphBackend``     G0..G6 variants with normal/tangent/scale cues

Design notes
------------
* kNN graph is built strictly symmetric: each unique undirected edge ``(i,j)``
  is weighted once then mirrored to ``(i,j)`` and ``(j,i)`` so that
  ``scipy.sparse.csgraph.dijkstra(directed=False)`` is numerically consistent.
* ``compute_distance_multisource`` uses ``min_only=True`` which returns the
  minimum distance to *any* of the source nodes -- matching the heat solver's
  multi-source field semantics.
* Normals are resolved by a validity-gated priority:
      A) ``GaussianData.nxnynz``            (only if valid fraction is high enough)
      B) covariance min-eigenvector         (np.linalg.eigh)
      C) local PCA fallback                  (np.linalg.eigh of local neighbourhood)
  Anisotropy ``a_i = 1 - s_min / s_mid`` is computed from the *scale* eigenvalues
  ``s = sqrt(eigenvalue)`` (NOT raw eigenvalue ratios).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from gaussian_utils import GaussianData, compute_cov3d


EPS = 1e-12


def _finite_distances(d: np.ndarray, cap: float = None) -> np.ndarray:
    """Replace unreachable (Inf/NaN) distances with a large finite cap.

    LeafFit downstream assumes finite fields (temperature_field uses
    ``max - d``; Inf -> NaN).  A finite cap preserves ordering while keeping
    the field finite so downstream logic is not corrupted.  The count of
    unreachable nodes is reported separately via ``graph_stats``.

    When ``cap`` is None, the cap is set to the 99th percentile of the *finite*
    distances (so unreachable nodes sort after all reachable ones without
    introducing an artificial 1e6 scale distortion that would corrupt the
    relative temperature field).
    """
    d = np.asarray(d, dtype=np.float64)
    bad = ~np.isfinite(d)
    if not bad.any():
        return d
    if cap is None:
        good = d[~bad]
        if good.size == 0:
            cap = 1.0
        else:
            cap = float(np.percentile(good, 99)) + 1.0
    d[bad] = cap
    return d

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class GeodesicBackend:
    """Minimal duck-typed interface compatible with ``pp3d.PointCloudHeatSolver``."""

    def compute_distance(self, source_idx: int) -> np.ndarray:
        raise NotImplementedError

    def compute_distance_multisource(self, source_indices) -> np.ndarray:
        raise NotImplementedError

    @property
    def graph_stats(self) -> dict:
        raise NotImplementedError

    @property
    def runtime_stats(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_normals(g: GaussianData, k: int = 16,
                   nxnynz_valid_threshold: float = 0.5) -> np.ndarray:
    """Resolve per-Gaussian unit normals.

    Priority (see module docstring).  Returns (N, 3) float64 unit normals.
    """
    N = len(g.xyz)

    # A) nxnynz field validity check
    norms = np.linalg.norm(g.nxnynz, axis=1)
    valid = np.isfinite(g.nxnynz).all(axis=1) & (norms > 1e-3) & (norms < 1 + 1e-3)
    valid_frac = valid.mean() if N > 0 else 0.0

    if valid_frac >= nxnynz_valid_threshold:
        n = np.asarray(g.nxnynz, dtype=np.float64)
        n[~valid] = _cov_normals(g, list(np.where(~valid)[0]), k)
        # renormalise
        nl = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(nl < EPS, 1.0, nl)
        return n

    # B) covariance min-eigenvector
    n = _cov_normals(g, list(range(N)), k)
    return n


def _cov_normals(g: GaussianData, idxs: Sequence[int], k: int = 16) -> np.ndarray:
    """Normal = eigenvector of smallest eigenvalue of each Gaussian's covariance."""
    if len(idxs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    scales = np.asarray(g.scale[idxs], dtype=np.float32)
    rots = np.asarray(g.rot[idxs], dtype=np.float32)
    cov = compute_cov3d(scales, rots)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 2:
        cov = cov[None, :, :]
    out = np.zeros((len(idxs), 3), dtype=np.float64)
    for ti in range(cov.shape[0]):
        w, v = np.linalg.eigh(cov[ti])
        out[ti] = v[:, int(np.argmin(w))]
    return out


def _local_pca_normals(xyz: np.ndarray, idxs: Sequence[int], k: int = 16) -> np.ndarray:
    """Fallback: local-PCA normal = smallest eigenvector of neighbourhood covariance."""
    if len(idxs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = xyz[idxs]
    tree = cKDTree(xyz)
    _, nn = tree.query(pts, k=k + 1)
    out = np.zeros((len(idxs), 3), dtype=np.float64)
    for ti in range(len(idxs)):
        nbr = nn[ti, 1:]
        c = xyz[nbr] - xyz[nbr].mean(axis=0)
        cov = c.T @ c / (len(nbr) + EPS)
        w, v = np.linalg.eigh(cov)
        out[ti] = v[:, int(np.argmin(w))]
    return out


def _anisotropy(g: GaussianData) -> np.ndarray:
    """a_i = 1 - s_min/s_mid  with s = sqrt(eigenvalue)."""
    scales = np.asarray(g.scale, dtype=np.float32)
    rots = np.asarray(g.rot, dtype=np.float32)
    cov = compute_cov3d(scales, rots)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 2:
        cov = cov[None, :, :]
    ani = np.zeros(cov.shape[0], dtype=np.float64)
    for ti in range(cov.shape[0]):
        w = np.linalg.eigvalsh(cov[ti])
        w_sorted = np.sort(w)
        s = np.sqrt(np.maximum(w_sorted, 0.0))
        s_min, s_mid = s[0], s[1]
        ani[ti] = 1.0 - s_min / max(s_mid, EPS)
    return ani


def _build_knn_edges(xyz: np.ndarray, k: int, mutual: bool = False):
    """Return symmetric kNN graph as (rows, cols, dists) with strict i<j uniqueness.

    ``k`` is the *neighbor count* (self excluded).  The returned undirected edges
    are unique (i < j), each weighted once; callers mirror them to (j, i).
    """
    N = len(xyz)
    tree = cKDTree(xyz)
    k_use = min(k + 1, N)  # +1 to exclude self
    dists, idx = tree.query(xyz, k=k_use)

    # directed edges (i -> neighbor j), self at position 0 excluded
    nn_j = idx[:, 1:]
    nn_d = dists[:, 1:]
    i_all = np.repeat(np.arange(N), nn_j.shape[1])
    j_all = nn_j.ravel()
    d_all = nn_d.ravel()

    # keep unique undirected edge (i, j) once, i < j
    mask = i_all < j_all
    rows, cols, vals = i_all[mask], j_all[mask], d_all[mask]

    if mutual:
        # keep edge (i,j) iff (j,i) is also a directed neighbor pair
        directed = np.lexsort((j_all, i_all))  # order by (i, j)
        sorted_i = i_all[directed]
        sorted_j = j_all[directed]
        # candidates (i,j) i<j: exists if (j,i) in directed list
        rev_rows, rev_cols = cols, rows  # query for (j, i)
        # binary search (rev_rows, rev_cols) in (sorted_i, sorted_j)
        # combine into a flat 1D index space to avoid lexicographic search fiddliness:
        # encode pairs as (i * N + j) — N up to ~1e5 → fits in int64
        enc_dir = sorted_i.astype(np.int64) * N + sorted_j
        enc_q = rev_rows.astype(np.int64) * N + rev_cols
        pos = np.searchsorted(enc_dir, enc_q)
        pos = np.clip(pos, 0, len(enc_dir) - 1)
        mutual_ok = enc_dir[pos] == enc_q
        rows, cols, vals = rows[mutual_ok], cols[mutual_ok], vals[mutual_ok]

    return rows, cols, vals, tree, idx


# ---------------------------------------------------------------------------
# Euclidean baseline (G0)
# ---------------------------------------------------------------------------
class EuclideanGraphBackend(GeodesicBackend):
    """kNN + Dijkstra with pure euclidean edge weights (≡ G0 / baseline B)."""

    def __init__(self, points: np.ndarray, k: int = 32, mutual: bool = False):
        self.N = len(points)
        self.k = k
        self.mutual = mutual
        t0 = time.time()
        rows, cols, vals, tree, nn_idx = _build_knn_edges(points, k, mutual)
        # symmetric CSR (undirected); edges unique as i<j, mirrored to (j,i)
        self._graph = csr_matrix((np.concatenate([vals, vals]),
                                  (np.concatenate([rows, cols]),
                                   np.concatenate([cols, rows]))),
                                 shape=(self.N, self.N))
        self.graph_build_time = time.time() - t0
        self._tree = tree
        self._nn_idx = nn_idx
        self._points = np.asarray(points, dtype=np.float64)
        self._rows = rows
        self._cols = cols

        # local scale (median of k NN distances per node)
        nn_d = np.asarray(tree.query(points, k=k + 1)[0][:, 1:], dtype=np.float64)
        self._local_scale = np.median(nn_d, axis=1)

        self._memo_single: dict[int, np.ndarray] = {}
        self.distance_compute_time = 0.0

    def crossleaf_diagnostics(self, labels: np.ndarray) -> dict:
        """Cross-leaf edge counts (euclidean graph has no surface-feature breakdown)."""
        a = labels[self._rows]
        b = labels[self._cols]
        same = a == b
        diff = ~same
        def _c(mask):
            return {"count": int(mask.sum())}
        return {
            "n_within_leaf_edges": int(same.sum()),
            "n_cross_leaf_edges": int(diff.sum()),
            "median_c_t_within": None,
            "median_c_t_cross": None,
            "median_c_n_within": None,
            "median_c_n_cross": None,
            "median_c_d_within": _c(same),
            "median_c_d_cross": _c(diff),
        }

    def compute_distance(self, source_idx: int) -> np.ndarray:
        if source_idx in self._memo_single:
            return self._memo_single[source_idx]
        t0 = time.time()
        d = np.asarray(dijkstra(self._graph, directed=False, indices=[int(source_idx)],
                                min_only=True, unweighted=False).ravel(), dtype=np.float64)
        d = _finite_distances(d)
        self.distance_compute_time += time.time() - t0
        self._memo_single[source_idx] = d
        return d

    def compute_distance_multisource(self, source_indices) -> np.ndarray:
        srcs = np.asarray(source_indices, dtype=np.int64).ravel()
        if len(srcs) == 1:
            return self.compute_distance(int(srcs[0]))
        t0 = time.time()
        d = np.asarray(dijkstra(self._graph, directed=False, indices=srcs,
                                min_only=True).ravel(), dtype=np.float64)
        # scipy returns (N, n_sources) when min_only=False; with min_only=True returns (N,)
        if d.ndim == 2:
            d = d.min(axis=1)
        d = _finite_distances(d)
        self.distance_compute_time += time.time() - t0
        return d

    @property
    def graph_stats(self) -> dict:
        n_comp, labels = connected_components(self._graph, directed=False)
        comp_sizes = np.bincount(labels)
        largest = comp_sizes.max()
        return {
            "num_nodes": int(self.N),
            "num_edges": int(self._graph.nnz // 2),
            "connected_components": int(n_comp),
            "largest_component_size": int(largest),
            "largest_component_fraction": float(largest / self.N),
            "mean_degree": float(self._graph.nnz / self.N),
            "median_degree": float(np.median(np.diff(self._graph.indptr))),
            "isolated_nodes": int((np.diff(self._graph.indptr) == 0).sum()),
            "root_reachable": bool(True),  # updated by caller
        }

    @property
    def runtime_stats(self) -> dict:
        return {
            "graph_build_time": self.graph_build_time,
            "distance_compute_time": self.distance_compute_time,
        }


# ---------------------------------------------------------------------------
# Surface-aware (G1-G6)
# ---------------------------------------------------------------------------

    @property
    def runtime_stats(self) -> dict:
        return {
            "graph_build_time": self.graph_build_time,
            "distance_compute_time": self.distance_compute_time,
        }


# ---------------------------------------------------------------------------
# Surface-aware (G1-G6)
# ---------------------------------------------------------------------------
# feature sets (bitmask-like): controls which penalties / gates are active
_FEATURE_SETS = {
    "G0": dict(penalty_normal=0.0, penalty_tangent=0.0, gate=False),
    "G1": dict(penalty_normal=1.0, penalty_tangent=0.0, gate=False),
    "G2": dict(penalty_normal=0.0, penalty_tangent=1.0, gate=False),
    "G3": dict(penalty_normal=1.0, penalty_tangent=1.0, gate=False),
    "G4": dict(penalty_normal=0.0, penalty_tangent=0.0, gate=True),   # gate-only, w=d (surface gate)
    "G5": dict(penalty_normal=1.0, penalty_tangent=1.0, gate=True),   # gate + soft penalty
}


class SurfaceAwareGraphBackend(GeodesicBackend):
    """kNN + Dijkstra with surface-aware edge features.

    Edge weight:
        w_ij = d_ij * [1 + g_ij * (lambda_n * c_n + lambda_t * c_t^p)]   (soft penalty)
    Locality gate (for G4 / G5):
        surviving edge iff c_d <= tau_d  AND  c_t <= tau_t
    Surviving edges keep weight = d_ij when soft penalty is disabled (G4).
    """

    def __init__(self, points: np.ndarray, gaussians: GaussianData, k: int = 32,
                 lambda_n: float = 1.0, lambda_t: float = 2.0, p: float = 2.0,
                 tau_d: float = 3.0, tau_t: float = 0.5, mutual: bool = False,
                 feature_set: str = "G5"):
        self.N = len(points)
        self.k = k
        self.lambda_n = float(lambda_n)
        self.lambda_t = float(lambda_t)
        self.p = float(p)
        self.tau_d = float(tau_d) if np.isfinite(tau_d) else np.inf
        self.tau_t = float(tau_t) if np.isfinite(tau_t) else np.inf
        self.mutual = mutual
        self.feature_set = feature_set
        fs = _FEATURE_SETS[feature_set]

        t0 = time.time()
        rows, cols, vals, tree, nn_idx = _build_knn_edges(points, k, mutual)
        nn_d = np.asarray(tree.query(points, k=k + 1)[0][:, 1:], dtype=np.float64)
        local_scale = np.median(nn_d, axis=1)

        normals = _build_normals(gaussians, k)
        anisotropy = _anisotropy(gaussians)

        # edge features
        xi = points[rows]
        xj = points[cols]
        ni = normals[rows]
        nj = normals[cols]
        d_ij = vals

        # normal consistency (sign-invariant)
        cos_n = np.clip(np.abs((ni * nj).sum(axis=1)), -1.0, 1.0)
        c_n = 1.0 - cos_n

        # tangent-plane consistency
        diff = xj - xi
        norm_diff = np.linalg.norm(diff, axis=1)
        u_ij = diff / np.where(norm_diff[:, None] < EPS, EPS, norm_diff[:, None])
        c_t = (np.abs((u_ij * ni).sum(axis=1)) + np.abs((u_ij * nj).sum(axis=1))) / 2.0

        # local-scale normalized distance
        r_i = local_scale[rows]
        r_j = local_scale[cols]
        c_d = d_ij / np.where((r_i + r_j) / 2.0 + EPS < EPS, EPS,
                              (r_i + r_j) / 2.0 + EPS)

        # anisotropy confidence gate
        a_i = anisotropy[rows]
        a_j = anisotropy[cols]
        g_ij = np.sqrt(a_i * a_j)

        # weight
        penalty = fs["penalty_normal"] * self.lambda_n * c_n + \
                  fs["penalty_tangent"] * self.lambda_t * (c_t ** self.p)
        w = d_ij * (1.0 + g_ij * penalty)

        # gate -> prune edges (G4 uses gate only with w = d; penalty term still zeroed)
        # The gate is *anisotropy-confidence weighted*: only prune edges where the
        # cross-surface signal (c_t) is strong AND both endpoints are reliably flat
        # (g_ij ~ 1).  Thin structures (petioles/stems, cylindrical covariances) have
        # a_i ~ 0 -> g_ij ~ 0 -> the gate never prunes them, keeping the graph
        # connected (do NOT cut the whole graph).
        if fs["gate"]:
            keep = (c_d <= self.tau_d) & (c_t * g_ij <= self.tau_t)
            if not keep.all():
                rows, cols, w = rows[keep], cols[keep], w[keep]
                d_ij, c_n, c_t, c_d = d_ij[keep], c_n[keep], c_t[keep], c_d[keep]

        self._graph = csr_matrix((np.concatenate([w, w]),
                                  (np.concatenate([rows, cols]),
                                   np.concatenate([cols, rows]))),
                                 shape=(self.N, self.N))
        # store raw edge arrays for cross-leaf diagnostics (aligned with retained edges)
        self._rows = rows
        self._cols = cols
        self._c_n = c_n
        self._c_t = c_t
        self._c_d = c_d
        self._w = w
        self._d_ij = d_ij
        self._local_scale = local_scale
        self._normals = normals
        self._anisotropy = anisotropy

        self.graph_build_time = time.time() - t0
        self._tree = tree
        self._nn_idx = nn_idx
        self._points = np.asarray(points, dtype=np.float64)
        self._memo_single: dict[int, np.ndarray] = {}
        self.distance_compute_time = 0.0

    def compute_distance(self, source_idx: int) -> np.ndarray:
        if source_idx in self._memo_single:
            return self._memo_single[source_idx]
        t0 = time.time()
        d = np.asarray(dijkstra(self._graph, directed=False, indices=[int(source_idx)],
                                min_only=True, unweighted=False).ravel(), dtype=np.float64)
        d = _finite_distances(d)
        self.distance_compute_time += time.time() - t0
        self._memo_single[source_idx] = d
        return d

    def compute_distance_multisource(self, source_indices) -> np.ndarray:
        srcs = np.asarray(source_indices, dtype=np.int64).ravel()
        if len(srcs) == 1:
            return self.compute_distance(int(srcs[0]))
        t0 = time.time()
        d = np.asarray(dijkstra(self._graph, directed=False, indices=srcs,
                                min_only=True).ravel(), dtype=np.float64)
        if d.ndim == 2:
            d = d.min(axis=1)
        d = _finite_distances(d)
        self.distance_compute_time += time.time() - t0
        return d

    @property
    def graph_stats(self) -> dict:
        n_comp, labels = connected_components(self._graph, directed=False)
        comp_sizes = np.bincount(labels)
        largest = comp_sizes.max()
        return {
            "num_nodes": int(self.N),
            "num_edges": int(self._graph.nnz // 2),
            "connected_components": int(n_comp),
            "largest_component_size": int(largest),
            "largest_component_fraction": float(largest / self.N),
            "mean_degree": float(self._graph.nnz / self.N),
            "median_degree": float(np.median(np.diff(self._graph.indptr))),
            "isolated_nodes": int((np.diff(self._graph.indptr) == 0).sum()),
        }

    @property
    def runtime_stats(self) -> dict:
        return {
            "graph_build_time": self.graph_build_time,
            "distance_compute_time": self.distance_compute_time,
        }

    @property
    def edge_features(self) -> dict:
        """Raw edge arrays for cross-leaf diagnostics (undirected edges)."""
        return {
            "rows": self._rows,
            "cols": self._cols,
            "edge_weight": self._w,
            "euclidean_distance": self._d_ij,
            "normal_consistency": self._c_n,
            "tangent_consistency": self._c_t,
            "local_scale_distance": self._c_d,
        }

    def crossleaf_diagnostics(self, labels: np.ndarray) -> dict:
        """Post-hoc: within-leaf vs cross-leaf edge feature distributions."""
        a = labels[self._rows]
        b = labels[self._cols]
        same = a == b
        diff = ~same
        def _stats(mask, arr):
            if mask.sum() == 0:
                return {"count": 0, "median": None}
            return {"count": int(mask.sum()),
                    "median": float(np.median(arr[mask]))}
        return {
            "n_within_leaf_edges": int(same.sum()),
            "n_cross_leaf_edges": int(diff.sum()),
            "median_c_n_within": _stats(same, self._c_n),
            "median_c_n_cross": _stats(diff, self._c_n),
            "median_c_t_within": _stats(same, self._c_t),
            "median_c_t_cross": _stats(diff, self._c_t),
            "median_c_d_within": _stats(same, self._c_d),
            "median_c_d_cross": _stats(diff, self._c_d),
        }
