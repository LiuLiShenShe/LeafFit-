"""Geometry utilities for the controlled overlap benchmark (Task 2).

Pure geometry — zero algorithm changes, zero imports of auto_segment/apex_grouping/petiole_detection.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

# Reuse existing quaternion utilities (wxyz convention).
from gaussian_utils import (
    GaussianData,
    save_gaussian_data_as_ply,
    matrix_to_quaternion_wxyz,
    quat_multiply,
)


# ---------------------------------------------------------------------------
# PCA plane fitting
# ---------------------------------------------------------------------------

def fit_leaf_pca(xyz: NDArray[np.floating]) -> dict:
    """Fit PCA to a leaf point cloud.

    Returns
    -------
    dict with keys: centroid (3,), normal (3,), eigvecs (3,3), eigvals (3,).
    ``normal`` is the eigenvector of the **smallest** eigenvalue (plane normal).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    centroid = xyz.mean(axis=0)
    pca = PCA(n_components=3).fit(xyz)
    # pca.components_ shape (3,3), rows = principal axes, sorted by variance descending
    eigvals = pca.explained_variance_          # (3,)
    eigvecs = pca.components_                  # (3,3) rows
    normal = eigvecs[2]                        # smallest variance = plane normal
    return {"centroid": centroid, "normal": normal, "eigvecs": eigvecs, "eigvals": eigvals}


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(axis: NDArray[np.floating], angle_rad: float) -> NDArray[np.float64]:
    """Rodrigues rotation matrix from unit *axis* and *angle_rad*.

    Returns (3,3) rotation matrix.
    """
    axis = np.asarray(axis, dtype=np.float64)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = axis / ax_norm
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], dtype=np.float64)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.eye(3, dtype=np.float64) + s * K + (1 - c) * (K @ K)


def normal_align_rotation(n_from: NDArray[np.floating],
                           n_to: NDArray[np.floating]) -> NDArray[np.float64]:
    """Rotation matrix that maps unit vector *n_from* → unit vector *n_to*.

    Uses the axis = n_from × n_to, angle = arccos(dot).
    """
    n_from = np.asarray(n_from, dtype=np.float64)
    n_to = np.asarray(n_to, dtype=np.float64)
    n_from = n_from / (np.linalg.norm(n_from) + 1e-12)
    n_to = n_to / (np.linalg.norm(n_to) + 1e-12)
    cross = np.cross(n_from, n_to)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(n_from, n_to)
    if sin_a < 1e-12:
        return np.eye(3, dtype=np.float64) * np.sign(cos_a) if cos_a < 0 else np.eye(3, dtype=np.float64)
    axis = cross / sin_a
    return axis_angle_to_matrix(axis, np.arctan2(sin_a, cos_a))


# ---------------------------------------------------------------------------
# Gaussian rigid transform
# ---------------------------------------------------------------------------

def transform_leaf_gaussians(
    g: GaussianData,
    leaf_indices: NDArray[np.intp],
    pivot_xyz: NDArray[np.floating],
    R: NDArray[np.floating],
    t: NDArray[np.floating],
    *,
    also_rotate_sh: bool = False,
) -> GaussianData:
    """Apply rigid transform ``R(x - pivot) + pivot + t`` to specified leaf indices.

    - xyz:   ``x' = R(x-p) + p + t``
    - rot:   ``q' = q_delta ⊗ q_old``  (left multiply via quat_multiply)
    - nxnynz: ``n' = R @ n``
    - sh:    optionally ``sh_rotate(sh, R)`` (segmentation does not read SH)
    - scale, opacity, filter_3Ds: unchanged
    - Non-leaf indices: copied verbatim.
    - N and Gaussian index ordering are preserved.

    Returns a **new** GaussianData (input is not modified).
    """
    leaf_indices = np.asarray(leaf_indices, dtype=np.intp)
    pivot_xyz = np.asarray(pivot_xyz, dtype=np.float64).ravel()
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    N = len(g)
    mask = np.zeros(N, dtype=bool)
    mask[leaf_indices] = True

    # -- xyz --
    xyz_new = g.xyz.copy()
    leaf_xyz = g.xyz[leaf_indices].astype(np.float64)
    xyz_new[leaf_indices] = (R @ (leaf_xyz - pivot_xyz).T).T + pivot_xyz + t

    # -- rot (wxyz): q' = q_delta ⊗ q_old --
    q_delta = matrix_to_quaternion_wxyz(R).astype(np.float32)  # (4,)
    rot_new = g.rot.copy()
    rot_new[leaf_indices] = quat_multiply(rot_new[leaf_indices], q_delta)

    # -- nxnynz: n' = R @ n --
    nxnynz_new = g.nxnynz.copy()
    leaf_n = g.nxnynz[leaf_indices].astype(np.float64)
    nxnynz_new[leaf_indices] = (R @ leaf_n.T).T

    # -- sh (optional) --
    if also_rotate_sh:
        from gaussian_utils import sh_rotate
        sh_new = g.sh.copy()
        sh_new[leaf_indices] = sh_rotate(g.sh[leaf_indices], R)
    else:
        sh_new = g.sh

    # -- unchanged fields --
    return GaussianData(
        xyz=xyz_new.astype(np.float32),
        rot=rot_new.astype(np.float32),
        scale=g.scale.copy(),
        opacity=g.opacity.copy(),
        sh=sh_new.copy(),
        nxnynz=nxnynz_new.astype(np.float32),
        filter_3Ds=g.filter_3Ds.copy(),
    )


# ---------------------------------------------------------------------------
# Projected overlap fraction (2D Jaccard)
# ---------------------------------------------------------------------------

def _build_projection_frame(
    normal_a: NDArray[np.floating],
    normal_b: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build orthonormal (u, v, n_avg) frame from two normals (sign-aligned)."""
    n_a = normal_a.astype(np.float64)
    n_b = normal_b.astype(np.float64)
    if np.dot(n_a, n_b) < 0:
        n_b = -n_b                                  # unify sign (CRITICAL)
    n_avg = n_a + n_b
    n_avg_norm = np.linalg.norm(n_avg)
    if n_avg_norm < 1e-12:
        n_avg = n_a                                 # degenerate: both normals cancel
        n_avg_norm = 1.0
    n_avg = n_avg / n_avg_norm
    # pick an arbitrary reference not parallel to n_avg
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_avg[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n_avg) * n_avg
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n_avg, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v, n_avg


def compute_projected_overlap_fraction(
    xyz_a: NDArray[np.floating],
    xyz_b: NDArray[np.floating],
    normal_a: NDArray[np.floating] | None = None,
    normal_b: NDArray[np.floating] | None = None,
    voxel_size: float | None = None,
) -> dict:
    """2D projected overlap (Jaccard) between two leaf point clouds.

    If normals are not provided, PCA is fitted internally.
    """
    xyz_a = np.asarray(xyz_a, dtype=np.float64)
    xyz_b = np.asarray(xyz_b, dtype=np.float64)

    if normal_a is None or normal_b is None:
        pca = PCA(n_components=3).fit(np.vstack([xyz_a, xyz_b]))
        if normal_a is None:
            normal_a = pca.components_[2]
        if normal_b is None:
            normal_b = pca.components_[2]

    u, v, _ = _build_projection_frame(normal_a, normal_b)
    uv = np.stack([u, v], axis=1)  # (3,2)

    proj_a = xyz_a @ uv  # (Na, 2)
    proj_b = xyz_b @ uv  # (Nb, 2)

    # voxel size: median NN spacing / 2
    if voxel_size is None:
        all_pts = np.vstack([proj_a, proj_b])
        nn = cKDTree(all_pts)
        dists, _ = nn.query(all_pts, k=min(7, len(all_pts)))
        voxel_size = float(np.median(dists[:, -1])) / 2.0 + 1e-12

    cells_a = set(map(tuple, (proj_a / voxel_size).astype(int)))
    cells_b = set(map(tuple, (proj_b / voxel_size).astype(int)))

    # fallback if too few cells
    if len(cells_a) < 10 or len(cells_b) < 10:
        voxel_size = voxel_size / 5.0 + 1e-12
        cells_a = set(map(tuple, (proj_a / voxel_size).astype(int)))
        cells_b = set(map(tuple, (proj_b / voxel_size).astype(int)))

    inter = len(cells_a & cells_b)
    union = len(cells_a | cells_b)
    fraction = inter / max(union, 1)

    return {
        "overlap_fraction": fraction,
        "intersection_cells": inter,
        "union_cells": union,
        "voxel_size": float(voxel_size),
        "cells_a": len(cells_a),
        "cells_b": len(cells_b),
    }


# ---------------------------------------------------------------------------
# Cross-leaf proximity (Horizontal helper)
# ---------------------------------------------------------------------------

def compute_contact_fraction(
    xyz_a: NDArray[np.floating],
    xyz_b: NDArray[np.floating],
    spacing: float | None = None,
    c: float = 1.0,
) -> dict:
    """Contact fraction and cross-leaf distance statistics.

    *spacing*: median kNN distance within one leaf.  If None, computed from xyz_a.
    *c*: contact threshold = c × spacing.
    """
    xyz_a = np.asarray(xyz_a, dtype=np.float64)
    xyz_b = np.asarray(xyz_b, dtype=np.float64)

    tree_b = cKDTree(xyz_b)
    dist_a_to_b, _ = tree_b.query(xyz_a, k=1)

    if spacing is None:
        tree_a = cKDTree(xyz_a)
        k_nn = min(7, len(xyz_a))
        d_a_nn, _ = tree_a.query(xyz_a, k=k_nn)
        spacing = float(np.median(d_a_nn[:, -1])) + 1e-12

    contact = float(np.mean(dist_a_to_b < c * spacing))
    return {
        "contact_fraction": contact,
        "min_cross_leaf_distance": float(dist_a_to_b.min()),
        "min_cross_leaf_distance_ratio": float(dist_a_to_b.min() / spacing),
        "p05_cross_leaf_distance_ratio": float(np.percentile(dist_a_to_b, 5) / spacing),
        "median_cross_leaf_distance_ratio": float(np.median(dist_a_to_b) / spacing),
        "median_nn_spacing": spacing,
    }


# ---------------------------------------------------------------------------
# Vertical gap measurement
# ---------------------------------------------------------------------------

def compute_apex_gap(
    upper_apex_xyz: NDArray[np.floating],
    lower_xyz: NDArray[np.floating],
    lower_normal: NDArray[np.floating],
    upper_xyz: NDArray[np.floating] | None = None,
    local_k: int = 50,
) -> dict:
    """Apex-to-lower-surface gap (euclidean + normal-direction).

    Primary metric: apex_euclidean_gap_ratio.

    If *upper_xyz* is given, also computes local and whole median gaps.
    """
    upper_apex = np.asarray(upper_apex_xyz, dtype=np.float64).ravel()
    lower_xyz = np.asarray(lower_xyz, dtype=np.float64)
    lower_normal = np.asarray(lower_normal, dtype=np.float64)

    tree_lower = cKDTree(lower_xyz)

    # --- apex gap ---
    d_apex, idx_apex = tree_lower.query(upper_apex, k=1)
    apex_normal_gap = abs(np.dot(upper_apex - lower_xyz[idx_apex], lower_normal))

    out = {
        "apex_euclidean_gap": float(d_apex),
        "apex_normal_gap": float(apex_normal_gap),
    }

    if upper_xyz is not None and len(upper_xyz) > 0:
        upper_xyz = np.asarray(upper_xyz, dtype=np.float64)
        dists_upper, _ = tree_lower.query(upper_xyz, k=1)
        out["whole_leaf_median_euclidean_gap"] = float(np.median(dists_upper))

        # local gap: upper apex's nearest upper neighbours
        tree_upper = cKDTree(upper_xyz)
        k = min(local_k, len(upper_xyz) - 1)
        local_indices = tree_upper.query(upper_apex, k=k + 1)[1]  # skip self
        local_dists = dists_upper[local_indices]
        out["local_median_euclidean_gap"] = float(np.median(local_dists))
        out["local_apex_region_size"] = k

    return out


def compute_vertical_gap_and_spacing(
    upper_xyz: NDArray[np.floating],
    lower_xyz: NDArray[np.floating],
    lower_normal: NDArray[np.floating],
    k_nn: int = 6,
) -> dict:
    """Full vertical gap statistics: Euclidean + normal-direction gaps, spacing."""
    upper_xyz = np.asarray(upper_xyz, dtype=np.float64)
    lower_xyz = np.asarray(lower_xyz, dtype=np.float64)
    lower_normal = np.asarray(lower_normal, dtype=np.float64)

    tree_lower = cKDTree(lower_xyz)
    dists, idxs = tree_lower.query(upper_xyz, k=1)
    normals_at = lower_xyz[idxs]
    normal_dists = np.abs(np.sum((upper_xyz - normals_at) * lower_normal, axis=1))

    # median NN spacing within lower leaf
    k = min(k_nn, len(lower_xyz) - 1)
    tree_lower_k = cKDTree(lower_xyz)
    d_nn, _ = tree_lower_k.query(lower_xyz, k=k + 1)
    spacing = float(np.median(d_nn[:, -1]))

    return {
        "median_euclidean_gap": float(np.median(dists)),
        "median_normal_gap": float(np.median(normal_dists)),
        "min_euclidean_gap": float(dists.min()),
        "median_nn_spacing": spacing,
        "normal_alignment_cos": float(np.abs(np.dot(
            _fit_normal(upper_xyz), lower_normal
        ))) if len(upper_xyz) > 10 else 0.0,
    }


def _fit_normal(xyz: NDArray[np.floating]) -> NDArray[np.float64]:
    """Quick PCA normal for a point cloud."""
    if len(xyz) < 10:
        return np.zeros(3, dtype=np.float64)
    pca = PCA(n_components=3).fit(xyz)
    return pca.components_[2]
