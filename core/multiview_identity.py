#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-view (synthetic) identity / consistency features for Gaussian splats.

This module produces **view-synthesized evidence** — NOT recovered observations.
The LeafFit 3DGS assets ship as `data/<plant>.ply` (xyz, rot, scale, opacity, sh, nxnynz)
with **no camera poses** (no COLMAP cameras/images.bin, no c2w/w2c). Therefore all
"multi-view" features here are computed from orbit-synthesized cameras around the
plant centroid. Per the Task 4 claim discipline:

    "View-synthesized visibility/appearance cues provide information beyond
     local pairwise surface geometry."

NOT identity-observed-from-training-cameras. appearance-based-identity language is
ONLY warranted if the appearance ablation shows independent held-out gain, or if
real capture images/masks are introduced (out of scope here).

Design notes
------------
* Pure NumPy / SciPy only — NO torch, NO diff_gaussian_rasterization, NO CUDA.
  (core/gen_template_leaf.py and core/template_transform.py import both at module
  top, so importing them would transitively pull the CUDA rasterizer. This module
  must be importable from geodesic_backends.py during headless segmentation, where
  the rasterizer is forbidden.)
* Deterministic: no RNG. Synthetic orbit is a fixed azimuth ring at fixed elevation.
* Memory: designed for chunked consumption by the backend (~122k gaussians × k=256
  candidate edges can be tens of millions — features must be computable in ≤1M-edge
  chunks). View signatures per-case are cached on disk by run_task4_case.py.
* The SH→RGB and camera math re-implement the formulas from gen_template_leaf.py:17
  (SH eval) and core/template_transform.py:18-82 (projection), but in NumPy.

References for Gaussian-aware visibility:
    3DGS projects a Gaussian's 3D covariance Σ to screen space via the Jacobian of
    perspective projection: Σ' = J W Σ Wᵀ Jᵀ  (W = world->cam), and the 2D ellipse
    from Σ' defines the splat footprint / opacity accumulation. This module uses the
    ellipse major radius as an occlusion proxy (per-pixel min-depth buffer over
    Gaussian footprint boxes), keeping it CPU-only and approximate rather than
    photorealistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

# reuse Gaussian covariance + quaternion math (pure numpy capable)
from gaussian_utils import compute_cov3d, quaternion_wxyz_to_matrix


# --------------------------------------------------------------------------- #
# Camera synthesis (orbit ring)
# --------------------------------------------------------------------------- #
def synthesize_orbit_cameras(centroid: np.ndarray, radius: float,
                             n_views: int = 36,
                             elevation_deg: float = 25.0,
                             up: np.ndarray = None) -> np.ndarray:
    """Synthesize a deterministic ring of cameras orbiting ``centroid``.

    Cameras are placed on a ring at constant elevation around the centroid,
    uniformly sampled in azimuth; each camera looks at the centroid. The ring is
    fixed (no RNG) so the same signature is reproducible.

    Returns (n_views, 4, 4) world->view (Rt) matrices where Rt[:3,:3] is the
    camera rotation (world->cam) and Rt[:3,3] is the translation (t = -R @ cam_pos).
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    up = up / np.linalg.norm(up)
    el = np.radians(elevation_deg)
    az = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)

    cams = np.zeros((n_views, 4, 4), dtype=np.float64)
    for i, a in enumerate(az):
        # camera position on the ring at elevation
        cam = centroid + radius * np.array([
            np.cos(el) * np.cos(a),
            np.cos(el) * np.sin(a),
            np.sin(el),
        ])
        # look-at direction toward centroid
        f = centroid - cam
        f = f / np.linalg.norm(f)
        # camera looks along -z (standard look-at view matrix): z row = -f.
        # With this convention points in front of the camera have cam_z < 0,
        # which is what project_points()/ellipse_visibility() expect.
        zrow = -f
        x = np.cross(up, zrow)
        x = x / np.linalg.norm(x)
        y = np.cross(zrow, x)
        R = np.stack([x, y, zrow], axis=0)  # 3x3, world->cam
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = -R @ cam
        cams[i] = Rt
    return cams


# --------------------------------------------------------------------------- #
# Pure-numpy SH->RGB  (re-implements gen_template_leaf.compute_sh_color_with_direction)
# --------------------------------------------------------------------------- #
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2_0 = 1.0925484305920792
_SH_C2_1 = -1.0925484305920792
_SH_C2_2 = 0.31539156525252005
_SH_C2_3 = -1.0925484305920792
_SH_C2_4 = 0.5462742152960396
_SH_C3_0 = -0.5900435899266435
_SH_C3_1 = 2.890611442640554
_SH_C3_2 = -0.4570457994644658
_SH_C3_3 = 0.3731763325901154
_SH_C3_4 = -0.4570457994644658
_SH_C3_5 = 1.445305721320277
_SH_C3_6 = -0.5900435899266435


def eval_sh_numpy(shs: np.ndarray, positions: np.ndarray,
                  camera_pos: np.ndarray, sh_degree: int = 3) -> np.ndarray:
    """Evaluate spherical-harmonic color for gaussians, pure numpy.

    Mirrors gen_template_leaf.compute_sh_color_with_direction: RGB = clamp(
    sum_l SH_l(direction) * coeff + 0.5, 0, 1). The SH coeffs are interleaved
    [R,G,B per band] in ACN order: bands are (l=0:1, l=1:3, l=2:5, l=3:7) =>
    16 coeffs per channel => 48 total (sh_dim=48).

    Args:
        shs: (N, 48) interleaved SH coeffs.
        positions: (N, 3) world-space gaussian centers.
        camera_pos: (3,) or broadcast.
        sh_degree: max SH degree (0..3).

    Returns: (N, 3) RGB in [0,1].
    """
    shs = np.asarray(shs, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    camera_pos = np.asarray(camera_pos, dtype=np.float32).reshape(1, 3)
    directions = positions - camera_pos
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]

    colors = _SH_C0 * shs[:, :3]  # DC
    nch = shs.shape[1]
    if nch >= 12 and sh_degree >= 1:
        colors += (-_SH_C1 * y[:, None] * shs[:, 3:6]
                   + _SH_C1 * z[:, None] * shs[:, 6:9]
                   - _SH_C1 * x[:, None] * shs[:, 9:12])
        if nch >= 27 and sh_degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            colors += (_SH_C2_0 * xy[:, None] * shs[:, 12:15]
                       + _SH_C2_1 * yz[:, None] * shs[:, 15:18]
                       + _SH_C2_2 * (2.0 * zz - xx - yy)[:, None] * shs[:, 18:21]
                       + _SH_C2_3 * xz[:, None] * shs[:, 21:24]
                       + _SH_C2_4 * (xx - yy)[:, None] * shs[:, 24:27])
            if nch >= 48 and sh_degree >= 3:
                colors += (_SH_C3_0 * (y * (3.0 * xx - yy))[:, None] * shs[:, 27:30]
                           + _SH_C3_1 * (xy * z)[:, None] * shs[:, 30:33]
                           + _SH_C3_2 * (y * (4.0 * zz - xx - yy))[:, None] * shs[:, 33:36]
                           + _SH_C3_3 * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None] * shs[:, 36:39]
                           + _SH_C3_4 * (x * (4.0 * zz - xx - yy))[:, None] * shs[:, 39:42]
                           + _SH_C3_5 * (z * (xx - yy))[:, None] * shs[:, 42:45]
                           + _SH_C3_6 * (x * (xx - 3.0 * yy))[:, None] * shs[:, 45:48])
    colors += 0.5
    return np.clip(colors, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Projection + Gaussian-aware visibility
# --------------------------------------------------------------------------- #
def project_points(xyz: np.ndarray, Rt: np.ndarray,
                   fov_deg: float = 40.0, image_h: int = 1024
                   ) -> dict:
    """Project gaussian centers + (optional) ellipses into a camera view.

    Args:
        xyz: (N,3) world-space centers.
        Rt:  (4,4) world->view matrix (from synthesize_orbit_cameras).
        fov_deg: vertical field of view in degrees.
        image_h: image height in pixels.

    Returns dict with ndc_xy (N,2), depth (N,) camera-space depth (negative = in
    front), in_frustum (N,) bool, pixel (N,2) ints, W,H (image dims), Rc (camera
    center in world coords).
    """
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    cam = (Rt @ xyz_h.T).T  # (N,4) camera space
    depth = cam[:, 2]
    Z = np.maximum(np.abs(depth), 1e-6)
    f = (image_h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    # perspective divide to NDC-like ([-1,1])
    ndc_x = cam[:, 0] * (f / Z) / (image_h / 2.0)
    ndc_y = cam[:, 1] * (f / Z) / (image_h / 2.0)
    W = image_h
    H = image_h  # square
    # pixel coords (0..W-1)
    px = np.clip(((ndc_x * 0.5 + 0.5) * W).astype(np.int64), 0, W - 1)
    py = np.clip(((ndc_y * 0.5 + 0.5) * H).astype(np.int64), 0, H - 1)
    in_frustum = (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0) & (depth < 0)
    # camera center in world coords
    Rc = -Rt[:3, :3].T @ Rt[:3, 3]
    return dict(ndc_xy=np.stack([ndc_x, ndc_y], axis=1),
                depth=depth, in_frustum=in_frustum,
                pixel=np.stack([px, py], axis=1),
                W=W, H=H, Rc=Rc)


def project_ellipse_radii(xyz: np.ndarray, scales: np.ndarray, rots: np.ndarray,
                        Rt: np.ndarray, fov_deg: float = 40.0,
                        image_h: int = 1024) -> np.ndarray:
    """2D ellipse major-axis half-size (pixels) from Gaussian 3D covariance.

    Sigma_3d = R diag(s^2) R^T  (compute_cov3d). Projected 2D covariance via
    the perspective Jacobian J at the camera-space point (standard 3DGS):
      J = [[f/Z, 0, -f*X/Z^2], [0, f/Z, -f*Y/Z^2]]   (f in pixels)
      Sigma' = J Sigma_3d_cam J^T
    ellipse major radius = sqrt(largest eigenvalue of Sigma').

    Returns (N,) float32 major radius in pixels (0 where not in front).
    """
    cov3 = compute_cov3d(scales.astype(np.float32), rots.astype(np.float32))  # (N,3,3)
    xyz_h = np.concatenate([xyz.astype(np.float64),
                            np.ones((len(xyz), 1))], axis=1)
    cam = (Rt @ xyz_h.T).T
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    f = (image_h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    Zsafe = np.maximum(np.abs(Z), 1e-6)
    # view-space covariance: R @ cov3 @ R^T   (cam is world->view, so rotate cov)
    Rwc = Rt[:3, :3]  # this is already world->cam rotation
    cov_cam = Rwc @ cov3 @ Rwc.T  # (N,3,3)
    # Jacobian (N,2,3)
    J = np.zeros((len(xyz), 2, 3))
    J[:, 0, 0] = f / Zsafe
    J[:, 0, 2] = -f * X / (Zsafe ** 2)
    J[:, 1, 1] = f / Zsafe
    J[:, 1, 2] = -f * Y / (Zsafe ** 2)
    # Sigma_2d = J @ cov_cam @ J^T   -> (N,2,2)
    Sigma2d = np.einsum("nij,njk,nlk->nil", J, cov_cam, J)
    # eigenvalues of 2x2
    a = Sigma2d[:, 0, 0]
    b = Sigma2d[:, 0, 1]
    c = Sigma2d[:, 1, 1]
    tr = a + c
    det = np.maximum(a * c - b * b, 0.0)
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
    lam_max = (tr + disc) / 2.0
    return np.sqrt(lam_max).astype(np.float32)


#: Cap for the footprint box radius (pixels) in ellipse_visibility. Larger splats
#: (near-camera giants or far background) get a capped box; their exact occlusion
#: contribution at leaf-scale kNN resolution is irrelevant, and an uncapped box for
#: a whole-image splat would blow up memory (W*H pixels per gaussian).
_MAX_FOOTPRINT_R = 16


def ellipse_visibility(xyz: np.ndarray, Rt: np.ndarray, scales: np.ndarray,
                       rots: np.ndarray, opacity: np.ndarray,
                       radii: np.ndarray, pixel: np.ndarray, W: int, H: int,
                       alpha_thresh: float = 0.05,
                       vis_mode: str = "winner_take_all") -> np.ndarray:
    """Ellipse-aware footprint visibility (vectorized, pure numpy).

    Selects between:

    * ``"winner_take_all"`` (default) — each Gaussian claims the axis-aligned
      box ``[px±ceil(r_i)] x [py±ceil(r_i)]`` (radius capped at
      ``_MAX_FOOTPRINT_R``; r_i = projected 2D ellipse major radius). Claimed
      (gaussian, pixel) pairs are expanded (O(P) rows, P = total footprint
      pixels) and grouped by pixel via a single ``lexsort((-depth, pixel))``.
      The first claim per pixel in front-to-back order is that pixel's front-most
      Gaussian. A Gaussian is *visible* iff it is the front-most owner of >=1 of
      its own footprint pixels *and* its opacity is above ``alpha_thresh``
      (near-fully-transparent splats are neither visible nor occluding).
    * ``"transmittance_k2"`` — same footprint claim + front-to-back ordering, but
      instead of winner-take-all, keep the frontmost ``_TRANSMIT_DEPTH=2``
      claimants per pixel and accumulate transmittance front-to-back: a Gaussian
      is *visible* at a pixel iff ``T_before × alpha_i > alpha_thresh``, where
      ``T_before`` is the product of ``(1 - alpha_k)`` over all claimants strictly
      in front of it. Captures partial occlusion (semi-transparent front lets
      through enough light that a partially-hidden back splat stays visible).
      Use this variant when winner-take-all collapses visibility (see memory note:
      dense splat cloud -> 62% of surface gaussians visible in 0 views under
      winner-take-all, c_vis Jaccard ~0 for both within and cross).

    Returns (N,) int 0/1 visibility for this single view.
    """
    N = len(xyz)
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    if vis_mode == "transmittance_k2":
        return _ellipse_visibility_transmittance(
            xyz, Rt, scales, rots, opacity, radii, pixel, W, H, alpha_thresh)

    cam = (Rt @ np.concatenate([xyz.astype(np.float64),
                                np.ones((N, 1))], axis=1).T).T
    depth = cam[:, 2]
    alpha = np.asarray(opacity, dtype=np.float32).reshape(-1)
    # near-transparent splats claim/occlude nothing
    in_front = (depth < 0) & (alpha >= alpha_thresh)
    visible = np.zeros(N, dtype=np.int64)
    if not in_front.any():
        return visible

    px = np.asarray(pixel, dtype=np.int64)[:, 0]
    py = np.asarray(pixel, dtype=np.int64)[:, 1]
    r = np.maximum(np.minimum(
        np.ceil(np.maximum(np.asarray(radii, dtype=np.float64), 0.0)).astype(np.int64),
        _MAX_FOOTPRINT_R), 1)
    x0 = np.clip(px - r, 0, W - 1)
    x1 = np.clip(px + r, 0, W - 1)
    y0 = np.clip(py - r, 0, H - 1)
    y1 = np.clip(py + r, 0, H - 1)
    dx = np.where(in_front, (x1 - x0 + 1).astype(np.int64), 0)
    dy = np.where(in_front, (y1 - y0 + 1).astype(np.int64), 0)
    tot = dx * dy
    P = int(tot.sum())
    if P == 0:
        return visible
    if P > 100_000_000:
        raise ValueError(f"footprint expansion too large (P={P}); lower _MAX_FOOTPRINT_R")

    # ---- expand to one row per (gaussian, claimed pixel) ----
    gid = np.repeat(np.arange(N, dtype=np.int64), tot)
    cum = np.concatenate([[0], np.cumsum(tot)])
    block_pos = np.arange(P, dtype=np.int64) - np.repeat(cum[:-1], tot)
    dxx = np.repeat(dx, tot)
    row = block_pos // dxx          # 0..dy_i-1
    col = block_pos - row * dxx     # 0..dx_i-1
    flat = (np.repeat(x0, tot) + col) * W + (np.repeat(y0, tot) + row)
    del row, col, dxx, block_pos, cum

    # ---- per-pixel front-most: lexsort by (pixel, depth), first occurrence is owner ----
    sidx = np.lexsort((-depth[gid], flat))  # NEAREST (largest cam z) first per pixel
    flat_s = flat[sidx]
    first = np.concatenate([[0], np.flatnonzero(flat_s[1:] != flat_s[:-1]) + 1])
    pix_to_gid = np.full(W * H, -1, dtype=np.int64)
    pix_to_gid[flat_s[first]] = gid[sidx[first]]
    del flat_s, first, sidx

    # ---- visible iff owns >=1 of its own footprint pixels ----
    wins = (pix_to_gid[flat] == gid)
    visible[gid[wins]] = 1
    return visible


#: How many front-to-back claimants per pixel to consider for transmittance
#: accumulation in ``_ellipse_visibility_transmittance``. 2 = frontmost splat + the
#: next splat behind it (T = 1 - alpha_front). More depth layers cost linearly more
#: memory; 2 captures the dominant occlusion.
_TRANSMIT_DEPTH = 2
#: Transparent splats below this opacity are excluded entirely (claim no pixels).
_TRANSPARENT_THRESH = 0.01


def _ellipse_visibility_transmittance(xyz, Rt, scales, rots, opacity,
                                      radii, pixel, W, H,
                                      alpha_thresh=0.05) -> np.ndarray:
    """Transmittance-K2 variant of ellipse_visibility (see its docstring)."""
    N = len(xyz)
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    cam = (Rt @ np.concatenate([xyz.astype(np.float64),
                                np.ones((N, 1))], axis=1).T).T
    depth = cam[:, 2]
    alpha = np.asarray(opacity, dtype=np.float32).reshape(-1)
    in_front = (depth < 0) & (alpha >= _TRANSPARENT_THRESH)
    visible = np.zeros(N, dtype=np.int64)
    if not in_front.any():
        return visible

    px = np.asarray(pixel, dtype=np.int64)[:, 0]
    py = np.asarray(pixel, dtype=np.int64)[:, 1]
    r = np.maximum(np.minimum(
        np.ceil(np.maximum(np.asarray(radii, dtype=np.float64), 0.0)).astype(np.int64),
        _MAX_FOOTPRINT_R), 1)
    x0 = np.clip(px - r, 0, W - 1)
    x1 = np.clip(px + r, 0, W - 1)
    y0 = np.clip(py - r, 0, H - 1)
    y1 = np.clip(py + r, 0, H - 1)
    dx = np.where(in_front, (x1 - x0 + 1).astype(np.int64), 0)
    dy = np.where(in_front, (y1 - y0 + 1).astype(np.int64), 0)
    tot = dx * dy
    P = int(tot.sum())
    if P == 0:
        return visible
    if P > 100_000_000:
        raise ValueError(f"footprint expansion too large (P={P}); lower _MAX_FOOTPRINT_R")

    # ---- expand to one row per (gaussian, claimed pixel) ----
    gid = np.repeat(np.arange(N, dtype=np.int64), tot)
    cum = np.concatenate([[0], np.cumsum(tot)])
    block_pos = np.arange(P, dtype=np.int64) - np.repeat(cum[:-1], tot)
    dxx = np.repeat(dx, tot)
    row = block_pos // dxx          # 0..dy_i-1
    col = block_pos - row * dxx     # 0..dx_i-1
    flat = (np.repeat(x0, tot) + col) * W + (np.repeat(y0, tot) + row)
    del row, col, dxx, block_pos, cum

    # ---- front-to-back ordering per pixel (nearest first) ----
    sidx = np.lexsort((-depth[gid], flat))   # ascending cam z => nearest first
    flat_s = flat[sidx]
    gid_s = gid[sidx]
    alpha_s = alpha[gid_s]
    group_starts = np.concatenate([[0], np.flatnonzero(flat_s[1:] != flat_s[:-1]) + 1])
    run_lens = np.diff(np.concatenate([group_starts, [P]]))
    rank = np.arange(P, dtype=np.int64) - np.repeat(group_starts, run_lens)
    keep = rank < _TRANSMIT_DEPTH

    flat_k = flat_s[keep]
    gid_k = gid_s[keep]
    alpha_k = alpha_s[keep]
    rank_k = rank[keep]

    # transmittance before this claimant: 0 => 1.0; rank>=1 => 1 - alpha of the
    # frontmost claimant per pixel (rank 0 rows retained).
    T_before = np.ones(flat_k.shape, dtype=np.float32)
    if _TRANSMIT_DEPTH > 1:
        front_alpha = np.zeros(W * H, dtype=np.float32)
        front_rows = rank_k == 0
        front_alpha[flat_k[front_rows]] = alpha_k[front_rows]
        T_before[~front_rows] = 1.0 - front_alpha[flat_k[~front_rows]]

    contribution = T_before * alpha_k
    vis_px = contribution > alpha_thresh
    if vis_px.any():
        visible[gid_k[vis_px]] = 1
    return visible



# --------------------------------------------------------------------------- #
# View signature
# --------------------------------------------------------------------------- #
@dataclass
class ViewSignature:
    """Per-Gaussian multi-view evidence (synthetic orbit cameras).

    Attributes
    ----------
    n_views : int
        Number of orbit cameras.
    visible : (n_views, N) uint8
        Per-view visibility (0/1), ellipse-aware occlusion.
    uv : (n_views, N, 2) float32
        Projected NDC position per view (NaN if not in frustum).
    depth : (n_views, N) float32
        Camera-space depth per view (positive = behind, negative = in front).
    rgb : (n_views, N, 3) float32
        SH-eval view-dependent RGB per view (0 where not visible).
    appear_sig : (N, 3) float32
        Mean RGB over *visible* views (identity appearance signature).
    sh_dc : (N, 3) float32
        SH DC term `sh[:, :3]` (viewpoint-invariant appearance).
    visibility_fraction : (N,) float32
        f_i = |V_i| / n_views  (visibility confidence per gaussian).
    cameras : (n_views, 4, 4) float64
        World->view matrices (provenance).
    """
    n_views: int
    visible: np.ndarray          # (n_views, N) uint8
    uv: np.ndarray               # (n_views, N, 2)
    depth: np.ndarray            # (n_views, N)
    rgb: np.ndarray              # (n_views, N, 3)
    appear_sig: np.ndarray       # (N, 3)
    sh_dc: np.ndarray            # (N, 3)
    visibility_fraction: np.ndarray  # (N,)
    cameras: np.ndarray          # (n_views, 4, 4)

    def to_dict(self) -> dict:
        return dict(n_views=self.n_views,
                    visible=self.visible, uv=self.uv, depth=self.depth,
                    rgb=self.rgb, appear_sig=self.appear_sig,
                    sh_dc=self.sh_dc, visibility_fraction=self.visibility_fraction,
                    cameras=self.cameras)


def build_view_signature(g, n_views: int = 36, radius_frac: float = 3.0,
                         elevation_deg: float = 25.0, fov_deg: float = 40.0,
                         image_h: int = 1024,
                         vis_mode: str = "winner_take_all") -> 'ViewSignature':
    """Build the per-Gaussian view signature for a (already-transformed) GaussianData.

    Cameras are orbit-synthesized around the centroid; for each view we project
    centers, compute ellipse radii, run ellipse-aware visibility, and SH-eval RGB.

    ``vis_mode`` selects the visibility model — ``"winner_take_all"`` (default)
    or ``"transmittance_k2"`` (see ellipse_visibility). Transmittance-k2 is the
    designed remedy when winner-take-all collapses (dense splat cloud -> most
    surface gaussians visible in 0 views, c_vis Jaccard ~0 for both within and
    cross): keep it selectable so the Stage 0 checkpoint can A/B them without a
    code change.
    """
    xyz = np.asarray(g.xyz, dtype=np.float64)
    scales = np.asarray(g.scale, dtype=np.float32)
    rots = np.asarray(g.rot, dtype=np.float32)
    opacity = np.asarray(g.opacity, dtype=np.float32).reshape(-1)
    shs = np.asarray(g.sh, dtype=np.float32)
    N = xyz.shape[0]
    centroid = xyz.mean(axis=0)
    # camera radius = radius_frac * max distance from centroid
    radius = radius_frac * float(np.max(np.linalg.norm(xyz - centroid, axis=1)))
    if radius <= 0:
        radius = 1.0

    cameras = synthesize_orbit_cameras(centroid, radius, n_views, elevation_deg)
    W = image_h
    H = image_h

    visible = np.zeros((n_views, N), dtype=np.uint8)
    uv = np.full((n_views, N, 2), np.nan, dtype=np.float32)
    depth = np.zeros((n_views, N), dtype=np.float32)
    rgb = np.zeros((n_views, N, 3), dtype=np.float32)

    sh_dc = shs[:, :3].astype(np.float32) if shs.shape[1] >= 3 else np.zeros((N, 3), dtype=np.float32)

    for v in range(n_views):
        Rt = cameras[v]
        proj = project_points(xyz, Rt, fov_deg, image_h)
        radii = project_ellipse_radii(xyz, scales, rots, Rt, fov_deg, image_h)
        vis = ellipse_visibility(xyz, Rt, scales, rots, opacity,
                                 radii, proj["pixel"], W, H, vis_mode=vis_mode)
        visible[v] = vis
        uv[v] = proj["ndc_xy"].astype(np.float32)
        depth[v] = proj["depth"].astype(np.float32)
        # SH eval only for in-frustum + visible gaussians
        cam_pos = proj["Rc"]
        in_fr = proj["in_frustum"]
        rgb_v = np.zeros((N, 3), dtype=np.float32)
        if in_fr.any():
            rgb_all = eval_sh_numpy(shs, xyz, cam_pos, sh_degree=3)
            rgb_v[in_fr] = rgb_all[in_fr]
        rgb[v] = rgb_v * vis[:, None]

    # identity appearance signature: mean RGB over visible views
    visible_f = visible.astype(np.float32)  # (n_views, N)
    denom = np.maximum(visible_f.sum(axis=0), 1.0)  # (N,)
    appear_sig = (rgb * visible_f[:, :, None]).sum(axis=0) / denom[:, None]
    vis_frac = visible_f.sum(axis=0) / n_views

    return ViewSignature(
        n_views=n_views,
        visible=visible,
        uv=uv,
        depth=depth,
        rgb=rgb,
        appear_sig=appear_sig.astype(np.float32),
        sh_dc=sh_dc,
        visibility_fraction=vis_frac.astype(np.float32),
        cameras=cameras,
    )


def viewsign_cache_hash(g, n_views: int, radius_frac: float,
                        elevation_deg: float, fov_deg: float,
                        image_h: int, vis_version: str = "v1",
                        vis_mode: str = "winner_take_all") -> str:
    """Stable hash over the inputs that make a view signature case-specific.

    Per Transformed-case cache (M3): the signature depends on the transformed
    geometry (xyz/rot/scale/opacity), so the hash MUST include transformed
    arrays, not just the plant name. ``vis_mode`` is folded into the version key
    so winner-take-all vs transmittance-k2 caches never collide.
    """
    import hashlib
    h = hashlib.sha256()
    for arr in (np.asarray(g.xyz).tobytes(), np.asarray(g.rot).tobytes(),
                np.asarray(g.scale).tobytes(), np.asarray(g.opacity).tobytes(),
                np.asarray(g.sh).tobytes() if hasattr(g.sh, 'shape') and g.sh.ndim == 2 else np.asarray(g.sh).tobytes()):
        h.update(arr)
    h.update(str((n_views, radius_frac, elevation_deg, fov_deg, image_h,
                  f"{vis_version}:{vis_mode}")).encode())
    return h.hexdigest()[:16]
