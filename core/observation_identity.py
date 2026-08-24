#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 5R — corrected, occlusion-aware real-observation identity features.

This module replaces the invalidated frustum-only "visibility" of Task 5 with a
physically-motivated approximation:

  1. each Gaussian's mean/covariance is transformed into camera space;
  2. it is projected through K (COLMAP +z-in-front convention);
  3. contributors to each PIXEL BUCKET are sorted front-to-back by depth;
  4. front-to-back alpha compositing accumulates transmittance;
  5. a Gaussian's contribution = opacity * T_before at its own projected pixel
     (center-sample of its footprint; footprint radii are also computed and
     stored for downstream ellipse-overlap occlusion tests);
  6. visible[v,i] uses a FROZEN contribution threshold (not image bounds).

Approximation (documented, version-tagged):
  * center-pixel sampling of the footprint (not full rasterization). On dense
    leaf surfaces adjacent Gaussians share buckets, so depth competition at the
    projected pixel captures the dominant occlusion physics while remaining
    O(N log N) per view (the full-rasterization variant was >24 min/plant and
    abandoned; see outputs/task5r/task5_validity_audit.json follow-up).
  * alpha at the projected center = sigmoid-range opacity (already applied at
    load time); transmittance update uses log1p(-alpha) in log space.

Coordinate conventions (single convention throughout this module):
  * uv_pixel : float pixel coordinates in the DOWNSCALED image grid.
  * uv_ndc   : (uv_pixel - c) / (0.5 * wh) in [-1, 1].
  Both stored; consumers must pick ONE and scale thresholds accordingly.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

VISIBILITY_VERSION = "task5r-alpha-v2-blockbucket"

# Frozen constants (NOT tuned against any scientific outcome):
CONTRIBUTION_THRESHOLD = 0.01   # min accumulated alpha contribution to call visible
RGB_MIN_CONTRIBUTION = 0.05     # min contribution for an RGB sample to be valid
MIN_OPACITY = 1e-4              # clamp: Gaussians below this never contribute
ALPHA_MAX = 1.0 - 1e-6          # log-domain clamp
BLOCK_PX = 4                    # footprint-quantized depth-competition block size (downscaled px)


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------
def git_commit(repo_root) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    a = np.ascontiguousarray(np.asarray(a))
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def ordered_name_hash(names: List[str]) -> str:
    """Hash of the ORDERED image-name list (order matters for view alignment)."""
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def viewsig_cache_key(g, rt, K, names, downscale: int,
                      visibility_version: str = VISIBILITY_VERSION,
                      extra: Optional[dict] = None) -> str:
    """Cache key over EVERYTHING that changes the signature.

    Changing xyz/rot/scale/opacity (e.g. any leaf transform), camera poses,
    intrinsics, image list, downscale or algorithm version invalidates it.
    """
    h = hashlib.sha256()
    h.update(sha256_array(g.xyz).encode())
    if getattr(g, "rot", None) is not None:
        h.update(sha256_array(g.rot).encode())
    if getattr(g, "scale", None) is not None:
        h.update(sha256_array(g.scale).encode())
    if getattr(g, "opacity", None) is not None:
        h.update(sha256_array(g.opacity).encode())
    h.update(sha256_array(rt).encode())
    h.update(sha256_array(K).encode())
    h.update(ordered_name_hash(names).encode())
    h.update(f"downscale={downscale}".encode())
    h.update(f"version={visibility_version}".encode())
    if extra:
        for k in sorted(extra):
            h.update(f"{k}={extra[k]}".encode())
    return h.hexdigest()[:32]


def cache_key_for_plant_state(xyz, rot, scale, opacity, rt, K, names,
                              downscale, visibility_version=VISIBILITY_VERSION):
    """Array-level cache key without requiring a full GaussianData object."""

    class _G:  # minimal duck-type
        pass
    g = _G()
    g.xyz, g.rot, g.scale, g.opacity = xyz, rot, scale, opacity
    return viewsig_cache_key(g, rt, K, names, downscale, visibility_version)


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class CorrectedRealViewSignature:
    n_views: int
    n_points: int
    # raw membership vs physical visibility:
    in_frustum: np.ndarray          # (V, N) bool — image-bounds test only
    visible: np.ndarray             # (V, N) bool — contribution >= CONTRIBUTION_THRESHOLD
    max_alpha: np.ndarray           # (V, N) float — per-view own-pixel contribution
    acc_alpha: np.ndarray           # (V, N) float — same as max_alpha under center sampling
    uv_pixel: np.ndarray            # (V, N, 2) float, NaN where not in_frustum
    uv_ndc: np.ndarray              # (V, N, 2) float, NaN where not in_frustum
    depth: np.ndarray               # (V, N) float camera-space z (>0 COLMAP conv), inf elsewhere
    footprint_radius_px: np.ndarray  # (V, N) float 2-sigma projected radius (downscaled px)
    # per-view RGB evidence (kept per-view; NOT collapsed to one global mean)
    rgb_views: np.ndarray           # (V, N, 3) float sampled RGB at OWN pixels
    rgb_valid: np.ndarray           # (V, N) bool — sampled only where own contribution >= RGB_MIN_CONTRIBUTION
    visibility_fraction: np.ndarray  # (N,) fraction of views with visible=True
    meta: dict = field(default_factory=dict)

    def appear_sig(self) -> np.ndarray:
        """Convenience: contribution-weighted mean RGB over valid views.

        Per-view evidence remains available in rgb_views/rgb_valid; downstream
        consumers may use either. Points with no valid view get NaN.
        """
        w = self.max_alpha * self.rgb_valid       # (V, N)
        num = np.einsum("vnc,vn->nc", self.rgb_views.astype(np.float64), w.astype(np.float64))
        den = w.sum(axis=0).astype(np.float64)
        out = np.full((self.n_points, 3), np.nan)
        ok = den > 0
        out[ok] = num[ok] / den[ok, None]
        return out


# ---------------------------------------------------------------------------
# core projection / compositing
# ---------------------------------------------------------------------------
def _project_cov_radius(cov3d_sub: np.ndarray, Rm: np.ndarray,
                        xs, ys, zs, fx, fy) -> np.ndarray:
    """Project 3D covariances to a 2-sigma footprint radius (downscaled px).

    cov2d = J cov_cam J^T + 0.3*I with J the standard perspective Jacobian;
    radius = ELLIPSE_SIGMA * sqrt(median eigenvalue-ish) via sqrt(max(eig)).
    """
    cov_cam = np.einsum("ij,njk,lk->nil", Rm, cov3d_sub, Rm)
    ix = fx / zs; iy = fy / zs
    jx = -fx * xs / zs ** 2; jy = -fy * ys / zs ** 2
    sx = sy = 1.0
    c00 = ix**2 * cov_cam[:, 0, 0] + 2*ix*jx*cov_cam[:, 0, 2] + jx**2 * cov_cam[:, 2, 2] + 0.3 * (sx*sx)
    c01 = ix*iy*cov_cam[:, 0, 1] + ix*jy*cov_cam[:, 0, 2] + jx*iy*cov_cam[:, 1, 2] + jx*jy*cov_cam[:, 2, 2]
    c11 = iy**2 * cov_cam[:, 1, 1] + 2*iy*jy*cov_cam[:, 1, 2] + jy**2 * cov_cam[:, 2, 2] + 0.3 * (sy*sy)
    det = c00 * c11 - c01**2
    tr = c00 + c11
    lmid = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    radius = 2.0 * np.sqrt(lmid)
    return np.clip(radius, 0.7, 64.0)


def build_occlusion_aware_real_view_signature(
    g, obs, decoded_images: Optional[np.ndarray] = None,
    downscale: int = 4,
    contribution_threshold: float = CONTRIBUTION_THRESHOLD,
    rgb_min_contribution: float = RGB_MIN_CONTRIBUTION,
    progress=None,
) -> CorrectedRealViewSignature:
    """Occlusion-aware real view signature (Task5R Phase 2, v2 center-bucket)."""
    from gaussian_utils import compute_cov3d

    xyz = np.asarray(g.xyz, dtype=np.float64)
    N = len(xyz)
    V = obs.n_views

    opacity = np.clip(np.asarray(g.opacity, dtype=np.float64).ravel(), MIN_OPACITY, ALPHA_MAX)
    cov3d = compute_cov3d(np.asarray(g.scale, dtype=np.float32),
                          np.asarray(g.rot, dtype=np.float32)).astype(np.float64)

    in_frustum = np.zeros((V, N), dtype=bool)
    visible = np.zeros((V, N), dtype=bool)
    max_alpha = np.zeros((V, N), dtype=np.float32)
    acc_alpha = np.zeros((V, N), dtype=np.float32)
    uv_pixel = np.full((V, N, 2), np.nan)
    uv_ndc = np.full((V, N, 2), np.nan)
    depth = np.full((V, N), np.inf)
    radius_arr = np.zeros((V, N), dtype=np.float32)
    rgb_views = np.zeros((V, N, 3), dtype=np.float32)
    rgb_valid = np.zeros((V, N), dtype=bool)

    if decoded_images is None:
        from core.real_observation import load_or_cache_decoded_images
        decoded_images = load_or_cache_decoded_images(obs, downscale=downscale)
    dH, dW = decoded_images.shape[1], decoded_images.shape[2]

    for v in range(V):
        if progress is not None:
            progress(v, V)
        Rt = obs.rt[v]; Kv = obs.K[v]
        w_full, h_full = obs.image_wh[v]
        Rm = Rt[:3, :3]; t = Rt[:3, 3]
        cam_xyz = (Rm @ xyz.T).T + t                      # world->cam (+z front)
        z = cam_xyz[:, 2]
        infr = (z > 1e-9)
        px = np.full(N, np.nan); py = np.full(N, np.nan)
        px[infr] = Kv[0, 0] * cam_xyz[infr, 0] / z[infr] + Kv[0, 2]
        py[infr] = Kv[1, 1] * cam_xyz[infr, 1] / z[infr] + Kv[1, 2]
        infr &= (px >= 0) & (px < w_full) & (py >= 0) & (py < h_full)
        in_frustum[v] = infr
        if not infr.any():
            continue
        idxs = np.where(infr)[0]

        sx = dW / float(w_full); sy = dH / float(h_full)
        pxd = px[idxs] * sx
        pyd = py[idxs] * sy
        zd = z[idxs]
        fx = Kv[0, 0] * sx; fy = Kv[1, 1] * sy
        rad = _project_cov_radius(cov3d[idxs], Rm,
                                  cam_xyz[idxs, 0], cam_xyz[idxs, 1], zd, fx, fy)

        # --- footprint-quantized block front-to-back compositing ---
        # Each Gaussian competes for depth within BLOCK_PX x BLOCK_PX blocks of
        # its projected center (footprint quantization): a rear Gaussian whose
        # center lands in the same block as an opaque foreground Gaussian is
        # occluded even if the exact pixels differ by a fraction of a block.
        bxi = np.clip(pxd.astype(np.int64), 0, dW - 1)
        byi = np.clip(pyd.astype(np.int64), 0, dH - 1)
        nbx = (dW + BLOCK_PX - 1) // BLOCK_PX
        nby = (dH + BLOCK_PX - 1) // BLOCK_PX
        bxc = np.clip(bxi // BLOCK_PX, 0, nbx - 1)
        byc = np.clip(byi // BLOCK_PX, 0, nby - 1)
        bucket = byc * nbx + bxc
        order = np.lexsort((zd, bucket))          # bucket asc, then depth asc (front first)
        bof = idxs[order]                         # absolute gaussian ids in compositing order
        bk = bucket[order]
        al = opacity[bof]
        new_group = np.empty(len(bk), dtype=bool)
        new_group[0] = True
        new_group[1:] = bk[1:] != bk[:-1]
        gid = np.cumsum(new_group) - 1
        # exclusive-within-group cumsum of log(1-alpha):
        log1pa = np.log1p(-al)
        cs = np.cumsum(log1pa)
        # start-of-group prefix value:
        starts = np.zeros(gid[-1] + 1, dtype=np.float64)
        first_pos = np.flatnonzero(new_group)
        starts[gid[first_pos]] = cs[first_pos]
        excl = cs - log1pa - starts[gid]
        T_before = np.exp(excl)                   # transmittance BEFORE each gaussian in its bucket
        contrib_order = al * T_before
        # scatter back to absolute indices
        max_alpha[v, bof] = contrib_order.astype(np.float32)
        acc_alpha[v, bof] = contrib_order.astype(np.float32)
        vis_order = contrib_order >= contribution_threshold
        visible[v, bof] = vis_order

        # --- RGB at own pixel where own contribution suffices ---
        img = decoded_images[v]
        sel_rgb = vis_order & (contrib_order >= rgb_min_contribution)
        gx = bxi[order][sel_rgb]
        gy = byi[order][sel_rgb]
        rgb_views[v, bof[sel_rgb]] = img[gy, gx].astype(np.float32)
        rgb_valid[v, bof[sel_rgb]] = True

        uv_pixel[v, idxs, 0] = pxd
        uv_pixel[v, idxs, 1] = pyd
        uv_ndc[v, idxs, 0] = (pxd - dW / 2.0) / (dW / 2.0)
        uv_ndc[v, idxs, 1] = (pyd - dH / 2.0) / (dH / 2.0)
        depth[v, idxs] = zd
        radius_arr[v, idxs] = rad

    vis_frac = visible.sum(axis=0).astype(np.float64) / float(V)
    meta = {
        "visibility_version": VISIBILITY_VERSION,
        "contribution_threshold": contribution_threshold,
        "rgb_min_contribution": rgb_min_contribution,
        "downscale": downscale,
        "sampling": "own-pixel center bucket",
        "compositing": "front-to-back log-space within pixel bucket",
        "min_opacity": MIN_OPACITY,
        "low_pass": "0.3*I in downscaled px^2 (radius only)",
        "footprint_radius_clip": [0.7, 64.0],
    }
    return CorrectedRealViewSignature(
        n_views=V, n_points=N, in_frustum=in_frustum, visible=visible,
        max_alpha=max_alpha, acc_alpha=acc_alpha, uv_pixel=uv_pixel,
        uv_ndc=uv_ndc, depth=depth, footprint_radius_px=radius_arr,
        rgb_views=rgb_views, rgb_valid=rgb_valid,
        visibility_fraction=vis_frac, meta=meta)


def visibility_prior(g) -> np.ndarray:
    """Deterministic prior ordering when subsampling is needed (unused by default)."""
    return np.asarray(g.opacity, dtype=np.float64).ravel()
