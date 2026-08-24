#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 5R-v3 — occlusion-aware real-observation identity features.

v3 (task5r-alpha-v3-ellipseblock) replaces the v2 center-bucket compositing
with APPROXIMATE ELLIPSE-BLOCK COMPOSITING:

  1. each Gaussian's full 3D covariance is transformed into camera space and
     projected to a 2x2 covariance through K (COLMAP +z-in-front);
  2. lambda_max = tr/2 + sqrt(tr^2/4 - det); footprint radius = ELLIPSE_SIGMA
     * sqrt(lambda_max) in downscaled pixels;
  3. pixel BLOCKS whose centers fall within the 2-sigma ellipse (Mahalanobis
     d^2 <= ELLIPSE_SIGMA^2) are enumerated per Gaussian;
  4. alpha_eff = opacity * exp(-0.5 * d^2) — the Gaussian footprint weight —
     so the footprint genuinely participates in occlusion;
  5. each block's contributors are sorted front-to-back by depth and composed
     with CORRECT EXCLUSIVE transmittance: T_before of the first contributor
     is exactly 1; two alpha=0.9 contributors yield [0.9, 0.09]; three yield
     [0.9, 0.09, 0.009]; different blocks never interact;
  6. a Gaussian's max_alpha = max contribution over its blocks; acc_alpha =
     sum over its blocks (v2 conflated these; v3 separates them);
  7. RGB is sampled at the representative pixel of the block where the
     Gaussian's own contribution peaks; valid only if that peak contribution
     >= RGB_MIN_CONTRIBUTION.

This is an APPROXIMATION, not full 2D Gaussian rasterization: coverage is
evaluated at block centers only (block_px quantization), not per-pixel. It is
deterministic, unit-tested against exact analytic compositing sequences, and
chunked/vectorized (no unbounded dense matrices).

Coordinate conventions (single convention throughout this module):
  * uv_pixel : float pixel coordinates in the DOWNSCALED image grid.
  * uv_ndc   : (uv_pixel - c) / (0.5 * wh) in [-1, 1].
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

VISIBILITY_VERSION = "task5r-alpha-v3-ellipseblock"

# Frozen constants (NOT tuned against any scientific outcome):
CONTRIBUTION_THRESHOLD = 0.01   # min max-block alpha contribution to call visible
RGB_MIN_CONTRIBUTION = 0.05     # min peak contribution for an RGB sample to be valid
MIN_OPACITY = 1e-4              # clamp: Gaussians below this never contribute
ALPHA_MAX = 1.0 - 1e-6          # log-domain clamp
ALPHA_FLOOR = 1e-6              # drop alpha_eff below this before log-domain work
ELLIPSE_SIGMA = 2.0             # FROZEN: Mahalanobis acceptance d^2 <= sigma^2
BLOCK_PX = 4                    # ellipse-quantized compositing block size (downscaled px)
MAX_RADIUS_PX = 64.0            # hard clip bounding worst-case blocks per Gaussian
PAIR_CHUNK_GAUSS = 16384        # gaussian chunk size for (gaussian, block) pair generation


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


def git_tree_dirty(repo_root) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return True   # unknown tree state must NOT be treated as clean


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


def frozen_constants() -> dict:
    return {
        "contribution_threshold": CONTRIBUTION_THRESHOLD,
        "rgb_min_contribution": RGB_MIN_CONTRIBUTION,
        "min_opacity": MIN_OPACITY,
        "alpha_max": ALPHA_MAX,
        "alpha_floor": ALPHA_FLOOR,
        "ellipse_sigma": ELLIPSE_SIGMA,
        "block_px": BLOCK_PX,
        "max_radius_px": MAX_RADIUS_PX,
    }


def algorithm_extra() -> dict:
    """Extra terms folded into every v3 cache key (must change with any constant)."""
    fc = frozen_constants()
    return {
        "ellipse_sigma": fc["ellipse_sigma"],
        "block_px": fc["block_px"],
        "max_radius_px": fc["max_radius_px"],
        "contribution_threshold": fc["contribution_threshold"],
        "rgb_min_contribution": fc["rgb_min_contribution"],
        "compositing": "exclusive_log_front_to_back",
        "footprint_weight": "exp(-0.5*mahalanobis_d2)",
    }


def viewsig_cache_key(g, rt, K, names, downscale: int,
                      visibility_version: str = VISIBILITY_VERSION,
                      extra: Optional[dict] = None) -> str:
    """Cache key over EVERYTHING that changes the signature.

    Changing xyz/rot/scale/opacity (e.g. any leaf transform), camera poses,
    intrinsics, image list, downscale, any frozen algorithm constant or the
    algorithm version invalidates it.
    """
    merged = dict(algorithm_extra())
    if extra:
        merged.update(extra)
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
    for k in sorted(merged):
        h.update(f"{k}={merged[k]}".encode())
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
    visible: np.ndarray             # (V, N) bool — max_alpha >= CONTRIBUTION_THRESHOLD
    max_alpha: np.ndarray           # (V, N) float — max contribution over own blocks
    acc_alpha: np.ndarray           # (V, N) float — SUM of contributions over own blocks
    uv_pixel: np.ndarray            # (V, N, 2) float, NaN where not in_frustum
    uv_ndc: np.ndarray              # (V, N, 2) float, NaN where not in_frustum
    depth: np.ndarray               # (V, N) float camera-space z (>0 COLMAP conv), inf elsewhere
    footprint_radius_px: np.ndarray  # (V, N) float ELLIPSE_SIGMA-sigma projected radius (downscaled px)
    # per-view RGB evidence (kept per-view; NOT collapsed to one global mean)
    rgb_views: np.ndarray           # (V, N, 3) float RGB at peak-contribution block pixel
    rgb_valid: np.ndarray           # (V, N) bool — peak contribution >= RGB_MIN_CONTRIBUTION
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
# core projection
# ---------------------------------------------------------------------------
def _project_cov2d(cov3d_sub: np.ndarray, Rm: np.ndarray,
                   xs, ys, zs, fx, fy):
    """Project 3D covariances to FULL 2x2 covariances in downscaled px^2.

    cov2d = J cov_cam J^T + 0.3*I (low-pass), J the standard perspective
    Jacobian. Returns (c00, c01, c11) float64 arrays of shape (M,).
    """
    cov_cam = np.einsum("ij,njk,lk->nil", Rm, cov3d_sub, Rm)
    ix = fx / zs; iy = fy / zs
    jx = -fx * xs / zs ** 2; jy = -fy * ys / zs ** 2
    c00 = ix**2 * cov_cam[:, 0, 0] + 2*ix*jx*cov_cam[:, 0, 2] + jx**2 * cov_cam[:, 2, 2] + 0.3
    c01 = ix*iy*cov_cam[:, 0, 1] + ix*jy*cov_cam[:, 0, 2] + jx*iy*cov_cam[:, 1, 2] + jx*jy*cov_cam[:, 2, 2]
    c11 = iy**2 * cov_cam[:, 1, 1] + 2*iy*jy*cov_cam[:, 1, 2] + jy**2 * cov_cam[:, 2, 2] + 0.3
    return c00, c01, c11


def cov2d_lambda_max(c00, c01, c11):
    """Largest eigenvalue of [[c00,c01],[c01,c11]]:
    lambda_max = tr/2 + sqrt(max(tr^2/4 - det, 0)). Exact, unclipped."""
    tr = c00 + c11
    det = c00 * c11 - c01 ** 2
    return tr / 2.0 + np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))


def cov2d_radius_px(c00, c01, c11, sigma: float = ELLIPSE_SIGMA):
    """sigma-sigma footprint radius in downscaled px, clipped only against
    pathological values [0.7, MAX_RADIUS_PX]. Unit tests assert the UNCLIPPED
    formula directly (clip must never mask math errors)."""
    lam = cov2d_lambda_max(c00, c01, c11)
    return np.clip(sigma * np.sqrt(np.maximum(lam, 0.0)), 0.7, MAX_RADIUS_PX)


def _project_cov_radius(cov3d_sub: np.ndarray, Rm: np.ndarray,
                        xs, ys, zs, fx, fy,
                        sigma: float = ELLIPSE_SIGMA) -> np.ndarray:
    """Backward-compatible wrapper returning the sigma-sigma radius."""
    c00, c01, c11 = _project_cov2d(cov3d_sub, Rm, xs, ys, zs, fx, fy)
    return cov2d_radius_px(c00, c01, c11, sigma)


def exclusive_transmittance(alpha_sorted: np.ndarray, group_ids: np.ndarray):
    """Exact grouped front-to-back transmittance (vectorized, log domain).

    For contributors already sorted front-to-back WITHIN each group:
      T_before(first member of a group) == 1 exactly;
      contribution_i = alpha_i * T_before_i.
    Two alpha=0.9 in one group -> contributions [0.9, 0.09];
    three -> [0.9, 0.09, 0.009]. Groups never interact.
    Returns (T_before, contribution).
    """
    al = np.asarray(alpha_sorted, dtype=np.float64)
    gid = np.asarray(group_ids)
    log1pa = np.log1p(-al)
    cs = np.cumsum(log1pa)
    new_group = np.empty(len(gid), dtype=bool)
    new_group[0] = True
    new_group[1:] = gid[1:] != gid[:-1]
    first_pos = np.flatnonzero(new_group)
    starts = np.zeros(int(gid[-1]) + 1, dtype=np.float64)
    # EXCLUSIVE prefix at group start = inclusive cumsum BEFORE the first
    # member's own term:
    starts[gid[first_pos]] = cs[first_pos] - log1pa[first_pos]
    excl = cs - log1pa - starts[gid]
    T_before = np.exp(excl)
    return T_before, al * T_before


def _ellipse_block_pairs(pxd, pyd, zd, opac, c00, c01, c11,
                         nbx, nby, block_px: int = BLOCK_PX,
                         chunk: int = PAIR_CHUNK_GAUSS):
    """Enumerate (gaussian, block) pairs whose block centers fall inside the
    ELLIPSE_SIGMA ellipse (Mahalanobis d^2 <= sigma^2).

    Chunked/vectorized; returns compacted (gi, block_id, d2) arrays where gi
    indexes into the frustum-local arrays passed in.
    """
    M = len(pxd)
    sig2 = ELLIPSE_SIGMA ** 2
    half = (block_px - 1) / 2.0
    ext_x = ELLIPSE_SIGMA * np.sqrt(c00)          # marginal bound on |dx|
    ext_y = ELLIPSE_SIGMA * np.sqrt(c11)
    bx_lo = np.floor((pxd - ext_x) / block_px).astype(np.int64)
    bx_hi = np.floor((pxd + ext_x) / block_px).astype(np.int64)
    by_lo = np.floor((pyd - ext_y) / block_px).astype(np.int64)
    by_hi = np.floor((pyd + ext_y) / block_px).astype(np.int64)
    np.clip(bx_lo, 0, nbx - 1, out=bx_lo); np.clip(bx_hi, 0, nbx - 1, out=bx_hi)
    np.clip(by_lo, 0, nby - 1, out=by_lo); np.clip(by_hi, 0, nby - 1, out=by_hi)
    nbx_i = bx_hi - bx_lo + 1
    nby_i = by_hi - by_lo + 1

    parts_gi, parts_bid, parts_d2 = [], [], []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        nb_i = (nbx_i[s:e] * nby_i[s:e]).astype(np.int64)
        P = int(nb_i.sum())
        if P <= 0:
            continue
        gi = np.repeat(np.arange(s, e, dtype=np.int64), nb_i)
        offs = np.arange(P, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(nb_i)[:-1]]), nb_i)
        kx = offs % np.repeat(nbx_i, nb_i)
        ky = offs // np.repeat(nbx_i, nb_i)
        bxp = np.repeat(bx_lo[s:e], nb_i) + kx
        byp = np.repeat(by_lo[s:e], nb_i) + ky
        bid = byp * nbx + bxp
        cx = bxp * block_px + half
        cy = byp * block_px + half
        dx = cx - pxd[gi]
        dy = cy - pyd[gi]
        det = np.maximum(c00[gi] * c11[gi] - c01[gi] ** 2, 1e-12)
        d2 = (c11[gi] * dx * dx - 2.0 * c01[gi] * dx * dy + c00[gi] * dy * dy) / det
        keep = d2 <= sig2
        parts_gi.append(gi[keep])
        parts_bid.append(bid[keep])
        parts_d2.append(d2[keep])
    if not parts_gi:
        z6 = np.zeros(0, dtype=np.int64); z0 = np.zeros(0, dtype=np.float64)
        return z6, z6.astype(np.int32), z0
    return (np.concatenate(parts_gi),
            np.concatenate(parts_bid).astype(np.int64),
            np.concatenate(parts_d2))


def build_occlusion_aware_real_view_signature(
    g, obs, decoded_images=None,
    downscale: int = 4,
    contribution_threshold: float = CONTRIBUTION_THRESHOLD,
    rgb_min_contribution: float = RGB_MIN_CONTRIBUTION,
    source_commit: str = "",
    source_tree_dirty: bool = False,
    progress=None,
) -> CorrectedRealViewSignature:
    """Approximate ellipse-block compositing view signature (Task5R-v3).

    NOT full 2D Gaussian rasterization: coverage is evaluated at block centers
    under a Mahalanobis gate; see module docstring for the approximation and
    its unit tests.
    """
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
    nbx = (dW + BLOCK_PX - 1) // BLOCK_PX
    nby = (dH + BLOCK_PX - 1) // BLOCK_PX

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

        c00, c01, c11 = _project_cov2d(cov3d[idxs], Rm,
                                       cam_xyz[idxs, 0], cam_xyz[idxs, 1],
                                       zd, fx, fy)
        rad = cov2d_radius_px(c00, c01, c11)

        # --- approximate ellipse-block pairs ---
        gi, bid, d2 = _ellipse_block_pairs(pxd, pyd, zd,
                                           opacity[idxs], c00, c01, c11,
                                           nbx, nby, BLOCK_PX)
        if len(gi) == 0:
            continue
        # footprint weight: opacity attenuated by Mahalanobis distance
        a_eff = opacity[idxs[gi]] * np.exp(-0.5 * d2)
        keep_a = a_eff >= ALPHA_FLOOR
        gi, bid, a_eff = gi[keep_a], bid[keep_a], a_eff[keep_a]
        a_eff = np.minimum(a_eff, ALPHA_MAX)
        zp = zd[gi]

        # --- exclusive front-to-back compositing per block ---
        order = np.lexsort((zp, bid))         # block asc, depth asc (front first)
        bk_o = bid[order]
        gi_o = gi[order]
        T_before, contrib = exclusive_transmittance(a_eff[order], bk_o)
        contrib = np.clip(contrib, 0.0, 1.0)  # guard: contributions must be in [0,1]

        # scatter back to absolute gaussian ids (multiple rows per gaussian now)
        loc = idxs[gi_o]
        np.maximum.at(max_alpha[v], loc, contrib.astype(np.float32))
        acc_alpha[v] += np.bincount(loc, weights=contrib,
                                    minlength=N).astype(np.float32)
        vis_rows = contrib >= contribution_threshold
        if vis_rows.any():
            visible[v, np.unique(loc[vis_rows])] = True

        # --- RGB at peak-contribution block ---
        img = decoded_images[v]
        best_pos = {}
        # argmax of contrib within each gaussian group of `loc` (loc rows are
        # scattered across groups after sorting by block, so reduce via lexsort)
        ord2 = np.lexsort((-contrib, loc))
        loc_s = loc[ord2]
        last = np.append(np.diff(loc_s) != 0, True)     # last row of each run
        peak_rows = ord2[last]
        pk_gauss = loc[peak_rows]
        pk_contrib = contrib[peak_rows]
        sel_rgb = pk_contrib >= rgb_min_contribution
        if sel_rgb.any():
            pbid = bid[peak_rows[sel_rgb]]
            gx = ((pbid % nbx) * BLOCK_PX + BLOCK_PX // 2).astype(np.int64)
            gy = ((pbid // nbx) * BLOCK_PX + BLOCK_PX // 2).astype(np.int64)
            gx = np.clip(gx, 0, dW - 1); gy = np.clip(gy, 0, dH - 1)
            pg = pk_gauss[sel_rgb]
            rgb_views[v, pg] = img[gy, gx].astype(np.float32)
            rgb_valid[v, pg] = True

        uv_pixel[v, idxs, 0] = pxd
        uv_pixel[v, idxs, 1] = pyd
        uv_ndc[v, idxs, 0] = (pxd - dW / 2.0) / (dW / 2.0)
        uv_ndc[v, idxs, 1] = (pyd - dH / 2.0) / (dH / 2.0)
        depth[v, idxs] = zd
        radius_arr[v, idxs] = rad

    vis_frac = visible.sum(axis=0).astype(np.float64) / float(V)
    meta = {
        "visibility_version": VISIBILITY_VERSION,
        "algorithm_name": "approximate ellipse-block compositing",
        "not_full_rasterization": True,
        "contribution_threshold": contribution_threshold,
        "rgb_min_contribution": rgb_min_contribution,
        "ellipse_sigma": ELLIPSE_SIGMA,
        "block_px": BLOCK_PX,
        "downscale": downscale,
        "source_commit": source_commit,
        "source_tree_dirty": source_tree_dirty,
        "frozen_constants": frozen_constants(),
        "compositing": "exclusive log-space front-to-back within ellipse-accepted blocks",
        "footprint_weight": "alpha_eff = opacity * exp(-0.5 * mahalanobis_d2)",
        "low_pass": "0.3*I in downscaled px^2",
        "footprint_radius_clip": [0.7, MAX_RADIUS_PX],
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
