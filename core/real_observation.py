#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-observation ingest for Task 5 — build GaussianData + ViewSignature from COLMAP.

This is the observation-grounded identity drop-site. Unlike Task 4 (which used
synthetic orbit cameras), this module loads REAL captures:
  * geometry  : COLMAP points3D.bin (already in the same world frame as the images)
  * poses     : COLMAP images.bin  (world->cam Rt, calibrated)
  * intrinsics: COLMAP cameras.bin (PINHOLE K)
  * RGB       : the actual captured images (datasets/04-COLMAP/<plant>/images/)

The ViewSignature's appearance cue (c_app) uses REAL pixel color sampled at each
point's projected location across the views that actually see it — not synthetic
SH. This is the genuine observation-grounded evidence the spec asks for.

Conventions: all camera math follows colmap_io (COLMAP +z-in-front). GaussianData
layout follows core.gaussian_utils.GaussianData (xyz, rot wxyz, scale, opacity,
sh(48), nxnynz, filter_3Ds).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from colmap_io import (
    read_cameras_bin, read_images_bin, read_points3d_bin,
    images_to_world2cam_rt, cameras_to_intrinsics, pinhole_project,
    colmap_plant_paths,
)
from gaussian_utils import GaussianData

import core.headless_segmentation as _hs

# Dataset roots (repo-relative: repo is leaf_fit/, datasets/ is its sibling).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASETS = os.path.abspath(os.path.join(_REPO_ROOT, "..", "datasets"))
# v4 substrate: dense 3DGS Gaussian clouds (07-SuGaR-GS) + real captures (04-COLMAP).
_DENSE_ROOT = os.path.join(_DATASETS, "07-SuGaR-GS")
_COLMAP_ROOT = os.path.join(_DATASETS, "04-COLMAP")
_DENSE_PLY = os.path.join("vanilla_gs", "point_cloud", "iteration_7000",
                          "point_cloud_clean_v2_rerun_20260304_181041.ply")


@dataclass
class RealObservations:
    """Real capture bundle for one plant."""
    plant_dir: str
    images_dir: str
    rt: np.ndarray               # (n_views, 4, 4) world->cam
    K: np.ndarray                # (n_views, 3, 3) intrinsics
    image_paths: List[str]       # per-view absolute image path (aligned to rt rows)
    image_wh: List[tuple]        # (w, h) per view
    names: List[str]             # per-view filename
    n_views: int


@dataclass
class RealViewSignature:
    """Observation-grounded identity features (mirrors multiview_identity.ViewSignature)."""
    n_views: int
    visible: np.ndarray          # (n_views, N) uint8 0/1
    uv: np.ndarray               # (n_views, N, 2) pixel coords (float, NaN if not seen)
    depth: np.ndarray            # (n_views, N) positive = in front
    appear_sig: np.ndarray       # (N, 3) mean REAL RGB over visible views (uint8 0-255)
    appear_sig_real: np.ndarray  # alias of appear_sig (real pixel colors)
    visibility_fraction: np.ndarray  # (N,) |V_i| / n_views
    rt: np.ndarray               # (n_views, 4, 4)
    image_wh: List[tuple]


def colmap_plant_to_gaussians(plant_dir: str,
                              nn_scale_factor: float = 0.6,
                              min_scale: float = 1e-3) -> GaussianData:
    """Convert a COLMAP plant's points3D.bin into a LeafFit GaussianData.

    The 3DGS Gaussians are SURROGATED by the SfM points (we do NOT retrain 3DGS).
    Each SfM point becomes a (near-)spherical Gaussian:
      * xyz       = points3D.xyz
      * rot       = identity quaternion (no orientation prior)
      * scale     = isotropic = nn_scale_factor * median(nearest-neighbor dist)
                    (a point occupies ~1 voxel of its local density)
      * opacity   = 1.0  (sigmoid-range, fully opaque surrogates)
      * sh        = (rgb/255 - 0.5) / C0  with C0=0.28209479177387814
                    so the SH DC term reconstructs the real color at view dir (0,0,1)
      * nxnynz    = PCA-normal of the local point neighborhood (or zeros if too sparse)
      * filter_3Ds= 1.0
    """
    paths = colmap_plant_paths(plant_dir)
    xyz, rgb, _pid = read_points3d_bin(paths["points3D"])
    N = len(xyz)
    xyz = xyz.astype(np.float32)

    # scale: median nearest-neighbor distance (robust local density)
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(xyz).query(xyz, k=2)  # self + nearest
        nn = d[:, 1]
        med = float(np.median(nn[np.isfinite(nn) & (nn > 0)]))
    except Exception:
        med = float(np.percentile(np.linalg.norm(xyz - xyz.mean(0), axis=1), 50))
    med = max(med, min_scale)
    scale_val = float(max(med * nn_scale_factor, min_scale))
    scale = np.full((N, 3), scale_val, dtype=np.float32)

    # rotation: identity quaternion wxyz
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    opacity = np.ones((N, 1), dtype=np.float32)

    # SH DC from real RGB: sh_dc = (c/255 - 0.5) / C0, rest zeros
    C0 = 0.28209479177387814
    rgb_f = rgb.astype(np.float64) / 255.0
    sh_dc = (rgb_f - 0.5) / C0
    sh = np.zeros((N, 48), dtype=np.float32)
    sh[:, :3] = sh_dc.astype(np.float32)

    # normals: PCA over k-nearest neighbors
    nxnynz = np.zeros((N, 3), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        kdt = cKDTree(xyz)
        k = min(16, N - 1)
        _, nbr = kdt.query(xyz, k=k + 1)
        for i in range(N):
            nb = xyz[nbr[i, 1:]] - xyz[i]
            cov = nb.T @ nb
            w, V = np.linalg.eigh(cov)
            nrm = V[:, 0]  # smallest eigenvalue -> normal direction
            if nrm[2] < 0:
                nrm = -nrm
            nxnynz[i] = nrm.astype(np.float32)
    except Exception:
        pass

    filter_3Ds = np.ones((N, 1), dtype=np.float32)

    return GaussianData(xyz=xyz, rot=rot, scale=scale, opacity=opacity,
                        sh=sh, nxnynz=nxnynz, filter_3Ds=filter_3Ds)


def load_real_observations(plant_dir: str) -> RealObservations:
    """Load real RGB + calibrated poses + intrinsics for one COLMAP plant."""
    paths = colmap_plant_paths(plant_dir)
    cams = read_cameras_bin(paths["cameras"])
    imgs = read_images_bin(paths["images"])
    images_to_world2cam_rt(imgs, cams)
    K = cameras_to_intrinsics(cams)

    # align by image name
    rt_list = []
    K_list = []
    names = []
    image_paths = []
    wh = []
    for im in imgs:
        rt_list.append(im.rt)
        K_list.append(K[im.cid])
        names.append(im.name)
        image_paths.append(os.path.join(paths["images_dir"], im.name))
        wh.append((cams[im.cid].w, cams[im.cid].h))
    return RealObservations(
        plant_dir=plant_dir,
        images_dir=paths["images_dir"],
        rt=np.array(rt_list, dtype=np.float64),
        K=np.array(K_list, dtype=np.float64),
        image_paths=image_paths,
        image_wh=wh,
        names=names,
        n_views=len(imgs),
    )


def dense_gaussian_ply_path(plant_name: str, dense_root: Optional[str] = None) -> str:
    """Absolute path to the dense 3DGS Gaussian PLY for a plant (v4 substrate)."""
    root = dense_root if dense_root else _DENSE_ROOT
    return os.path.join(root, plant_name, _DENSE_PLY)


def observation_dir_for(plant_name: str, colmap_root: Optional[str] = None) -> str:
    """COLMAP capture dir (poses/RGB) for a plant — SAME world frame as the dense cloud."""
    root = colmap_root if colmap_root else _COLMAP_ROOT
    return os.path.join(root, plant_name)


def load_dense_gaussian_plant(plant_name: str,
                              dense_root: Optional[str] = None) -> GaussianData:
    """Load a plant's DENSE 3DGS Gaussian cloud (07-SuGaR-GS) as a GaussianData.

    This is the v4 geometry substrate: a 3DGS trained on the real COLMAP captures
    (task.md: 'use COLMAP 04-COLMAP to train 3DGS'). It is byte-identical schema to
    the released LeafFit baseline plants and segments into 8-28 leaves. The
    identity evidence still comes ONLY from real RGB+pose, never from this geometry.
    `dense_root` overrides the module default dataset root (CLI --dense-root must
    actually flow into loading, not only into provenance metadata).
    """
    ply = dense_gaussian_ply_path(plant_name, dense_root)
    if not os.path.exists(ply):
        raise FileNotFoundError(f"dense Gaussian PLY not found: {ply}")
    return _hs.load_gaussian_data(ply)


def load_dense_observations(plant_name: str,
                            colmap_root: Optional[str] = None) -> RealObservations:
    """Real RGB + COLMAP pose bundle for a plant, in the dense cloud's world frame.

    `colmap_root` overrides the module default dataset root (CLI --colmap-root
    must actually control which poses/images are loaded).
    """
    return load_real_observations(observation_dir_for(plant_name, colmap_root))


def load_or_cache_decoded_images(obs: RealObservations,
                                 downscale: int = 4,
                                 cache_dir: Optional[str] = None) -> np.ndarray:
    """One-time decode of all real images -> (nv, H//d, W//d, 3) uint8 array.

    Decoding 215 x 8MP JPEGs is ~50s; we do it ONCE and cache to disk so that
    per-case re-projection (which only moves a couple of leaves) reuses them
    without re-decoding. `downscale=4` keeps color-sampling quality ample for
    appearance while cutting memory ~16x.
    """
    if cache_dir is None:
        cache_dir = "/data/fj/LeafFit论文复现及修改/leaf_fit/outputs/task5/projection_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "decoded_images_d%d.npz" % downscale)
    if os.path.exists(cache_path):
        return np.load(cache_path)["imgs"]
    from PIL import Image
    nv = obs.n_views
    sample = np.asarray(Image.open(obs.image_paths[0]))
    H, W = sample.shape[0], sample.shape[1]
    dH, dW = H // downscale, W // downscale
    out = np.empty((nv, dH, dW, 3), dtype=np.uint8)
    for v in range(nv):
        try:
            im = Image.open(obs.image_paths[v]).convert("RGB")
            im = im.resize((dW, dH), Image.BILINEAR)
            out[v] = np.asarray(im)
        except Exception:
            out[v] = 0
    np.savez(cache_path, imgs=out)
    return out


def build_real_view_signature(g: GaussianData, obs: RealObservations,
                              cache_dir: Optional[str] = None,
                              cache_key: Optional[str] = None,
                              use_real_rgb: bool = True,
                              decoded_images: Optional[np.ndarray] = None,
                              downscale: int = 4) -> RealViewSignature:
    """Project Gaussians into the REAL images and build observation-grounded features.

    visible[i,v] = point i seen in view v (in_frustum; ellipse-visibility optional later)
    appear_sig[i] = mean REAL RGB sampled at projected locations over views seeing i
    c_app (computed downstream in the backend) = cosine similarity of appear_sig.

    For efficiency we project ALL points into ALL views (10764 x 215 is ~2.3M
    projections — tractable). Real-pixel color is sampled lazily per visible entry.
    If `decoded_images` (downscaled) is provided, color sampling skips PIL decode.
    """
    xyz = g.xyz.astype(np.float64)
    N = len(xyz)
    nv = obs.n_views

    visible = np.zeros((nv, N), dtype=np.uint8)
    uv = np.full((nv, N, 2), np.nan, dtype=np.float64)
    depth = np.zeros((nv, N), dtype=np.float64)
    acc_color = np.zeros((N, 3), dtype=np.float64)
    acc_count = np.zeros((N,), dtype=np.int64)

    if decoded_images is None:
        decoded_images = load_or_cache_decoded_images(obs, downscale=downscale, cache_dir=cache_dir)
    dH = decoded_images.shape[1]
    dW = decoded_images.shape[2]
    fx_scale = dW / float(obs.image_wh[0][0])
    fy_scale = dH / float(obs.image_wh[0][1])
    for v in range(nv):
        Kv = obs.K[v]
        rtv = obs.rt[v]
        w, h = obs.image_wh[v]
        pr = pinhole_project(xyz, Kv, rtv, w, h)
        infr = pr["in_frustum"]
        if not infr.any():
            continue
        idx = np.where(infr)[0]
        visible[v, idx] = 1
        uv[v, idx] = pr["pixel"][idx].astype(np.float64)
        depth[v, idx] = pr["depth"][idx]
        # sample real RGB at projected pixels from cached downscaled image
        img = decoded_images[v]
        px = np.clip(np.rint(pr["pixel"][idx, 0] * fx_scale), 0, dW - 1).astype(np.int64)
        py = np.clip(np.rint(pr["pixel"][idx, 1] * fy_scale), 0, dH - 1).astype(np.int64)
        cols = img[py, px].astype(np.float64)
        acc_color[idx] += cols
        acc_count[idx] += 1

    # appear_sig = mean real RGB (fallback to g.sh DC if no views saw a point)
    appear_sig = np.zeros((N, 3), dtype=np.float64)
    seen = acc_count > 0
    appear_sig[seen] = acc_color[seen] / acc_count[seen, None]
    # fallback: from SH DC (reconstructed color) for unseen points
    if getattr(g, "sh", None) is not None:
        C0 = 0.28209479177387814
        sh_dc = g.sh[:, :3].astype(np.float64)
        recon = (sh_dc * C0 + 0.5) * 255.0
        appear_sig[~seen] = recon[~seen]
    appear_sig = np.clip(appear_sig, 0, 255).astype(np.float64)

    vis_frac = visible.sum(0) / float(nv)

    return RealViewSignature(
        n_views=nv, visible=visible, uv=uv, depth=depth,
        appear_sig=appear_sig, appear_sig_real=appear_sig,
        visibility_fraction=vis_frac.astype(np.float64),
        rt=obs.rt, image_wh=obs.image_wh,
    )


if __name__ == "__main__":
    import sys
    plant = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/fj/LeafFit论文复现及修改/datasets/04-COLMAP/DouBanLv1"
    g = colmap_plant_to_gaussians(plant)
    print("GaussianData:", len(g), "scale=%.4f" % float(g.scale[0, 0]),
          "normal non-zero frac=%.2f" % float((np.linalg.norm(g.nxnynz, axis=1) > 0).mean()))
    obs = load_real_observations(plant)
    print("observations: n_views=%d  wh=%s" % (obs.n_views, obs.image_wh[0]))
    vs = build_real_view_signature(g, obs)
    print("visible mean per view: %.3f" % float(vs.visible.mean()))
    print("appear_sig sample (first 3 pts):", vs.appear_sig[:3].round(1).tolist())
    print("visibility_fraction mean: %.2f" % float(vs.visibility_fraction.mean()))
