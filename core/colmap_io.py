#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COLMAP binary SfM reader -- cameras / images / points3D.

Pure stdlib `struct` parser for the COLMAP binary reconstruction format
(`sparse/0/{cameras.bin, images.bin, points3D.bin}`). Verified against the
LeafFit `datasets/04-COLMAP/<plant>/sparse/0/*.bin` files (magic 19642,
little-endian). No COLMAP binary / pycolmap dependency required.

Camera models we support for intrinsic extraction (only the PINHOLE family
matters for the leaf_fit datasets; `sparse/0` is already de-distorted):
    model_id 1  PINHOLE        params = [fx, fy, cx, cy]
    model_id 0  SIMPLE_PINHOLE params = [fx, fy, cx, cy]  (fx == fy)
    model_id 2  SIMPLE_RADIAL  params = [fx, fy, cx, cy, k]
    model_id 3  RADIAL         params = [fx, fy, cx, cy, k1, k2]
    model_id 4  OPENCV         params = [fx, fy, cx, cy, k1, k2, p1, p2]
We build the full 3x3 K from [fx, fy, cx, cy]. Radial / fisheye distortion is
NOT modelled here because `sparse/0` is already de-distorted (images/ align).

World<->camera convention (COLMAP):
    X_cam = R @ (X_world - C),   C = -R^T t
    q = (qw, qx, qy, qz)  (wxyz), R = quat_to_matrix(q)
    world->cam (4,4) Rt = [[R, -R@C], [0,0,0,1]]

This module is the observation-ingest drop-site for Task 5 (real RGB +
calibrated poses). It replaces the synthetic `synthesize_orbit_cameras` used
in Task 4 with real capture Rt. See real_observation.py for the downstream
ViewSignature built on top of these reads.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# COLMAP camera model -> number of params.
_MODEL_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 4, 7: 8, 8: 12, 9: 13}
_MODEL_NAME = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE",
}


@dataclass
class Camera:
    cid: int
    model: str
    model_id: int
    w: int
    h: int
    params: List[float]
    # Derived intrinsics (filled by cameras_to_intrinsics if present).
    K: np.ndarray = None


@dataclass
class ImagePose:
    iid: int
    qvec: Tuple[float, float, float, float]  # wxyz (QW, QX, QY, QZ)
    tvec: Tuple[float, float, float]
    cid: int
    name: str
    # Filled later:
    R: np.ndarray = None
    t: np.ndarray = None
    rt: np.ndarray = None  # 4x4 world->cam


def read_cameras_bin(path: str) -> Dict[int, Camera]:
    """Read COLMAP cameras.bin -> {cid: Camera}."""
    out: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            np_ = _MODEL_NPARAMS.get(model_id, 4)
            params = list(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            out[cid] = Camera(cid, _MODEL_NAME.get(model_id, f"MODEL_{model_id}"),
                             model_id, int(w), int(h), params)
    return out


def read_images_bin(path: str) -> List[ImagePose]:
    """Read COLMAP images.bin -> list[ImagePose] (qw,qx,qy,qz wxyz order)."""
    out: List[ImagePose] = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            iid = struct.unpack("<I", f.read(4))[0]
            q = struct.unpack("<dddd", f.read(32))  # QW,QX,QY,QZ
            t = struct.unpack("<ddd", f.read(24))   # TX,TY,TZ
            cid = struct.unpack("<I", f.read(4))[0]  # colmap2.2 uses unsigned
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            # each 2D point: x (double), y (double), point3d_id (int64) = 24 bytes
            f.seek(24 * npts, os.SEEK_CUR)
            out.append(ImagePose(iid, q, t, cid, name.decode("latin-1", errors="replace")))
    return out


def read_points3d_bin(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin -> (xyz (N,3) float64, rgb (N,3) uint8, pid (N,) int64)."""
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
    pid: List[int] = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            p = struct.unpack("<Q", f.read(8))[0]       # point3d_id (uint64)
            x, y, z = struct.unpack("<ddd", f.read(24))  # xyz (3 double)
            r, g, b = struct.unpack("<BBB", f.read(3))   # rgb (3 uint8)
            struct.unpack("<d", f.read(8))[0]            # error (double)
            ntrack = struct.unpack("<Q", f.read(8))[0]   # track_length (uint64)
            # each track entry: image_id (int64) + point2d_idx (int64) = 8 bytes
            f.seek(8 * ntrack, os.SEEK_CUR)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            pid.append(int(p))
    xyz_arr = np.array(xyz, dtype=np.float64)
    rgb_arr = np.array(rgb, dtype=np.uint8)
    pid_arr = np.array(pid, dtype=np.int64)
    return xyz_arr, rgb_arr, pid_arr


# ---- small math helpers (no scipy rotation dep) ----

def _quat_wxyz_to_rot(q: Tuple[float, float, float, float]) -> np.ndarray:
    """COLMAP qvec (qw,qx,qy,qz) -> world->cam rotation matrix R (3x3).

    COLMAP convention: X_cam = R @ (X_world - C), R = R(q). Quaternion wxyz.
    """
    qw, qx, qy, qz = q
    # Normalize defensively.
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        n = 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


def cameras_to_intrinsics(cameras: Dict[int, Camera]) -> Dict[int, np.ndarray]:
    """cameras.bin cid -> 3x3 intrinsics K = [[fx, 0, cx],[0, fy, cy],[0,0,1]]."""
    K: Dict[int, np.ndarray] = {}
    for cid, c in cameras.items():
        params = c.params
        fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        K[cid] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def images_to_world2cam_rt(imgs: List[ImagePose], cameras: Dict[int, Camera]) -> List[ImagePose]:
    """Populate R, t, rt (4x4 world->cam) on each ImagePose in place."""
    for img in imgs:
        R = _quat_wxyz_to_rot(img.qvec)
        t = np.array(img.tvec, dtype=np.float64)
        C = -R.T @ t  # camera center in world
        img.R = R
        img.t = t
        # world->cam: X_cam = R @ X_world + (-R @ C) = R @ (X_world - C)
        rt = np.eye(4, dtype=np.float64)
        rt[:3, :3] = R
        rt[:3, 3] = R @ (-C)  # = t  actually: -R@C
        # Sanity: -R @ C == -R @ (-R^T t) == t. So rt[:3,3] = t. Keep via C for clarity.
        img.rt = rt
    return imgs


def pinhole_project(xyz: np.ndarray, K: np.ndarray, rt: np.ndarray, w: int, h: int) -> dict:
    """Project xyz (N,3) by world->cam rt + K into pixels.

    COLMAP camera convention: the camera looks along **+z** in camera space, so a
    point is in front iff cam_z > 0 (this is the OPPOSITE of multiview_identity's
    -z-in-front convention). We report `depth` as cam_z (positive = in front) to
    match COLMAP, and `in_frustum` requires cam_z > 0 AND (px,py) in [0,W-1]x[0,H-1].
    """
    N = len(xyz)
    homog = np.hstack([xyz, np.ones((N, 1), dtype=np.float64)])  # (N,4)
    cam = (rt @ homog.T).T  # (N,4) -> camera coords
    cam_z = cam[:, 2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = cam[:, 0]; y = cam[:, 1]
    # COLMAP: px = fx * x / z + cx with z = cam_z > 0 in front.
    px = fx * x / cam_z + cx
    py = fy * y / cam_z + cy
    pixel = np.stack([px, py], axis=1)  # (N,2)
    W, H = int(w), int(h)
    in_frustum = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (cam_z > 0)
    # NDC in [-1,1] over the image plane.
    ndc_x = (px - cx) / (W / 2.0)
    ndc_y = (py - cy) / (H / 2.0)
    return {
        "ndc_xy": np.stack([ndc_x, ndc_y], axis=1).astype(np.float64),
        "depth": cam_z.astype(np.float64),  # positive = in front (COLMAP convention)
        "in_frustum": in_frustum.astype(bool),
        "pixel": np.stack([
            np.clip(np.rint(px), 0, W - 1).astype(np.int64),
            np.clip(np.rint(py), 0, H - 1).astype(np.int64),
        ], axis=1),
        "W": W, "H": H,
    }


def colmap_plant_paths(plant_dir: str) -> dict:
    """Locate the standard COLMAP undistorted sparse/0 + images."""
    sparse0 = os.path.join(plant_dir, "sparse", "0")
    cameras_path = os.path.join(sparse0, "cameras.bin")
    images_path = os.path.join(sparse0, "images.bin")
    points_path = os.path.join(sparse0, "points3D.bin")
    return {
        "sparse0": sparse0,
        "cameras": cameras_path,
        "images": images_path,
        "points3D": points_path,
        "points_ply": os.path.join(sparse0, "points3D.ply"),
        "images_dir": _find_images_dir(plant_dir),
    }


def _find_images_dir(plant_dir: str) -> str:
    for cand in ("images", "images_rgba", "distorted/image"):
        p = os.path.join(plant_dir, cand)
        if os.path.isdir(p):
            return p
    return os.path.join(plant_dir, "images")


if __name__ == "__main__":
    # Self-test: read one plant end-to-end.
    import sys
    plant_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not plant_dir:
        # default example that exists in the leaf_fit datasets
        plant_dir = "/data/fj/LeafFit论文复现及修改/datasets/04-COLMAP/DouBanLv1"
    paths = colmap_plant_paths(plant_dir)
    print("plant_dir:", paths["images_dir"])
    cams = read_cameras_bin(paths["cameras"])
    imgs = read_images_bin(paths["images"])
    xyz, rgb, pid = read_points3d_bin(paths["points3D"])
    images_to_world2cam_rt(imgs, cams)
    K = cameras_to_intrinsics(cams)
    print("cameras:", {k: (v.model, v.w, v.h) for k, v in cams.items()})
    print("K[1]:", K[imgs[0].cid])
    print("images:", len(imgs), "  first:", imgs[0].name, "rt[:3,:3] orth?",
          np.allclose(imgs[0].rt[:3, :3] @ imgs[0].rt[:3, :3].T, np.eye(3)))
    proj = pinhole_project(xyz[:5], K[imgs[0].cid], imgs[0].rt, cams[imgs[0].cid].w, cams[imgs[0].cid].h)
    print("points3d:", xyz.shape, "rgb", rgb.shape, "sample pixel:", proj["pixel"][0].tolist())
