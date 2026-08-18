#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeafFit CPU smoke test suite.

Runs the self-contained (CPU, no CUDA / no diff_gaussian_rasterization / no pytorch3d)
geometry paths of the LeafFit pipeline so that the core logic can be verified even
without the GPU rasterizer submodule.

Run:
    export PYTHONPATH=<repo>/core:<repo>
    python tests/smoke_test.py [data/plant1_green_pepper.ply]

Covered:
  1. cylinder_generator — create_test_cylinder() mesh validity
  2. cylinder_generator — generate_stem_cylinders() end-to-end on synthetic skeleton
  3. core/gaussian_utils + real .ply load — parse 3DGS fields into GaussianData
  4. core/pca_utils — PCA leaf alignment (tilted plane -> flat on xy)

Skipped (require missing submodule / GPU build — not present in this environment):
  S. diff_gaussian_rasterization import (orphaned gitlink in libs/)
  T. template_transform / gen_template_leaf (need rasterizer + pytorch3d)
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np


def _env_deps_ok():
    """Return list of {name: error} for required optional C deps."""
    missing = []
    for mod in ("open3d", "fpsample", "plyfile", "e3nn"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}: {e}")
    return missing


def test_cylinder_create() -> bool:
    import cylinder_generator as cg

    m = cg.create_test_cylinder()
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    assert len(v) == 16 and len(f) == 16, f"unexpected mesh: v={len(v)} f={len(f)}"
    m.compute_vertex_normals()
    print("[OK] create_test_cylinder -> verts=16 tris=16")
    return True


def test_cylinder_stems() -> bool:
    import cylinder_generator as cg

    pts = {
        0: np.array([0, 0, 0]),
        1: np.array([0, 0, 0.5]),
        2: np.array([0, 0, 1.0]),
        3: np.array([0.5, 0, 1.0]),
        4: np.array([-0.5, 0, 1.0]),
        5: np.array([0, 0, 1.5]),
    }
    seg_usage = {(0, 1): 30, (1, 2): 25, (2, 3): 8, (2, 4): 8, (2, 5): 15}
    pt_usage = {0: 30, 1: 25, 2: 22, 3: 8, 4: 8, 5: 15}
    cols = {i: np.array([0.2, 0.5, 0.15]) for i in pts}
    path_info = {
        "segment_usage": seg_usage,
        "point_usage_counter": pt_usage,
        "point_coordinates": pts,
        "point_colors": cols,
    }
    cyls = cg.generate_stem_cylinders(path_info)
    assert len(cyls) == 5, f"expected 5 cylinder segments, got {len(cyls)}"
    for i, c in enumerate(cyls):
        v = np.asarray(c.vertices)
        f = np.asarray(c.triangles)
        assert len(v) > 0 and len(f) > 0 and c.has_vertex_colors(), f"seg{i} invalid"
    print(f"[OK] generate_stem_cylinders -> {len(cyls)} valid colored cylinder meshes")
    return True


def test_real_ply_load(path: str) -> bool:
    """Faithful reproduction of viewer/utils.load_ply_gaussian using core.GaussianData."""
    from plyfile import PlyData
    from gaussian_utils import GaussianData

    max_sh_degree = 3
    ply = PlyData.read(path)
    el = ply.elements[0]
    n = len(el)
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    nxyz = np.stack([el["nx"], el["ny"], el["nz"]], 1)
    opac = np.asarray(el["opacity"])[..., None]
    fd = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1)
    extra = sorted(
        (p.name for p in el.properties if p.name.startswith("f_rest_")),
        key=lambda x: int(x.split("_")[-1]),
    )
    fe = np.stack([el[a] for a in extra], 1).reshape(
        -1, 3, (max_sh_degree + 1) ** 2 - 1
    ).transpose(0, 2, 1)
    shs = np.concatenate([fd.reshape(-1, 3), fe.reshape(n, -1)], -1).astype(np.float32)
    scale_names = sorted(
        (p.name for p in el.properties if p.name.startswith("scale_")),
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.stack([el[a] for a in scale_names], 1)
    rot_names = sorted(
        (p.name for p in el.properties if p.name.startswith("rot")),
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.stack([el[a] for a in rot_names], 1)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    filt = np.asarray(el["filter_3D"])[..., None]
    g = GaussianData(
        xyz.astype(np.float32),
        rots.astype(np.float32),
        np.exp(scales).astype(np.float32),
        (1 / (1 + np.exp(-opac))).astype(np.float32),
        shs,
        nxyz.astype(np.float32),
        filt.astype(np.float32),
    )
    assert n > 0 and len(g) == n
    assert np.allclose(np.linalg.norm(g.rot, axis=-1), 1), "quats not unit"
    assert g.opacity.min() >= 0.0 and g.opacity.max() <= 1.0, "opacity out of range"
    assert g.sh_dim == 48, f"SH dim {g.sh_dim}"
    print(f"[OK] real .ply load: {n} gaussians, sh_dim={g.sh_dim}, flat={g.flat().shape}")
    return True


def test_pca_align() -> bool:
    import scipy.spatial.transform as st
    from pca_utils import align_to_xy_plane

    rng = np.random.default_rng(0)
    n = 300
    pts = np.column_stack([(rng.random(n) - 0.5) * 2, (rng.random(n) - 0.5) * 1.2, np.zeros(n)])
    R = st.Rotation.from_euler("zyx", [25, 10, -40], degrees=True).as_matrix()
    pts = np.ascontiguousarray((pts @ R.T + np.array([0.1, 0.2, 5])).astype(np.float64))
    _T, aligned, _info = align_to_xy_plane(pts)
    zstd = float(np.asarray(aligned)[:, 2].std())
    assert zstd < 0.05, f"leaf not flattened, z-std={zstd}"
    print(f"[OK] pca_utils.align_to_xy_plane -> z-std={zstd:.6f} (flat on xy)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply_path", nargs="?", help="a real data/*.ply dataset path")
    args = ap.parse_args()

    missing = _env_deps_ok()
    if missing:
        print("MISSING DEPS:")
        for m in missing:
            print("  ", m)
        return 2

    ok = True
    try:
        ok &= test_cylinder_create()
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] create_test_cylinder: {e}")
    try:
        ok &= test_cylinder_stems()
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] generate_stem_cylinders: {e}")
    if args.ply_path and os.path.exists(args.ply_path):
        try:
            ok &= test_real_ply_load(args.ply_path)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[FAIL] real ply load: {e}")
    try:
        ok &= test_pca_align()
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] pca_align: {e}")

    print("\n--- BLOCKED ON GPU (not run here) ---")
    try:
        import diff_gaussian_rasterization  # noqa: F401
        print("[SKIP] diff_gaussian_rasterization present (not expected)")
    except Exception as e:  # noqa: BLE001
        print(f"[BLOCKED] diff_gaussian_rasterization import: {type(e).__name__}: {e}")
    for mod in ("template_transform", "gen_template_leaf"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            print(f"[BLOCKED] {mod}: {type(e).__name__}: {e}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())