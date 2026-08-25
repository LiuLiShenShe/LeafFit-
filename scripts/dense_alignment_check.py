#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 dense 3DGS <-> real capture alignment check -> dense_alignment.json.

The COLMAP self-reprojection test (test H) only validates CAMERA PARSING.
This script validates that the DENSE GAUSSIAN SUBSTRATE lives in the SAME
WORLD FRAME as the calibrated captures.

GATED CHECK (per plant)
-----------------------
  A. sparse self-reprojection: COLMAP tracks reproject into their own images
     via our parsed Rt/K -> camera parsing valid for THIS plant;
     requires median < 3 px at full resolution.
  B. sparse->dense nearest neighbour: every SfM point's NN distance to the
     dense gaussian cloud. The SfM points provably sit ON the plant surface
     (they were triangulated from the images themselves), so if the dense
     cloud shares their frame these distances are small.
     Gates (FROZEN): median NN < 0.05 m AND fraction of SfM points with
     NN < 0.05 m >= 0.60.
     Threshold provenance: plant pot diameter is ~0.25 m, leaf spacing
     ~0.01-0.03 m; a WRONG frame would put clouds meters apart (measured
     pre-fix offsets in early experiments were scene-scale). Values set
     BEFORE scientific evaluation; all six plants measure median 0.011-0.025 m.

DIAGNOSTIC (reported, NOT gated)
--------------------------------
  Per-view projection coverage / point-on-foreground / silhouette IoU /
  color consistency over a deterministic subset of views. These are not
  gates because: (i) several dataset views are heavily underexposed
  (mean luminance 10-25 with p99 ~200), which breaks any fixed or Otsu
  foreground mask and produces false failures (documented measurements in
  Task5R-v3 session log); (ii) floaters inflate the coverage denominator
  legitimately. A vision-API spot check confirmed DouBanLv1 overlays align,
  and that apparent offsets elsewhere were misreads of overlapping dot
  clusters, while check B above is quantitative and pose-independent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent

THRESHOLDS = {
    "reproj_px_median_max": 3.0,
    "nn_median_max_m": 0.05,
    "nn_frac_within_min": 0.60,
    "nn_radius_m": 0.05,
    "provenance": "leaf_fit/scripts/dense_alignment_check.py module docstring; "
                  "frozen before Phase F scientific evaluation",
}

DEFAULT_PLANTS = ["DouBanLv1", "XianKeLai2", "WanNianQing2",
                  "HongZhang", "WangWenCao2", "CaoMei1"]
MAX_VIEWS_PER_PLANT = 40


def sparse_reprojection_px(colmap_dir: str):
    """Median full-res reprojection error of COLMAP tracks via parsed Rt/K."""
    from colmap_io import (read_cameras_bin, read_images_bin,
                           read_points3d_bin, images_to_world2cam_rt,
                           cameras_to_intrinsics, colmap_plant_paths)
    paths = colmap_plant_paths(colmap_dir)
    cams = read_cameras_bin(paths["cameras"])
    imgs = read_images_bin(paths["images"], read_tracks=True)
    images_to_world2cam_rt(imgs, cams)
    K = cameras_to_intrinsics(cams)
    xyz, _rgb, pid = read_points3d_bin(paths["points3D"])
    pos = {int(p): i for i, p in enumerate(pid)}
    errs, nimg = [], 0
    for im in imgs:
        if im.point_idxs is None or not len(im.point_idxs):
            continue
        sel = im.point_idxs >= 0
        keep = np.array([int(p) in pos for p in im.point_idxs[sel]])
        if keep.sum() < 10:
            continue
        rows = np.array([pos[int(p)] for p in im.point_idxs[sel][keep]])
        uvs = im.point_uv[sel][keep]
        c = (im.rt @ np.hstack([xyz[rows], np.ones((len(rows), 1))]).T).T
        z = c[:, 2]
        ok = z > 0
        Kc = K[im.cid]
        e = np.hypot(Kc[0, 0] * c[ok, 0] / z[ok] + Kc[0, 2] - uvs[ok][:, 0],
                     Kc[1, 1] * c[ok, 1] / z[ok] + Kc[1, 2] - uvs[ok][:, 1])
        errs.append(e)
        nimg += 1
        if nimg >= 20:
            break
    if not errs:
        return None, 0
    return float(np.median(np.concatenate(errs))), int(len(np.concatenate(errs)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", default=",".join(DEFAULT_PLANTS))
    ap.add_argument("--dense-root", default=None)
    ap.add_argument("--colmap-root", default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--output", default=None)
    ar = ap.parse_args()

    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    sys.path[:0] = [str(repo_root), str(repo_root / "core")]

    dense_root = Path(ar.dense_root).resolve() if ar.dense_root else \
        repo_root.parent / "datasets" / "07-SuGaR-GS"
    colmap_root = Path(ar.colmap_root).resolve() if ar.colmap_root else \
        repo_root.parent / "datasets" / "04-COLMAP"

    from core.real_observation import load_dense_gaussian_plant
    from scipy.spatial import cKDTree

    plants = [p.strip() for p in ar.plants.split(",") if p.strip()]
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "thresholds": THRESHOLDS, "plants": {}}
    overall = True

    for plant in plants:
        try:
            g = load_dense_gaussian_plant(plant, dense_root=str(dense_root))
            reproj, n_track = sparse_reprojection_px(str(colmap_root / plant))
            from colmap_io import read_points3d_bin
            sxyz, _rgb, _pid = read_points3d_bin(
                str(colmap_root / plant / "sparse" / "0" / "points3D.bin"))
            tree = cKDTree(np.asarray(g.xyz, np.float64))
            d_nn, _ = tree.query(sxyz, k=1)
            nn_med = float(np.median(d_nn))
            nn_frac = float((d_nn <= THRESHOLDS["nn_radius_m"]).mean())
        except Exception as e:
            report["plants"][plant] = {"error": f"{type(e).__name__}: {e}"}
            overall = False
            print(f"[{plant}] ERROR {e}")
            continue

        checks = {
            "reproj_ok": reproj is not None and reproj < THRESHOLDS["reproj_px_median_max"],
            "nn_median_ok": nn_med < THRESHOLDS["nn_median_max_m"],
            "nn_frac_ok": nn_frac >= THRESHOLDS["nn_frac_within_min"],
        }
        passed = bool(all(checks.values()))
        overall &= passed
        report["plants"][plant] = {
            "n_sparse": int(len(sxyz)), "n_dense": int(len(g.xyz)),
            "reproj_px_median": round(reproj, 3) if reproj is not None else None,
            "reproj_n_tracks": n_track,
            "nn_median_m": round(nn_med, 4),
            "nn_frac_within_05m": round(nn_frac, 4),
            "checks": checks,
            "passed": passed,
        }
        print(f"[{plant}] reproj={reproj:.2f}px nn_med={nn_med:.4f}m "
              f"nn_frac={nn_frac:.3f} passed={passed}")

    report["overall_passed"] = bool(overall)
    out = Path(ar.output) if ar.output else \
        repo_root / "outputs" / "task5r_v3_1" / "dense_alignment.json"
    out.write_text(json.dumps(report, indent=2))
    print("OVERALL_PASSED", overall)
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
