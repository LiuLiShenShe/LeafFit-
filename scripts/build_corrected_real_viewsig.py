#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R — build the CORRECTED (occlusion-aware) real view signature for a plant.

Phase 1+2 entry point: portable paths, full provenance, content-hash cache keys,
plant-specific decoded-RGB cache isolation.

Outputs (heavy, gitignored): <cache-dir>/<plant>/corrected_viewsig_<key>.npz
Manifest (light, committable):  <output>  (default outputs/task5r/corrected_viewsig_manifest.jsonl)

Usage:
  python scripts/build_corrected_real_viewsig.py --plant DouBanLv1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent

for p in (str(REPO_DEFAULT), str(REPO_DEFAULT / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.observation_identity import (  # noqa: E402
    build_occlusion_aware_real_view_signature,
    viewsig_cache_key, git_commit, file_sha256, ordered_name_hash,
    VISIBILITY_VERSION,
)
from core.real_observation import load_or_cache_decoded_images  # noqa: E402


def resolve_roots(args):
    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_DEFAULT
    dense_root = Path(args.dense_root).resolve() if args.dense_root else \
        repo_root.parent / "datasets" / "07-SuGaR-GS"
    colmap_root = Path(args.colmap_root).resolve() if args.colmap_root else \
        repo_root.parent / "datasets" / "04-COLMAP"
    return repo_root, dense_root, colmap_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dense-root", default=None)
    ap.add_argument("--colmap-root", default=None)
    ap.add_argument("--plant", required=True)
    ap.add_argument("--output", default=None,
                    help="manifest jsonl path (committable)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--visibility-version", default=VISIBILITY_VERSION)
    ap.add_argument("--seed", type=int, default=0)
    ar = ap.parse_args()

    np.random.seed(ar.seed)  # deterministic; algorithm itself is deterministic anyway

    repo_root, dense_root, colmap_root = resolve_roots(ar)
    sys.path[:0] = [str(repo_root), str(repo_root / "core")]

    from core.real_observation import load_dense_gaussian_plant, load_dense_observations

    cache_dir = Path(ar.cache_dir).resolve() if ar.cache_dir else \
        repo_root / "outputs" / "task5r_v3" / "projection_cache"
    out_path = Path(ar.output).resolve() if ar.output else \
        repo_root / "outputs" / "task5r_v3" / "corrected_viewsig_manifest.jsonl"

    # CLI roots MUST control data loading (not just provenance metadata).
    g = load_dense_gaussian_plant(ar.plant, dense_root=str(dense_root))
    obs = load_dense_observations(ar.plant, colmap_root=str(colmap_root))

    from core.observation_identity import git_tree_dirty, algorithm_extra
    dirty = git_tree_dirty(repo_root)
    key = viewsig_cache_key(g, obs.rt, obs.K, obs.names, ar.downscale,
                            ar.visibility_version)

    # plant-specific decoded-image cache (never shared across plants)
    img_cache = cache_dir / ar.plant
    img_cache.mkdir(parents=True, exist_ok=True)

    zdir = cache_dir / ar.plant
    zpath = zdir / f"corrected_viewsig_{key}.npz"
    t0 = time.time()
    if zpath.exists():
        z = np.load(zpath, allow_pickle=False)
        vs_meta = json.loads(str(z["meta_json"]))
        status = "cache_hit"
    else:
        imgs = load_or_cache_decoded_images(obs, downscale=ar.downscale,
                                            cache_dir=str(img_cache))
        vs = build_occlusion_aware_real_view_signature(
            g, obs, decoded_images=imgs, downscale=ar.downscale,
            source_commit=git_commit(repo_root),
            source_tree_dirty=dirty)
        vs_meta = dict(vs.meta)
        zdir.mkdir(parents=True, exist_ok=True)
        tmp = zpath.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            in_frustum=vs.in_frustum, visible=vs.visible,
            max_alpha=vs.max_alpha, acc_alpha=vs.acc_alpha,
            uv_pixel=vs.uv_pixel, uv_ndc=vs.uv_ndc,
            depth=vs.depth, footprint_radius_px=vs.footprint_radius_px,
            rgb_views=vs.rgb_views, rgb_valid=vs.rgb_valid,
            visibility_fraction=vs.visibility_fraction,
            meta_json=np.array(json.dumps(vs.meta)),
        )
        tmp.rename(zpath)
        status = "computed"

    dt = time.time() - t0
    vis_frac = float(np.load(zpath)["visibility_fraction"].mean())

    record = {
        "plant": ar.plant,
        "status": status,
        "cache_key": key,
        "viewsig_path": str(zpath.relative_to(repo_root)) if str(zpath).startswith(str(repo_root)) else str(zpath),
        "git_commit": git_commit(repo_root),
        "source_tree_dirty": dirty,
        "dense_cloud": {"root": str(dense_root), "plant": ar.plant,
                        "ply_sha256": file_sha256(dense_root / ar.plant /
                                                  "vanilla_gs" / "point_cloud" /
                                                  "iteration_7000" /
                                                  "point_cloud_clean_v2_rerun_20260304_181041.ply")},
        "colmap_dir": str(colmap_root / ar.plant),
        "camera_bin_sha256": file_sha256(colmap_root / ar.plant / "sparse" / "0" / "cameras.bin"),
        "images_bin_sha256": file_sha256(colmap_root / ar.plant / "sparse" / "0" / "images.bin"),
        "ordered_image_name_hash": ordered_name_hash(obs.names),
        "n_images": int(obs.n_views),
        "n_points": int(len(g)),
        "image_wh": list(obs.image_wh[0]),
        "downscale": ar.downscale,
        "visibility_version": ar.visibility_version,
        "visibility_algorithm": vs_meta,
        "visibility_fraction_mean": vis_frac,
        "seed": ar.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(dt, 1),
    }
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: record[k] for k in
                      ["plant", "status", "cache_key", "n_points", "n_images",
                       "visibility_fraction_mean", "runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
