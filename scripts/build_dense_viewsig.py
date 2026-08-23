#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #40 — build the real-view signature on a DENSE 3DGS cloud (07-SuGaR-GS)
using REAL captures from 04-COLMAP (same world frame). Cache to
outputs/task5/projection_cache/<plant>/real_viewsig_dense.npz.
"""
import sys, os, time, json
import numpy as np

REPO = "/data/fj/LeafFit论文复现及修改/leaf_fit"
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.real_observation import (
    load_dense_gaussian_plant, load_dense_observations,
    build_real_view_signature,
)

_DENSE_ROOT = "/data/fj/LeafFit论文复现及修改/datasets/07-SuGaR-GS"


def main(plant):
    g = load_dense_gaussian_plant(plant)
    obs = load_dense_observations(plant)
    print("[%s] loaded dense cloud: N=%d; observations: n_views=%d wh=%s"
          % (plant, len(g), obs.n_views, obs.image_wh[0]), flush=True)

    t0 = time.time()
    vs = build_real_view_signature(g, obs, downscale=4)
    dt = time.time() - t0
    print("[%s] view signature built in %.1fs" % (plant, dt), flush=True)

    vis_mean = float(vs.visible.mean())
    vis_frac_mean = float(vs.visibility_fraction.mean())
    n_never_seen = int((vs.visibility_fraction == 0).sum())
    print("[%s] visible.mean=%.3f  visibility_fraction.mean=%.3f  never_seen=%d/%d"
          % (plant, vis_mean, vis_frac_mean, n_never_seen, len(g)), flush=True)

    # appearance cue coherence: mean per-point real-RGB std across visible views
    # (low std => stable color => usable appearance identity)
    cache_dir = "/data/fj/LeafFit论文复现及修改/leaf_fit/outputs/task5/projection_cache/%s" % plant
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, "real_viewsig_dense.npz")
    np.savez(out,
             visible=vs.visible,
             uv=vs.uv,
             depth=vs.depth,
             appear_sig=vs.appear_sig,
             visibility_fraction=vs.visibility_fraction)
    print("[%s] saved %s" % (plant, out), flush=True)

    json.dump({
        "plant": plant, "n_points": len(g), "n_views": vs.n_views,
        "visible_mean": vis_mean, "visibility_fraction_mean": vis_frac_mean,
        "never_seen": n_never_seen, "build_seconds": round(dt, 1),
        "appear_sig_sample": vs.appear_sig[:3].round(1).tolist(),
    }, open(os.path.join(cache_dir, "real_viewsig_dense_meta.json"), "w"), indent=2)
    print("[%s] DONE" % plant, flush=True)


if __name__ == "__main__":
    plant = sys.argv[1] if len(sys.argv) > 1 else "DouBanLv1"
    main(plant)
