#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 5 dev/held-out segmentation on the DENSE 3DGS substrate (07-SuGaR-GS).

Fork of the Task 1/2 baseline segmenter but reads the dense Gaussian cloud
(via core.real_observation.load_dense_gaussian_plant) and runs
headless_segmentation in AUTO-root mode (root_index=None) so the petiole /
base detection populates base_gaussian_index (required by the overlap
transform search). Writes outputs/task5/dense_baseline/<plant>/{labels.npy,
apexes.json,status.json,graph_meta.json}.

Usage:
    python scripts/seg_dense_plant.py --plant DouBanLv1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

REPO = "/data/fj/LeafFit论文复现及修改/leaf_fit"
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
import core.headless_segmentation as hs  # noqa: E402
from core.real_observation import load_dense_gaussian_plant  # noqa: E402

OUT = os.path.join(REPO, "outputs", "task5", "dense_baseline")
MIN_LEAF_POINTS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True)
    ap.add_argument("--root-index", type=int, default=None,
                    help="None = auto-root (recommended).")
    ap.add_argument("--min-leaf", type=int, default=MIN_LEAF_POINTS)
    ar = ap.parse_args()
    plant = ar.plant

    g = load_dense_gaussian_plant(plant)
    out = os.path.join(OUT, plant)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    if ar.root_index is None:
        res = hs.run_headless_segmentation(g)  # auto-root
    else:
        res = hs.run_headless_segmentation(g, root_index=ar.root_index)
    seg_dt = time.time() - t0

    raw = np.zeros(res.N, dtype=np.int64)
    for k, seg in enumerate(res.found_segs):
        raw[np.asarray(seg, dtype=np.int64)] = k + 1
    lc = np.bincount(raw)
    leaves = [(i, int(lc[i])) for i in range(1, len(lc)) if lc[i] > ar.min_leaf]
    n_with_base = 0
    apexes = []
    for c in res.final_cluster_results:
        tip = int(c["selected_tip"])
        base = c.get("base_idx")
        if base is not None:
            n_with_base += 1
        apexes.append({
            "gaussian_index": tip,
            "base_gaussian_index": int(base) if base is not None else None,
            "type": c.get("type", "single_tip"),
        })
    sizes = [s for _, s in leaves]
    print("[%s] N=%d seg=%.1fs n_leaves>%d=%d sizes=%s with_base=%d/%d"
          % (plant, len(g), seg_dt, ar.min_leaf, len(leaves),
             sizes[:12], n_with_base, len(apexes)), flush=True)

    np.save(os.path.join(out, "labels.npy"), raw)
    json.dump(apexes, open(os.path.join(out, "apexes.json"), "w"), indent=2)
    json.dump({
        "plant": plant, "n_points": len(g), "n_leaves_total": len(res.final_cluster_results),
        "n_leaves_gt": len(leaves), "leaf_sizes_gt": sizes, "n_with_base": n_with_base,
        "segmentation_seconds": round(seg_dt, 1), "status": "ok",
        "root_index": int(res.root_idx) if ar.root_index is None else ar.root_index,
        "root_source": res.root_source if ar.root_index is None else "fixed",
    }, open(os.path.join(out, "status.json"), "w"), indent=2)
    # keep a copy of the full HeadlessResult for downstream reuse if available
    print("[%s] saved %s" % (plant, out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
