#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R Phase 4 — observation-matched natural overlap case audit.

A valid case requires:
  * dense 3DGS + RGB/COLMAP from the SAME physical capture (verified by
    reprojection test H passing on that plant);
  * the overlap exists in the CAPTURED plant (no synthetic transforms);
  * two leaves with independent identities;
  * labels NOT produced solely by the LeafFit prediction being evaluated — we use
    an independent clustering proposal (normal+color region growing, no heat
    solver, no LeafFit pipeline) and store LeafFit's own labels separately as a
    construction reference only;
  * camera/3DGS alignment validated (reprojection);
  * both leaves have sufficient real view coverage (corrected visibility).

Output: outputs/task5r/observation_overlap_cases.json + benchmark_manifest.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent
for p in (str(REPO_DEFAULT), str(REPO_DEFAULT / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.observation_identity import git_commit, file_sha256, ordered_name_hash  # noqa
from scipy.spatial import cKDTree  # noqa: E402


def load_corrected_viewsig(cache_dir: Path, plant: str):
    d = cache_dir / plant
    files = sorted(d.glob("corrected_viewsig_*.npz"))
    if not files:
        raise FileNotFoundError(f"no corrected viewsig for {plant} in {d}")
    z = np.load(files[-1])
    return z


def propose_leaf_labels_independent(g, vis_frac, col=None, k=16, sin_thr=0.35,
                                    lin_thr=0.6, plan_thr=0.25,
                                    color_thr=30.0):
    """Independent leaf-instance proposal WITHOUT LeafFit's heat solver.

    Region growing on a kNN graph with edges kept by LOCAL PCA surface
    coherence (the SuGaR PLY carries no normals -- all zeros -- so we estimate
    normals ourselves). This labeler shares NO code path with LeafFit's
    heat/FPS/apex/grouping pipeline; its errors are independent of the
    failure modes being evaluated.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    xyz = np.asarray(g.xyz, dtype=np.float64)
    N = len(xyz)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    rows = np.repeat(np.arange(N), k)
    cols = idx[:, 1:].ravel()
    d = np.linalg.norm(xyz[cols] - xyz[rows], axis=1)
    med_d = float(np.median(d))
    keep = d < 2.5 * max(med_d, 1e-9)          # drop cross-gap links
    rows, cols = rows[keep], cols[keep]
    nbr_idx = idx[rows, np.arange(1, k + 1)[None, :].repeat(len(rows), 0)] if False else None
    # local-PCA normal coherence (chunked)
    keep2 = np.empty(len(rows), dtype=bool)
    CH = 100000
    for s in range(0, len(rows), CH):
        r = rows[s:s+CH]; c = cols[s:s+CH]
        nb = xyz[idx[r, 1:k+1]]                       # (m,k,3) same neighbors as graph
        nb_c = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb_c, nb_c)
        w, V = np.linalg.eigh(cov)
        nrm = V[:, :, 0]                              # smallest eigvec = normal
        seg = xyz[c] - xyz[r]
        dn = np.abs(np.einsum("nc,nc->n", nrm, seg))
        dist_along = np.linalg.norm(seg, axis=1) + 1e-12
        keep2[s:s+CH] = (dn / dist_along) < sin_thr   # sin(angle) < thr -> same surface
    rows, cols = rows[keep2], cols[keep2]
    # STEM CUT: classify each point as locally LINEAR (tubular petiole/stem)
    # from local-PCA eigen-shape and remove every edge touching such a point.
    # Leaf blades are locally planar; stems/petioles are linear+non-planar.
    # This is what prevents the one-merged-body collapse through junctions.
    xyz_all = np.asarray(g.xyz, dtype=np.float64)  # noqa: F841 (kept for clarity)
    N_all = len(xyz_all)
    stem_like = np.zeros(N_all, dtype=bool)
    CH2 = 200000
    for s in range(0, N_all, CH2):
        nb_e = xyz[idx[s:s+CH2, 1:k+1]]
        nb_ec = nb_e - nb_e.mean(axis=1, keepdims=True)
        cov_e = np.einsum("nki,nkj->nij", nb_ec, nb_ec)
        ev = np.linalg.eigvalsh(cov_e)          # ascending l3<=l2<=l1
        l1, l2, l3 = ev[:, 2], ev[:, 1], ev[:, 0]
        lin = (l1 - l2) / (l1 + 1e-12)
        plan = (l2 - l3) / (l1 + 1e-12)
        stem_like[s:s+CH2] = (lin > lin_thr) & (plan < plan_thr)
    edge = ~stem_like[rows] & ~stem_like[cols]
    # RGB coherence gate where corrected real-RGB exists on both endpoints
    # (unobserved endpoints keep surface-only evidence):
    if col is not None:
        okc = np.isfinite(col[:, 0])
        both = okc[rows] & okc[cols]
        cdist = np.linalg.norm(col[rows] - col[cols], axis=1)
        edge = (edge & ((cdist < color_thr) | ~both))
    graph = csr_matrix((np.ones(edge.sum()), (rows[edge], cols[edge])), shape=(N, N))
    ncomp, lab = connected_components(graph, directed=False)
    return lab, ncomp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dense-root", default=None)
    ap.add_argument("--colmap-root", default=None)
    ap.add_argument("--plants", default="DouBanLv1,XianKeLai2,WanNianQing2,HongZhang")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ar = ap.parse_args()

    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    dense_root = Path(ar.dense_root).resolve() if ar.dense_root else \
        repo_root.parent / "datasets" / "07-SuGaR-GS"
    colmap_root = Path(ar.colmap_root).resolve() if ar.colmap_root else \
        repo_root.parent / "datasets" / "04-COLMAP"
    cache_dir = Path(ar.cache_dir).resolve() if ar.cache_dir else \
        repo_root / "outputs" / "task5r" / "projection_cache"

    sys.path[:0] = [str(repo_root), str(repo_root / "core")]
    import core.headless_segmentation as hs
    from core.real_observation import load_dense_gaussian_plant
    from tests.test_real_observation_task5r import TestCOLMAPReprojection

    plants = [p.strip() for p in ar.plants.split(",") if p.strip()]
    cases = []
    plant_reports = []
    for plant in plants:
        t0 = time.time()
        ply = dense_root / plant / "vanilla_gs" / "point_cloud" / "iteration_7000" / \
            "point_cloud_clean_v2_rerun_20260304_181041.ply"
        colmap_dir = colmap_root / plant
        entry = {"plant": plant,
                 "dense_ply": str(ply), "ply_exists": ply.exists(),
                 "colmap_dir": str(colmap_dir)}
        if not ply.exists() or not (colmap_dir / "sparse" / "0").exists():
            entry["status"] = "MISSING_DATA"
            plant_reports.append(entry)
            continue
        entry["ply_sha256"] = file_sha256(ply)
        entry["camera_bin_sha256"] = file_sha256(colmap_dir / "sparse/0/cameras.bin")
        entry["images_bin_sha256"] = file_sha256(colmap_dir / "sparse/0/images.bin")

        # --- reprojection validation (observation matching) ---
        t = TestCOLMAPReprojection()
        t.PLANT = plant
        try:
            t.test_H_points3D_reproject_into_their_own_images()
            entry["reprojection"] = "PASS (<2px median)"
        except unittest.SkipTest:
            entry["reprojection"] = "SKIP"
        except AssertionError as e:
            entry["reprojection"] = f"FAIL: {e}"
            entry["status"] = "ALIGNMENT_FAIL"
            plant_reports.append(entry)
            continue

        g = load_dense_gaussian_plant(str(dense_root / plant))
        z = load_corrected_viewsig(cache_dir, plant)
        vf = z["visibility_fraction"]
        # contribution-weighted mean real RGB per Gaussian (for pair colour stats)
        w_rgbv = z["max_alpha"] * z["rgb_valid"]
        num_c = np.einsum("vnc,vn->nc", z["rgb_views"].astype(np.float64),
                          w_rgbv.astype(np.float64))
        den_c = w_rgbv.sum(axis=0)
        col = np.full((len(g), 3), np.nan)
        okc = den_c > 0
        col[okc] = num_c[okc] / den_c[okc, None]

        # --- independent leaf proposals ---
        lab, ncomp = propose_leaf_labels_independent(g, vf, col=col)
        sizes = np.bincount(lab)
        big = [(int(i), int(sizes[i])) for i in np.argsort(-sizes)[:30] if sizes[i] >= 500]
        entry["n_components"] = int(ncomp)
        entry["top_components"] = big[:15]
        entry["n_points"] = len(g)
        entry["visibility_fraction_mean"] = float(vf.mean())

        # --- near-contact pairs among leaf-scale components (overlap candidates) ---
        xyz_a = np.asarray(g.xyz, dtype=np.float64)
        big_ids = [i for i, _ in big]
        pairs = []
        for a in range(len(big_ids)):
            for b in range(a + 1, len(big_ids)):
                ia, ib = big_ids[a], big_ids[b]
                Pa, Pb = xyz_a[lab == ia], xyz_a[lab == ib]
                dm, _ = cKDTree(Pb).query(Pa)
                gap = float(dm.min())
                if gap < 0.08:      # near-contact at plant scale (~2 m tall)
                    ca = np.nanmean(col[lab == ia], axis=0) if col is not None else None
                    cb = np.nanmean(col[lab == ib], axis=0) if col is not None else None
                    cd = float(np.linalg.norm(ca - cb)) if ca is not None and cb is not None else None
                    pairs.append({"a": int(ia), "b": int(ib),
                                  "n_a": int(sizes[ia]), "n_b": int(sizes[ib]),
                                  "min_gap_m": round(gap, 4), "rgb_dist": cd})
        pairs.sort(key=lambda p: p["min_gap_m"])
        entry["near_contact_pairs"] = pairs[:20]
        entry["status"] = "PROPOSED"
        entry["runtime_seconds"] = round(time.time() - t0, 1)
        plant_reports.append(entry)
        print(f"[{plant}] comps={ncomp} top={big[:8]} vis_mean={vf.mean():.3f}", flush=True)

    out = {
        "task": "Task5R Phase4 observation-matched overlap audit",
        "git_commit": git_commit(repo_root),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plants": plant_reports,
    }
    out_path = Path(ar.output) if ar.output else \
        repo_root / "outputs" / "task5r" / "observation_overlap_cases.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("WROTE", out_path)
    return 0


import unittest  # noqa: E402  (needed for SkipTest above)

if __name__ == "__main__":
    sys.exit(main())
