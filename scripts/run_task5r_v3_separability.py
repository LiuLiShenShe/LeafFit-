#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 Phase D2/D3 — matched-edge separability runner.

DO NOT RUN until outputs/task5r_v3/human_verification.json exists with
approved=true (labels are PROPOSER_DIAGNOSTIC before that). The verdict gate
stops at BENCHMARK_NOT_HUMAN_VERIFIED anyway; this script exists so the
post-review rerun is one command.

Differences from the invalidated v2 runner:
  * viewsig loaded by EXACT cache_key from the v3 manifest;
  * candidate pairs come from candidate_benchmark.json (v3 viewsigs);
  * score-blind 1:1 distance matching (core.edge_matching) replaces raw
    prevalence ~0.9995 edge pools; 1:5 sensitivity variant also emitted;
  * statistics via core.task_stats (Cliff's delta = 2*AUROC-1, midrank ties,
    cluster bootstrap over contact pairs);
  * split assignment frozen: dev=DouBanLv1(+zero-pair plants),
    heldout=HongZhang,WangWenCao2;
  * per-edge scores persisted for bootstrap re-computation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent

DEV_PLANTS = {"DouBanLv1"}
HELDOUT_PLANTS = {"HongZhang", "WangWenCao2"}
MV_WEIGHTS = dict(vis=0.4, app=0.3, occ=0.3)   # frozen (unchanged from v2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dense-root", default=None)
    ap.add_argument("--colmap-root", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=int, nargs="+", default=[1, 5])
    ar = ap.parse_args()

    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    sys.path[:0] = [str(repo_root), str(repo_root / "core")]

    hv = repo_root / "outputs" / "task5r_v3" / "human_verification.json"
    if not hv.exists() or not json.loads(hv.read_text()).get("approved", False):
        print("REFUSING TO RUN: human_verification.json missing/not approved.\n"
              "Labels are PROPOSER_DIAGNOSTIC; obtain human review first.")
        return 2

    out_dir = repo_root / "outputs" / "task5r_v3"
    dense_root = Path(ar.dense_root) if ar.dense_root else \
        repo_root.parent / "datasets" / "07-SuGaR-GS"
    cand = json.loads((Path(ar.candidates) if ar.candidates else
                       out_dir / "candidate_benchmark.json").read_text())
    # NOTE: after human review, regenerate candidate_benchmark filtered to
    # reviewer KEEP/RELABEL decisions before invoking this script.

    from core.real_observation import load_dense_gaussian_plant
    from core.observation_identity import VISIBILITY_VERSION
    from core.geodesic_backends import _mv_edge_features
    from core import task_stats as ts
    from core import edge_matching as em
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_task5r_separability import local_pca_normals, surface_coherence
    from scripts.audit_real_overlap_cases import (
        propose_leaf_labels_independent)
    from scipy.spatial import cKDTree

    manifest_path = Path(ar.manifest) if ar.manifest else \
        out_dir / "corrected_viewsig_manifest.jsonl"
    latest = {}
    for line in manifest_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("visibility_version") == VISIBILITY_VERSION:
            latest[r["plant"]] = r

    rows_out = []
    pair_scores = []
    for plant in sorted({c["plant"] for c in cand["cases"]}):
        rec = latest[plant]
        zpath = Path(rec["viewsig_path"])
        if not zpath.is_absolute() and not zpath.exists():
            zpath = repo_root / rec["viewsig_path"]
        z = np.load(zpath, allow_pickle=False)
        g = load_dense_gaussian_plant(plant, dense_root=str(dense_root))
        visible = z["visible"]
        w_rgbv = z["max_alpha"] * z["rgb_valid"]
        num_c = np.einsum("vnc,vn->nc", z["rgb_views"].astype(np.float64),
                          w_rgbv.astype(np.float64))
        den_c = w_rgbv.sum(axis=0)
        col = np.full((len(g), 3), np.nan)
        okc = den_c > 0
        col[okc] = num_c[okc] / den_c[okc, None]
        appear = np.nan_to_num(col, nan=0.0).astype(np.float32)
        depth = np.nan_to_num(z["depth"], posinf=0.0).astype(np.float32)
        uv = np.nan_to_num(z["uv_ndc"].astype(np.float32), nan=0.0)
        xyz = np.asarray(g.xyz, dtype=np.float64)
        nrm = local_pca_normals(xyz)
        lab, _ = propose_leaf_labels_independent(
            g, z["visibility_fraction"], col=col, **cand["proposer_params"])

        for case in [c for c in cand["cases"] if c["plant"] == plant]:
            ids_a = np.where(lab == case["component_a"])[0]
            ids_b = np.where(lab == case["component_b"])[0]
            sel = np.concatenate([ids_a, ids_b])
            P = xyz[sel]
            tree = cKDTree(P)
            k = min(33, len(sel))
            dists, nbrs = tree.query(P, k=k)
            er = np.repeat(np.arange(len(sel)), k - 1)
            ec = nbrs[:, 1:].ravel()
            lo, hi = np.minimum(er, ec), np.maximum(er, ec)
            key = lo.astype(np.int64) * len(sel) + hi
            uniq = np.unique(key)
            r_abs = sel[uniq // len(sel)]
            c_abs = sel[uniq % len(sel)]

            is_within = lab[r_abs] == lab[c_abs]
            c_vis, c_app, c_occ = _mv_edge_features(
                r_abs, c_abs, visible, appear, depth, uv, visible.shape[0])
            sin_sc = surface_coherence(nrm, r_abs, c_abs)
            c_mv = (MV_WEIGHTS["vis"] * c_vis + MV_WEIGHTS["app"] * c_app +
                    MV_WEIGHTS["occ"] * c_occ)
            sc01 = np.clip(1.0 - sin_sc, 0.0, 1.0)
            d_edge = np.linalg.norm(xyz[c_abs] - xyz[r_abs], axis=1)
            ablations = {
                "R0_dist": -d_edge,
                "R1_c_vis": c_vis, "R2_c_app": c_app, "R3_c_occ": c_occ,
                "R4_c_mv": c_mv, "R5_surface": sc01,
                "R6_mv_and_surface": np.minimum(c_mv, sc01),
            }
            for name, sc in ablations.items():
                rows_out.append({
                    "case_id": case["case_id"], "plant": plant,
                    "split": ("dev" if plant in DEV_PLANTS else "heldout"),
                    "ablation": name,
                    "n_edges": int(len(is_within)),
                    "n_within": int(is_within.sum()),
                    "n_cross": int((~is_within).sum()),
                })
            # persist per-edge scores for matching + bootstrap re-computation
            for i in range(len(is_within)):
                pair_scores.append({
                    "case_id": case["case_id"], "gauss_a": int(r_abs[i]),
                    "gauss_b": int(c_abs[i]),
                    "distance_m": float(d_edge[i]),
                    "label": int(bool(is_within[i])),
                    **{n: float(s[i]) for n, s in ablations.items()},
                })
        print(f"[{plant}] scored", flush=True)

    csv_path = out_dir / "edge_scores_v3.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_scores[0].keys()))
        w.writeheader()
        w.writerows(pair_scores)

    # ---- distance matching + gates per ratio -------------------------------
    d_all = np.array([r["distance_m"] for r in pair_scores])
    l_all = np.array([bool(r["label"]) for r in pair_scores])
    cid = np.array([r["case_id"] for r in pair_scores])
    ga = np.array([r["gauss_a"] for r in pair_scores])
    gb = np.array([r["gauss_b"] for r in pair_scores])
    gates_report = {}
    for ratio in ar.ratio:
        me = em.match_within_for_cross(d_all, l_all, cid, ga, gb,
                                       seed=ar.seed, ratio=ratio)
        gates = em.matching_gates(me, -me.distance_m, me.distance_m)
        gates_report[f"1:{ratio}"] = {
            **gates,
            "matched_csv_sha256_pending": True,
        }
        mpath = out_dir / f"matched_edges_1to{ratio}.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "label", "gauss_a", "gauss_b",
                        "distance_m", "dist_bin_lo", "match_group",
                        "variant"])
            w.writerows(zip(me.case_id, me.label.astype(int), me.gauss_a,
                            me.gauss_b, me.distance_m, me.dist_bin_lo,
                            me.match_group, me.variant))
        gates_report[f"1:{ratio}"]["matched_csv"] = str(mpath)
    (out_dir / "matched_edges_gates.json").write_text(
        json.dumps({"matcher_version": em.MATCHER_VERSION,
                    "seed": ar.seed, "gates": gates_report}, indent=2))
    print(f"WROTE {csv_path} ({len(pair_scores)} edges) + matched edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
