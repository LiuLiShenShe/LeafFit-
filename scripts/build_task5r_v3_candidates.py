#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 Phase D1 — candidate benchmark + HUMAN REVIEW PACKAGE.

Runs the FROZEN independent proposer (audit_real_overlap_cases.
propose_leaf_labels_independent, parameters unchanged) on the v3 corrected
viewsigs, builds near-contact candidate pairs, and writes:

  outputs/task5r_v3/candidate_benchmark.json      machine-readable candidates
  outputs/task5r_v3/benchmark_review_queue.csv    per-pair review queue
  outputs/task5r_v3/benchmark_review_guide.md     review instructions
  outputs/task5r_v3/review_crops/<plant>/         multi-view contact-zone crops
                                                  (heavy, gitignored)

LABEL DISCIPLINE: these are PROPOSER_DIAGNOSTIC candidate labels. They are NOT
Gold, NOT a formal benchmark, until a human reviewer fills human_verification.json.
This script NEVER sets human_verified=true.
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

# frozen proposer parameters (unchanged from audit_real_overlap_cases.py)
PROPOSER_PARAMS = dict(k=16, sin_thr=0.35, lin_thr=0.6, plan_thr=0.25,
                       color_thr=30.0)
NEAR_CONTACT_M = 0.08          # frozen: near-contact threshold at plant scale
MIN_COMPONENT_POINTS = 500     # frozen: leaf-scale component floor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dense-root", default=None)
    ap.add_argument("--colmap-root", default=None)
    ap.add_argument("--plants",
                    default="DouBanLv1,XianKeLai2,WanNianQing2,HongZhang,"
                            "WangWenCao2,CaoMei1")
    ap.add_argument("--manifest", default=None,
                    help="v3 corrected_viewsig_manifest.jsonl")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ar = ap.parse_args()

    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    sys.path[:0] = [str(repo_root), str(repo_root / "core")]

    dense_root = Path(ar.dense_root) if ar.dense_root else \
        repo_root.parent / "datasets" / "07-SuGaR-GS"
    colmap_root = Path(ar.colmap_root) if ar.colmap_root else \
        repo_root.parent / "datasets" / "04-COLMAP"
    out_dir = Path(ar.output_dir).resolve() if ar.output_dir else \
        repo_root / "outputs" / "task5r_v3"
    manifest_path = Path(ar.manifest) if ar.manifest else \
        out_dir / "corrected_viewsig_manifest.jsonl"
    cache_dir = Path(ar.cache_dir) if ar.cache_dir else \
        out_dir / "projection_cache"

    from core.real_observation import load_dense_gaussian_plant
    from core.observation_identity import VISIBILITY_VERSION, git_commit
    from scripts.audit_real_overlap_cases import propose_leaf_labels_independent
    from scipy.spatial import cKDTree

    # exact cache-key lookup from the v3 manifest (NEVER glob[-1])
    latest = {}
    for line in manifest_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("visibility_version") == VISIBILITY_VERSION:
            latest[r["plant"]] = r          # last entry per plant wins

    crops_dir = out_dir / "review_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    plants = [p.strip() for p in ar.plants.split(",") if p.strip()]
    cases = []
    plant_summaries = {}

    for plant in plants:
        rec = latest.get(plant)
        if rec is None:
            print(f"[{plant}] NO_V3_VIEWSIG in manifest — skipping")
            continue
        zpath = repo_root / rec["viewsig_path"]
        if not str(rec["viewsig_path"]).startswith(str(repo_root)):
            zpath = Path(rec["viewsig_path"])
        if not zpath.exists():
            print(f"[{plant}] missing npz {zpath}")
            continue
        z = np.load(zpath, allow_pickle=False)
        vf = z["visibility_fraction"]

        g = load_dense_gaussian_plant(plant, dense_root=str(dense_root))
        w_rgbv = z["max_alpha"] * z["rgb_valid"]
        num_c = np.einsum("vnc,vn->nc", z["rgb_views"].astype(np.float64),
                          w_rgbv.astype(np.float64))
        den_c = w_rgbv.sum(axis=0)
        col = np.full((len(g), 3), np.nan)
        okc = den_c > 0
        col[okc] = num_c[okc] / den_c[okc, None]

        lab, ncomp = propose_leaf_labels_independent(g, vf, col=col,
                                                     **PROPOSER_PARAMS)
        sizes = np.bincount(lab)
        big_ids = [int(i) for i in np.argsort(-sizes)[:30]
                   if sizes[i] >= MIN_COMPONENT_POINTS]
        xyz_a = np.asarray(g.xyz, dtype=np.float64)

        pairs = []
        trees = {i: cKDTree(xyz_a[lab == i]) for i in big_ids}
        pts = {i: xyz_a[lab == i] for i in big_ids}
        for ai in range(len(big_ids)):
            for bi in range(ai + 1, len(big_ids)):
                ia, ib = big_ids[ai], big_ids[bi]
                dm, _ = trees[ib].query(pts[ia])
                gap = float(dm.min())
                if gap < NEAR_CONTACT_M:
                    ca = np.nanmean(col[lab == ia], axis=0)
                    cb = np.nanmean(col[lab == ib], axis=0)
                    cd = float(np.linalg.norm(ca - cb)) \
                        if np.isfinite(ca).all() and np.isfinite(cb).all() else None
                    pairs.append({
                        "plant": plant,
                        "component_a": int(ia), "component_b": int(ib),
                        "n_points_a": int(sizes[ia]),
                        "n_points_b": int(sizes[ib]),
                        "min_gap_m": round(gap, 4), "rgb_dist": cd,
                    })
        pairs.sort(key=lambda p: p["min_gap_m"])
        pairs = pairs[:20]
        for rank, p in enumerate(pairs):
            p["case_id"] = f"{plant}_c{p['component_a']}_c{p['component_b']}"
            p["proposer_rank"] = rank
        cases.extend(pairs)

        # contact-region crops for the review queue
        pdir = crops_dir / plant
        pdir.mkdir(parents=True, exist_ok=True)
        n_crops = 0
        try:
            from core.real_observation import load_dense_observations
            obs = load_dense_observations(plant, colmap_root=str(colmap_root))
            for p in pairs[:10]:
                mid = 0.5 * (pts[p["component_a"]].mean(0) +
                             pts[p["component_b"]].mean(0))
                made = []
                from PIL import Image
                for vi in np.linspace(0, obs.n_views - 1, 24).astype(int):
                    try:
                        im = Image.open(obs.image_paths[vi]).convert("RGB")
                    except Exception:
                        continue
                    W, H = im.size
                    d = max(1, int(round(max(W, H) / 1024)))
                    cam = obs.rt[vi] @ np.hstack([mid, 1.0])
                    if cam[2] <= 1e-6:
                        continue
                    px = obs.K[vi][0, 0] * cam[0] / cam[2] + obs.K[vi][0, 2]
                    py = obs.K[vi][1, 1] * cam[1] / cam[2] + obs.K[vi][1, 2]
                    sx, sy = px / d, py / d
                    sw, sh = W // d, H // d
                    if not (60 < sx < sw - 60 and 60 < sy < sh - 60):
                        continue
                    half = max(48, int(round(0.12 * min(sw, sh))))
                    box = (int(sx - half), int(sy - half),
                           int(sx + half), int(sy + half))
                    crop = im.crop((box[0] * d, box[1] * d,
                                    box[2] * d, box[3] * d))
                    crop.thumbnail((512, 512))
                    name = pdir / f"{p['case_id']}_v{obs.names[vi]}"
                    crop.save(name.with_suffix(".jpg"), quality=85)
                    made.append(name.name)
                    if len(made) >= 6:
                        break
                p["review_crops"] = made
                n_crops += len(made)
        except Exception as e:
            p["review_crops_error"] = str(e)
        plant_summaries[plant] = {
            "n_components": int(ncomp),
            "n_leaf_scale_components": len(big_ids),
            "n_candidate_pairs": len(pairs),
            "n_review_crops": n_crops,
        }
        print(f"[{plant}] comps={ncomp} leaf-scale={len(big_ids)} "
              f"pairs={len(pairs)} crops={n_crops}", flush=True)

    # ---- candidate benchmark (machine-readable; PROPOSER_DIAGNOSTIC tier) --
    bench = {
        "tier": "PROPOSER_DIAGNOSTIC",
        "human_verified": False,
        "note": "Candidate labels from the frozen independent proposer. "
                "NOT Gold. Requires manual review of benchmark_review_queue.csv.",
        "proposer_params": PROPOSER_PARAMS,
        "near_contact_m": NEAR_CONTACT_M,
        "min_component_points": MIN_COMPONENT_POINTS,
        "viewsig_version": VISIBILITY_VERSION,
        "git_commit": git_commit(repo_root),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plant_summaries": plant_summaries,
        "cases": cases,
    }
    (out_dir / "candidate_benchmark.json").write_text(json.dumps(bench, indent=2))

    # ---- review queue CSV ---------------------------------------------------
    with open(out_dir / "benchmark_review_queue.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "plant", "component_a", "component_b",
                    "n_points_a", "n_points_b", "min_gap_m", "rgb_dist",
                    "suggestion", "reviewer_decision",
                    "reviewer_note", "review_crops"])
        for p in cases:
            suggestion = ("KEEP" if p["min_gap_m"] < 0.03 and
                          p["n_points_a"] >= MIN_COMPONENT_POINTS and
                          p["n_points_b"] >= MIN_COMPONENT_POINTS
                          else "UNCERTAIN")
            w.writerow([p["case_id"], p["plant"], p["component_a"],
                        p["component_b"], p["n_points_a"], p["n_points_b"],
                        p["min_gap_m"],
                        round(p["rgb_dist"], 1) if p["rgb_dist"] is not None
                        else "NA",
                        suggestion, "", "", ";".join(p.get("review_crops", []))])

    # ---- review guide -------------------------------------------------------
    guide = f"""# Task5R-v3 基准人工复核指南

生成时间: {bench['timestamp']} | viewsig: `{VISIBILITY_VERSION}` |
提议器参数(冻结): `{PROPOSER_PARAMS}`

## 你在审什么
独立提议器（normal+color 区域生长，与 LeafFit 热求解器零共享代码路径）
把每株植物的稠密 3DGS 点云切成连通组件，并提出"近接触叶片对"候选
（min_gap < {NEAR_CONTACT_M} m）。这些候选标签目前只是 **PROPOSER_DIAGNOSTIC**，
不是 Gold、不是正式 benchmark。

## 复核步骤
1. 打开 `benchmark_review_queue.csv`，逐行查看 `review_crops` 列列出的截图
   （位于 `review_crops/<plant>/`，每个 case 至多 6 张多视角接触区放大图）。
2. 判断两个组件是否为**两片真实存在的不同叶片**的接触：
   - 同一片叶子被错误切成两块 → RELABEL（合并）
   - 其中一块其实是茎/土/盆/漂浮噪声 → REJECT
   - 接触关系真实但几何上不构成"上下遮挡接触" → UNCERTAIN
   - 两片真实叶片的真实近接触 → KEEP
3. 在 `reviewer_decision` 列填 `KEEP` / `REJECT` / `RELABEL` / `UNCERTAIN`，
   需要时在 `reviewer_note` 写一句话理由。
4. 全部复核完成后，把本 CSV 转成交付格式并填写
   `outputs/task5r_v3/human_verification.json`：

```json
{{
  "approved": true,
  "reviewer": "<姓名>",
  "date": "<YYYY-MM-DD>",
  "queue_csv_sha256": "<对已复核 CSV 的 sha256>",
  "decisions_summary": {{"KEEP": N, "REJECT": N, "RELABEL": N, "UNCERTAIN": N}}
}}
```

`suggestions` 列只是机器建议，**不构成决定**；请以截图为准。
未完成人工复核前，Task5R-v3 的正式判决将停在
BENCHMARK_NOT_HUMAN_VERIFIED（task6_allowed=false）。
"""
    (out_dir / "benchmark_review_guide.md").write_text(guide)
    print(f"WROTE {out_dir/'candidate_benchmark.json'} "
          f"({len(cases)} candidate cases)")
    print(f"WROTE {out_dir/'benchmark_review_queue.csv'}")
    print(f"WROTE {out_dir/'benchmark_review_guide.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
