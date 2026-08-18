#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeafFit headless leaf-instance segmentation CLI.

Runs the OFFICIAL LeafFit automatic leaf-instance segmentation (paper Section 3.1)
on a single plant Gaussian PLY, WITHOUT the GUI / GPU rasterizer / private modules.

    python scripts/run_leaf_segmentation.py \
        --input  data/plant1_green_pepper.ply \
        --output outputs/baseline/plant1_green_pepper \
        --root-index 12345            # Mode A: fixed root Gaussian index
    # or
        --root auto                   # Mode B: official PCA auto root (frozen after)

All intermediate state is saved under <output>/:
  config.json, root.json, sample_indices.npy, root_geodesic_single.npy,
  root_geodesic_multisource.npy, root_basin_indices.npy, root_geodesic_stats.json,
  temperature_field.npy, apexes.json, paths.json, tree.json, apex_grouping.json,
  petioles.json, raw_labels.npy, labels.npy, segmentation_result.ply,
  segmentation_points.ply, runtime.json, metadata.json, status.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# Ensure repo root on path (for core/ imports) regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Only import the headless wrapper (never viewer/GL/rasterizer).
import core.headless_segmentation as hs  # noqa: E402
from gaussian_utils import GaussianData, save_gaussian_data_as_ply  # noqa: E402

# Deterministic per-leaf colors (fixed palette so runs are reproducible).
_LEAF_COLORS = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.10], [0.10, 0.10, 0.90],
    [0.90, 0.60, 0.10], [0.60, 0.10, 0.90], [0.10, 0.90, 0.90],
    [0.90, 0.90, 0.10], [0.95, 0.45, 0.75], [0.30, 0.60, 0.95],
    [0.95, 0.30, 0.45], [0.45, 0.85, 0.30], [0.70, 0.30, 0.20],
    [0.20, 0.70, 0.55], [0.80, 0.55, 0.20], [0.55, 0.20, 0.80],
], dtype=np.float32)
_STEM_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)
SH_C0 = 0.28209479177387814


def forbid_module_leak() -> None:
    """Assert no private/GPU/GUI module has leaked into the process."""
    leaked = [m for m in hs.FORBIDDEN_MODULES
              if m in sys.modules or any(k.startswith(m + ".") for k in sys.modules)]
    if leaked:
        raise RuntimeError(f"forbidden module leaked into headless process: {leaked}")


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Inverse SH-C0: SH DC coeff such that C0*coeff + 0.5 = rgb."""
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def git_commits() -> dict:
    """Record upstream + local commit hashes for provenance."""
    info = {}
    try:
        r = subprocess.run(["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        info["ours_commit"] = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        info["ours_commit"] = "unknown"
    # upstream HEAD (remote) from the local clone's refs (read-only)
    try:
        r = subprocess.run(["git", "-C", _REPO_ROOT, "rev-parse", "origin/main"],
                           capture_output=True, text=True, timeout=10)
        info["upstream_commit"] = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        info["upstream_commit"] = "unknown"
    return info


def write_config_json(outdir: str, res: hs.HeadlessResult) -> None:
    cfg = {
        "method": res.method,
        "frozen_baseline": {
            "Ns": hs.BASELINE_NS,
            "h": hs.BASELINE_H,
            "t_coef": hs.BASELINE_T_COEF,
            "root_basin_radius": hs.BASELINE_ROOT_BASIN,
            "opacity_threshold": hs.BASELINE_OPACITY_THRESHOLD,
            "center_gaussians": True,
        },
        "effective_runtime_params": {
            # official call-site values inside get_segment_mask / calculate_cluster_base_indices
            "tips_k": len(res.sparse_indices) // 64,
            "path_k": len(res.sparse_indices) // 32,
            "overlap_cut": 0.8,
            "triangle_cut": 0.62,
            "slack_eps": 1e-9,
            "max_iters": 50,
            "multi_tips_distance_factor": 1.25,
            "single_tip_distance_factor": 1.0,
            "euclidean_offset": 0.25,
            "geodesic_offset": 0.5,
            "petiole_method": "geodesic_tip_graph",
            "petiole_min_distance_threshold": 0.05,
            "petiole_tolerance_percentage": 0.02,
            "petiole_last_virtual_path_distance_factor": 2.5,
            "petiole_protection_period_ratio": 0.25,
        },
        "paper_vs_code_discrepancies": [
            {
                "param": "tips neighbor k",
                "paper_says": "Nk=512 (flat)",
                "code_does": f"len(sparse)//64 = {len(res.sparse_indices)//64}",
                "decision": "keep code behavior (baseline = upstream source)",
            },
            {
                "param": "path neighbor k",
                "paper_says": "512/128 (flat)",
                "code_does": f"len(sparse)//32 = {len(res.sparse_indices)//32}",
                "decision": "keep code behavior (baseline = upstream source)",
            },
            {
                "param": "apex grouping margin tau",
                "paper_says": "tau=0.5",
                "code_does": "triangle_cut=0.62",
                "decision": "keep code (upstream group_apexes_by_inequality default)",
            },
            {
                "param": "petiole epsilon",
                "paper_says": "epsilon=0.05",
                "code_does": "tolerance_percentage=0.02 at call site (signature default 0.005)",
                "decision": "keep call-site value 0.02 (official calculate_cluster_base_indices)",
            },
            {
                "param": "petiole delta",
                "paper_says": "delta=0.01",
                "code_does": "min_distance_threshold=0.05",
                "decision": "keep code (official call site)",
            },
            {
                "param": "petiole rho",
                "paper_says": "rho=0.25",
                "code_does": "protection_period_ratio=0.25",
                "decision": "matches paper",
            },
        ],
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def write_root_json(outdir: str, res: hs.HeadlessResult) -> None:
    root = {
        "root_index": int(res.root_idx),
        "root_source": res.root_source,
        "xyz": np.asarray(res.root_xyz, dtype=float).tolist(),
        "basin_radius": hs.BASELINE_ROOT_BASIN,
        "basin_size": int(len(res.root_basin_indices)),
        "note": "Mode A: exact Gaussian index. Mode B: official PCA-asymmetry auto root.",
    }
    with open(os.path.join(outdir, "root.json"), "w") as f:
        json.dump(root, f, indent=2, ensure_ascii=False)


def write_geodesic(outdir: str, res: hs.HeadlessResult) -> None:
    np.save(os.path.join(outdir, "root_geodesic_single.npy"), res.root_geodesic_single)
    np.save(os.path.join(outdir, "root_geodesic_multisource.npy"), res.root_geodesic_multisource)
    np.save(os.path.join(outdir, "root_basin_indices.npy"), res.root_basin_indices)
    np.save(os.path.join(outdir, "sample_indices.npy"), res.sparse_indices)
    np.save(os.path.join(outdir, "temperature_field.npy"), res.temperature_field)
    stats = {
        "single": {"min": float(res.root_geodesic_single.min()),
                   "max": float(res.root_geodesic_single.max()),
                   "mean": float(res.root_geodesic_single.mean()),
                   "finite": bool(np.isfinite(res.root_geodesic_single).all())},
        "multisource": {"min": float(res.root_geodesic_multisource.min()),
                        "max": float(res.root_geodesic_multisource.max()),
                        "mean": float(res.root_geodesic_multisource.mean()),
                        "finite": bool(np.isfinite(res.root_geodesic_multisource).all())},
        "note": "multisource min may be slightly negative near sources (potpourri3d behavior); "
                "official code consumes as-is.",
    }
    with open(os.path.join(outdir, "root_geodesic_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def write_apexes(outdir: str, res: hs.HeadlessResult) -> None:
    dtos = hs.build_dense_to_sparse_map(res)
    apexes = []
    for i, c in enumerate(res.final_cluster_results):
        tip = int(c["selected_tip"])
        apexes.append({
            "apex_id": i,
            "gaussian_index": tip,
            "sample_index": int(dtos[tip]),
            "xyz": np.asarray(res.gaussians_centered.xyz[tip], dtype=float).tolist(),
            "root_geodesic": float(res.root_geodesic_multisource[tip]),
            "type": c.get("type", "unknown"),
            "tips": [int(t) for t in c.get("tips", [])],
            "base_gaussian_index": int(c["base_idx"]) if c.get("base_idx") is not None else None,
        })
    with open(os.path.join(outdir, "apexes.json"), "w") as f:
        json.dump(apexes, f, indent=2, ensure_ascii=False)


def write_paths(outdir: str, res: hs.HeadlessResult) -> None:
    dtos = hs.build_dense_to_sparse_map(res)
    paths = []
    for i, c in enumerate(res.final_cluster_results):
        dense_path = [int(x) for x in c.get("path", [])]
        paths.append({
            "apex_id": i,
            "tip_gaussian_index": int(c["selected_tip"]),
            "path_sample_indices": [int(dtos[d]) for d in dense_path],
            "path_gaussian_indices": dense_path,
            "path_xyz": np.asarray(res.gaussians_centered.xyz[dense_path], dtype=float).tolist(),
            "base_gaussian_index": int(c["base_idx"]) if c.get("base_idx") is not None else None,
        })
    with open(os.path.join(outdir, "paths.json"), "w") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)


def write_tree(outdir: str, res: hs.HeadlessResult) -> None:
    with open(os.path.join(outdir, "tree.json"), "w") as f:
        json.dump(hs.build_tree(res), f, indent=2, ensure_ascii=False)


def write_apex_grouping(outdir: str, res: hs.HeadlessResult) -> None:
    with open(os.path.join(outdir, "apex_grouping.json"), "w") as f:
        json.dump(hs.build_apex_grouping(res), f, indent=2, ensure_ascii=False)


def write_petioles(outdir: str, res: hs.HeadlessResult) -> None:
    with open(os.path.join(outdir, "petioles.json"), "w") as f:
        json.dump(hs.build_petioles(res), f, indent=2, ensure_ascii=False)


def write_pre_grouping_replay(outdir: str, res: hs.HeadlessResult) -> None:
    """Pre-grouping diagnostic replay (Task 2 Figure-13-style analysis). Additive;
    does NOT change the segmentation outputs (labels/status/root/geodesics)."""
    with open(os.path.join(outdir, "pre_grouping_replay.json"), "w") as f:
        json.dump(hs.build_pre_grouping_replay(res), f, indent=2, ensure_ascii=False)


def write_labels(outdir: str, res: hs.HeadlessResult) -> None:
    """Write raw_labels.npy (official) + labels.npy (unified 0=stem, 1..K=leaf)."""
    raw = np.zeros(res.N, dtype=np.int64)     # 0 = unassigned/stem (official)
    for k, seg in enumerate(res.found_segs):
        raw[seg] = k + 1
    np.save(os.path.join(outdir, "raw_labels.npy"), raw)
    # unified: stem=0, leaf instances 1..K (same here since official uses 0=unassigned)
    labels = raw.copy()
    np.save(os.path.join(outdir, "labels.npy"), labels)


def write_colored_ply(outdir: str, res: hs.HeadlessResult) -> None:
    """Write segmentation_result.ply (full GaussianData with instance colors in SH-DC)."""
    g = res.gaussians_centered
    # build per-point RGB
    rgb = np.tile(_STEM_COLOR, (res.N, 1)).astype(np.float32)
    for k, seg in enumerate(res.found_segs):
        color = _LEAF_COLORS[k % len(_LEAF_COLORS)]
        rgb[seg] = color
    # encode into a fresh GaussianData's SH-DC so PLY renders with instance colors
    sh = g.sh.copy()
    sh[:, :3] = rgb_to_sh_dc(rgb)
    colored = GaussianData(xyz=g.xyz.copy(), rot=g.rot.copy(), scale=g.scale.copy(),
                           opacity=g.opacity.copy(), sh=sh, nxnynz=g.nxnynz.copy(),
                           filter_3Ds=g.filter_3Ds.copy())
    save_gaussian_data_as_ply(os.path.join(outdir, "segmentation_result.ply"), colored)
    # minimal xyz+rgb PLY for easy viewing
    write_points_ply(os.path.join(outdir, "segmentation_points.ply"),
                     np.asarray(g.xyz, dtype=np.float32), rgb)


def write_points_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyElement, PlyData
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    n = len(xyz)
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"] = np.clip(rgb[:, 0] * 255, 0, 255).astype(np.uint8)
    arr["green"] = np.clip(rgb[:, 1] * 255, 0, 255).astype(np.uint8)
    arr["blue"] = np.clip(rgb[:, 2] * 255, 0, 255).astype(np.uint8)
    PlyData([PlyElement.describe(arr, "vertex")], text=True).write(path)


def write_runtime(outdir: str, res: hs.HeadlessResult) -> None:
    with open(os.path.join(outdir, "runtime.json"), "w") as f:
        json.dump({"phases": {k: round(v, 4) for k, v in res.phases.items()},
                   "total_sec": round(res.phases.get("total", 0.0), 4)}, f, indent=2)


def write_metadata(outdir: str, plant_name: str, res: hs.HeadlessResult,
                   input_path: str, status: dict) -> None:
    import importlib.metadata as md
    commits = git_commits()
    meta = {
        "plant": plant_name,
        "input_ply": input_path,
        "num_gaussians": res.N,
        "method": res.method,
        "root_index": int(res.root_idx),
        "root_source": res.root_source,
        "num_apexes": len(res.final_cluster_results),
        "num_leaf_instances": res.num_leaves,
        "centered": True,
        "coordinate_space": "centered (official pipeline centers before segmentation)",
        "status": status,
        "python": sys.version.split()[0],
        "numpy": md.version("numpy"),
        "potpourri3d": md.version("potpourri3d"),
        "fpsample": md.version("fpsample"),
        "scipy": md.version("scipy"),
        "sklearn": md.version("scikit-learn"),
        **commits,
    }
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def write_status(outdir: str, status: dict) -> None:
    with open(os.path.join(outdir, "status.json"), "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LeafFit Section 3.1 headless segmentation")
    ap.add_argument("--input", required=True, help="input Gaussian PLY path")
    ap.add_argument("--output", required=True, help="output directory (created)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--root-index", type=int, help="Mode A: exact Gaussian index (fixed root)")
    grp.add_argument("--root", choices=["auto"], help="Mode B: official PCA auto root")
    ap.add_argument("--save-debug", action="store_true", help="(all debug artifacts always written; kept for parity)")
    ap.add_argument("--no-pre-grouping-replay", action="store_true",
                    help="skip the additive pre-grouping diagnostic replay json")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (informational; Mode A deterministic)")
    ap.add_argument("--device", default="cpu", help="CPU only (documented); no GPU path")
    args = ap.parse_args(argv)

    forbid_module_leak()

    outdir = args.output
    os.makedirs(outdir, exist_ok=True)
    plant_name = os.path.splitext(os.path.basename(args.input))[0]

    t0 = time.time()
    status = {"status": "RUNNING", "error": None, "failure_stage": None}
    try:
        # 1) load (GL-free loader)
        t_load = time.time()
        g = hs.load_gaussian_data(args.input)
        load_sec = time.time() - t_load

        # root selection mode
        root_index = args.root_index if args.root is None else None
        res = hs.run_headless_segmentation(g, root_index=root_index, seed=args.seed)
        res.phases["load"] = load_sec

        # 2) write all artifacts
        t_w = time.time()
        write_config_json(outdir, res)
        write_root_json(outdir, res)
        write_geodesic(outdir, res)
        write_apexes(outdir, res)
        write_paths(outdir, res)
        write_tree(outdir, res)
        write_apex_grouping(outdir, res)
        write_petioles(outdir, res)
        if not args.no_pre_grouping_replay:
            write_pre_grouping_replay(outdir, res)
        write_labels(outdir, res)
        write_colored_ply(outdir, res)
        res.phases["writers"] = time.time() - t_w
        write_runtime(outdir, res)

        # 3) segmentation status semantics
        if res.num_leaves == 0:
            status = {"status": "SEGMENTATION_FAILED_NO_LEAVES",
                      "error": "0 leaf instances produced by official algorithm",
                      "failure_stage": "segment"}
        else:
            status = {"status": "SUCCESS", "error": None, "failure_stage": None}
        write_status(outdir, status)
        write_metadata(outdir, plant_name, res, args.input, status)

        print(f"[{status['status']}] {plant_name}: N={res.N} root={res.root_idx} "
              f"({res.root_source}) leaves={res.num_leaves} apexes={len(res.final_cluster_results)} "
              f"total={res.phases.get('total', time.time()-t0):.2f}s -> {outdir}")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        status = {"status": "FAILED", "error": str(e), "failure_stage": "unknown",
                  "traceback": traceback.format_exc()}
        with open(os.path.join(outdir, "error.log"), "w") as f:
            f.write(traceback.format_exc())
        write_status(outdir, status)
        print(f"[FAILED] {plant_name}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
