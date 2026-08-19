#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a single controlled-overlap case: transform → segmentation → metrics.

Reads frozen benchmark_transforms.json, applies the specified transform to
leaf_a and/or leaf_b, saves the transformed PLY, runs headless segmentation,
and saves all outputs + failure metrics.

Usage:
    python scripts/run_overlap_case.py \
        --pair-key plant2_rubber_tree_pair_3_12 \
        --mode horizontal --severity H3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.headless_segmentation as hs  # noqa: E402
from gaussian_utils import GaussianData, save_gaussian_data_as_ply  # noqa: E402
from overlap_geometry import (  # noqa: E402
    transform_leaf_gaussians,
    axis_angle_to_matrix,
    compute_projected_overlap_fraction,
    compute_contact_fraction,
    compute_apex_gap,
    fit_leaf_pca,
)
from scipy.spatial import cKDTree  # noqa: E402

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_BLROOT = os.path.join(_REPO_ROOT, "outputs", "baseline")
_DATA = os.path.join(_REPO_ROOT, "data")


def load_transforms() -> dict:
    with open(os.path.join(_OUTROOT, "benchmark_transforms.json")) as f:
        return json.load(f)


def load_plant(plant: str):
    g = hs.load_gaussian_data(os.path.join(_DATA, f"{plant}.ply"))
    gc = hs.center_gaussians(g)
    labels = np.load(os.path.join(_BLROOT, plant, "labels.npy"))
    apexes = json.load(open(os.path.join(_BLROOT, plant, "apexes.json")))
    return {"g": g, "gc": gc, "labels": labels, "apexes": apexes}


def apply_transform_entry(
    g: GaussianData,
    transform: dict,
    leaf_indices: NDArray[np.intp],
) -> GaussianData:
    """Apply a transform entry (pivot, R, t) to specified leaf indices.

    If pivot is None, returns the input unchanged (identity / no-op).
    """
    if transform.get("pivot") is None:
        return g  # identity — no rotation needed
    R = np.asarray(transform["R"], dtype=np.float64)
    t = np.asarray(transform["t"], dtype=np.float64).ravel()
    pivot = np.asarray(transform["pivot"], dtype=np.float64).ravel()
    return transform_leaf_gaussians(g, leaf_indices, pivot, R, t)


def compute_construction_metrics(
    g: GaussianData,
    labels: NDArray,
    transform_a: dict,
    transform_b: dict,
    leaf_a_id: int,
    leaf_b_id: int,
    apexes: list,
) -> dict:
    """Compute pre-segmentation geometry metrics for this case."""
    xyz = np.asarray(g.xyz, dtype=np.float64)
    xa = xyz[labels == leaf_a_id]
    xb = xyz[labels == leaf_b_id]
    pca_a = fit_leaf_pca(xa)
    pca_b = fit_leaf_pca(xb)

    # spacing
    tree_a = cKDTree(xa)
    tree_b = cKDTree(xb)
    d_a, _ = tree_a.query(xa, k=min(7, len(xa)))
    d_b, _ = tree_b.query(xb, k=min(7, len(xb)))
    spacing_a = float(np.median(d_a[:, -1])) + 1e-12
    spacing_b = float(np.median(d_b[:, -1])) + 1e-12
    avg_spacing = (spacing_a + spacing_b) / 2.0

    # overlap (projected 2D Jaccard)
    ov = compute_projected_overlap_fraction(xa, xb)

    # contact
    cf = compute_contact_fraction(xa, xb, spacing=avg_spacing)

    # min cross-leaf distance
    tree_b_xyz = cKDTree(xb)
    d_a_to_b, _ = tree_b_xyz.query(xa, k=1)
    min_cross = float(d_a_to_b.min())

    out = {
        "projected_overlap_fraction": ov["overlap_fraction"],
        "contact_fraction": cf["contact_fraction"],
        "min_cross_leaf_distance_ratio": min_cross / avg_spacing,
        "spacing_a": spacing_a,
        "spacing_b": spacing_b,
        "centroid_dist": float(np.linalg.norm(xa.mean(0) - xb.mean(0))),
        "leaf_a_count": int(len(xa)),
        "leaf_b_count": int(len(xb)),
    }
    return out


def save_case_outputs(
    outdir: str,
    result: hs.HeadlessResult,
    g_transformed: GaussianData,
    labels: NDArray,
    apexes: list,
    construction_gt_labels: NDArray,
    transforms_used: dict,
    construction_metrics: dict,
    plant: str,
    pair_key: str,
    mode: str,
    severity: str,
    root_index: int,
    runtime: float,
) -> None:
    """Save all outputs for a single case."""
    os.makedirs(outdir, exist_ok=True)

    # Save transformed PLY (for visual inspection)
    save_gaussian_data_as_ply(
        os.path.join(outdir, "input_transformed.ply"),
        g_transformed,
    )

    # Save construction GT labels (same as original baseline labels)
    np.save(os.path.join(outdir, "construction_gt_labels.npy"), construction_gt_labels)

    # Save segmentation labels (convert found_segs list → flat (N,) int array)
    raw = np.zeros(result.N, dtype=np.int64)
    for k, seg in enumerate(result.found_segs):
        raw[seg] = k + 1
    np.save(os.path.join(outdir, "labels.npy"), raw)
    np.save(os.path.join(outdir, "raw_labels.npy"), raw)

    # Save geodesic fields
    np.save(os.path.join(outdir, "root_geodesic_single.npy"), result.root_geodesic_single)
    np.save(os.path.join(outdir, "root_geodesic_multisource.npy"), result.root_geodesic_multisource)
    np.save(os.path.join(outdir, "root_basin_indices.npy"), result.root_basin_indices)
    np.save(os.path.join(outdir, "sample_indices.npy"), result.sparse_indices)
    np.save(os.path.join(outdir, "temperature_field.npy"), result.temperature_field)

    # Save transforms used
    with open(os.path.join(outdir, "transforms.json"), "w") as f:
        json.dump(transforms_used, f, indent=2, ensure_ascii=False)

    # Save construction metrics
    with open(os.path.join(outdir, "construction_metrics.json"), "w") as f:
        json.dump(construction_metrics, f, indent=2, ensure_ascii=False)

    # Save apexes
    dtos = hs.build_dense_to_sparse_map(result)
    apexes_out = []
    for i, c in enumerate(result.final_cluster_results):
        tip = int(c["selected_tip"])
        dense_idx = int(result.sparse_indices[tip]) if tip < len(result.sparse_indices) else -1
        apexes_out.append({
            "id": i + 1,
            "selected_tip_sparse": tip,
            "gaussian_index": dense_idx,
            "type": c.get("type", "unknown"),
            "base_gaussian_index": c.get("base_gaussian_index"),
        })
    with open(os.path.join(outdir, "apexes.json"), "w") as f:
        json.dump(apexes_out, f, indent=2)

    # Save paths (from final_cluster_results, matching baseline format)
    dtos = hs.build_dense_to_sparse_map(result)
    paths_out = []
    for k, c in enumerate(result.final_cluster_results):
        dense_path = [int(x) for x in c.get("path", [])]
        paths_out.append({
            "id": k + 1,
            "path_sample_indices": [int(dtos[d]) for d in dense_path if 0 <= d < len(dtos)],
            "path_gaussian_indices": dense_path,
            "path_xyz": np.asarray(g_transformed.xyz[dense_path], dtype=float).tolist()
                if len(dense_path) > 0 else [],
        })
    with open(os.path.join(outdir, "paths.json"), "w") as f:
        json.dump(paths_out, f, indent=2, ensure_ascii=False)

    # Save tree (if available)
    tree = hs.build_tree(result)
    with open(os.path.join(outdir, "tree.json"), "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    # Save apex grouping
    apex_grouping = hs.build_apex_grouping(result)
    with open(os.path.join(outdir, "apex_grouping.json"), "w") as f:
        json.dump(apex_grouping, f, indent=2, ensure_ascii=False)

    # Save petioles
    petioles = hs.build_petioles(result)
    with open(os.path.join(outdir, "petioles.json"), "w") as f:
        json.dump(petioles, f, indent=2, ensure_ascii=False)

    # Save pre-grouping replay
    pre_grp = hs.build_pre_grouping_replay(result)
    with open(os.path.join(outdir, "pre_grouping_replay.json"), "w") as f:
        json.dump(pre_grp, f, indent=2, ensure_ascii=False)

    # Save config
    config = {
        "pair_key": pair_key,
        "mode": mode,
        "severity": severity,
        "plant": plant,
        "root_index": root_index,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save runtime
    runtime_info = {
        "total_seconds": runtime,
        "phases": result.phases,
    }
    with open(os.path.join(outdir, "runtime.json"), "w") as f:
        json.dump(runtime_info, f, indent=2)

    # Save status
    status = {"status": "completed"}
    with open(os.path.join(outdir, "status.json"), "w") as f:
        json.dump(status, f)


def run_one_case(
    pair_key: str,
    mode: str,
    severity: str,
    dry_run: bool = False,
) -> dict:
    """Run a single controlled-overlap case."""
    transforms = load_transforms()
    pk_data = transforms[pair_key]
    plant = pk_data["plant"]
    a_id = pk_data["leaf_a_id"]
    b_id = pk_data["leaf_b_id"]
    root_index = pk_data["root_index"]

    # Find the severity entry in the mode list
    mode_data = pk_data[mode]
    sev_entry = None
    for se in mode_data:
        if se["severity"] == severity:
            sev_entry = se
            break
    if sev_entry is None:
        raise ValueError(f"Severity {severity} not found in {mode} for {pair_key}")

    # Load plant data
    pd = load_plant(plant)
    gc = pd["gc"]
    labels = pd["labels"]
    apexes = pd["apexes"]

    # Get leaf indices
    leaf_a_indices = np.where(labels == a_id)[0]
    leaf_b_indices = np.where(labels == b_id)[0]

    # Apply transforms
    transform_a = sev_entry["leaf_a_transform"]
    transform_b = sev_entry["leaf_b_transform"]

    g_out = gc
    if transform_a.get("pivot") is not None:
        g_out = apply_transform_entry(g_out, transform_a, leaf_a_indices)
    if transform_b.get("pivot") is not None:
        g_out = apply_transform_entry(g_out, transform_b, leaf_b_indices)

    # Compute construction metrics (pre-segmentation geometry)
    construction_metrics = compute_construction_metrics(
        g_out, labels, transform_a, transform_b, a_id, b_id, apexes)

    if dry_run:
        return {
            "pair_key": pair_key,
            "mode": mode,
            "severity": severity,
            "construction_metrics": construction_metrics,
            "status": "dry_run",
        }

    # Run segmentation — gc is already centered; pass transformed g directly
    outdir = os.path.join(_OUTROOT, "controlled", pair_key, mode, severity)
    t0 = time.time()
    result = hs.run_headless_segmentation(
        hs.GaussianData(
            xyz=np.asarray(g_out.xyz, dtype=np.float32),
            rot=np.asarray(g_out.rot, dtype=np.float32),
            scale=np.asarray(g_out.scale, dtype=np.float32),
            opacity=np.asarray(g_out.opacity, dtype=np.float32),
            sh=np.asarray(g_out.sh, dtype=np.float32),
            nxnynz=np.asarray(g_out.nxnynz, dtype=np.float32),
            filter_3Ds=np.asarray(g_out.filter_3Ds, dtype=np.float32),
        ),
        root_index=root_index,
    )
    runtime = time.time() - t0

    # Save all outputs
    save_case_outputs(
        outdir, result, g_out, labels, apexes,
        construction_gt_labels=labels.copy(),
        transforms_used=sev_entry,
        construction_metrics=construction_metrics,
        plant=plant, pair_key=pair_key, mode=mode, severity=severity,
        root_index=root_index, runtime=runtime,
    )

    return {
        "pair_key": pair_key,
        "mode": mode,
        "severity": severity,
        "construction_metrics": construction_metrics,
        "num_segments": len(result.found_segs),
        "runtime": runtime,
        "status": "completed",
    }


def main() -> int:
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-key", required=True)
    parser.add_argument("--mode", choices=["horizontal", "vertical"], required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute geometry only, skip segmentation")
    args = parser.parse_args()

    try:
        result = run_one_case(args.pair_key, args.mode, args.severity, args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
