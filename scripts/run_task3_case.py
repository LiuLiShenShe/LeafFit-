#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a single Task 3 case with an arbitrary geodesic backend.

Extends run_overlap_case.run_one_case / save_case_outputs to:
  * accept a backend config (kind + params) passed via --backend / --lambda-n /
    --lambda-t / --tau-d / --tau-t / --k / --mutual / --feature-set,
  * inject the chosen backend via solver_factory into run_headless_segmentation,
  * freeze the root basin from Task 2 (outputs/task2/controlled/<pair>/<mode>/<sev>/
    root_basin_indices.npy) via frozen_root_basin_indices so the only variable
    between backends is the geodesic solver,
  * write outputs to outputs/task3/test|dev|ablation/<pair>/<mode>/<sev>/<backend>,

Reuses run_overlap_case.apply_transform_entry + compute_construction_metrics +
save_case_outputs verbatim (geometry/methods untouched).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core"),
           os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.headless_segmentation as hs  # noqa: E402
from run_overlap_case import (  # noqa: E402
    load_plant, apply_transform_entry, compute_construction_metrics,
    save_case_outputs,
)
import compute_failure_metrics as cfm  # noqa: E402
from geodesic_backends import (  # noqa: E402
    EuclideanGraphBackend, SurfaceAwareGraphBackend,
)

_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")
_T2TRANSFORMS = os.path.join(_REPO_ROOT, "outputs", "task2", "benchmark_transforms.json")
_T2CONT = os.path.join(_REPO_ROOT, "outputs", "task2", "controlled")


def backend_factory(kind: str, cfg: dict):
    """Return a solver_factory(points, new_gaussians) closure for run_headless_segmentation."""
    holder = {"backend": None}  # run_case reads the instantiated backend for diagnostics
    def factory(points, new_gaussians):
        if kind == "heat":
            import potpourri3d as pp3d
            b = pp3d.PointCloudHeatSolver(points, t_coef=1e+8)
            holder["backend"] = None  # heat has no graph diagnostics
            return b
        if kind == "euclidean":
            b = EuclideanGraphBackend(points, k=cfg.get("k", 32),
                                      mutual=cfg.get("mutual", False))
            holder["backend"] = b
            return b
        fs = cfg.get("feature_set", "G5")
        b = SurfaceAwareGraphBackend(
            points, new_gaussians,
            k=cfg.get("k", 64),
            lambda_n=cfg.get("lambda_n", 1.0),
            lambda_t=cfg.get("lambda_t", 2.0),
            p=cfg.get("p", 2.0),
            tau_d=cfg.get("tau_d", 3.0),
            tau_t=cfg.get("tau_t", 0.5),
            mutual=cfg.get("mutual", False),
            feature_set=fs)
        holder["backend"] = b
        return b
    factory.holder = holder
    return factory


def resolve_transforms(mode: str, severity: str) -> str:
    """Return path to the transforms file that defines the case.

    Uses fine_boundary_transforms.json when severity is HF1-4/VF1-4, else the
    task2 benchmark_transforms.json (H0/V0/H1-4/V1-4).
    """
    fine = os.path.join(_T3, "fine_boundary_transforms.json")
    coarse = _T2TRANSFORMS
    # fine severities are HF1-HF4 / VF1-VF4 (severity[2] is the digit:
    # "HF1"[2]="1", "VF3"[2]="3"; severity[1] is "F"/"V" and never a digit)
    if severity.startswith("HF") and severity[2] in "1234":
        return fine if os.path.exists(fine) else coarse
    if severity.startswith("VF") and severity[2] in "1234":
        return fine if os.path.exists(fine) else coarse
    return coarse


def load_case_transforms(pair_key: str, mode: str, severity: str):
    path = resolve_transforms(mode, severity)
    with open(path) as f:
        transforms = json.load(f)
    return transforms[pair_key]


def load_frozen_basin(pair_key: str, mode: str, severity: str):
    """Frozen root basin from Task 2 (byte-identical across backends).

    For fine severities (HF1-4/VF1-4) the basin is taken from the matching
    coarse boundary (HF1<-H1, VF1<-V1 ...) since the basin definition is a
    function of the root only, which does not change with the fine geometry.
    """
    # map fine -> coarse boundary for basin lookup (basin is root-defined, not
    # geometry-defined, so any V0/H0 basin in the pair is byte-identical)
    basin_path = os.path.join(_T2CONT, pair_key, mode, "V0" if mode == "vertical" else "H0",
                              "root_basin_indices.npy")
    if os.path.exists(basin_path):
        return np.load(basin_path)
    return None


def compute_task3_metrics(pair_key: str, mode: str, severity: str, case_dir: str) -> dict:
    """Task-3-aware failure metrics.

    Uses the fine/coarse transforms (pair-level GT leaf ids) and a
    self-contained shortcut computation reading THIS case's paths.json and the
    task2 V0 heat reference.
    """
    pk_data = load_case_transforms(pair_key, mode, severity)
    gt_leaf_a = pk_data["leaf_a_id"]
    gt_leaf_b = pk_data["leaf_b_id"]
    data = cfm.load_case(case_dir)
    labels, gt_labels, apexes = data["labels"], data["gt_labels"], data["apexes"]

    instance_metrics = cfm.compute_hungarian_iou(labels, gt_labels)
    pair_metrics = cfm.compute_pair_metrics(labels, gt_labels, gt_leaf_a, gt_leaf_b)
    geo_path = os.path.join(case_dir, "root_geodesic_multisource.npy")
    root_geodesic = np.load(geo_path) if os.path.exists(geo_path) else None
    geodesic_metrics = cfm.compute_geodesic_metrics(
        labels, gt_labels, root_geodesic, apexes, gt_leaf_a, gt_leaf_b, pair_key)

    shortcut_metrics = None
    if mode == "vertical":
        upper_id = pk_data["upper_leaf_id"]
        lower_id = pk_data["lower_leaf_id"]
        shortcut_metrics = _task3_shortcut(labels, gt_labels, root_geodesic,
                                           apexes, upper_id, lower_id,
                                           pair_key, severity, case_dir)

    first_failure_stage = "NONE"
    if geodesic_metrics["wrong_grouping"]:
        first_failure_stage = "GROUPING"
    elif geodesic_metrics["reference_apex_recall"] < 1.0:
        first_failure_stage = "APEX"
    elif instance_metrics["mIoU"] < 0.9:
        first_failure_stage = "SEGMENTATION"

    metrics = {
        "pair_key": pair_key,
        "mode": mode,
        "severity": severity,
        "instance": instance_metrics,
        "pair_local": pair_metrics,
        "geodesic": geodesic_metrics,
        "first_failure_stage": first_failure_stage,
        "dominant_failure_stage": first_failure_stage,
        "construction": {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in data["construction"].items()},
    }
    if shortcut_metrics is not None:
        metrics["shortcut"] = shortcut_metrics
    return metrics


def _task3_shortcut(labels, gt_labels, root_geodesic, apexes,
                    upper_id, lower_id, pair_key, severity, case_dir):
    """Self-contained vertical shortcut (reads this case's own paths.json)."""
    upper_apex_idx = None
    sp = json.load(open(os.path.join(_REPO_ROOT, "outputs", "task2", "source_pairs.json")))
    plant_name = pair_key.split("_pair_")[0]
    for p in sp["pairs"]:
        if p["plant"] != plant_name:
            continue
        if p["leaf_a_id"] == upper_id:
            upper_apex_idx = p["leaf_a"]["apex_gaussian_index"]
            break
        elif p["leaf_b_id"] == upper_id:
            upper_apex_idx = p["leaf_b"]["apex_gaussian_index"]
            break

    if upper_apex_idx is None or root_geodesic is None or upper_apex_idx >= len(root_geodesic):
        return {"shortcut_ratio": None, "reason": "upper_apex_not_found"}

    # Reference V0: prefer the SAME backend's own V0 field (fair for graph
    # backends, whose Dijkstra field differs from heat even on clean V0 —
    # otherwise graph backends are penalized by field shape, not by actual
    # shortcut). Fall back to the Task 2 heat V0 reference.
    v0_dir = case_dir.replace(os.sep + severity + os.sep, os.sep + "V0" + os.sep)
    v0_path = os.path.join(v0_dir, "root_geodesic_multisource.npy")
    if not os.path.exists(v0_path):
        v0_path = os.path.join(_T2CONT, pair_key, "vertical", "V0",
                               "root_geodesic_multisource.npy")
    if not os.path.exists(v0_path):
        return {"shortcut_ratio": None, "reason": "no_v0_reference"}
    d_v0 = float(np.load(v0_path)[upper_apex_idx])
    d_case = float(root_geodesic[upper_apex_idx])
    shortcut_ratio = d_case / d_v0 if d_v0 > 0 else None

    has_cross_leaf_path = False
    paths_path = os.path.join(case_dir, "paths.json")
    if os.path.exists(paths_path):
        with open(paths_path) as f:
            paths = json.load(f)
        for path in paths:
            pg = np.asarray(path.get("path_gaussian_indices", []), dtype=int)
            if pg.size == 0:
                continue
            gt_along = gt_labels[pg]
            if ((gt_along == upper_id).sum() > 0 and
                    (gt_along == lower_id).sum() > 0):
                has_cross_leaf_path = True
                break

    upper_mask = gt_labels == upper_id
    lower_mask = gt_labels == lower_id
    shared = set(int(x) for x in labels[upper_mask] if x > 0) & \
             set(int(x) for x in labels[lower_mask] if x > 0)
    lower_med = float(np.median(root_geodesic[lower_mask])) if lower_mask.sum() > 0 else None
    upper_d = root_geodesic[upper_mask]
    below_lower = int((upper_d < lower_med).sum()) if lower_med else 0

    return {
        "shortcut_ratio": float(shortcut_ratio) if shortcut_ratio is not None else None,
        "shortcut_confirmed": (shortcut_ratio is not None and shortcut_ratio < 1.0 - 1e-6),
        "cross_leaf_path": bool(has_cross_leaf_path),
        "cross_leaf_merge": bool(len(shared) > 0),
        "shared_instances": sorted(shared),
        "upper_apex_idx": int(upper_apex_idx),
        "upper_below_lower_ratio": below_lower / max(len(upper_mask), 1),
        "d_case": float(d_case),
        "d_v0": float(d_v0),
    }


def _backend_dirname(backend: str, cfg: dict) -> str:
    """Encode the config into the output directory name so distinct configs
    don't clobber each other.

    e.g. surface/G4_k64_td3.0_tt0.5_mFalse  (G4), euclidean/k32_mFalse, heat
    """
    if backend == "heat":
        return "heat"
    if backend == "euclidean":
        return f"euclidean/k{cfg.get('k', 32)}_m{str(cfg.get('mutual', False))}"
    fs = cfg.get("feature_set", "G5")
    k = cfg.get("k", 64)
    m = str(cfg.get("mutual", False))
    if fs in ("G4", "G0"):
        return f"surface/{fs}_k{k}_m{m}_td{cfg.get('tau_d', 3.0)}_tt{cfg.get('tau_t', 0.5)}"
    # G1/G2/G3/G5/G6 with penalty terms
    ln = cfg.get("lambda_n", 1.0)
    lt = cfg.get("lambda_t", 2.0)
    return (f"surface/{fs}_k{k}_m{m}_ln{ln}_lt{lt}_"
            f"td{cfg.get('tau_d', 3.0)}_tt{cfg.get('tau_t', 0.5)}")


def run_case(pair_key: str, mode: str, severity: str, backend: str, cfg: dict,
             out_subdir: str, skip_if_exists: bool = True) -> dict:
    pk_data = load_case_transforms(pair_key, mode, severity)
    plant = pk_data["plant"]
    a_id = pk_data["leaf_a_id"]
    b_id = pk_data["leaf_b_id"]
    root_index = pk_data["root_index"]
    sev_entry = next(s for s in pk_data[mode] if s["severity"] == severity)
    pd = load_plant(plant)
    gc, labels, apexes = pd["gc"], pd["labels"], pd["apexes"]
    leaf_a = np.where(labels == a_id)[0]
    leaf_b = np.where(labels == b_id)[0]
    g = gc
    if sev_entry["leaf_a_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev_entry["leaf_a_transform"], leaf_a)
    if sev_entry["leaf_b_transform"].get("pivot") is not None:
        g = apply_transform_entry(g, sev_entry["leaf_b_transform"], leaf_b)
    construction_metrics = compute_construction_metrics(
        g, labels, sev_entry["leaf_a_transform"], sev_entry["leaf_b_transform"],
        a_id, b_id, apexes)

    outdir = os.path.join(_T3, out_subdir, pair_key, mode, severity,
                          _backend_dirname(backend, cfg))
    os.makedirs(outdir, exist_ok=True)
    if skip_if_exists and os.path.exists(os.path.join(outdir, "failure_metrics.json")):
        with open(os.path.join(outdir, "failure_metrics.json")) as f:
            metrics = json.load(f)
            return {"pair_key": pair_key, "mode": mode, "severity": severity,
                    "backend": backend, "status": "cached",
                    "metrics": metrics, "config": cfg}

    factory = backend_factory(backend, cfg)
    t0 = time.time()
    result = hs.run_headless_segmentation(
        hs.GaussianData(
            xyz=np.asarray(g.xyz, dtype=np.float32),
            rot=np.asarray(g.rot, dtype=np.float32),
            scale=np.asarray(g.scale, dtype=np.float32),
            opacity=np.asarray(g.opacity, dtype=np.float32),
            sh=np.asarray(g.sh, dtype=np.float32),
            nxnynz=np.asarray(g.nxnynz, dtype=np.float32),
            filter_3Ds=np.asarray(g.filter_3Ds, dtype=np.float32),
        ),
        root_index=root_index,
        solver_factory=factory,
        frozen_root_basin_indices=load_frozen_basin(pair_key, mode, severity),
    )
    runtime = time.time() - t0

    save_case_outputs(
        outdir, result, g, labels, apexes,
        construction_gt_labels=labels.copy(),
        transforms_used=sev_entry,
        construction_metrics=construction_metrics,
        plant=plant, pair_key=pair_key, mode=mode, severity=severity,
        root_index=root_index, runtime=runtime,
    )

    # task-3-aware failure metrics (fine/coarse transforms, self-contained shortcut)
    metrics = compute_task3_metrics(pair_key, mode, severity, outdir)
    with open(os.path.join(outdir, "failure_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # attach graph stats / backend config to a sidecar
    sidecar = {"backend": backend, "cfg": cfg, "runtime": runtime}
    with open(os.path.join(outdir, "graph_stats.json"), "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    # graph-backend topology diagnostics (PASS criterion 5 / cross-leaf suppression)
    inst = factory.holder.get("backend")
    if inst is not None and hasattr(inst, "graph_stats"):
        gs = dict(inst.graph_stats)  # property -> dict
        gs.update({"backend": backend, "cfg": cfg})
        with open(os.path.join(outdir, "graph_stats.json"), "w") as f:
            json.dump(gs, f, indent=2, ensure_ascii=False)
        if hasattr(inst, "crossleaf_diagnostics"):
            diag = inst.crossleaf_diagnostics(labels)
            with open(os.path.join(outdir, "crossleaf_diagnostics.json"), "w") as f:
                json.dump(diag, f, indent=2, ensure_ascii=False)

    return {"pair_key": pair_key, "mode": mode, "severity": severity,
            "backend": backend, "status": "completed", "runtime": runtime,
            "config": cfg, "metrics": metrics}


def _parse_float(v):  # allow "inf"
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-key", required=True)
    ap.add_argument("--mode", choices=["horizontal", "vertical"], required=True)
    ap.add_argument("--severity", required=True)
    ap.add_argument("--backend", choices=["heat", "euclidean", "surface"],
                    default="surface")
    ap.add_argument("--feature-set", default="G5",
                    choices=["G0", "G1", "G2", "G3", "G4", "G5", "G6"])
    ap.add_argument("--lambda-n", type=float, default=1.0)
    ap.add_argument("--lambda-t", type=float, default=2.0)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--tau-d", type=_parse_float, default=3.0)
    ap.add_argument("--tau-t", type=_parse_float, default=0.5)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--mutual", type=lambda x: x.lower() in ("true", "1", "yes"),
                    default=False)
    ap.add_argument("--out-subdir", default="dev", choices=["dev", "test", "ablation"])
    ap.add_argument("--force", action="store_true")
    ar = ap.parse_args()

    # for euclidean backend treat feature_set/G4/G5 as the same euclidean weight
    if ar.backend == "euclidean":
        cfg = {"k": ar.k, "mutual": ar.mutual}
    elif ar.backend == "heat":
        cfg = {}
    else:
        cfg = {"feature_set": ar.feature_set, "lambda_n": ar.lambda_n,
               "lambda_t": ar.lambda_t, "p": ar.p, "tau_d": ar.tau_d,
               "tau_t": ar.tau_t, "k": ar.k, "mutual": ar.mutual}
    r = run_case(ar.pair_key, ar.mode, ar.severity, ar.backend, cfg,
                 ar.out_subdir, skip_if_exists=not ar.force)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
