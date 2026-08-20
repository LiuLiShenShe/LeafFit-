#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0: baseline (heat) runs on the frozen fine boundary transforms.

Reads outputs/task3/fine_boundary_transforms.json (HF1-4 / VF1-4, FROZEN),
runs the DEFAULT heat-method segmentation on every pair × fine level, saves
outputs under outputs/task3/phase0/, computes failure metrics (same schema as
Task 2), and aggregates outputs/task3/phase0_fine_baseline_summary.json.

The vertical shortcut reference (V0 geodesic) is read from this phase's own
V0 case when present, else from the Task 2 heat V0 (identical — same backend,
same identity transform).

NO surface-aware backend is used anywhere here. This fixes the BASELINE
failure boundary location precisely, before Ours is observed.
"""
from __future__ import annotations

import argparse
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
from gaussian_utils import GaussianData  # noqa: E402
from run_overlap_case import (  # noqa: E402
    load_plant,
    apply_transform_entry,
    compute_construction_metrics,
    save_case_outputs,
)
import compute_failure_metrics as cfm  # noqa: E402

_T3ROOT = os.path.join(_REPO_ROOT, "outputs", "task3")
_T2ROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_PHASE0 = os.path.join(_T3ROOT, "phase0")
_TRANSFORMS = os.path.join(_T3ROOT, "fine_boundary_transforms.json")
_MODES = ["horizontal", "vertical"]


def load_fine_transforms() -> dict:
    with open(_TRANSFORMS) as f:
        return json.load(f)


def _find_severity(pk_data: dict, mode: str, severity: str) -> dict:
    for se in pk_data[mode]:
        if se["severity"] == severity:
            return se
    # H0/V0 identity controls are not stored in fine_boundary_transforms.json;
    # synthesize the identity entry (apply_transform_entry treats pivot=None
    # as a no-op, so this is exactly the baseline identity case).
    if severity in ("H0", "V0"):
        return {
            "severity": severity, "mode": mode,
            "leaf_a_transform": {"pivot": None, "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 "t": [0.0, 0.0, 0.0], "angle_deg": 0.0},
            "leaf_b_transform": {"pivot": None, "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 "t": [0.0, 0.0, 0.0], "angle_deg": 0.0},
            "case_type": "identity_control",
        }
    raise ValueError(f"{severity} not found in {mode}")


def run_case(pair_key: str, mode: str, severity: str,
             skip_if_exists: bool = True) -> dict:
    transforms = load_fine_transforms()
    pk_data = transforms[pair_key]
    plant = pk_data["plant"]
    a_id = pk_data["leaf_a_id"]
    b_id = pk_data["leaf_b_id"]
    root_index = pk_data["root_index"]
    sev_entry = _find_severity(pk_data, mode, severity)

    outdir = os.path.join(_PHASE0, pair_key, mode, severity)
    if skip_if_exists and os.path.exists(os.path.join(outdir, "failure_metrics.json")):
        with open(os.path.join(outdir, "failure_metrics.json")) as f:
            return {"pair_key": pair_key, "mode": mode, "severity": severity,
                    "status": "cached", "metrics": json.load(f)}

    pd = load_plant(plant)
    gc, labels, apexes = pd["gc"], pd["labels"], pd["apexes"]
    leaf_a_idx = np.where(labels == a_id)[0]
    leaf_b_idx = np.where(labels == b_id)[0]

    g_out = gc
    ta = sev_entry["leaf_a_transform"]
    tb = sev_entry["leaf_b_transform"]
    if ta.get("pivot") is not None:
        g_out = apply_transform_entry(g_out, ta, leaf_a_idx)
    if tb.get("pivot") is not None:
        g_out = apply_transform_entry(g_out, tb, leaf_b_idx)

    construction_metrics = compute_construction_metrics(
        g_out, labels, ta, tb, a_id, b_id, apexes)

    t0 = time.time()
    result = hs.run_headless_segmentation(
        GaussianData(
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

    os.makedirs(outdir, exist_ok=True)
    save_case_outputs(
        outdir, result, g_out, labels, apexes,
        construction_gt_labels=labels.copy(),
        transforms_used=sev_entry,
        construction_metrics=construction_metrics,
        plant=plant, pair_key=pair_key, mode=mode, severity=severity,
        root_index=root_index, runtime=runtime,
    )

    metrics = compute_metrics(pair_key, mode, severity, outdir)
    with open(os.path.join(outdir, "failure_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return {"pair_key": pair_key, "mode": mode, "severity": severity,
            "status": "completed", "runtime": runtime, "metrics": metrics}


def compute_metrics(pair_key: str, mode: str, severity: str, case_dir: str) -> dict:
    """Task-3-aware failure metrics (mirrors compute_all_metrics, but the V0
    reference and GT leaf ids come from the fine transforms / phase0 root)."""
    transforms = load_fine_transforms()
    pk_data = transforms[pair_key]
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
        shortcut_metrics = _compute_shortcut(labels, gt_labels, root_geodesic,
                                             apexes, upper_id, lower_id,
                                             pair_key, severity, case_dir)

    first_failure_stage = "NONE"
    if geodesic_metrics["wrong_grouping"]:
        first_failure_stage = "GROUPING"
    elif geodesic_metrics["reference_apex_recall"] < 1.0:
        first_failure_stage = "APEX"
    elif instance_metrics["mIoU"] < 0.9:
        first_failure_stage = "SEGMENTATION"

    dominant_failure_stage = first_failure_stage

    metrics = {
        "pair_key": pair_key,
        "mode": mode,
        "severity": severity,
        "instance": instance_metrics,
        "pair_local": pair_metrics,
        "geodesic": geodesic_metrics,
        "first_failure_stage": first_failure_stage,
        "dominant_failure_stage": dominant_failure_stage,
        "construction": {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in data["construction"].items()},
    }
    if shortcut_metrics is not None:
        metrics["shortcut"] = shortcut_metrics
    return metrics


def _compute_shortcut(labels, gt_labels, root_geodesic, apexes,
                      upper_id, lower_id, pair_key, severity, case_dir):
    """Task-3 vertical shortcut metrics (self-contained, not Task-2-bound).

    V0 reference: Task 2 heat V0 root_geodesic_multisource — same deterministic
    heat backend, same centered identity input, same frozen root -> byte-identical
    to a fresh identity run.

    cross_leaf_path is detected from THIS case's own paths.json (Task 2's
    compute_shortcut_metrics reads task2/controlled/..., which has no VF levels).
    """
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

    if upper_apex_idx is None or upper_apex_idx >= len(root_geodesic):
        return {"shortcut_ratio": None, "reason": "upper_apex_not_found"}

    v0_geodesic_path = os.path.join(
        _T2ROOT, "controlled", pair_key, "vertical", "V0",
        "root_geodesic_multisource.npy")
    if not os.path.exists(v0_geodesic_path):
        return {"shortcut_ratio": None, "reason": "no_v0_reference"}
    d_v0 = float(np.load(v0_geodesic_path)[upper_apex_idx])
    d_case = float(root_geodesic[upper_apex_idx])
    shortcut_ratio = d_case / d_v0 if d_v0 > 0 else None

    # cross-leaf path from THIS case's paths.json
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
    lower_med = float(np.median(root_geodesic[lower_mask]))
    upper_d = root_geodesic[upper_mask]
    below_lower = int((upper_d < lower_med).sum())

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


def find_boundary(summaries: list[dict], mode: str, criterion) -> int | None:
    """First severity index where criterion(metrics) is True, else None."""
    for i, s in enumerate(summaries):
        if s["mode"] == mode and criterion(s.get("metrics", {})):
            return i
    return None


def aggregate(pair_keys: list[str]) -> dict:
    """Build phase0_fine_baseline_summary.json with plan-strict boundaries.

    mechanism_onset_boundary   (horizontal): first level with wrong_grouping
                               (vertical): first level with cross_leaf_path AND
                               shortcut_ratio < 1-eps AND downstream instance error
    final_instance_failure_boundary (horizontal): wrong_grouping
                               (vertical): wrong_grouping OR PQ drop >=0.1 vs V0 OR
                               (merge_level>=1 with cross-leaf evidence)
    clean_fidelity_pq = PQ at H0 / V0 (identity control).
    """
    transforms = load_fine_transforms()
    rows = []
    for pk in pair_keys:
        for mode in _MODES:
            for sev in (["H0", "HF1", "HF2", "HF3", "HF4"] if mode == "horizontal"
                        else ["V0", "VF1", "VF2", "VF3", "VF4"]):
                fp = os.path.join(_PHASE0, pk, mode, sev, "failure_metrics.json")
                if not os.path.exists(fp):
                    continue
                with open(fp) as f:
                    m = json.load(f)
                row = {
                    "pair_key": pk,
                    "plant": pk.split("_pair_")[0],
                    "mode": mode,
                    "severity": sev,
                    "contact_fraction": m["construction"].get("contact_fraction"),
                    "apex_gap_ratio": None,
                    "wrong_grouping": m["geodesic"]["wrong_grouping"],
                    "merge_level": m["geodesic"]["merge_level"],
                    "reference_apex_recall": m["geodesic"]["reference_apex_recall"],
                    "PQ": m["instance"]["PQ"],
                    "mIoU": m["instance"]["mIoU"],
                    "first_failure_stage": m["first_failure_stage"],
                }
                # achieved apex gap ratio from the frozen fine transforms
                for se in transforms[pk][mode]:
                    if se["severity"] == sev:
                        row["apex_gap_ratio"] = se.get("achieved_apex_gap_ratio")
                        break
                if mode == "vertical" and "shortcut" in m:
                    s = m["shortcut"]
                    row["shortcut_ratio"] = s.get("shortcut_ratio")
                    row["cross_leaf_path"] = s.get("cross_leaf_path", False)
                    row["cross_leaf_merge"] = s.get("cross_leaf_merge", False)
                    row["shortcut_confirmed"] = s.get("shortcut_confirmed", False)
                rows.append(row)

    boundaries = {}
    for pk in pair_keys:
        for mode in _MODES:
            key = f"{pk}|{mode}"
            lv = rows if False else [r for r in rows if r["pair_key"] == pk and r["mode"] == mode]
            clean_row = next((r for r in lv if r["severity"] == "V0" or r["severity"] == "H0"), None)
            clean_pq = clean_row["PQ"] if clean_row else None
            fin_levels = [r for r in lv if r["severity"] != "V0" and r["severity"] != "H0"]

            onset = None
            final = None
            if mode == "horizontal":
                for r in fin_levels:
                    if onset is None and r["wrong_grouping"]:
                        onset = r["severity"]
                    if final is None and r["wrong_grouping"]:
                        final = r["severity"]
            else:  # vertical strict criterion
                for r in fin_levels:
                    has_shortcut = r.get("cross_leaf_path", False) and r.get("shortcut_confirmed", False)
                    inst_err = (r.get("wrong_grouping", False)
                                or (clean_pq is not None and r["PQ"] is not None
                                    and r["PQ"] < clean_pq - 0.1))
                    if onset is None and has_shortcut and inst_err:
                        onset = r["severity"]
                    if final is None and (r.get("wrong_grouping", False)
                                          or inst_err):
                        final = r["severity"]

            boundaries[key] = {
                "mechanism_onset": onset,
                "final_instance_failure": final,
                "clean_fidelity_pq": clean_pq,
            }

    out = {
        "transforms_source": "outputs/task3/fine_boundary_transforms.json",
        "backend": "heat (default)",
        "pairs": pair_keys,
        "levels": {
            "horizontal": ["H0", "HF1", "HF2", "HF3", "HF4"],
            "vertical": ["V0", "VF1", "VF2", "VF3", "VF4"],
        },
        "cases": rows,
        "boundaries": boundaries,
    }
    with open(os.path.join(_T3ROOT, "phase0_fine_baseline_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-key")
    parser.add_argument("--mode", choices=_MODES)
    parser.add_argument("--severity")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if failure_metrics.json exists")
    args = parser.parse_args()

    transforms = load_fine_transforms()
    pair_keys = [args.pair_key] if args.pair_key else list(transforms.keys())

    if args.pair_key or args.mode or args.severity:
        # single/filtered case run
        modes = [args.mode] if args.mode else _MODES
        for pk in pair_keys:
            for mode in modes:
                sevs = ([args.severity] if args.severity
                        else (["HF1", "HF2", "HF3", "HF4"] if mode == "horizontal"
                              else ["VF1", "VF2", "VF3", "VF4"]))
                for sev in sevs:
                    r = run_case(pk, mode, sev, skip_if_exists=not args.force)
                    print(f"  {r['status']}: {pk} {mode} {sev} "
                          f"PQ={r.get('metrics', {}).get('instance', {}).get('PQ')}")
        aggregate(pair_keys)
        print(f"[OK] aggregated -> phase0_fine_baseline_summary.json")
        return 0

    # full run: all pairs × all fine levels (+ V0/H0 references)
    all_levels = []
    for pk in pair_keys:
        for mode in _MODES:
            for sev in (["H0", "HF1", "HF2", "HF3", "HF4"] if mode == "horizontal"
                        else ["V0", "VF1", "VF2", "VF3", "VF4"]):
                all_levels.append((pk, mode, sev))

    n = len(all_levels)
    for i, (pk, mode, sev) in enumerate(all_levels):
        t0 = time.time()
        r = run_case(pk, mode, sev, skip_if_exists=not args.force)
        dt = time.time() - t0
        m = r.get("metrics", {})
        print(f"[{i+1}/{n}] {r['status']}: {pk} {mode} {sev} "
              f"PQ={m.get('instance', {}).get('PQ'):.3f} "
              f"merge={m.get('geodesic', {}).get('merge_level')} "
              f"shortcut={m.get('shortcut', {}).get('shortcut_ratio') if m.get('shortcut') else '-'} "
              f"({dt:.1f}s)")

    summary = aggregate(pair_keys)
    print(f"\n[OK] full Phase 0 baseline run. "
          f"{len(summary['cases'])} cases -> {os.path.join(_T3ROOT, 'phase0_fine_baseline_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
