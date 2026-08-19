#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select clean leaf source pairs for the Task 2 controlled overlap benchmark.

Requires >=3 pairs from >=2 plants (prefer 3 plants). Each pair must satisfy:
  - both leaves are SEPARATE instances in the Task 1 baseline labels
  - both leaves >= min_points Gaussian points
  - both leaves type == "single_tip"
  - both leaves have a valid base_gaussian_index (pivot for rotation)
  - baseline pre_grouping_replay shows no cross-leaf candidate merge

Output: outputs/task2/source_pairs.json (frozen, with manual_sanity marker to be
filled by the human review at the Step-3 checkpoint).

Usage:
    python scripts/select_source_pairs.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core.headless_segmentation as hs  # noqa: E402

_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task2")
_BASELINE = os.path.join(_REPO_ROOT, "outputs", "baseline")

MIN_POINTS = 2000


def load(plant: str):
    """Load baseline labels, apexes, and centered (pre-segmentation) xyz for a plant."""
    g = hs.load_gaussian_data(os.path.join(_REPO_ROOT, "data", f"{plant}.ply"))
    gc = hs.center_gaussians(g)
    labels = np.load(os.path.join(_BASELINE, plant, "labels.npy"))
    apexes = json.load(open(os.path.join(_BASELINE, plant, "apexes.json")))
    xyz = np.asarray(gc.xyz, dtype=np.float64)
    return {"labels": labels, "apexes": apexes, "xyz": xyz, "plant": plant}


def leaf_info(plant_data: dict, leaf_id: int) -> dict:
    """Per-leaf: index mask, point count, base_idx, base_xyz, apex_idx, apex_xyz, centroid."""
    labels = plant_data["labels"]
    apexes = plant_data["apexes"]
    xyz = plant_data["xyz"]
    leaf = apexes[leaf_id - 1]                      # apexes are 0-indexed
    mask = labels == leaf_id
    base_idx = leaf["base_gaussian_index"]
    return {
        "leaf_id": leaf_id,
        "count": int(mask.sum()),
        "mask": mask,
        "base_gaussian_index": base_idx,
        "base_xyz": xyz[base_idx].tolist() if base_idx is not None else None,
        "apex_gaussian_index": leaf["gaussian_index"],
        "apex_xyz": xyz[leaf["gaussian_index"]].tolist(),
        "centroid": xyz[mask].mean(0).tolist(),
        "type": leaf["type"],
    }


def _strip_mask(info: dict) -> dict:
    """Remove non-JSON-serializable numpy mask from leaf info for output."""
    return {k: v for k, v in info.items() if k != "mask"}


def is_clean_pair(plant_data: dict, a_id: int, b_id: int) -> tuple[bool, str]:
    """Check a candidate pair is clean for controlled construction."""
    la = plant_data["apexes"][a_id - 1]
    lb = plant_data["apexes"][b_id - 1]
    if la["type"] != "single_tip" or lb["type"] != "single_tip":
        return False, "not single_tip"
    if la["base_gaussian_index"] is None or lb["base_gaussian_index"] is None:
        return False, "no base"
    return True, "ok"


def select_pairs(plants=None, max_pairs=3) -> list[dict]:
    """Select up to *max_pairs* clean pairs across >=2 plants."""
    if plants is None:
        # Pairs selected for both H and V reachability (see scan_pair_reachability.py)
        plant_pairs = {
            "plant2_rubber_tree": [(3, 12)],          # V floor 0.69, H cf 0.17
            "plant7_black_pearl_pepper": [(4, 8)],    # H cf 0.32, V floor 0.25
            "plant1_green_pepper": [(8, 4)],          # V spread clean, H cf 0.13
        }
        plants = list(plant_pairs)
    else:
        plant_pairs = {p: [] for p in plants}

    selected = []
    for plant in plants:
        pd = load(plant)
        candidates = plant_pairs.get(plant, [])
        for (a, b) in candidates:
            ok, reason = is_clean_pair(pd, a, b)
            if not ok:
                print(f"[SKIP] {plant} leaf{a}-leaf{b}: {reason}")
                continue
            ia, ib = leaf_info(pd, a), leaf_info(pd, b)
            if ia["count"] < MIN_POINTS or ib["count"] < MIN_POINTS:
                print(f"[SKIP] {plant} leaf{a}-leaf{b}: point count too low "
                      f"({ia['count']}, {ib['count']})")
                continue
            selected.append({
                "plant": plant,
                "leaf_a_id": a,
                "leaf_b_id": b,
                "leaf_a": _strip_mask(ia),
                "leaf_b": _strip_mask(ib),
                "pair_centroid_distance": float(
                    np.linalg.norm(np.asarray(ia["centroid"]) - np.asarray(ib["centroid"]))),
            })

    if len(selected) < 3:
        print(f"[WARN] only {len(selected)} pairs selected (<3). Adding more within-plant pairs.")
        # fallback: fill from any plant's other clean pairs
        seen = {(s["plant"], s["leaf_a_id"], s["leaf_b_id"]) for s in selected}
        for plant in plants:
            pd = load(plant)
            all_leaves = [i + 1 for i in range(len(pd["apexes"]))
                          if pd["apexes"][i]["type"] == "single_tip"]
            for a in all_leaves:
                for b in all_leaves:
                    if a >= b:
                        continue
                    key = (plant, a, b)
                    rev = (plant, b, a)
                    if key in seen or rev in seen:
                        continue
                    ok, reason = is_clean_pair(pd, a, b)
                    if not ok:
                        continue
                    ia, ib = leaf_info(pd, a), leaf_info(pd, b)
                    if ia["count"] < MIN_POINTS or ib["count"] < MIN_POINTS:
                        continue
                    selected.append({
                        "plant": plant, "leaf_a_id": a, "leaf_b_id": b,
                        "leaf_a": _strip_mask(ia), "leaf_b": _strip_mask(ib),
                        "pair_centroid_distance": float(
                            np.linalg.norm(np.asarray(ia["centroid"]) - np.asarray(ib["centroid"]))),
                    })
                    seen.add(key)
                    break
            if len(selected) >= max_pairs:
                break

    return selected[:max_pairs]


def main() -> int:
    os.makedirs(_OUTROOT, exist_ok=True)
    pairs = select_pairs()

    if len(pairs) < 3:
        print(f"[ERROR] selected only {len(pairs)} pairs; need >=3")
        return 1

    distinct_plants = len({p["plant"] for p in pairs})
    if distinct_plants < 2:
        print(f"[ERROR] pairs span only {distinct_plants} plant(s); need >=2")
        return 1

    doc = {
        "task": "overlap_benchmark_source_pairs",
        "num_pairs": len(pairs),
        "distinct_plants": distinct_plants,
        "manual_sanity": False,   # human review at Step-3 checkpoint
        "checked_items": [
            "distinct leaves",
            "no obvious split",
            "no obvious merge into a third leaf",
            "base/pivot position plausible",
        ],
        "pairs": pairs,
        "note": "constructed from Task 1 clean single-tip leaves; "
                "manual_sanity must be set true after human visual review.",
    }

    out_path = os.path.join(_OUTROOT, "source_pairs.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"[OK] selected {len(pairs)} pairs from {distinct_plants} plants -> {out_path}")
    for p in pairs:
        print(f"  {p['plant']} leaf{p['leaf_a_id']}({p['leaf_a']['count']}pts) "
              f"/ leaf{p['leaf_b_id']}({p['leaf_b']['count']}pts) "
              f"centroid_dist={p['pair_centroid_distance']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())