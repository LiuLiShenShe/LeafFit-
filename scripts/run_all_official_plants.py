#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-run the LeafFit headless segmentation over ALL official plants (data/*.ply).

Phase 1 (root freeze): for each plant, run `--root auto` once to obtain the official
    PCA-estimated root_idx, and persist it under outputs/frozen_roots.json.
Phase 2 (formal batch): for each plant, run with the FROZEN root (`--root-index`)
    into outputs/baseline/<stem>/. This is the canonical baseline Task 2/3 must use.

    python scripts/run_all_official_plants.py

A per-plant failure does not abort the batch: each plant gets status
SUCCESS / FAILED / SEGMENTATION_FAILED_NO_LEAVES, with error.log on failure.
A combined manifest is written to outputs/baseline_manifest.json.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO_ROOT, "data")
_OUTROOT = os.path.join(_REPO_ROOT, "outputs")
_BASELINE = os.path.join(_OUTROOT, "baseline")
_FROZEN = os.path.join(_OUTROOT, "frozen_roots.json")
_MANIFEST = os.path.join(_OUTROOT, "baseline_manifest.json")
_PY = sys.executable
_CLI = os.path.join(_REPO_ROOT, "scripts", "run_leaf_segmentation.py")


def plant_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def run_cli(input_path: str, out_dir: str, root_specargs) -> int:
    cmd = [_PY, _CLI, "--input", input_path, "--output", out_dir] + list(root_specargs)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(_REPO_ROOT, "core"), _REPO_ROOT]))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.stdout:
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.strip(), file=sys.stderr)
    return r.returncode


def read_status(out_dir: str) -> dict:
    p = os.path.join(out_dir, "status.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"status": "FAILED_NO_STATUS", "error": "no status.json", "failure_stage": "unknown"}


def load_plants():
    return sorted(glob.glob(os.path.join(_DATA, "*.ply")))


def main():
    os.makedirs(_BASELINE, exist_ok=True)
    plants = load_plants()
    if not plants:
        print(f"no data/*.ply under {_DATA}")
        return 1

    # ---- Phase 1: freeze roots (only for plants missing from frozen_roots.json) ----
    frozen = {}
    if os.path.exists(_FROZEN):
        with open(_FROZEN) as f:
            frozen = json.load(f)

    changed = False
    for pp in plants:
        stem = plant_stem(pp)
        if stem in frozen:
            continue
        print(f"[FREEZE] {stem}: running auto root ...")
        tmp = os.path.join(_BASELINE, f"{stem}__freeze")
        rc = run_cli(pp, tmp, ["--root", "auto"])
        if rc != 0:
            print(f"[FREEZE] {stem}: FAILED (rc={rc}); plant will be skipped in formal batch")
            frozen[stem] = {"root_index": None, "root_source": "auto", "status": "FAILED"}
        else:
            with open(os.path.join(tmp, "root.json")) as f:
                root = json.load(f)
            frozen[stem] = {"root_index": int(root["root_index"]),
                            "root_source": root["root_source"], "status": "ROOT_FROZEN"}
        changed = True

    if changed:
        with open(_FROZEN, "w") as f:
            json.dump(frozen, f, indent=2, ensure_ascii=False)
        print(f"frozen roots written -> {_FROZEN}")

    # ---- Phase 2: formal batch with frozen roots ----
    manifest = {"method": "geodesic_tip_graph"}
    plants_record = {}
    for pp in plants:
        stem = plant_stem(pp)
        fr = frozen.get(stem, {})
        root_index = fr.get("root_index")
        out = os.path.join(_BASELINE, stem)
        if root_index is None:
            plants_record[stem] = {"status": "FAILED", "root_index": None,
                                   "error": "frozen root unavailable", "input": pp}
            print(f"[SKIP] {stem}: no frozen root")
            continue
        print(f"[RUN] {stem}: root={root_index}")
        rc = run_cli(pp, out, ["--root-index", str(root_index)])
        st = read_status(out)
        rec = {"status": st.get("status"), "error": st.get("error"),
               "root_index": root_index, "root_source": "frozen(auto)"}
        # enrich with counts from metadata
        mpath = os.path.join(out, "metadata.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                meta = json.load(f)
            rec.update({"num_gaussians": meta.get("num_gaussians"),
                        "num_apexes": meta.get("num_apexes"),
                        "num_leaf_instances": meta.get("num_leaf_instances"),
                        "source": "fixed(meta)"})
        else:
            rec["error"] = (rec.get("error") or "") + " | no metadata.json"
            if rec["status"] == "SUCCESS":
                rec["status"] = "PARTIAL"
        rec["rc"] = rc
        plants_record[stem] = rec

    # merge env into manifest
    import importlib.metadata as md
    manifest.update({
        "python": sys.version.split()[0],
        "numpy": md.version("numpy"),
        "potpourri3d": md.version("potpourri3d"),
        "plants": plants_record,
    })
    with open(_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # summary table
    print("\n=== Batch Summary ===")
    header = f"{'Plant':<26} {'N':>7} {'Root':>7} {'Apex':>5} {'Leaf':>5} {'Status':<26}"
    print(header)
    for stem in sorted(plants_record):
        r = plants_record[stem]
        print(f"{stem:<26} {str(r.get('num_gaussians','-')):>7} "
              f"{str(r.get('root_index','-')):>7} {str(r.get('num_apexes','-')):>5} "
              f"{str(r.get('num_leaf_instances','-')):>5} {str(r.get('status','-')):<26}")
    print(f"\nmanifest -> {_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())