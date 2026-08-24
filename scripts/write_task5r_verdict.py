#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 verdict gate — machine-readable ordered checks, NO hard-coded verdict.

Reads measured artifacts and produces exactly one of:
  IMPLEMENTATION_INVALID / ALIGNMENT_FAIL / BENCHMARK_NOT_HUMAN_VERIFIED /
  MATCHING_FAIL / INSUFFICIENT_SAMPLE / SEPARABILITY_FAIL / SEPARABILITY_PASS

task6_allowed = (verdict == "SEPARABILITY_PASS") — derived, never asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent

MIN_CROSS_PER_UNIT = 30          # frozen sample floor per analysis unit
HELDOUT_MIN_AUROC = 0.55         # frozen: signed AUROC must exceed this on held-out
DEV_NULL_BAND = (0.45, 0.55)     # R0 control must be null on dev
SIGN_DELTA_MIN = 0.05            # frozen direction-freezing margin


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def evaluate(repo_root: Path) -> dict:
    out_dir = repo_root / "outputs" / "task5r_v3"
    checks = []
    verdict = None
    task6_allowed = None  # derived below; never hard-asserted

    def record(name, status, artifact, details=None):
        checks.append({"order": len(checks) + 1, "name": name,
                       "status": status, "artifact": str(artifact),
                       "details": details or {}})

    def stop(v, name, status, artifact, details=None):
        record(name, status, artifact, details)
        return {"verdict": v, "task6_allowed": False, "checks": checks,
                "first_failure": name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # --- 1. implementation self-test -------------------------------------
    st = load_json(out_dir / "selftest.json")
    if not st or not st.get("all_passed", False):
        return stop("IMPLEMENTATION_INVALID", "implementation_selftest",
                    "FAIL", out_dir / "selftest.json",
                    {"missing": st is None})
    record("implementation_selftest", "PASS", out_dir / "selftest.json")

    # --- 2. source tree cleanliness + commit provenance -------------------
    sys.path[:0] = [str(repo_root), str(repo_root / "core")]
    from core.observation_identity import git_commit, git_tree_dirty
    dirty = git_tree_dirty(repo_root)
    src_commit = git_commit(repo_root)
    man = load_json(out_dir / "benchmark_manifest.json") or {}
    recorded = man.get("source_commit")
    if dirty or (recorded and recorded != src_commit):
        return stop("IMPLEMENTATION_INVALID", "source_tree_clean",
                    "FAIL", repo_root / ".git",
                    {"dirty": dirty, "recorded_commit": recorded,
                     "current_commit": src_commit})
    record("source_tree_clean", "PASS", repo_root / ".git",
           {"commit": src_commit})

    # --- 3. viewsig cache-key consistency (v3 only) ----------------------
    from core.observation_identity import VISIBILITY_VERSION
    manifest_path = out_dir / "corrected_viewsig_manifest.jsonl"
    stale = []
    n_plants = 0
    if manifest_path.exists():
        seen_plants = {}
        for line in manifest_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("visibility_version") != VISIBILITY_VERSION:
                continue
            key_seen = seen_plants.get(r["plant"])
            # last entry per plant wins; must match its own cache_key file
            seen_plants[r["plant"]] = r["cache_key"]
        n_plants = len(seen_plants)
        for plant, key in seen_plants.items():
            z = out_dir / "projection_cache" / plant / f"corrected_viewsig_{key}.npz"
            if not z.exists():
                stale.append(plant)
    else:
        stale = ["<no manifest>"]
    if stale or n_plants < 2:
        return stop("ALIGNMENT_FAIL", "viewsig_cache_consistency",
                    "FAIL", manifest_path,
                    {"stale_or_missing": stale, "n_plants": n_plants})
    record("viewsig_cache_consistency", "PASS", manifest_path,
           {"n_plants": n_plants})

    # --- 4. dense alignment check ----------------------------------------
    al = load_json(out_dir / "dense_alignment.json")
    if not al or not al.get("overall_passed", False):
        return stop("ALIGNMENT_FAIL", "dense_alignment",
                    "FAIL", out_dir / "dense_alignment.json")
    record("dense_alignment", "PASS", out_dir / "dense_alignment.json")

    # --- 5. human verification of benchmark labels -----------------------
    hv = load_json(out_dir / "human_verification.json")
    if not hv or not hv.get("approved", False):
        return stop("BENCHMARK_NOT_HUMAN_VERIFIED", "human_verification",
                    "PENDING", out_dir / "human_verification.json",
                    {"note": "proposer labels are PROPOSER_DIAGNOSTIC until "
                             "a human reviewer fills human_verification.json"})
    record("human_verification", "PASS", out_dir / "human_verification.json")

    # --- 6. matching gates -------------------------------------------------
    mg = load_json(out_dir / "matched_edges_gates.json")
    formal = (mg or {}).get("gates", {}).get("1:1", {})
    if not mg or not formal.get("gates_passed", False):
        return stop("MATCHING_FAIL", "matching_gates",
                    "FAIL", out_dir / "matched_edges_gates.json",
                    mg or {})
    gates_passed = bool(formal.get("gates_passed"))
    units = mg.get("units", {})
    insufficient = [u for u, f in units.items()
                    if f.get("n_cross", 0) < MIN_CROSS_PER_UNIT]
    gating_units_insufficient = [u for u in insufficient
                                 if u.split(":")[0] in ("dev", "heldout")]
    record("matching_gates", "PASS" if gates_passed else "FAIL",
           out_dir / "matched_edges_gates.json",
           {"insufficient_pairs_excluded": insufficient,
            "formal_variant": "1:1"})

    # --- 7. dev direction freezing + dev gate ------------------------------
    sep = load_json(out_dir / "edge_separability_summary_v3.json")
    if not sep:
        return stop("MATCHING_FAIL", "separability_results",
                    "MISSING", out_dir / "edge_separability_summary_v3.json")
    dev = sep["splits"]["dev"]
    frozen_signs = {}
    for ab, stats in dev["by_ablation"].items():
        au = stats["signed_auroc"] if "signed_auroc" in stats else stats["auroc"]
        frozen_signs[ab] = 1 if au >= 0.5 else -1
        stats["frozen_sign"] = frozen_signs[ab]
    # R0 control must be null on dev under the preset convention (-distance)
    r0_au = dev["by_ablation"].get("R0_dist", {}).get("auroc")
    if r0_au is None or not (DEV_NULL_BAND[0] <= r0_au <= DEV_NULL_BAND[1]):
        return stop("MATCHING_FAIL", "r0_control_null_on_dev",
                    "FAIL", out_dir / "edge_separability_summary_v3.json",
                    {"R0_dist_dev_auroc": r0_au})
    # composite gates must carry signal on dev in the PRESET direction
    for gate_ab in ("R4_c_mv", "R6_mv_and_surface"):
        s = dev["by_ablation"][gate_ab]
        if abs(s["auroc"] - 0.5) < SIGN_DELTA_MIN:
            return stop("SEPARABILITY_FAIL", "dev_gate_no_signal",
                        "FAIL", out_dir / "edge_separability_summary_v3.json",
                        {gate_ab: s["auroc"], "delta_min": SIGN_DELTA_MIN})
    record("dev_gate_direction_frozen", "PASS",
           out_dir / "edge_separability_summary_v3.json",
           {"frozen_signs": frozen_signs})

    # --- 8. held-out evaluated ONCE with frozen signs ----------------------
    ho_sha = sha256_file(out_dir / "edge_separability_summary_v3.json")
    heldout = sep["splits"]["heldout"]
    sign_flips = []
    for gate_ab in ("R4_c_mv", "R6_mv_and_surface"):
        s = heldout["by_ablation"][gate_ab]
        signed = s["auroc"] * frozen_signs[gate_ab]
        s["heldout_signed_auroc"] = signed
        if signed < HELDOUT_MIN_AUROC:
            # distinguish preset-direction failure vs reversed signal
            if s["auroc"] < 0.5 - SIGN_DELTA_MIN and frozen_signs[gate_ab] == 1:
                s["failure_mode"] = "reversed_signal_present"
            elif abs(s["auroc"] - 0.5) < SIGN_DELTA_MIN:
                s["failure_mode"] = "no_signal"
            else:
                s["failure_mode"] = "preset_direction_failed"
            return stop("SEPARABILITY_FAIL", "heldout_gate_once",
                        "FAIL", out_dir / "edge_separability_summary_v3.json",
                        {gate_ab: {"auroc": s["auroc"],
                                   "signed": signed,
                                   "mode": s["failure_mode"]}})
        if np.sign(s["auroc"] - 0.5) != np.sign(dev["by_ablation"][gate_ab]["auroc"] - 0.5):
            sign_flips.append(gate_ab)
    if sign_flips:
        return stop("SEPARABILITY_FAIL", "sign_consistency",
                    "FAIL", out_dir / "edge_separability_summary_v3.json",
                    {"sign_flips": sign_flips})
    record("heldout_gate_once", "PASS",
           out_dir / "edge_separability_summary_v3.json", {"sha256": ho_sha})

    # --- 9. sample size downgrade ------------------------------------------
    gating_units_ok = all(
        units.get(f"{split}:ALL", {}).get("n_cross", 0) >= MIN_CROSS_PER_UNIT
        for split in ("dev", "heldout"))
    verdict = ("INSUFFICIENT_SAMPLE" if (not gating_units_ok or gating_units_insufficient)
               else "SEPARABILITY_PASS")

    task6_allowed = (verdict == "SEPARABILITY_PASS")
    return {"verdict": verdict, "task6_allowed": task6_allowed,
            "checks": checks, "first_failure": None,
            "frozen_signs": frozen_signs,
            "summary_sha256": ho_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--output", default=None)
    ar = ap.parse_args()
    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    result = evaluate(repo_root)
    out = Path(ar.output) if ar.output else \
        repo_root / "outputs" / "task5r_v3" / "verdict.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("verdict", "task6_allowed", "first_failure")},
                     indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
