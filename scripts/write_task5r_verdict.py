#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3.1 verdict gate — machine-readable ordered checks, NO hard-coded verdict.

Reads measured artifacts and produces exactly one of:
  IMPLEMENTATION_INVALID / ALIGNMENT_FAIL / BENCHMARK_NOT_HUMAN_VERIFIED /
  MATCHING_FAIL / INSUFFICIENT_SAMPLE / SEPARABILITY_NOT_DEMONSTRATED /
  SEPARABILITY_FAIL / SEPARABILITY_PASS

v3.1 changes (statistical governance):
  * formal point estimate is the PAIR-MACRO AUROC; pooled-edge AUROC is
    descriptive only and never gates;
  * held-out signed transform: signed = auc if frozen_sign==+1 else (1-auc)
    — v3's `auc * -1` was mathematically wrong on [0,1];
  * upstream manifest chain validation: EVERY viewsig record, benchmark
    manifest, matching gates, dense alignment and selftest must carry a clean
    source tree at the CURRENT commit — any dirty/stale upstream fails
    IMPLEMENTATION_INVALID (v3 only checked the current working tree);
  * provenance: matched-edge CSVs, summary and review queue SHA256s are
    recorded in the verdict instead of remaining "pending";
  * sample floor: n_pairs >= MIN_PAIRS_PER_SPLIT per split, else
    INSUFFICIENT_SAMPLE;
  * no-signal outcomes default to SEPARABILITY_NOT_DEMONSTRATED;
    SEPARABILITY_FAIL requires demonstrated REVERSED signal (macro AUROC CI
    fully below the null band) with sufficient pairs.

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

MIN_CROSS_PER_UNIT = 30          # frozen sample floor per analysis unit (edges)
# FROZEN by scripts/power_analysis_min_pairs.py BEFORE any v3.1 measurement
# (outputs/task5r_v3_1/min_pairs_freeze.json, frozen 2026-08-25T10:56:15+0800;
# pilot = the consumed/exploratory v3 run, per-pair AUROC var 0.1334 → K=205
# for 95% CI half-width <= SIGN_DELTA_MIN). Do not tune after seeing results.
MIN_PAIRS_PER_SPLIT = 205
HELDOUT_MIN_AUROC = 0.55         # frozen: signed macro AUROC must exceed this
DEV_NULL_BAND = (0.45, 0.55)     # R0 control must be null on dev (pooled, descriptive)
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


def _signed_auc(auc: float, frozen_sign: int) -> float:
    """Direction transform: positive direction keeps auc; negative direction
    maps to 1 - auc (NOT auc * -1, which leaves [0,1])."""
    return float(auc) if frozen_sign >= 0 else 1.0 - float(auc)


def evaluate(repo_root: Path) -> dict:
    out_dir = repo_root / "outputs" / "task5r_v3_1"
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

    # --- 2b. UPSTREAM MANIFEST CHAIN VALIDATION ---------------------------
    # Every upstream measured artifact must itself claim a clean source tree
    # at the current commit. v3 only checked the current working tree.
    chain_problems = []

    def check_upstream(name, obj, commit_key="source_commit",
                       dirty_key="source_tree_dirty"):
        if not isinstance(obj, dict):
            chain_problems.append(f"{name}: <missing or malformed>")
            return
        c = obj.get(commit_key)
        d = obj.get(dirty_key)
        if c is not None and c != src_commit:
            # two-phase commits are legitimate: accept an ANCESTOR commit
            # whose SOURCE tree (core/scripts/tests) is identical to HEAD's.
            same_source = False
            try:
                import subprocess as _sp
                _anc = _sp.run(["git", "merge-base", "--is-ancestor",
                                str(c), src_commit], cwd=str(repo_root),
                               capture_output=True).returncode == 0
                if _anc:
                    _diff = _sp.run(
                        ["git", "diff", "--stat", str(c), src_commit,
                         "--", "core", "scripts", "tests"],
                        cwd=str(repo_root), capture_output=True)
                    same_source = _anc and _diff.returncode == 0 and \
                        not _diff.stdout.strip()
            except Exception:
                same_source = False
            if not same_source:
                chain_problems.append(
                    f"{name}: source_commit {str(c)[:12]} differs from HEAD "
                    f"{src_commit[:12]} in tracked source (core/scripts/tests)")
        if d:
            chain_problems.append(f"{name}: source_tree_dirty=true")

    ms_path = out_dir / "corrected_viewsig_manifest.jsonl"
    if ms_path.exists():
        last_rows = {}
        for line in ms_path.read_text().splitlines():
            r = json.loads(line)
            # append-only manifest: LAST row per plant is authoritative
            # (matches the cache-consistency check's last-wins semantics)
            if r.get("visibility_version") == VISIBILITY_VERSION:
                last_rows[r["plant"]] = r
        for plant, r in last_rows.items():
            check_upstream(f"viewsig[{plant}]", r)
            alg = r.get("visibility_algorithm") or {}
            check_upstream(f"viewsig[{plant}].algorithm", alg)
    else:
        chain_problems.append("corrected_viewsig_manifest.jsonl: <missing>")
    check_upstream("benchmark_manifest", man)
    mg_probe = load_json(out_dir / "matched_edges_gates.json")
    check_upstream("matched_edges_gates", mg_probe or {})
    al_probe = load_json(out_dir / "dense_alignment.json")
    check_upstream("dense_alignment", al_probe or {})
    st_probe = load_json(out_dir / "selftest.json")
    check_upstream("selftest", st_probe or {})
    if chain_problems:
        return stop("IMPLEMENTATION_INVALID", "upstream_manifest_chain",
                    "FAIL", out_dir,
                    {"problems": chain_problems})
    record("upstream_manifest_chain", "PASS", out_dir,
           {"n_viewsig_records": len(list(out_dir.glob("projection_cache/*")))})

    # --- 3. viewsig cache-key consistency ---------------------------------
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

    # --- 5b. review queue provenance (SHA256, migration audit) ------------
    rq_path = out_dir / "benchmark_review_queue.csv"
    prov = {
        "review_queue_csv_sha256": sha256_file(rq_path) if rq_path.exists() else None,
        "matched_edges_1to1_sha256":
            sha256_file(out_dir / "matched_edges_1to1.csv"),
        "matched_edges_1to5_sha256":
            sha256_file(out_dir / "matched_edges_1to5.csv"),
    }
    missing_prov = [k for k, v in prov.items() if v is None]
    if missing_prov:
        return stop("IMPLEMENTATION_INVALID", "provenance_sha256_complete",
                    "FAIL", out_dir, {"missing": missing_prov})
    record("provenance_sha256_complete", "PASS", out_dir, prov)

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

    # --- 7. dev direction freezing + dev gate (PAIR MACRO) ------------------
    sep = load_json(out_dir / "edge_separability_summary_v3.json")
    if not sep:
        return stop("MATCHING_FAIL", "separability_results",
                    "MISSING", out_dir / "edge_separability_summary_v3.json")
    dev = sep["splits"]["dev"]
    frozen_signs = {}
    for ab, stats in dev["by_ablation"].items():
        au = stats["macro_auroc"]
        frozen_signs[ab] = 1 if au >= 0.5 else -1
        stats["frozen_sign"] = frozen_signs[ab]
    # R0 control must be null on dev under the preset convention (-distance);
    # pooled value used here is DESCRIPTIVE (matching-quality diagnostic).
    r0_au = dev["by_ablation"].get("R0_dist", {}).get("pooled_auroc_descriptive")
    if r0_au is None or not (DEV_NULL_BAND[0] <= r0_au <= DEV_NULL_BAND[1]):
        return stop("MATCHING_FAIL", "r0_control_null_on_dev",
                    "FAIL", out_dir / "edge_separability_summary_v3.json",
                    {"R0_dist_dev_pooled_auroc": r0_au,
                     "note": "descriptive matching-quality diagnostic"})
    # composite gates must carry signal on dev in the PRESET direction
    # (pair-macro point estimate; pooled never gates)
    for gate_ab in ("R4_c_mv", "R6_mv_and_surface"):
        s = dev["by_ablation"][gate_ab]
        if abs(s["macro_auroc"] - 0.5) < SIGN_DELTA_MIN:
            return stop("SEPARABILITY_NOT_DEMONSTRATED", "dev_gate_no_signal",
                        "FAIL", out_dir / "edge_separability_summary_v3.json",
                        {gate_ab: s["macro_auroc"],
                         "delta_min": SIGN_DELTA_MIN,
                         "note": "|macro_auroc-0.5|<0.05 means NOT "
                                 "DEMONSTRATED, not proven absence of signal"})
    record("dev_gate_direction_frozen", "PASS",
           out_dir / "edge_separability_summary_v3.json",
           {"frozen_signs": frozen_signs})

    # --- 8. held-out evaluated ONCE with frozen signs ----------------------
    ho_sha = sha256_file(out_dir / "edge_separability_summary_v3.json")
    heldout = sep["splits"]["heldout"]
    sign_flips = []
    for gate_ab in ("R4_c_mv", "R6_mv_and_surface"):
        s = heldout["by_ablation"][gate_ab]
        signed = _signed_auc(s["macro_auroc"], frozen_signs[gate_ab])
        s["heldout_signed_macro_auroc"] = round(signed, 4)
        cb = s.get("cluster_bootstrap") or {}
        lo, hi = cb.get("lo"), cb.get("hi")
        if signed < HELDOUT_MIN_AUROC:
            # classify failure mode against the RAW macro AUROC
            raw = s["macro_auroc"]
            if frozen_signs[gate_ab] == 1 and hi is not None and hi < 0.5 \
                    and raw < 0.5 - SIGN_DELTA_MIN:
                s["failure_mode"] = "reversed_signal_present"
            elif abs(raw - 0.5) < SIGN_DELTA_MIN:
                s["failure_mode"] = "no_signal_not_demonstrated"
            else:
                s["failure_mode"] = "preset_direction_failed"
            return stop("SEPARABILITY_FAIL"
                        if s["failure_mode"] == "reversed_signal_present"
                        else "SEPARABILITY_NOT_DEMONSTRATED",
                        "heldout_gate_once",
                        "FAIL", out_dir / "edge_separability_summary_v3.json",
                        {gate_ab: {"macro_auroc": raw,
                                   "signed": signed,
                                   "mode": s["failure_mode"]}})
        if np.sign(s["macro_auroc"] - 0.5) != np.sign(
                dev["by_ablation"][gate_ab]["macro_auroc"] - 0.5):
            sign_flips.append(gate_ab)
    if sign_flips:
        return stop("SEPARABILITY_NOT_DEMONSTRATED", "sign_consistency",
                    "FAIL", out_dir / "edge_separability_summary_v3.json",
                    {"sign_flips": sign_flips,
                     "note": "dev/held-out directions disagree; neither "
                             "direction is confirmed"})
    record("heldout_gate_once", "PASS",
           out_dir / "edge_separability_summary_v3.json", {"sha256": ho_sha})

    # --- 9. sample floors (edges AND contact pairs) -------------------------
    gating_units_ok = all(
        units.get(f"{split}:ALL", {}).get("n_cross", 0) >= MIN_CROSS_PER_UNIT
        for split in ("dev", "heldout"))
    pairs_ok = True
    pair_counts = {}
    for split in ("dev", "heldout"):
        ab = sep["splits"][split]["by_ablation"].get("R4_c_mv", {})
        n_pairs = int(ab.get("n_pairs_degenerate_excluded_from_macro") is not None
                      and ab.get("n_pairs", 0))
        usable = ab.get("n_pairs", 0) - ab.get("n_pairs_degenerate_excluded_from_macro", 0)
        pair_counts[split] = usable
        if usable < MIN_PAIRS_PER_SPLIT:
            pairs_ok = False
    verdict = ("INSUFFICIENT_SAMPLE"
               if (not gating_units_ok or gating_units_insufficient or not pairs_ok)
               else "SEPARABILITY_PASS")

    task6_allowed = (verdict == "SEPARABILITY_PASS")
    return {"verdict": verdict, "task6_allowed": task6_allowed,
            "checks": checks, "first_failure": None,
            "frozen_signs": frozen_signs,
            "min_pairs_per_split_frozen": MIN_PAIRS_PER_SPLIT,
            "usable_pairs_per_split": pair_counts,
            "summary_sha256": ho_sha,
            "provenance": prov,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--output", default=None)
    ar = ap.parse_args()
    repo_root = Path(ar.repo_root).resolve() if ar.repo_root else REPO_DEFAULT
    result = evaluate(repo_root)
    out = Path(ar.output) if ar.output else \
        repo_root / "outputs" / "task5r_v3_1" / "verdict.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("verdict", "task6_allowed", "first_failure")},
                     indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
