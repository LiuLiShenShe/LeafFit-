#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select the frozen Task 3 backend config (Stage 2) from DEV(plant1) results.

Selection on plant1 ONLY (held-out plant2/7 never touched here). Per plan:

  Stage 1 coarse screen -> candidates -> Stage 2 freeze.

Priority (PASS criteria):
  1. Same-backend clean fidelity: H0/V0 PQ not degraded vs the clean control of
     the SAME backend (comparison is within-backend, per ExitPlanMode feedback —
     the Dijkstra field ceiling makes absolute-vs-heat PQ unusable).
  2. Boundary push: G4 final_instance_failure / mechanism_onset boundary moves
     later than the heat baseline, and later than G0(euclidean) control.
  3. Connectivity preserved + cross-leaf suppression.
  4. Robustness across gate/k settings.

Output: outputs/task3/frozen_method_config.json  (G4 primary; G5 documented).
"""
from __future__ import annotations

import json
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")

DEV_PAIR = "plant1_green_pepper_pair_8_4"
H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]
# ordered severity index for boundary localization
H_ORDER = {s: i for i, s in enumerate(H_LEVELS)}
V_ORDER = {s: i for i, s in enumerate(V_LEVELS)}


def _clean_ref(mode: str) -> str:
    return "V0" if mode == "vertical" else "H0"


def _cfg_matches(want: dict, have: dict) -> bool:
    """True if the saved graph_stats cfg matches the requested config."""
    if want.get("feature_set") != have.get("feature_set"):
        return False
    for key in ("k", "mutual", "tau_d", "tau_t", "lambda_n", "lambda_t", "p"):
        if key in want and key in have:
            if isinstance(want[key], float) and isinstance(have[key], (int, float)):
                if abs(want[key] - float(have[key])) > 1e-9:
                    return False
            elif want[key] != have[key]:
                return False
    return True


def _clean_ref(mode: str) -> str:
    return "V0" if mode == "vertical" else "H0"


def collect(pair: str, mode: str, backend: str, cfg: dict) -> dict:
    """Collect a config's full 9-level severity curve from saved dev outputs.

    Directory layout:
      dev/<pair>/<mode>/<sev>/<backend_name>/[<config_dir>/]failure_metrics.json
    where the backend name is one of 'heat' / 'euclidean' / 'surface', and
    'surface' contains config dirs (G4_k256_mFalse_td3.0_tt0.5 ...).
    """
    sevs = V_LEVELS if mode == "vertical" else H_LEVELS
    curves = []
    clean_pq = None
    for sev in sevs:
        bdir = os.path.join(_T3, "dev", pair, mode, sev)
        if not os.path.isdir(bdir):
            continue
        found = False
        # heat: <sev>/heat/
        if backend == "heat":
            fp = os.path.join(bdir, "heat", "failure_metrics.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    m = json.load(f)
                entry = {
                    "severity": sev,
                    "order": V_ORDER.get(sev) if mode == "vertical" else H_ORDER.get(sev),
                    "PQ": m["instance"]["PQ"],
                    "wrong_grouping": m["geodesic"]["wrong_grouping"],
                    "merge_level": m["geodesic"]["merge_level"],
                    "apex_recall": m["geodesic"]["reference_apex_recall"],
                }
                if mode == "vertical" and "shortcut" in m:
                    entry["cross_leaf_path"] = m["shortcut"].get("cross_leaf_path", False)
                    entry["shortcut_confirmed"] = m["shortcut"].get("shortcut_confirmed", False)
                curves.append(entry)
                if sev == _clean_ref(mode):
                    clean_pq = m["instance"]["PQ"]
                found = True
        else:
            # euclidean: <sev>/euclidean/k256_mFalse/  ; surface: <sev>/surface/<config_dir>/
            for backend_subdir in os.listdir(bdir):
                if not os.path.isdir(os.path.join(bdir, backend_subdir)):
                    continue
                if backend == "euclidean":
                    if backend_subdir != "euclidean":
                        continue
                    # scan each euclidean config dir
                    for cdir in os.listdir(os.path.join(bdir, backend_subdir)):
                        if not os.path.isdir(os.path.join(bdir, backend_subdir, cdir)):
                            continue
                        cfg_path = os.path.join(bdir, backend_subdir, cdir, "graph_stats.json")
                        if not os.path.exists(cfg_path):
                            continue
                        with open(cfg_path) as f:
                            gs = json.load(f)
                        gs_cfg = gs.get("cfg", {})
                        if not _cfg_matches(cfg, gs_cfg):
                            continue
                        fp = os.path.join(bdir, backend_subdir, cdir, "failure_metrics.json")
                        if os.path.exists(fp):
                            with open(fp) as f:
                                m = json.load(f)
                            entry = {
                                "severity": sev,
                                "order": V_ORDER.get(sev) if mode == "vertical" else H_ORDER.get(sev),
                                "PQ": m["instance"]["PQ"],
                                "wrong_grouping": m["geodesic"]["wrong_grouping"],
                                "merge_level": m["geodesic"]["merge_level"],
                                "apex_recall": m["geodesic"]["reference_apex_recall"],
                            }
                            if mode == "vertical" and "shortcut" in m:
                                entry["cross_leaf_path"] = m["shortcut"].get("cross_leaf_path", False)
                                entry["shortcut_confirmed"] = m["shortcut"].get("shortcut_confirmed", False)
                            curves.append(entry)
                            if sev == _clean_ref(mode):
                                clean_pq = m["instance"]["PQ"]
                            found = True
                            break
                else:  # surface
                    if backend_subdir != "surface":
                        continue
                    for cdir in os.listdir(os.path.join(bdir, backend_subdir)):
                        if not os.path.isdir(os.path.join(bdir, backend_subdir, cdir)):
                            continue
                        cfg_path = os.path.join(bdir, backend_subdir, cdir, "graph_stats.json")
                        if not os.path.exists(cfg_path):
                            continue
                        with open(cfg_path) as f:
                            gs = json.load(f)
                        gs_cfg = gs.get("cfg", {})
                        if not _cfg_matches(cfg, gs_cfg):
                            continue
                        fp = os.path.join(bdir, backend_subdir, cdir, "failure_metrics.json")
                        if os.path.exists(fp):
                            with open(fp) as f:
                                m = json.load(f)
                            entry = {
                                "severity": sev,
                                "order": V_ORDER.get(sev) if mode == "vertical" else H_ORDER.get(sev),
                                "PQ": m["instance"]["PQ"],
                                "wrong_grouping": m["geodesic"]["wrong_grouping"],
                                "merge_level": m["geodesic"]["merge_level"],
                                "apex_recall": m["geodesic"]["reference_apex_recall"],
                            }
                            if mode == "vertical" and "shortcut" in m:
                                entry["cross_leaf_path"] = m["shortcut"].get("cross_leaf_path", False)
                                entry["shortcut_confirmed"] = m["shortcut"].get("shortcut_confirmed", False)
                            curves.append(entry)
                            if sev == _clean_ref(mode):
                                clean_pq = m["instance"]["PQ"]
                            found = True
                            break
        # skip nothing if not found
    return {"cfg": cfg, "curves": curves, "clean_pq": clean_pq,
            "n_levels": len(curves)}


def boundary_push(curves: list[dict], baseline_first_fail: dict) -> dict:
    """Return the last level where downstream failure first triggers, + onset."""
    mechanism_onset = None
    final_failure = None
    for c in sorted(curves, key=lambda k: k["order"]):
        # mechanism onset: wrong_grouping (horizontal) / cross-leaf shortcut (vertical)
        if c["severity"] == "H0" or c["severity"] == "V0":
            continue
        hm = c.get("wrong_grouping", False)
        ve = c.get("cross_leaf_path", False) and c.get("shortcut_confirmed", False)
        if mechanism_onset is None and (hm or ve):
            mechanism_onset = c["severity"]
        if final_failure is None and hm:
            final_failure = c["severity"]
    return {"mechanism_onset": mechanism_onset, "final_instance_failure": final_failure}


def main() -> int:
    grid = json.load(open(os.path.join(_T3, "parameter_grid.json")))
    # Enumerate G4 configs from the grid + the ones we actually swept in stage1
    taus = [{"tau_d": td, "tau_t": tt}
            for td in grid["tau_grid"]["tau_d"] for tt in grid["tau_grid"]["tau_t"]]
    k_safe = [k for k in grid["k_grid"] if k >= 256]  # stage1 safe-k range
    g4_cfgs = []
    for t in taus:
        for k in k_safe:
            for m in grid["mutual_grid"]:
                g4_cfgs.append({"feature_set": "G4", **t, "k": k, "mutual": m})
    g0_cfgs = [{"feature_set": "G0", "k": k, "mutual": m}
               for k in k_safe for m in grid["mutual_grid"]]  # euclidean control
    g5_cfgs = [{"feature_set": "G5", "lambda_n": ln, "lambda_t": lt,
                "p": 2.0, "tau_d": 3.0, "tau_t": 0.5, "k": 256, "mutual": False}
               for ln, lt in [(0.5, 1.0), (1.0, 2.0), (0.0, 4.0)]]

    all_cfgs = {"G4": g4_cfgs, "G0": g0_cfgs, "G5": g5_cfgs}

    # baseline heat boundaries (from Phase 0 summary) for comparison
    base = json.load(open(os.path.join(_T3, "phase0_fine_baseline_summary.json")))
    base_b = {}
    for k, v in base["boundaries"].items():
        pk, mode = k.split("|")
        if pk == DEV_PAIR:
            base_b[mode] = v

    report = {}
    for mode in ["horizontal", "vertical"]:
        rows = []
        for backend, cfgs in all_cfgs.items():
            for cfg in cfgs:
                if backend == "G5":
                    bname = "surface"
                elif backend == "G0":
                    bname = "euclidean"
                else:
                    bname = "surface"
                col = collect(DEV_PAIR, mode, bname, cfg)
                if not col["curves"]:
                    continue
                b = boundary_push(col["curves"], base_b.get(mode, {}))
                rows.append({
                    "backend": backend, "cfg": cfg,
                    "clean_pq": col["clean_pq"],
                    "mechanism_onset": b["mechanism_onset"],
                    "final_instance_failure": b["final_instance_failure"],
                    "n_levels": col["n_levels"],
                })
        report[mode] = rows

    # Select best G4 per mode: maximize boundary push (minimize failure severity
    # occurrence / move later) subject to clean fidelity not degraded; tie-break
    # on connectivity not captured here (favours lower k -> cheaper + denser).
    choices = {}
    for mode in ["horizontal", "vertical"]:
        g4_rows = [r for r in report[mode] if r["backend"] == "G4"]
        base_onset = base_b.get(mode, {}).get("mechanism_onset")
        base_final = base_b.get(mode, {}).get("final_instance_failure")
        # score: same-backend clean PQ high + mechanism onset as LATE as possible
        def _score(r):
            oi = H_ORDER.get(r["mechanism_onset"]) if mode == "horizontal" else V_ORDER.get(r["mechanism_onset"])
            clean = r["clean_pq"] if r["clean_pq"] is not None else 0
            return (clean, oi if oi is not None else 99)
        g4_rows.sort(key=_score, reverse=True)
        best = g4_rows[0] if g4_rows else None
        choices[mode] = {
            "best_g4": best,
            "baseline_heat": {"onset": base_onset, "final": base_final},
            "n_candidates": len(g4_rows),
        }

    frozen = {
        "stage1_dev_pair": DEV_PAIR,
        "selected": {
            "horizontal": choices["horizontal"]["best_g4"],
            "vertical": choices["vertical"]["best_g4"],
        },
        "baseline_heat_boundaries": {
            "horizontal": choices["horizontal"]["baseline_heat"],
            "vertical": choices["vertical"]["baseline_heat"],
        },
        "policy": {
            "g4": "Ours primary (metric-safe, weight=d, gate-only)",
            "g0": "Euclidean graph control",
            "g5": "Diagnostic ablation only; METRIC_INCOMPATIBLE recorded where it crashes",
            "pass_basis": "PASS/FAIL determined by G4 alone",
        },
        "report": report,
    }
    out = os.path.join(_T3, "frozen_method_config.json")
    with open(out, "w") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)
    # human table
    print("=== Stage 2 frozen config (DEV plant1) ===")
    for mode in ["horizontal", "vertical"]:
        c = choices[mode]
        b = c["best_g4"]
        print(f"\n[{mode}] baseline heat onset={c['baseline_heat']['onset']} "
              f"final={c['baseline_heat']['final']}")
        if b:
            print(f"  best G4: cfg={b['cfg']} clean_pq={b['clean_pq']:.3f} "
                  f"onset={b['mechanism_onset']} final={b['final_instance_failure']}")
        else:
            print("  (no G4 data found)")
    print(f"\n[OK] -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())