#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3 final report and summary (FAIL verdict).

Writes task3_final_report.json and prints a human-readable summary.
Aggregates: frozen config, held-out boundary comparison, cross-leaf
pruning evidence, ablation notes.
"""
from __future__ import annotations

import csv
import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")


def main() -> int:
    # --- Load core data ---
    frozen = json.load(open(os.path.join(_T3, "frozen_method_config.json")))
    phase0 = json.load(open(os.path.join(_T3, "phase0_fine_baseline_summary.json")))
    boundaries = list(csv.DictReader(open(os.path.join(_T3, "failure_boundary_summary.csv"))))
    benchmarks = list(csv.DictReader(open(os.path.join(_T3, "benchmark_summary.csv"))))

    # --- Dev (plant1) summary ---
    dev_pairs = ["plant1_green_pepper_pair_8_4"]
    # --- Held-out ---
    ho_pairs = ["plant2_rubber_tree_pair_3_12", "plant7_black_pearl_pepper_pair_4_8"]

    # Cross-leaf pruning fractions (from manual audit above)
    pruning = {
        "plant1": {"V0": {"within": 0.96, "cross": 1.87},
                    "V1": {"within": 0.97, "cross": 21.43}},
        "plant2": {"V0": {"within": 7.47, "cross": 5.94},
                    "V1": {"within": 7.51, "cross": 22.23}},
        "plant7": {"V0": {"within": 9.55, "cross": 4.43},
                    "V1": {"within": 9.61, "cross": 15.58}},
    }
    # Median c_t overlap (within vs cross at V0, G4 graph)
    ct_overlap = {
        "plant1": {"within": 0.103, "cross": 0.159},
        "plant2": {"within": 0.234, "cross": 0.246},
        "plant7": {"within": 0.215, "cross": 0.211},
    }

    # --- Summary ---
    summary = {
        "verdict": "FAIL",
        "frozen_config": frozen.get("selected", {}),
        "dev_findings": {
            "plant1_boundary": "HF1/VF1 (identical across G4/G0/heat)",
            "plant1_clean_pq": "0.889 (G4 m=False) vs 1.0 (heat baseline)",
            "gate_pruning_at_V0": "1.9% cross-leaf pruned; <1% within-leaf",
            "gate_pruning_at_V1": "21.4% cross-leaf pruned; <1% within-leaf",
            "conclusion": (
                "Gate prunes cross-leaf edges only when they are perturbed (V1). "
                "On clean geometry (V0), the gate cannot distinguish cross-leaf "
                "from within-leaf edges — median c_t differs by only 0.056. "
                "No boundary push on dev (all onsets = HF1/VF1)."
            ),
        },
        "heldout_findings": {
            "plant2": {
                "euclidean_boundary": "HF1 / VF1 (clean_pq=0.895)",
                "G4_boundary": "HF1 / VF1 (clean_pq=0.720)",
                "verdict": "G4 strictly worse (lower clean PQ, same boundary)",
                "pruning_V0": "7.5% within pruned vs 5.9% cross — INVERTED selectivity",
            },
            "plant7": {
                "heat_boundary": "HF3 / VF1 (clean_pq=1.0)  [heat pushes H boundary to HF3]",
                "euclidean_boundary": "HF1 / VF1 (clean_pq=0.735)",
                "G4_tt0.5_boundary": "HF1 / VF1 (clean_pq=0.039) — catastrophic",
                "G4_tt0.75_boundary": "HF1 / VF1 (clean_pq=0.735) — inert, same as euclid",
                "G4_tt1.0_boundary": "H1 / V1 (clean_pq=0.735) — slightly later",
                "verdict": (
                    "G4 at tt0.5 DESTROYS clean fidelity (PQ 0.039 vs heat 1.0). "
                    "At conservative gates (tt0.75/1.0) G4 = euclid (gate inert). "
                    "G4 does NOT improve over euclid; heat baseline (HF3) is already better."
                ),
                "pruning_V0": "9.6% within pruned vs 4.4% cross — gate destroys within-leaf",
            },
        },
        "gate_mechanism_fails": {
            "reason": (
                "The surface gate (G4) prunes a HIGHER fraction of within-leaf edges "
                "than cross-leaf edges on clean geometry (plant2 V0: 7.5% within vs "
                "5.9% cross; plant7 V0: 9.6% within vs 4.4% cross). This is the "
                "opposite of the desired behaviour."
            ),
            "c_t_overlap": (
                "At V0 (clean), median c_t (tangent) for within-leaf edges is "
                "essentially identical to cross-leaf edges: plant1 0.103 vs 0.159, "
                "plant2 0.234 vs 0.246 (near-identical), plant7 0.215 vs 0.211 "
                "(cross is LOWER). The tangent surface cue cannot distinguish "
                "topologically separate leaves from legitimate within-leaf connections "
                "when leaves are spatially adjacent."
            ),
        },
        "pass_criteria_checklist": [
            "Criterion 1 (Held-out boundary push): FAIL — G4 boundaries = HF1/VF1 everywhere, "
            "same as euclid; heat is at HF3 on plant7 horizontal.",
            "Criterion 2 (Clean fidelity preserved): FAIL — G4 tt0.5 on plant7: PQ=0.039 vs "
            "euclid 0.735 vs heat 1.0 (catastrophic degradation).",
            "Criterion 3 (Full > Euclidean): FAIL — G4 < euclid on plant2 (PQ 0.720 vs 0.895); "
            "G4 = euclid on plant7 at conservative gates.",
            "Criterion 4 (≥1 surface cue contributes): PARTIAL — c_n shows separation at V1 "
            "(plant1 cross cn=0.238 vs within 0.037), but the gate cannot selectively prune "
            "based on it at clean geometry due to c_t overlap.",
        ],
        "scientific_conclusion": (
            "Surface-aware Gaussian connectivity (G4 gate-only) FAILS to prevent "
            "geodesic shortcuts between spatially adjacent but topologically disconnected "
            "leaf surfaces. The tangent surface cue (c_t) does not separate cross-leaf "
            "from within-leaf edges at clean geometry — median c_t values overlap almost "
            "completely (plant2: within=0.234, cross=0.246; plant7: within=0.215, cross=0.211). "
            "The gate therefore either: (a) prunes more within-leaf than cross-leaf edges "
            "(plant7 V0: 9.6% within vs 4.4% cross), destroying clean fidelity; or "
            "(b) at conservative thresholds becomes completely inert (G4 tt0.75 = euclid). "
            "At perturbed V1, cross-leaf c_t increases and pruning becomes selective (21.4% "
            "cross vs <1% within on plant1), but the geodesic shortcut already exists at V1 — "
            "the gate is too late. Coplanar/adjacent leaves are geometrically "
            "indistinguishable at the kNN-graph level using only local surface normals and "
            "tangent angles. This motivates multi-view or semantic cues as the next stage."
        ),
        "next_stage_motivation": (
            "Since local surface geometry (normals, tangent, curvature) cannot resolve "
            "coplanar/adjacent leaves, the next stage should incorporate multi-view "
            "consistency (depth-based occlusion reasoning) or semantic identity features "
            "(appearance/textural differences between leaves) to identify cross-leaf "
            "connectivity. Pure geometry is insufficient for this problem."
        ),
    }

    # Write JSON
    out = os.path.join(_T3, "task3_final_report.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] {out}")

    # Print human-readable summary
    print("\n" + "="*72)
    print("TASK 3 FINAL REPORT")
    print("="*72)
    print(f"VERDICT: {summary['verdict']}")
    print()
    print("--- Frozen Config ---")
    for mode in ["horizontal", "vertical"]:
        c = summary["frozen_config"].get(mode, {})
        print(f"  [{mode}] cfg = {json.dumps(c.get('cfg', {}))}, "
              f"clean_pq = {c.get('clean_pq', '?')}, "
              f"onset = {c.get('mechanism_onset', '?')}")
    print()
    print("--- Held-out (plant2 + plant7) ---")
    hf = summary["heldout_findings"]
    for pk in ["plant2", "plant7"]:
        d = hf[pk]
        print(f"\n  {pk}:")
        for k, v in d.items():
            print(f"    {k}: {v}")
    print()
    print("--- Gate Mechanism Failure ---")
    gm = summary["gate_mechanism_fails"]
    print(f"  {gm['reason']}")
    print(f"  c_t overlap: {gm['c_t_overlap']}")
    print()
    print("--- PASS Criteria ---")
    for c in summary["pass_criteria_checklist"]:
        print(f"  {'[X]' if c.startswith('FAIL') else '[ ]'} {c}")
    print()
    print("--- Scientific Conclusion ---")
    print(f"  {summary['scientific_conclusion']}")
    print()
    print(f"--- Next Stage ---")
    print(f"  {summary['next_stage_motivation']}")
    print("="*72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
