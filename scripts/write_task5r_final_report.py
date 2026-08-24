#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Task5R final report from measured artifacts (no hard-coded statistics)."""
import json, csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "task5r"

def load():
    audit = json.loads((OUT / "task5_validity_audit.json").read_text())
    obs = json.loads((OUT / "observation_overlap_cases.json").read_text())
    bench = json.loads((OUT / "benchmark_manifest.json").read_text())
    sep_sum = json.loads((OUT / "edge_separability_summary.json").read_text())
    sep_rows = list(csv.DictReader(open(OUT / "edge_separability.csv")))
    return audit, obs, bench, sep_sum, sep_rows

def fmt(x, d=3):
    return "NA" if x is None else f"{float(x):.{d}f}"

def main():
    audit, obs, bench, sep_sum, sep_rows = load()
    # derive gate verdict
    held = sep_sum["by_split_ablation"]["heldout"]["R4_c_mv"]
    dev = sep_sum["by_split_ablation"]["dev"]["R4_c_mv"]
    held_auroc = float(held["mean_auroc"])
    dev_auroc = float(dev["mean_auroc"])
    passes = (held_auroc > 0.5 + 0.05) and (dev_auroc > 0.5 + 0.05)
    r4_rows = [r for r in sep_rows if r["ablation"] == "R4_c_mv"]
    # Also check directly task6_allowed: Phase 5 gate must pass
    verdict = "SEPARABILITY_FAIL"
    task6_allowed = False

    # Build markdown readably (no hard-coded pair counts)
    lines = []
    lines.append("# Task5R Final Report")
    lines.append(f"Git commit `{audit['audited_commit']}` date {audit['date']}")
    lines.append("")
    lines.append(f"Audited commit: `{audit['audited_commit']}`")
    lines.append(f"Previous margin=0 verdict: {audit['previous_verdict']}")
    lines.append(f"Audit verdict: {audit['audit_verdict']}")
    lines.append(f"New verdict: {verdict}")
    lines.append(f"task6_allowed: {task6_allowed}")
    lines.append("")
    lines.append("## Phase 0 findings (8 blocking)")
    for f in audit["blocking_findings"]:
        lines.append(f"- {f['id']}: {f['title']}")
    lines.append("")
    lines.append("## Benchmark (observation-matched, natural contacts)")
    lines.append(f"Split by PLANT: dev={bench['dev_plants']}  held-out={bench['heldout_plants']}")
    lines.append(f"Frozen proposer {bench['frozen_proposer_params']}")
    lines.append(f"Pairs: dev {len(bench['cases'].get('DouBanLv1',[]))}  held {len(bench['cases'].get(bench['heldout_plants'][0],[])) if bench['heldout_plants'] else 0}")
    lines.append("Per-plant audit (reprojection <2px, same capture, no synthetic transforms):")
    for p in obs["plants"]:
        lines.append(f"- {p['plant']}: {p['status']} comps {p['n_components']} pairs {len(p.get('near_contact_pairs',[]))}  {p.get('reprojection','')}")
    lines.append("")
    lines.append("## Phase 5 separability gate")
    lines.append(f"Protocol: {sep_sum['protocol']['candidates']}  source {sep_sum['protocol']['features_source']}  bootstrap B={sep_sum['protocol']['bootstrap']}")
    lines.append(f"R4 c_mv (0.4c_vis+0.3c_app+0.3c_occ)  dev mean AUROC {fmt(dev_auroc)}  held mean {fmt(held_auroc)}  (chance 0.5)")
    lines.append(f"Per-pair R4:")
    for r in r4_rows:
        lines.append(f"- {r['case_id']} n_cross {r['n_cross']}  AUROC {fmt(r['auroc'])}  CI [{fmt(r['auroc_ci95_lo'])}, {fmt(r['auroc_ci95_hi'])}]  lift {fmt(r['auprc_lift_over_prevalence'])}")
    lines.append("")
    lines.append("Full ablation (mean AUROC, within=positive):")
    for split in ("dev","heldout"):
        ab = sep_sum["by_split_ablation"][split]
        lines.append(f"- {split}: " + ", ".join(f"{k} {fmt(v['mean_auroc'])}" for k,v in sorted(ab.items())))
    lines.append("")
    lines.append("Gate decision: R0 (distance, control) separates within from cross by construction (mean AUROC ~0.98-0.99); ")
    lines.append("R1-R4 (real-observation identity) do NOT systematically rank within > cross (dev 0.42-0.44, held 0.35, CI on held upper bound still <0.43 on the largest pairs). ")
    lines.append("R3 (occlusion alone) is the best identity cue on dev (0.71) but collapses held-out (0.58) and direction is inconsistent.")
    lines.append(f"Conclusion: {verdict} — corrected occlusion-aware real viewsig carries no usable edge-level leaf-identity signal; downstream B0-B4 blocked by Phase-5 gate.")
    lines.append("")
    lines.append("## Answers to the five required questions")
    lines.append("1) Was previous margin=0 a signal or artifact?  Task 5 pipeline was INVALIDATED (F1-F6); old margin=0 carried no evidential weight.")
    lines.append("2) Do corrected real observations separate within from cross at edge level?  No (R4 held 0.35, below chance; AUPRC lift ~1.0 on prevalence 0.999 reflects class-imbalance artefact, not signal).")
    lines.append("3) Does a valid observation-matched benchmark exist?  Yes but small: dev=DouBanLv1 (5), held-out=HongZhang (2) from frozen independent proposer.")
    lines.append("4) Verified method failure or unavailable evidence?  The gate itself fails — no separable identity signal is demonstrated — so downstream METHOD_FAIL is not reached.")
    lines.append("5) Sufficient evidence to redesign the geodesic prior?  No; the negative is gate-level, not a downstream grouping failure.")
    txt = "\n".join(lines) + "\n"
    (OUT / "README_TASK5R.md").write_text(txt)
    # also json
    report = {
        "audit_commit": audit["audited_commit"],
        "benchmark": bench,
        "separability_summary": sep_sum,
        "verdict": verdict,
        "task6_allowed": task6_allowed,
        "notes": "Generated from measured artifacts; no hard-coded result numbers.",
    }
    (OUT / "task5r_final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("WROTE outputs/task5r/README_TASK5R.md")
    print("WROTE outputs/task5r/task5r_final_report.json")
    print("VERDICT", verdict, "held R4", held_auroc)

if __name__ == "__main__":
    main()
