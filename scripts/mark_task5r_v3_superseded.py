#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mark the Task5R-v3 verdict as superseded (INVALIDATED_BY_IMPLEMENTATION_AUDIT).

Programmatic invalidation — the original verdict.json is left BYTE-IDENTICAL
(it is the faithful record of what was measured under the buggy v3 code).
This script writes:
  * outputs/task5r_v3/verdict_superseded.json  — copy + invalidation fields;
  * outputs/task5r_v3/INVALIDATED_BY_TASK5R_V3_1.md — audit findings.

Run once; idempotent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
REPO_DEFAULT = _SCRIPT.parent.parent

AUDIT_FINDINGS = [
    {
        "id": "rgb_peak_block_argmax",
        "first_failure": True,
        "location": "core/observation_identity.py:448-467 (v3)",
        "defect": ("RGB peak-block selection lexsort'ed by (-contrib, loc) and "
                   "took the LAST row of each group run, which selects the "
                   "group-MINIMUM contribution, not the argmax. rgb_views / "
                   "rgb_valid were therefore keyed to the WORST block."),
    },
    {
        "id": "max_radius_not_enforced_in_enumeration",
        "location": "core/observation_identity.py:303-308 (v3)",
        "defect": ("MAX_RADIUS_PX clipped only the REPORTED radius "
                   "(cov2d_radius_px); _ellipse_block_pairs enumerated "
                   "candidates from the unclipped covariance extent, so the "
                   "manifest's footprint_radius_clip claim was false."),
    },
    {
        "id": "pooled_edge_statistics_as_formal",
        "location": "scripts/summarize_task5r_v3.py (v3) + verdict gate",
        "defect": ("Pooled-edge AUROC was used as the formal gate statistic "
                   "although contact pairs are the independent inference "
                   "unit; no pair-macro point estimate or per-pair table "
                   "was produced."),
    },
    {
        "id": "heldout_sign_transform_error",
        "location": "scripts/write_task5r_verdict.py:181 (v3)",
        "defect": ("Held-out signed AUROC computed as auc * (-1) for frozen-"
                   "negative directions; correct transform is 1 - auc "
                   "(AUROC lives on [0,1])."),
    },
]


def main() -> int:
    repo = REPO_DEFAULT
    out_dir = repo / "outputs" / "task5r_v3"
    vpath = out_dir / "verdict.json"
    if not vpath.exists():
        print("REFUSING: outputs/task5r_v3/verdict.json not found")
        return 2
    raw = vpath.read_bytes()
    verdict = json.loads(raw)
    sup = dict(verdict)
    sup["superseded_by"] = "task5r-v3.1"
    sup["invalidation"] = "INVALIDATED_BY_IMPLEMENTATION_AUDIT"
    sup["first_failure_audit"] = "rgb_peak_block_argmax"
    sup["task6_allowed_final"] = False
    sup["audit_findings"] = [f["id"] for f in AUDIT_FINDINGS]
    sup["original_verdict_sha256"] = hashlib.sha256(raw).hexdigest()
    sup["invalidated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    spath = out_dir / "verdict_superseded.json"
    spath.write_text(json.dumps(sup, indent=2))

    md = ["# Task5R-v3 INVALIDATED by implementation audit (v3.1)", "",
          f"Original verdict.json sha256: `{hashlib.sha256(raw).hexdigest()}`",
          f"Original verdict: **{verdict.get('verdict')}**, first_failure: "
          f"`{verdict.get('first_failure')}`", "",
          "The SEPARABILITY_FAIL verdict of v3 must NOT be cited as a final "
          "scientific conclusion. Audit findings:", ""]
    for i, f in enumerate(AUDIT_FINDINGS, 1):
        md.append(f"{i}. **{f['id']}** ({f['location']}) — {f['defect']}")
    md += ["",
           "Superseding run: Task5R-v3.1 (outputs/task5r_v3_1/), "
           "VISIBILITY_VERSION task5r-alpha-v3.1-rgbargmax.",
           "task6_allowed remains FALSE regardless of any downstream result."]
    (out_dir / "INVALIDATED_BY_TASK5R_V3_1.md").write_text("\n".join(md))
    assert vpath.read_bytes() == raw, "original verdict.json must stay byte-identical"
    print("WROTE", spath)
    print("WROTE", out_dir / "INVALIDATED_BY_TASK5R_V3_1.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
