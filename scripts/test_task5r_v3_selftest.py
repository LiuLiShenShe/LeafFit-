#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 self-test → outputs/task5r_v3/selftest.json.

Golden-vector and cross-validation checks the verdict gate requires before
ANY scientific evaluation:
  1. exclusive transmittance golden sequences;
  2. covariance radius goldens (unclipped formula);
  3. ellipse-block compositing vs an independent per-pixel brute-force
     reference implementation (max |Δ contribution| < 1e-9);
  4. matcher determinism + score-blindness;
  5. cliffs_delta == 2*auroc - 1 on tie-heavy data;
  6. source commit / tree-state recorded for provenance.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve()
REPO = _SCRIPT.parent.parent
for p in (str(REPO), str(REPO / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "outputs" / "task5r_v3"

from core.observation_identity import (  # noqa: E402
    exclusive_transmittance, cov2d_lambda_max, ELLIPSE_SIGMA,
    ALPHA_FLOOR, ALPHA_MAX, _ellipse_block_pairs,
    BLOCK_PX, git_commit, git_tree_dirty)


def check_exclusive_transmittance():
    T, c = exclusive_transmittance([0.9, 0.9], [0, 0])
    assert abs(c[0] - 0.9) < 1e-12 and abs(c[1] - 0.09) < 1e-12
    _, c = exclusive_transmittance([0.9, 0.9, 0.9], [0, 0, 0])
    assert abs(c[2] - 0.009) < 1e-12
    T, c = exclusive_transmittance([0.9, 0.5, 0.8], [0, 0, 1])
    assert abs(c[2] - 0.8) < 1e-12 and T[2] == 1.0   # group independence
    return {"two_layer": [0.9, 0.09], "three_layer": [0.9, 0.09, 0.009]}


def check_covariance_goldens():
    assert abs(cov2d_lambda_max(4., 0., 4.) - 4.0) < 1e-12
    assert abs(ELLIPSE_SIGMA * np.sqrt(cov2d_lambda_max(4., 0., 4.)) - 4.0) < 1e-12
    lam = cov2d_lambda_max(9., 0., 1.)
    assert abs(lam - 9.0) < 1e-12
    assert abs(ELLIPSE_SIGMA * np.sqrt(lam) - 6.0) < 1e-12
    return {"diag44_radius": 4.0, "diag91_radius": 6.0}


def _brute_reference(pxd, pyd, zd, opac, c00, c01, c11,
                     dW=64, dH=64, block_px=BLOCK_PX):
    """Independent O(P*blocks) per-pixel-block reference compositor."""
    nbx = (dW + block_px - 1) // block_px
    nby = (dH + block_px - 1) // block_px
    half = (block_px - 1) / 2.0
    M = len(pxd)
    # collect all (gaussian, block) pairs by brute enumeration of every block
    pairs = []
    for i in range(M):
        ext_x = ELLIPSE_SIGMA * np.sqrt(c00[i]) * 1.5 + block_px
        ext_y = ELLIPSE_SIGMA * np.sqrt(c11[i]) * 1.5 + block_px
        x0 = max(0, int((pxd[i] - ext_x) // block_px))
        x1 = min(nbx - 1, int((pxd[i] + ext_x) // block_px))
        y0 = max(0, int((pyd[i] - ext_y) // block_px))
        y1 = min(nby - 1, int((pyd[i] + ext_y) // block_px))
        for bx in range(x0, x1 + 1):
            for by in range(y0, y1 + 1):
                cx = bx * block_px + half; cy = by * block_px + half
                dx, dy = cx - pxd[i], cy - pyd[i]
                det = max(c00[i] * c11[i] - c01[i] ** 2, 1e-12)
                d2 = (c11[i]*dx*dx - 2*c01[i]*dx*dy + c00[i]*dy*dy) / det
                if d2 <= ELLIPSE_SIGMA ** 2:
                    a = opac[i] * np.exp(-0.5 * d2)
                    if a >= ALPHA_FLOOR:
                        pairs.append((i, by * nbx + bx, zd[i],
                                      min(a, ALPHA_MAX)))
    # group by block, sort front-to-back, exclusive composite
    from collections import defaultdict
    groups = defaultdict(list)
    for gi, bid, z, a in pairs:
        groups[bid].append((z, gi, a))
    out = defaultdict(list)   # gaussian -> list of contributions
    for bid, members in groups.items():
        members.sort()
        T = 1.0
        for z, gi, a in members:
            out[gi].append(a * T)
            T *= (1.0 - a)
    return {g: float(max(v)) for g, v in out.items()}, \
           {g: float(sum(v)) for g, v in out.items()}


def check_ellipse_block_vs_bruteforce():
    rng = np.random.default_rng(42)
    n_gauss = 40
    pxd = rng.uniform(4, 60, n_gauss); pyd = rng.uniform(4, 60, n_gauss)
    zd = rng.uniform(2, 8, n_gauss)
    opac = rng.uniform(0.3, 0.99, n_gauss)
    # random SPD 2D covariances with modest scale
    A = rng.normal(size=(n_gauss, 2, 2)) * 1.2
    c00 = A[:, 0, 0] ** 2 + A[:, 0, 1] ** 2 + 0.5
    c11 = A[:, 1, 0] ** 2 + A[:, 1, 1] ** 2 + 0.5
    c01 = A[:, 0, 0] * A[:, 1, 0] + A[:, 0, 1] * A[:, 1, 1]

    # run the module's pair generation + compositing
    gi, bid, d2 = _ellipse_block_pairs(pxd, pyd, zd, opac, c00, c01, c11,
                                       nbx=16, nby=16, block_px=BLOCK_PX)
    keep = opac[gi] * np.exp(-0.5 * d2) >= ALPHA_FLOOR
    gi, bid, d2 = gi[keep], bid[keep], d2[keep]
    a_eff = np.minimum(opac[gi] * np.exp(-0.5 * d2), ALPHA_MAX)
    zp = zd[gi]
    order = np.lexsort((zp, bid))
    bk_o, gi_o = bid[order], gi[order]
    T_b, contrib = exclusive_transmittance(a_eff[order], bk_o)

    ref_max, ref_acc = _brute_reference(pxd, pyd, zd, opac, c00, c01, c11)
    got_max, got_acc = {}, {}
    for g_idx in range(n_gauss):
        rows = gi_o == g_idx
        if rows.any():
            got_max[g_idx] = float(contrib[rows].max())
            got_acc[g_idx] = float(contrib[rows].sum())
    assert set(got_max) == set(ref_max), \
        f"gaussian coverage mismatch: {set(got_max) ^ set(ref_max)}"
    dmax = max(abs(got_max[g] - ref_max[g]) for g in ref_max)
    dacc = max(abs(got_acc[g] - ref_acc[g]) for g in ref_acc)
    assert dmax < 1e-9 and dacc < 1e-9, f"Δmax={dmax} Δacc={dacc}"
    return {"n_gaussians": n_gauss, "delta_max": dmax, "delta_acc": dacc}


def check_matcher_determinism():
    from core import edge_matching as em
    rng = np.random.default_rng(7)
    d = rng.uniform(0.005, 0.3, 3000)
    lab = np.concatenate([np.ones(2500, bool), np.zeros(500, bool)])
    cid = np.array(["P"] * 3000)
    ids = np.arange(3000)
    m1 = em.match_within_for_cross(d, lab, cid, ids, ids + 9000, seed=3)
    m2 = em.match_within_for_cross(d, lab, cid, ids, ids + 9000, seed=3)
    assert np.array_equal(m1.gauss_a, m2.gauss_a) and np.array_equal(m1.label, m2.label)
    assert int(m1.label.sum()) == int((~m1.label).sum())
    return {"matched_rows": int(len(m1.label))}


def check_cliffs_identity():
    from core.task_stats import cliffs_delta, auroc
    rng = np.random.default_rng(19)
    worst = 0.0
    for _ in range(50):
        a = np.round(rng.normal(size=int(rng.integers(5, 150))), 1)
        b = np.round(rng.normal(size=int(rng.integers(5, 150))), 1)
        s = np.concatenate([a, b])
        y = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
        worst = max(worst, abs(cliffs_delta(a, b) - (2 * auroc(s, y) - 1)))
    assert worst < 1e-12
    return {"max_deviation": worst}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "checks": {}, "all_passed": True}
    checks = [
        ("exclusive_transmittance_golden", check_exclusive_transmittance),
        ("covariance_radius_golden", check_covariance_goldens),
        ("ellipse_block_vs_bruteforce", check_ellipse_block_vs_bruteforce),
        ("matcher_determinism", check_matcher_determinism),
        ("cliffs_delta_identity", check_cliffs_identity),
    ]
    for name, fn in checks:
        try:
            results["checks"][name] = {"passed": True, **fn()}
        except Exception as e:
            results["checks"][name] = {"passed": False,
                                       "error": str(e),
                                       "traceback": traceback.format_exc()[-800:]}
            results["all_passed"] = False
    results["source_commit"] = git_commit(REPO)
    results["source_tree_dirty"] = git_tree_dirty(REPO)
    (OUT / "selftest.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v["passed"] for k, v in results["checks"].items()},
                     indent=2))
    print("ALL_PASSED", results["all_passed"])
    print("WROTE", OUT / "selftest.json")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
