#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 distance-matched edge construction.

Fixes the v2 class-imbalance defect (within prevalence ~99.985% made AUROC
dominated by the trivial distance covariate). For every cross-leaf edge we
sample within-leaf edges from the SAME plant + contact pair + 3D distance bin.

Hard architectural rule: matching is SCORE-BLIND. The function signature
admits only geometry (distances, labels, strata, seed) — no R1-R6 scores can
legally enter. A unit test passes poisoned score arrays through a side channel
and asserts the matched output is byte-identical.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MATCHER_VERSION = "task5r-v3-distmatch-1v1"
BIN_WIDTH_M = 0.01                 # frozen fixed-width distance bin
MAX_MERGE_BINS = 10                # upward merge cap before a cross edge is unmatchable
MIN_CROSS_PER_UNIT = 30            # analysis-unit sample floor
PREVALENCE_BAND = (0.45, 0.55)     # 1:1 gate
CONTROL_AUROC_BAND = (0.45, 0.55)  # -distance control gate after matching


def _stratum_rng(seed: int, case_id: str, bin_lo: float) -> np.random.Generator:
    """Deterministic per-stratum RNG derived from (seed, case, bin)."""
    h = hashlib.sha256(f"{seed}|{case_id}|{bin_lo:.6f}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


@dataclass
class MatchedEdges:
    case_id: np.ndarray          # (M,) str
    label: np.ndarray            # (M,) bool — True=within (matched), False=cross (anchor)
    gauss_a: np.ndarray          # (M,) int64 absolute endpoint ids
    gauss_b: np.ndarray
    distance_m: np.ndarray       # (M,) float64
    dist_bin_lo: np.ndarray      # (M,) float64
    match_group: np.ndarray      # (M,) int64 — group id; anchor cross edge shares it with its matches
    variant: np.ndarray          # (M,) str "1v1"/"1v5"
    unmatchable_cross: List[int] = field(default_factory=list)
    merged_bin_events: List[dict] = field(default_factory=list)


def match_within_for_cross(distances: np.ndarray,
                           labels_within: np.ndarray,
                           case_ids: np.ndarray,
                           gauss_a: np.ndarray,
                           gauss_b: np.ndarray,
                           seed: int = 0,
                           ratio: int = 1,
                           bin_width_m: float = BIN_WIDTH_M) -> MatchedEdges:
    """1:N distance matching of within edges to each cross edge.

    labels_within: True for within-leaf edges. Cross edges are anchors.
    Strata = (case_id, floor(distance/bin_width)); sparse strata merge upward.
    Sampling uses only distances/labels/ids — never any similarity score.
    """
    d = np.asarray(distances, dtype=np.float64)
    w = np.asarray(labels_within, dtype=bool)
    cid = np.asarray(case_ids)
    ga = np.asarray(gauss_a, dtype=np.int64)
    gb = np.asarray(gauss_b, dtype=np.int64)

    bins = np.floor(d / bin_width_m).astype(np.int64)
    out_case, out_lab, out_ga, out_gb, out_d, out_bin, out_grp, out_var = \
        [], [], [], [], [], [], [], []
    unmatchable: List[int] = []
    merged_events: List[dict] = []

    for ci_ in np.unique(cid):
        sel_c = np.where(cid == ci_)[0]
        cross_idx = sel_c[~w[sel_c]]
        within_idx = sel_c[w[sel_c]]
        if len(cross_idx) == 0:
            continue
        # per-stratum pools for this case
        pool: Dict[Tuple[str, int], List[int]] = {}
        for wi in within_idx:
            pool.setdefault((str(ci_), int(bins[wi])), []).append(wi)
        for anchor in cross_idx:
            b0 = int(bins[anchor])
            taken: List[int] = []
            merges = 0
            # expanding window [b0-m, b0+m] upward-biased merge
            for m in range(0, MAX_MERGE_BINS + 1):
                cand = []
                cand += pool.get((str(ci_), b0 - m), [])
                if m > 0:
                    cand += pool.get((str(ci_), b0 + m), [])
                rng = _stratum_rng(seed, str(ci_), b0 + 0.001 * m)
                order = rng.permutation(len(cand))
                need = ratio * 1 - len(taken)
                take = [cand[i] for i in order[:max(need, 0)]]
                taken += take
                if len(taken) >= ratio:
                    break
                if m > 0:
                    merges += 1
            if len(taken) < ratio:
                unmatchable.append(int(anchor))
                continue
            if merges:
                merged_events.append({"case": str(ci_), "anchor": int(anchor),
                                      "bin_lo": b0 * bin_width_m, "merges": merges})
            gid_base = len(out_case)
            grp = int(anchor)  # group id = absolute index of the cross anchor
            # anchor row first
            out_case.append(str(ci_)); out_lab.append(False)
            out_ga.append(ga[anchor]); out_gb.append(gb[anchor])
            out_d.append(d[anchor]); out_bin.append(b0 * bin_width_m)
            out_grp.append(grp); out_var.append("1v1" if ratio == 1 else f"1v{ratio}")
            for wi in taken:
                out_case.append(str(ci_)); out_lab.append(True)
                out_ga.append(ga[wi]); out_gb.append(gb[wi])
                out_d.append(d[wi]); out_bin.append(bins[wi] * bin_width_m)
                out_grp.append(grp); out_var.append("1v1" if ratio == 1 else f"1v{ratio}")

    return MatchedEdges(
        case_id=np.array(out_case), label=np.array(out_lab, dtype=bool),
        gauss_a=np.array(out_ga, dtype=np.int64), gauss_b=np.array(out_gb, dtype=np.int64),
        distance_m=np.array(out_d, dtype=np.float64),
        dist_bin_lo=np.array(out_bin, dtype=np.float64),
        match_group=np.array(out_grp, dtype=np.int64),
        variant=np.array(out_var),
        unmatchable_cross=unmatchable, merged_bin_events=merged_events)


def matching_gates(me: MatchedEdges, control_scores_neg_dist: np.ndarray,
                   control_scores_dist: np.ndarray) -> dict:
    """Post-matching gates: prevalence band + control-AUROC band.

    control_scores_neg_dist must be -distance (larger = more within-like under
    the preset convention). If either gate fails → MATCHING_FAIL downstream.
    """
    from core.task_stats import auroc
    n_within = int(me.label.sum())
    n_cross = int((~me.label).sum())
    prev = n_within / max(n_within + n_cross, 1)
    au_neg = auroc(control_scores_neg_dist, me.label)
    au_pos = auroc(control_scores_dist, me.label)
    gates = {
        "n_within": n_within, "n_cross": n_cross,
        "prevalence": prev,
        "control_auroc_negdist": au_neg,
        "control_auroc_dist": au_pos,
        "unmatchable_cross_count": len(me.unmatchable_cross),
        "merged_bin_event_count": len(me.merged_bin_events),
    }
    ok_prev = PREVALENCE_BAND[0] <= prev <= PREVALENCE_BAND[1] if n_cross else False
    ok_ctrl = (au_neg is not None and CONTROL_AUROC_BAND[0] <= au_neg <= CONTROL_AUROC_BAND[1]) \
        if n_cross else False
    frac_unmatch = len(me.unmatchable_cross) / max(n_cross + len(me.unmatchable_cross), 1)
    gates["gates_passed"] = bool(ok_prev and ok_ctrl and frac_unmatch <= 0.05)
    gates["prevalence_gate_passed"] = bool(ok_prev)
    gates["control_gate_passed"] = bool(ok_ctrl)
    gates["unmatchable_frac"] = frac_unmatch
    return gates


def sufficient_sample(n_cross_per_unit: Dict[str, int],
                      min_cross: int = MIN_CROSS_PER_UNIT) -> dict:
    """Per-analysis-unit sample floors. Units below min are excluded from
    macro claims and flagged; an empty/gating-failing unit ⇒ INSUFFICIENT_SAMPLE."""
    return {u: {"n_cross": int(n), "sufficient": bool(int(n) >= min_cross)}
            for u, n in n_cross_per_unit.items()}
