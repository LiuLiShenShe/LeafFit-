#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 canonical statistics.

Fixes the v2 defects:
  * cliffs_delta is EXACTLY 2*AUROC - 1 (midrank AUROC handles ties; the v2
    loop version truncated a[:4000] in array order and dropped tied mass);
  * cluster_bootstrap_auroc resamples CONTACT PAIRS (clusters), not edges —
    edge-level bootstrap is provided only as a descriptive extra.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def auroc(scores, labels) -> Optional[float]:
    """Rank-based AUROC with midranks for ties. Positive class = True."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0 or len(s) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1.0, len(s) + 1.0)
    # midrank correction for ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cliffs_delta(a, b) -> Optional[float]:
    """Cliff's delta = P(a>b) - P(a<b) with half-credit ties.
    Computed as 2*AUROC(a over b) - 1: algebraically identical for the full
    sample (no truncation, ties handled by midranks)."""
    au = auroc(np.concatenate([np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64)]),
               np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)]))
    return None if au is None else 2.0 * au - 1.0


def auprc(scores, labels) -> Optional[float]:
    """Average precision, positive class = True."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0 or len(s) == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y_sorted).sum() / n_pos)


def cluster_bootstrap_auroc(pairs: Sequence[Tuple[str, np.ndarray, np.ndarray]],
                            B: int = 1000, seed: int = 0,
                            ci=(2.5, 97.5)) -> dict:
    """Cluster bootstrap over contact pairs (the analysis clusters).

    pairs: list of (case_id, scores(E,), labels(E,) with positive=True).
    Point estimate = unweighted macro mean of per-pair AUROCs.
    Each bootstrap replicate resamples PAIRS with replacement and averages
    their per-pair AUROCs (a pair drawn twice counts twice).
    Returns {point, lo, hi, n_clusters, B}. NOTE: with few clusters (<5) the
    CI is descriptive only and must not support PASS/FAIL inference.
    """
    per_pair = []
    ids = []
    for cid, s, y in pairs:
        a = auroc(s, y)
        if a is not None:
            per_pair.append(a)
            ids.append(cid)
    n_clusters = len(per_pair)
    if n_clusters == 0:
        return {"point": None, "lo": None, "hi": None,
                "n_clusters": 0, "B": B, "descriptive_only": True}
    rng = np.random.default_rng(seed)
    vals = []
    arr = np.asarray(per_pair, dtype=np.float64)
    for _ in range(B):
        pick = rng.integers(0, n_clusters, n_clusters)
        vals.append(float(arr[pick].mean()))
    return {
        "point": float(arr.mean()),
        "lo": float(np.percentile(vals, ci[0])),
        "hi": float(np.percentile(vals, ci[1])),
        "n_clusters": n_clusters, "B": B,
        "descriptive_only": n_clusters < 5,
        "cluster_ids": ids,
    }


def edge_bootstrap_auroc(scores, labels, B: int = 1000, seed: int = 0,
                         ci=(2.5, 97.5)) -> dict:
    """DESCRIPTIVE ONLY — edge-level i.i.d. bootstrap. Never cite as an
    inferential CI: edges within a plant/pair are strongly clustered."""
    y = np.asarray(labels, dtype=bool)
    idx_pos = np.where(y)[0]
    idx_neg = np.where(~y)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return {"point": auroc(scores, labels), "lo": None, "hi": None,
                "B": B, "descriptive_only": True}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        take = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                               rng.choice(idx_neg, len(idx_neg), replace=True)])
        v = auroc(scores[take], labels[take])
        if v is not None:
            vals.append(v)
    return {
        "point": auroc(scores, labels),
        "lo": float(np.percentile(vals, ci[0])),
        "hi": float(np.percentile(vals, ci[1])),
        "B": B, "descriptive_only": True,
    }
