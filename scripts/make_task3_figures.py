#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3 figure generation (Agg headless, Chinese labels).

Figures:
  F1 Horizontal: merge_level & PQ vs severity, heat/euclidean/G4 + boundary markers
  F2 Vertical:   shortcut_ratio & cross_leaf_path vs severity, 4 curves
  F3 Failure boundary comparison across backends (bar chart)
  F4 Ablation:   ablation summary (G0-G6 comparison if available)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T3 = os.path.join(_REPO_ROOT, "outputs", "task3")
_FIGDIR = os.path.join(_T3, "figures")

H_LEVELS = ["H0", "HF1", "HF2", "HF3", "HF4", "H1", "H2", "H3", "H4"]
V_LEVELS = ["V0", "VF1", "VF2", "VF3", "VF4", "V1", "V2", "V3", "V4"]


def collect_backends(subdir: str, pair: str, mode: str):
    """Return {backend: {severity: metrics_dict}} for a given subdir/pair/mode.

    Handles both flat layout (heat: <sev>/heat/failure_metrics.json) and
    nested layout (graph backends: <sev>/<backend>/<config_dir>/failure_metrics.json).
    For nested configs, picks the FIRST config dir found under each backend.
    """
    bdir = os.path.join(_T3, subdir, pair, mode)
    if not os.path.isdir(bdir):
        return {}
    out = {}
    for backend in os.listdir(bdir):
        fp = os.path.join(bdir, backend, "failure_metrics.json")
        if os.path.exists(fp):
            with open(fp) as f:
                m = json.load(f)
            out.setdefault(backend, {})[m["severity"]] = m
            continue
        # nested: <backend>/<config_dir>/failure_metrics.json
        bpath = os.path.join(bdir, backend)
        if not os.path.isdir(bpath):
            continue
        for cdir in os.listdir(bpath):
            cfp = os.path.join(bpath, cdir, "failure_metrics.json")
            if os.path.exists(cfp):
                with open(cfp) as f:
                    m = json.load(f)
                out.setdefault(backend, {})[m["severity"]] = m
                break
    return out


def fig_horizontal_dev(pair: str, subdirs: list[str], outpath: str):
    """F1: Horizontal merge_level and PQ vs severity (heat/euclidean/G4)."""
    sevs = H_LEVELS
    backends_to_plot = ["heat", "euclidean", "surface"]
    backend_labels = {"heat": "Heat（baseline）", "euclidean": "Euclid-G0",
                      "surface": "Ours-G4（gate）"}
    colors = {"heat": "#888888", "euclidean": "#2196F3", "surface": "#E53935"}
    line_styles = {"heat": "--", "euclidean": "-", "surface": "-"}

    # merge level subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for sub in subdirs:
        for bname in backends_to_plot:
            data = collect_backends(sub, pair, "horizontal")
            if bname not in data:
                continue
            xs = []
            ys_merge = []
            ys_pq = []
            for s in sevs:
                if s in data[bname]:
                    m = data[bname][s]
                    xs.append(sevs.index(s))
                    ys_merge.append(m["geodesic"]["merge_level"])
                    ys_pq.append(m["instance"]["PQ"])
            if not xs:
                continue
            label = backend_labels.get(bname, bname)
            ax1.plot(xs, ys_merge, line_styles[bname], color=colors[bname],
                     marker="o", markersize=4, label=label)
            ax2.plot(xs, ys_pq, line_styles[bname], color=colors[bname],
                     marker="o", markersize=4, label=label)
    ax1.set_ylabel("merge_level（跨叶融合实例数）")
    ax2.set_ylabel("PQ（实例匹配质量）")
    ax2.set_xlabel("severity level（severity 等级）")
    ax2.set_xticks(range(len(sevs)))
    ax2.set_xticklabels(sevs, rotation=45, ha="right", fontsize=9)
    ax1.set_title(f"F1 Horizontal — {pair}")
    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[OK] {outpath}")


def fig_vertical_dev(pair: str, subdirs: list[str], outpath: str):
    """F2: Vertical shortcut_ratio vs severity."""
    sevs = V_LEVELS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    backend_labels = {"heat": "Heat（baseline）", "euclidean": "Euclid-G0",
                      "surface": "Ours-G4（gate）"}
    colors = {"heat": "#888888", "euclidean": "#2196F3", "surface": "#E53935"}
    for sub in subdirs:
        for bname in ["heat", "euclidean", "surface"]:
            data = collect_backends(sub, pair, "vertical")
            if bname not in data:
                continue
            xs, ys_ratio, ys_cross = [], [], []
            for s in sevs:
                if s in data[bname]:
                    m = data[bname][s]
                    idx = sevs.index(s)
                    xs.append(idx)
                    sc = m.get("shortcut", {})
                    ys_ratio.append(sc.get("shortcut_ratio"))
                    ys_cross.append(1 if sc.get("cross_leaf_path", False) else 0)
            label = backend_labels.get(bname, bname)
            valid = [(x, r) for x, r in zip(xs, ys_ratio) if r is not None]
            if valid:
                vx, vr = zip(*valid)
                ax1.plot(list(vx), list(vr), "o-", color=colors[bname], label=label)
            ax2.step(xs, ys_cross, where="mid", color=colors[bname], label=label)
    ax1.set_ylabel("shortcut_ratio（distance 比值 / V0）")
    ax2.set_ylabel("cross_leaf_path（是否穿越跨叶路径）")
    ax2.set_xlabel("severity level")
    ax2.set_xticks(range(len(sevs)))
    ax2.set_xticklabels(sevs, rotation=45, ha="right", fontsize=9)
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax1.set_title(f"F2 Vertical — {pair}")
    ax1.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[OK] {outpath}")


def fig_boundary_comparison(pair: str, outpath: str):
    """F3: Bar chart of mechanism_onset and final_failure level index per backend."""
    boundary = json.load(open(os.path.join(_T3, "failure_boundary_summary.csv").replace(".csv", ".json"))
                         ) if os.path.exists(os.path.join(_T3, "failure_boundary_summary.json")) else None
    # fallback: read from csv
    import csv
    rows = []
    with open(os.path.join(_T3, "failure_boundary_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["pair"] == pair:
                rows.append(r)
    if not rows:
        print(f"[skip F3] no boundary data for {pair}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode, sevs in zip(axes, ["horizontal", "vertical"], [H_LEVELS, V_LEVELS]):
        mode_rows = [r for r in rows if r["mode"] == mode]
        labels = []
        vals = []
        colors = []
        color_map = {"heat": "#888888", "euclidean": "#2196F3", "surface": "#E53935"}
        for r in mode_rows:
            bev = r["backend"]
            onset = r["mechanism_onset"]
            idx = sevs.index(onset) if onset and onset in sevs else -1
            labels.append(bev)
            vals.append(idx)
            colors.append(color_map.get(bev, "#999"))
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("机制触发severity索引（越晚越好）")
        ax.set_title(f"{'水平' if mode=='horizontal' else '垂直'} 失败边界 — {pair}")
        ax.set_ylim(-0.5, len(sevs))
        ax.set_yticks(range(len(sevs)))
        ax.set_yticklabels(sevs, fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[OK] {outpath}")


def main():
    os.makedirs(_FIGDIR, exist_ok=True)
    pair = "plant1_green_pepper_pair_8_4"
    subdirs = [d for d in ("dev", "test", "ablation")
               if os.path.isdir(os.path.join(_T3, d, pair))]
    fig_horizontal_dev(pair, subdirs, os.path.join(_FIGDIR, "F1_horizontal.png"))
    fig_vertical_dev(pair, subdirs, os.path.join(_FIGDIR, "F2_vertical.png"))
    fig_boundary_comparison(pair, os.path.join(_FIGDIR, "F3_boundaries.png"))
    print("[OK] all figures saved")


if __name__ == "__main__":
    main()
