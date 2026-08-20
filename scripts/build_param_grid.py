#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the full (lambda_n x lambda_t) parameter grid for Task 3.

Per plan Stage 1: a complete 5 x 6 = 30 grid search (NOT coordinate search):
  lambda_n in {0, 0.5, 1, 2, 4}
  lambda_t in {0, 0.5, 1, 2, 4, 8}

For G4 (gate-only) lambda is irrelevant (weight = d_ij), so G4 is run at a
fixed representative lambda with the full tau_d x tau_t gate sweep and k x mutual
operating grid.  G5 (gate + soft penalty) consumes the full grid.

Output: outputs/task3/parameter_grid.json  (FROZEN for the dev run).
"""
from __future__ import annotations

import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTROOT = os.path.join(_REPO_ROOT, "outputs", "task3")

LAMBDA_N = [0.0, 0.5, 1.0, 2.0, 4.0]
LAMBDA_T = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
P = 2.0

# Gate operating grid (dev small sweep)
TAU_D = [3.0, 4.0, float("inf")]   # inf = no distance gate
TAU_T = [0.5, 0.75, float("inf")]

# Graph operating grid
K_GRID = [16, 32, 64]
MUTUAL = [False, True]

# Representative euclidean k for baseline B sweep on dev
EUCLIDEAN_K = [16, 32, 64]

# Fixed k / gate for the full grid search (Stage 1 coarse screen).
# k=64 balances runtime (~30 lambdas x 6 levels x 3 pairs = 540 heat-equivalent
# seg runs) with connectivity (G4 needs k>=128 on SOME overlap cases, but those
# are checked at Stage 2 with the frozen high-k operating point).
STAGE1_K = 64
STAGE1_TAU = {"tau_d": 3.0, "tau_t": 0.5}


def build_grid() -> dict:
    grid = {"lambda_n": LAMBDA_N, "lambda_t": LAMBDA_T, "p": P,
            "tau_grid": {"tau_d": TAU_D, "tau_t": TAU_T},
            "k_grid": K_GRID, "mutual_grid": MUTUAL,
            "euclidean_k": EUCLIDEAN_K,
            "stage1": {"k": STAGE1_K, "tau_d": STAGE1_TAU["tau_d"],
                       "tau_t": STAGE1_TAU["tau_t"]},
            "configs": []}
    combos = []
    for ln in LAMBDA_N:
        for lt in LAMBDA_T:
            combos.append({"lambda_n": ln, "lambda_t": lt})
    # G5 uses the full grid; G4 is gate-only (lambda irrelevant — set to
    # representative value) with the gate on.
    grid["lambda_configs"] = {
        "G5": {"feature_set": "G5", "lambda_configs": combos},
        "G4": {"feature_set": "G4", "lambda_configs": [
            {"lambda_n": 1.0, "lambda_t": 2.0}]},
    }
    return grid


def main() -> int:
    os.makedirs(_OUTROOT, exist_ok=True)
    grid = build_grid()
    out = os.path.join(_OUTROOT, "parameter_grid.json")
    with open(out, "w") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)
    print(f"[OK] {len(grid['lambda_configs']['G5']['lambda_configs'])} lambda configs "
          f"(5x6) -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
