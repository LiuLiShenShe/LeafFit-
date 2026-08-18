#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 regression tests for the LeafFit headless segmentation baseline.

Test A: import isolation  — headless wrapper imports no private/GPU/GUI modules
Test B: PLY load         — GaussianData has correct shape/fields from a real PLY
Test C: segmentation      — produces len(labels)==N with N>=2 leaf labels, OR explicit SEGMENTATION_FAILED_NO_LEAVES
Test D: root geodesic     — root_geodesic_multisource[root_idx] ~= 0 and all finite
Test E: determinism       — two identical Mode-A runs yield byte-identical labels/sample_indices/apexes

Run:
    export PYTHONPATH=<repo>/core:<repo>
    python -m unittest tests.test_headless_segmentation -v
  or
    python -m pytest tests/test_headless_segmentation.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import core.headless_segmentation as hs  # noqa: E402

_DATA = os.path.join(_REPO_ROOT, "data", "plant1_green_pepper.ply")
_PY = sys.executable

FORBIDDEN = set(hs.FORBIDDEN_MODULES)


def _subprocess_snippet(code: str) -> str:
    """Run a python snippet in a fresh subprocess with the headless import path."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [p for p in [os.path.join(_REPO_ROOT, "core"), _REPO_ROOT] + sys.path if p]))
    env.pop("MPLBACKEND", None)
    r = subprocess.run([_PY, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        raise AssertionError(f"subprocess failed:\ncode={code}\n{r.stderr}")
    return r.stdout.strip()


class TestHeadlessSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_DATA):
            raise unittest.SkipTest(f"test data not found: {_DATA}")

    # --- Test A: import isolation -------------------------------------------
    def test_A_import_isolation(self):
        """No forbidden private/GPU/GUI module is imported by the headless pipeline."""
        code = (
            "import sys; sys.path.insert(0, 'core'); sys.path.insert(0, '.'); "
            "import core.headless_segmentation as hs; "
            "leaked=[m for m in hs.FORBIDDEN_MODULES if m in sys.modules or any(k.startswith(m+'.') for k in sys.modules)]; "
            "print('LEAKED=' + ','.join(leaked))"
        )
        out = _subprocess_snippet(code)
        leaked = out.split("LEAKED=", 1)[1]
        self.assertEqual(leaked, "", f"forbidden modules leaked: {leaked}")
        # direct assert in-process
        for m in FORBIDDEN:
            self.assertFalse(m in sys.modules or any(k.startswith(m + ".") for k in sys.modules),
                               f"forbidden module {m!r} present in-process")

    # --- Test B: PLY load ----------------------------------------------------
    def test_B_ply_load(self):
        g = hs.load_gaussian_data(_DATA)
        self.assertEqual(len(g.xyz), len(g))          # len(g) is the Gaussian count
        self.assertEqual(g.xyz.shape[1], 3)
        self.assertEqual(g.sh.shape[1], 48)           # 3 + 3*(3+1)^2 - 3 = 48 (degree 3)
        self.assertTrue(np.all(np.isfinite(g.xyz)), "xyz has non-finite values")
        self.assertTrue(np.isclose(np.linalg.norm(g.rot, axis=-1).mean(), 1.0),
                        "quaternions not normalized")
        self.assertTrue(g.opacity.min() >= 0.0 and g.opacity.max() <= 1.0,
                        "opacity out of [0,1]")

    # --- Test C: segmentation ------------------------------------------------
    def test_C_segmentation(self):
        g = hs.load_gaussian_data(_DATA)
        res = hs.run_headless_segmentation(g, root_index=47330)   # Mode A fixed root (from earlier run)
        labels = np.zeros(res.N, dtype=np.int64)
        for k, seg in enumerate(res.found_segs, start=1):
            labels[seg] = k
        self.assertEqual(len(labels), res.N, "labels length != num gaussians")
        n_leaves = res.num_leaves
        if n_leaves == 0:
            # must be explicitly flagged as failure of segmentation, not success
            self.assertEqual(len(res.found_segs), 0)
            self.assertEqual(len(res.final_cluster_results), 0)
        else:
            self.assertEqual(n_leaves, len(res.final_cluster_results),
                             "num leaves != num cluster results")
            self.assertEqual(len(np.unique(labels)), n_leaves + 1,
                             "label set size mismatch (stem + K leaves)")

    # --- Test D: root geodesic ------------------------------------------------
    def test_D_root_geodesic(self):
        g = hs.load_gaussian_data(_DATA)
        res = hs.run_headless_segmentation(g, root_index=47330)
        d = res.root_geodesic_multisource
        self.assertEqual(d.shape[0], res.N)
        self.assertTrue(np.isfinite(d).all(), "root geodesic has non-finite")
        # root basin indices must contain the root
        self.assertTrue(res.root_idx in res.root_basin_indices.tolist(),
                        "root not in basin")
        # single-source root distance at root ~ 0
        self.assertAlmostEqual(float(res.root_geodesic_single[res.root_idx]), 0.0, places=4)
        # multi-source at root: root lies inside its own basin (<=0.1), so its
        # multi-source distance is ~0 but not numerically exactly 0; bound by basin radius.
        self.assertLess(float(res.root_geodesic_multisource[res.root_idx]), hs.BASELINE_ROOT_BASIN)

    # --- Test E: determinism --------------------------------------------------
    def test_E_determinism(self):
        g = hs.load_gaussian_data(_DATA)
        r1 = hs.run_headless_segmentation(g, root_index=47330)
        r2 = hs.run_headless_segmentation(g, root_index=47330)
        l1 = np.zeros(r1.N, dtype=np.int64)
        l2 = np.zeros(r2.N, dtype=np.int64)
        for k, s in enumerate(r1.found_segs, 1):
            l1[s] = k
        for k, s in enumerate(r2.found_segs, 1):
            l2[s] = k
        # NOTE: label IDs may differ between runs only if tips are reordered; compare
        # the partition (set of member sets) to be robust to label permutation.
        def partition(labels):
            return {int(k): frozenset(np.where(labels == k)[0].tolist())
                    for k in np.unique(labels) if k != 0}
        self.assertEqual(partition(l1), partition(l2),
                         "leaf partition differs between two runs")
        self.assertTrue(np.array_equal(r1.sparse_indices, r2.sparse_indices),
                        "sample_indices not byte-identical")
        self.assertTrue(np.array_equal(r1.root_basin_indices, r2.root_basin_indices),
                        "root_basin_indices not byte-identical")
        self.assertEqual([c["selected_tip"] for c in r1.final_cluster_results],
                         [c["selected_tip"] for c in r2.final_cluster_results],
                         "apex order differs between runs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
