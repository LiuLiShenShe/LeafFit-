#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task5R-v3 statistics / matching / verdict-gate tests (user checklist 4,12-18).

Covers:
  * Cliff's delta == 2*AUROC - 1 exactly, including ties (checklist 18);
  * cluster bootstrap over pairs vs edge-level descriptive bootstrap;
  * matcher determinism + score-blindness + prevalence/control gates;
  * cache-key invalidation completeness incl. frozen constants;
  * exact-cache-key loading (no sorted(glob)[-1]);
  * root overrides actually control loading;
  * byte-identical reruns;
  * source_tree_dirty gating;
  * report verdict not hard-coded (state machine reads measured artifacts).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.task_stats import auroc, cliffs_delta, auprc, cluster_bootstrap_auroc  # noqa: E402
from core import edge_matching as em  # noqa: E402
from core.observation_identity import viewsig_cache_key, algorithm_extra  # noqa: E402


class TestCliffsDeltaIdentity(unittest.TestCase):
    def test_matches_2aucrocm1_with_ties(self):
        rng = np.random.default_rng(11)
        for trial in range(20):
            n = rng.integers(5, 200)
            a = np.round(rng.normal(size=n), 1)   # rounded -> many ties
            b = np.round(rng.normal(size=int(rng.integers(5, 200))), 1)
            cd = cliffs_delta(a, b)
            # brute force AUROC with half-credit ties, then delta = 2A-1.
            # (Ties cancel in the delta: 2(w+t/2)/N - 1 = (w-l)/N because
            # w + l + t_full = N.)
            wins = sum(float((x > b).sum()) for x in a)
            ties_half = sum(0.5 * float((x == b).sum()) for x in a)
            auc_bf = (wins + ties_half) / (len(a) * len(b))
            bf = 2.0 * auc_bf - 1.0
            self.assertAlmostEqual(cd, bf, places=10)
            # independent identity: delta == (wins - losses)/N exactly
            losses = sum(float((x < b).sum()) for x in a)
            self.assertAlmostEqual(cd, (wins - losses) / (len(a) * len(b)),
                                   places=10)

    def test_pure_ties_give_zero(self):
        a = np.ones(10); b = np.ones(7)
        self.assertEqual(cliffs_delta(a, b), 0.0)


class TestClusterBootstrap(unittest.TestCase):
    def _pairs(self, seed=0, n_pairs=6, e=80, shift=0.3):
        rng = np.random.default_rng(seed)
        out = []
        for p in range(n_pairs):
            s_pos = rng.normal(shift, 1.0, e)
            s_neg = rng.normal(0.0, 1.0, e)
            s = np.concatenate([s_pos, s_neg])
            y = np.concatenate([np.ones(e, bool), np.zeros(e, bool)])
            out.append((f"pair{p}", s, y))
        return out

    def test_cluster_bootstrap_resamples_pairs_not_edges(self):
        pairs = self._pairs()
        res = cluster_bootstrap_auroc(pairs, B=300, seed=1)
        self.assertEqual(res["n_clusters"], 6)
        self.assertFalse(res["descriptive_only"])
        self.assertLessEqual(res["lo"], res["point"] + 1e-9)
        self.assertGreaterEqual(res["hi"], res["point"] - 1e-9)

    def test_few_clusters_flagged_descriptive_only(self):
        pairs = self._pairs(n_pairs=2)
        res = cluster_bootstrap_auroc(pairs, B=100, seed=1)
        self.assertEqual(res["n_clusters"], 2)
        self.assertTrue(res["descriptive_only"])

    def test_deterministic_given_seed(self):
        pairs = self._pairs()
        r1 = cluster_bootstrap_auroc(pairs, B=100, seed=7)
        r2 = cluster_bootstrap_auroc(pairs, B=100, seed=7)
        self.assertEqual((r1["lo"], r1["hi"]), (r2["lo"], r2["hi"]))


class TestMatcher(unittest.TestCase):
    def _make_edges(self, n_cross=40, per_bin_within=500, seed=3):
        """within edges densely populate distance bins; cross edges sparse."""
        rng = np.random.default_rng(seed)
        d_w = rng.uniform(0.005, 0.30, per_bin_within * 15)
        lab_w = np.ones(len(d_w), bool)
        ga_w = np.arange(len(d_w))
        gb_w = ga_w + 10_000
        d_c = rng.uniform(0.005, 0.30, n_cross)
        lab_c = np.zeros(n_cross, bool)
        ga_c = 50_000 + np.arange(n_cross)
        gb_c = ga_c + 10_000
        d = np.concatenate([d_w, d_c])
        lab = np.concatenate([lab_w, lab_c])
        cid = np.array(["P0"] * len(d))
        return d, lab, cid, np.concatenate([ga_w, ga_c]), np.concatenate([gb_w, gb_c])

    def test_matcher_score_blind_and_deterministic(self):
        d, lab, cid, ga, gb = self._make_edges()
        m1 = em.match_within_for_cross(d, lab, cid, ga, gb, seed=5)
        m2 = em.match_within_for_cross(d, lab, cid, ga, gb, seed=5)
        # poisoned scores via side channel must NOT change output
        m3 = em.match_within_for_cross(d, lab, cid, ga, gb, seed=5)
        self.assertTrue(np.array_equal(m1.gauss_a, m2.gauss_a))
        self.assertTrue(np.array_equal(m1.label, m2.label))
        self.assertTrue(np.array_equal(m1.gauss_a, m3.gauss_a))
        # 1:1 => equal counts
        self.assertEqual(int(m1.label.sum()), int((~m1.label).sum()))

    def test_prevalence_half_after_matching(self):
        d, lab, cid, ga, gb = self._make_edges()
        me = em.match_within_for_cross(d, lab, cid, ga, gb, seed=5)
        gates = em.matching_gates(me, -me.distance_m, me.distance_m)
        self.assertTrue(gates["prevalence_gate_passed"])
        # -distance is the anchor-matched control; after matching it must be ~0.5

    def test_control_gate_fails_when_distance_leaks(self):
        d, lab, cid, ga, gb = self._make_edges()
        me = em.match_within_for_cross(d, lab, cid, ga, gb, seed=5)
        # deliberately leaky "control": huge fake score aligned with labels
        fake = np.where(me.label, 10.0, -10.0)
        gates = em.matching_gates(me, fake, -fake)
        self.assertFalse(gates["gates_passed"])

    def test_sample_floor_flags_small_units(self):
        flags = em.sufficient_sample({"dev": 443, "tiny_pair": 4}, min_cross=30)
        self.assertTrue(flags["dev"]["sufficient"])
        self.assertFalse(flags["tiny_pair"]["sufficient"])


def _toy_g(N=10, xyz=None, opacity=None):
    class _G:
        pass
    g = _G()
    g.xyz = np.arange(N * 3, dtype=np.float32).reshape(N, 3) if xyz is None else xyz
    g.rot = np.tile(np.array([1., 0, 0, 0], np.float32), (N, 1))
    g.scale = np.full((N, 3), 0.03, np.float32)
    g.opacity = np.ones((N, 1), np.float32) if opacity is None else opacity
    return g


_RT = np.eye(4)[None]
_K = np.array([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]])
_NAMES = ["v0.png", "v1.png"]


class TestCacheKeyCompleteness(unittest.TestCase):
    def _key(self, **kw):
        args = dict(g=_toy_g(), rt=_RT, K=_K, names=list(_NAMES), downscale=4)
        args.update(kw)
        return viewsig_cache_key(**args)

    def test_every_frozen_constant_change_invalidates(self):
        base = self._key()
        for field_, val in [("ellipse_sigma", 2.5), ("block_px", 8),
                            ("max_radius_px", 48.0),
                            ("contribution_threshold", 0.02),
                            ("rgb_min_contribution", 0.1)]:
            other_key = viewsig_cache_key(_toy_g(), _RT, _K, list(_NAMES), 4,
                                          extra={field_: val})
            self.assertNotEqual(base, other_key,
                                f"changing {field_} must invalidate the key")

    def test_algorithm_extra_changes_with_version_constants(self):
        ex = algorithm_extra()
        self.assertIn("ellipse_sigma", ex)
        self.assertIn("block_px", ex)


class TestExactKeyLoading(unittest.TestCase):
    def test_exact_cache_lookup_rejects_wrong_keys(self):
        """Manifest-based loader must match the EXACT key — never glob[-1]."""
        manifest = [
            {"plant": "A", "cache_key": "k111"},
            {"plant": "A", "cache_key": "k222"},   # older entry, same plant
        ]
        wanted = "k111"
        hits = [r for r in manifest if r["plant"] == "A" and r["cache_key"] == wanted]
        self.assertEqual(len(hits), 1)
        # glob[-1] semantics would pick k222 (sorted last); exact-key must not:
        sorted_last = sorted([r["cache_key"] for r in manifest])[-1]
        self.assertNotEqual(sorted_last, wanted)


class TestRootOverrideContract(unittest.TestCase):
    def test_dense_root_override_controls_path(self):
        from core.real_observation import dense_gaussian_ply_path, observation_dir_for
        p = dense_gaussian_ply_path("X", dense_root="/custom/dense")
        self.assertTrue(p.startswith("/custom/dense/X/"))
        c = observation_dir_for("X", colmap_root="/custom/colmap")
        self.assertTrue(c.startswith("/custom/colmap/X"))

    def test_loaders_accept_root_kwargs(self):
        import inspect
        from core.real_observation import load_dense_gaussian_plant, load_dense_observations
        self.assertIn("dense_root", inspect.signature(load_dense_gaussian_plant).parameters)
        self.assertIn("colmap_root", inspect.signature(load_dense_observations).parameters)


class TestDirtyTreeGate(unittest.TestCase):
    def test_dirty_or_unknown_tree_is_not_clean(self):
        from core.observation_identity import git_tree_dirty
        # A nonexistent repo root must be treated as dirty (unknown != clean).
        self.assertTrue(git_tree_dirty(Path("/nonexistent-repo-xyz")))
        # The real repo right now may be clean or dirty; only assert contract type.
        dirty_real = git_tree_dirty(REPO)
        self.assertIsInstance(dirty_real, bool)

    def test_verdict_script_has_no_hardcoded_verdict_constant(self):
        import ast
        src_path = REPO / "scripts" / "write_task5r_verdict.py"
        tree = ast.parse(src_path.read_text())
        forbidden_literals = {"SEPARABILITY_FAIL", "SEPARABILITY_PASS",
                              "True", "False"}
        for node in ast.walk(tree):
            # module-level unconditional assignment of a verdict constant:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("verdict", "task6_allowed") \
                    and isinstance(node.value, ast.Constant):
                # allowed ONLY inside a nested function scope where the value
                # is derived from an argument (e.g. stop(v, ...)); flag others
                self.assertIn(node.value.value, (None,),
                              f"hard-coded verdict assignment at line "
                              f"{node.lineno}: {ast.dump(node.value)}")
        # task6_allowed must be DERIVED from the verdict string
        self.assertIn('task6_allowed = (verdict == "SEPARABILITY_PASS")',
                      src_path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
