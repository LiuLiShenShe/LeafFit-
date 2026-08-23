"""
Headless wrapper for the LeafFit paper (Section 3.1) automatic leaf instance segmentation.

This module runs the OFFICIAL LeafFit segmentation algorithm with ZERO dependency on the
private / GPU components of the upstream repo:

    - diff_gaussian_rasterization   (orphaned gitlink submodule, source unavailable)
    - gsplat_bvh / my_rasterizer    (GPU rasterizer extensions)
    - OpenGL / GLFW / imgui / viewer GUI
    - template_transform / gen_template_leaf  (post-segmentation mesh stages)

It provides a CLI-friendly, batch-runnable, fully-instrumented entry that captures ALL
intermediate state (root geodesic fields, sample indices, apexes, paths, tree, grouping,
petioles, leaf labels) so the result can be frozen as a baseline for later algorithm
comparison (Tasks 2/3).

Design principle: ZERO modification of the core algorithm files
(`core/auto_segment.py`, `core/apex_grouping.py`, `core/petiole_detection.py`).
Everything is captured by calling the official functions and re-deriving, post-hoc,
any state that the official functions do not return, WITHOUT altering the algorithm.

This module intentionally does NOT import:
    viewer, glfw, imgui, OpenGL, diff_gaussian_rasterization,
    gsplat_bvh, my_rasterizer, template_transform, gen_template_leaf
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import potpourri3d as pp3d
import fpsample
from plyfile import PlyData
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# Silence tqdm progress bars in headless mode (official code uses tqdm internally).
if os.environ.get("HEADLESS_QUIET_TQDM", "1") == "1":
    try:
        from tqdm import tqdm as _tqdm

        def _quiet_tqdm(*args, **kwargs):
            if "disable" not in kwargs:
                kwargs["disable"] = True
            return _tqdm(*args, **kwargs)
        sys.modules["tqdm"].tqdm = _quiet_tqdm  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass

from gaussian_utils import GaussianData, apply_indices_to_gaussian_data
from auto_segment import (
    fix_plant_root_direction_legacy,
    get_segment_mask,
    get_temperature_field,
)

# ---------------------------------------------------------------------------
# Frozen baseline constants (official upstream behavior, not tunable parameters).
# These match viewer/main.py::load_gaussian_file EXACTLY. Do not change them for
# the baseline; anything else is an experiment, not the official baseline.
# ---------------------------------------------------------------------------
BASELINE_NS = 8192                       # FPS sample count (paper Ns)
BASELINE_H = 7                           # FPS kdline sampling grid factor (upstream)
BASELINE_T_COEF = 1e8                    # potpourri3d heat solver t_coef
BASELINE_ROOT_BASIN = 0.1                # root-source geodesic basin radius
BASELINE_OPACITY_THRESHOLD = 0.0         # no opacity filtering -> index space preserved
BASELINE_METHOD = "geodesic_tip_graph"   # upstream g_segmentation_method=8

# Private / GPU modules that must NEVER be imported by this package.
FORBIDDEN_MODULES = (
    "viewer", "glfw", "imgui", "OpenGL", "diff_gaussian_rasterization",
    "gsplat_bvh", "my_rasterizer", "template_transform", "gen_template_leaf",
)


# ---------------------------------------------------------------------------
# PLY loader (GL-free re-implementation of viewer/utils.py::load_ply_gaussian)
# ---------------------------------------------------------------------------
def load_gaussian_data(ply_path: str) -> GaussianData:
    """Load a standard 3DGS Gaussian PLY into core.GaussianData.

    Faithful re-implementation of viewer/utils.py::load_ply_gaussian (lines 368-416)
    WITHOUT importing the OpenGL-bound viewer module. Field transforms are identical:
      - opacity  = sigmoid(ply value)
      - scales   = exp(ply value)
      - rots     = normalized quaternion
      - sh       = concat(f_dc, detransposed f_rest)
    """
    max_sh_degree = 3
    ply = PlyData.read(ply_path)
    el = ply.elements[0]

    xyz = np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)           # (N,3)
    nxnynz = np.stack([el["nx"], el["ny"], el["nz"]], axis=1).astype(np.float32)     # (N,3)

    opacities = np.asarray(el["opacity"])[..., None].astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacities))).astype(np.float32)                # sigmoid

    f_dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1).astype(np.float32)  # (N,3)

    extra_names = sorted(
        (p.name for p in el.properties if p.name.startswith("f_rest_")),
        key=lambda x: int(x.split("_")[-1]),
    )
    assert len(extra_names) == 3 * (max_sh_degree + 1) ** 2 - 3, \
        f"unexpected SH degree: {len(extra_names)} extra coeffs"
    fe = np.stack([el[n] for n in extra_names], axis=1).astype(np.float32)           # (N,45)
    # (N,45) -> (N,3,15) -> (N,15,3) -> (N,45)  [same detranspose as upstream]
    fe = fe.reshape(fe.shape[0], 3, (max_sh_degree + 1) ** 2 - 1).transpose(0, 2, 1).reshape(fe.shape[0], -1)
    shs = np.concatenate([f_dc, fe], axis=-1).astype(np.float32)                     # (N,48)

    scale_names = sorted(
        (p.name for p in el.properties if p.name.startswith("scale_")),
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.stack([el[n] for n in scale_names], axis=1).astype(np.float32)
    scales = np.exp(scales).astype(np.float32)

    rot_names = sorted(
        (p.name for p in el.properties if p.name.startswith("rot")),
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.stack([el[n] for n in rot_names], axis=1).astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)

    # filter_3D is present on the released LeafFit plants but absent on some
    # vanilla-GS / SuGaR exports; default to all-ones (no filtering) when missing.
    if "filter_3D" in [p.name for p in el.properties]:
        filter_3Ds = np.asarray(el["filter_3D"])[..., None].astype(np.float32)
    else:
        filter_3Ds = np.ones((len(xyz), 1), dtype=np.float32)

    return GaussianData(xyz, rots, scales, opacities, shs, nxnynz, filter_3Ds)


def center_gaussians(g: GaussianData) -> GaussianData:
    """Subtract the centroid (frozen upstream behavior). Pure numpy."""
    if g is None or g.xyz is None:
        return g
    center = np.mean(g.xyz, axis=0)
    return GaussianData(
        xyz=g.xyz - center,
        rot=g.rot,
        scale=g.scale,
        opacity=g.opacity,
        sh=g.sh,
        nxnynz=g.nxnynz,
        filter_3Ds=g.filter_3Ds,
    )


def assert_index_preserved(original: GaussianData, corrected: GaussianData, opacity_threshold: float) -> None:
    """Guarantee that the segmentation operates on the SAME Gaussian index space as input.

    Upstream fix_plant_root_direction_legacy returns the identical array when
    opacity_threshold <= 0, so the corrected index space == input index space.
    This is a hard requirement for sample_indices / labels to be interpretable.
    """
    assert opacity_threshold <= 0.0, "opacity filtering changes index space; forbidden for baseline"
    assert len(corrected.xyz) == len(original.xyz), \
        f"index-space broken: N {len(original.xyz)} -> {len(corrected.xyz)}"
    # corrected = original centered (same order). Float rounding on the mean may leave
    # ~1e-7 residual, so compare via np.allclose row-wise ordering instead of exact equal.
    centered = np.asarray(original.xyz) - np.asarray(original.xyz).mean(axis=0)
    assert np.allclose(np.asarray(corrected.xyz), centered, atol=1e-5), \
        "corrected.xyz is not a pure centering of input xyz; index order changed"


# ---------------------------------------------------------------------------
# Main headless orchestration (mirrors viewer/main.py::load_gaussian_file 1907-1986)
# ---------------------------------------------------------------------------
@dataclass
class HeadlessResult:
    """All intermediate state of the official segmentation, captured for serialization."""
    gaussians_centered: GaussianData          # centered (what segmentation labeled)
    root_idx: int
    root_source: str                          # "fixed" | "auto"
    root_xyz: np.ndarray
    solver: pp3d.PointCloudHeatSolver          # dense full-resolution solver
    sparse_indices: np.ndarray                 # (Ns,) original Gaussian index per sampled point
    downsampled: GaussianData
    orig_to_sparse_mapping: np.ndarray         # (N,) dense -> nearest sparse index
    root_geodesic_single: np.ndarray           # (N,) single-source from root_idx
    root_basin_indices: np.ndarray             # (K,) indices with single_dist <= basin radius
    root_geodesic_multisource: np.ndarray      # (N,) multi-source from basin
    temperature_field: np.ndarray              # (N,) inverted field
    ckdtree: Any
    found_segs: List[np.ndarray]               # per-leaf dense index masks
    found_tips: List[List[int]]
    found_bases: List[int]
    found_geodist_from_tip: List[np.ndarray]
    final_cluster_results: List[dict]
    path_analysis_data: dict
    method: str
    phases: Dict[str, float] = field(default_factory=dict)
    root_pca_info: Optional[dict] = None       # info for auto-root report (informational)

    @property
    def N(self) -> int:
        return len(self.gaussians_centered.xyz)

    @property
    def num_leaves(self) -> int:
        return len(self.found_segs)


def run_headless_segmentation(
    g: GaussianData,
    root_index: Optional[int] = None,
    *,
    seed: Optional[int] = None,
    method: str = BASELINE_METHOD,
    solver_factory=None,
    frozen_root_basin_indices: Optional[np.ndarray] = None,
) -> HeadlessResult:
    """Run the official LeafFit segmentation headlessly on a loaded GaussianData.

    Parameters
    ----------
    g : GaussianData
        Loaded Gaussian data (via load_gaussian_data). Must carry full fields.
    root_index : int | None
        Mode A: exact Gaussian index to use as root. Mode B (None): use the official
        PCA-asymmetry auto root from fix_plant_root_direction_legacy.
    seed : int | None
        Optional RNG seed (informational; the algorithm is deterministic for Mode A;
        see determinism notes).
    method : str
        Petiole detection method; frozen to the upstream default "geodesic_tip_graph".
    solver_factory : callable | None
        If provided, called as ``solver_factory(points, new_gaussians)`` to build a
        geodesic backend (e.g. surface-aware or euclidean graph) that is a drop-in
        replacement for ``pp3d.PointCloudHeatSolver``.  When ``None`` the original
        potpourri3d heat solver is used (byte-identical baseline behaviour).
    frozen_root_basin_indices : np.ndarray | None
        Precomputed root-basin dense indices.  When given, the multisource root
        field is seeded from ``solver.compute_distance_multisource(basin)`` using
        these exact indices (the basin is the *same source set* used for the heat
        baseline), ensuring a fair topology comparison rather than re-deriving the
        basin from a different distance scale.  When ``None`` the basin is derived
        from ``root_geodesic_single <= BASELINE_ROOT_BASIN`` (baseline behaviour).

    Returns
    -------
    HeadlessResult with every intermediate field captured.
    """
    if seed is not None:
        np.random.seed(seed)
        import random as _random
        _random.seed(seed)

    t_all = time.time()
    phases: Dict[str, float] = {}

    # 1) center (frozen upstream preprocessing)
    t0 = time.time()
    cg = center_gaussians(g)
    phases["center"] = time.time() - t0

    # 2) root selection + dense heat solver
    t0 = time.time()
    corrected, root_idx, heat_solver = fix_plant_root_direction_legacy(
        cg, opacity_threshold=BASELINE_OPACITY_THRESHOLD, given_root_idx=root_index,
        solver_factory=solver_factory)
    # index-preservation guard (baseline: no opacity filtering). Compare corrected
    # against the PRE-CENTER original `g` so the only allowed difference is centering.
    assert_index_preserved(g, corrected, BASELINE_OPACITY_THRESHOLD)
    root_source = "fixed" if root_index is not None else "auto"
    root_xyz = np.asarray(corrected.xyz[root_idx])
    phases["root"] = time.time() - t0

    # 3) FPS downsampling (frozen Ns / h)
    t0 = time.time()
    sparse_indices = fpsample.bucket_fps_kdline_sampling(
        np.asarray(corrected.xyz, dtype=np.float64), BASELINE_NS, h=BASELINE_H, start_idx=0)
    sparse_indices = np.asarray(sparse_indices, dtype=np.int64)
    downsampled = apply_indices_to_gaussian_data(corrected, sparse_indices)
    # dense -> nearest sparse mapping (upstream main.py:1920-1924)
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(downsampled.xyz))
    _dist, idx = nn.kneighbors(np.asarray(corrected.xyz))
    orig_to_sparse_mapping = idx.flatten().astype(np.int64)
    phases["sampling"] = time.time() - t0

    # 4) root geodesic fields (single -> basin -> multi-source), official order
    t0 = time.time()
    root_geodesic_single = np.asarray(heat_solver.compute_distance(root_idx), dtype=np.float64)
    if frozen_root_basin_indices is not None:
        root_basin_indices = np.asarray(frozen_root_basin_indices, dtype=np.int64).ravel()
    else:
        root_basin_indices = np.where(root_geodesic_single <= BASELINE_ROOT_BASIN)[0].astype(np.int64)
    root_geodesic_multisource = np.asarray(
        heat_solver.compute_distance_multisource(root_basin_indices), dtype=np.float64)
    phases["root_field"] = time.time() - t0

    # 5) temperature field (captured WITHOUT modifying the algorithm: same inputs,
    #    called independently before get_segment_mask)
    t0 = time.time()
    temperature_field = np.asarray(
        get_temperature_field(heat_solver, root_basin_indices.tolist()), dtype=np.float64)
    phases["temperature"] = time.time() - t0

    # 6) cKDTree on the full corrected set (upstream builds on original_plant_gaussians == corrected)
    ckdtree = cKDTree(np.asarray(corrected.xyz))

    # 7) main segmentation (official)
    t0 = time.time()
    found_segs, found_tips, found_bases, found_geodist_from_tip, final_cluster_results, path_analysis = (
        get_segment_mask(
            corrected, sparse_indices, orig_to_sparse_mapping, heat_solver,
            ckdtree, root_basin_indices.tolist(), method,
            cached_root_distances=root_geodesic_multisource,
            debug_vis=False,
        )
    )
    phases["segment"] = time.time() - t0

    phases["total"] = time.time() - t_all

    result = HeadlessResult(
        gaussians_centered=corrected,
        root_idx=int(root_idx),
        root_source=root_source,
        root_xyz=root_xyz,
        solver=heat_solver,
        sparse_indices=sparse_indices,
        downsampled=downsampled,
        orig_to_sparse_mapping=orig_to_sparse_mapping,
        root_geodesic_single=root_geodesic_single,
        root_basin_indices=root_basin_indices,
        root_geodesic_multisource=root_geodesic_multisource,
        temperature_field=temperature_field,
        ckdtree=ckdtree,
        found_segs=found_segs,
        found_tips=found_tips,
        found_bases=found_bases,
        found_geodist_from_tip=found_geodist_from_tip,
        final_cluster_results=final_cluster_results,
        path_analysis_data=path_analysis,
        method=method,
        phases=phases,
    )
    return result


# ---------------------------------------------------------------------------
# Derived / capture helpers (post-hoc; do NOT alter the algorithm)
# ---------------------------------------------------------------------------
def build_dense_to_sparse_map(result: HeadlessResult) -> np.ndarray:
    """Return a (N,) array: dense index -> sparse position, or -1 if not sampled."""
    dense_to_sparse = np.full(result.N, -1, dtype=np.int64)
    dense_to_sparse[result.sparse_indices] = np.arange(len(result.sparse_indices), dtype=np.int64)
    return dense_to_sparse


def dense_to_sparse_indices(result: HeadlessResult, dense_indices) -> List[int]:
    """Map dense indices to their sparse positions (or -1 if absent)."""
    dtos = build_dense_to_sparse_map(result)
    return [int(dtos[d]) for d in np.asarray(dense_indices, dtype=np.int64)]


def build_tree(result: HeadlessResult) -> dict:
    """Post-hoc root->apex tree reconstruction from the OFFICIAL final paths.

    PROVENANCE: derived from official cluster paths + root; purely informational,
    does not change the algorithm. Nodes/edges are dense Gaussian indices.
    """
    paths = [c["path"] for c in result.final_cluster_results]      # tip->root paths (dense)
    edges: List[List[int]] = []
    junction_nodes: List[int] = []
    node_degree: Dict[int, int] = {}
    for path in paths:
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            edges.append([a, b])
            node_degree[a] = node_degree.get(a, 0) + 1
            node_degree[b] = node_degree.get(b, 0) + 1
    junction_nodes = [n for n, d in node_degree.items() if d > 1]
    nodes = sorted(node_degree.keys())
    return {
        "root": int(result.root_idx),
        "nodes": nodes,
        "edges": edges,
        "junction_nodes": junction_nodes,
        "apexes": [int(c["selected_tip"]) for c in result.final_cluster_results],
        "num_edges": len(edges),
        "num_nodes": len(nodes),
        "provenance": "post-hoc derived from official cluster paths; does not alter algorithm",
    }


def build_apex_grouping(result: HeadlessResult) -> dict:
    """Post-hoc pairwise apex-grouping report using official geodesics + final clusters.

    PROVENANCE: derived from official final clusters + solver geodesics; the official
    group_apexes_by_inequality merges clusters, so pairwise margins are reconstructed
    here for diagnostics WITHOUT changing the grouping decision.
    """
    tips = [c["selected_tip"] for c in result.final_cluster_results]
    # which cluster each tip belongs to (official grouping decision)
    tip_cluster = {int(c["selected_tip"]): i for i, c in enumerate(result.final_cluster_results)}
    cache: Dict[int, np.ndarray] = {}

    def geod_from(t: int) -> np.ndarray:
        if t not in cache:
            cache[t] = np.asarray(result.solver.compute_distance(int(t)), dtype=np.float64)
        return cache[t]

    root_dist = result.root_geodesic_multisource
    pairs = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            ti, tj = int(tips[i]), int(tips[j])
            d = float(geod_from(ti)[tj])                       # direct geodesic
            # LCA via deepest common node on tip->root paths
            p_i = [int(x) for x in result.final_cluster_results[i]["path"]]
            p_j = [int(x) for x in result.final_cluster_results[j]["path"]]
            set_j = set(p_j)
            lca = None
            for node in p_i:
                if node in set_j:
                    lca = node
                    break
            if lca is None:
                lca = result.root_idx
            tree_dist = float(abs(root_dist[ti] - root_dist[lca]) + abs(root_dist[tj] - root_dist[lca]))
            margin = d - tree_dist
            pairs.append({
                "pair": [i, j],
                "tip_1": ti,
                "tip_2": tj,
                "direct_geodesic": round(d, 6),
                "tree_path_distance": round(tree_dist, 6),
                "margin": round(float(margin), 6),
                "tau": 0.62,
                "same_leaf": tip_cluster.get(ti) == tip_cluster.get(tj),
                "cluster_1": tip_cluster.get(ti),
                "cluster_2": tip_cluster.get(tj),
            })
    return {
        "num_apexes": len(tips),
        "pairs": pairs,
        "provenance": "post-hoc derived from official geodesics and final clusters; does not alter grouping",
    }


def build_petioles(result: HeadlessResult) -> List[dict]:
    """Per-leaf petiole/base report from the OFFICIAL final_cluster_results."""
    petioles = []
    for i, c in enumerate(result.final_cluster_results):
        base = int(c["base_idx"]) if c.get("base_idx") is not None else None
        path = [int(x) for x in c.get("path", [])]
        pos = path.index(base) if base in path else None
        petioles.append({
            "leaf_id": i + 1,
            "primary_apex": int(c.get("selected_tip", -1)),
            "tips": [int(t) for t in c.get("tips", [])],
            "type": c.get("type", "unknown"),
            "base_gaussian_index": base,
            "base_xyz": np.asarray(result.gaussians_centered.xyz[base]).tolist() if base is not None else None,
            "path_position": pos,
            "provenance": "official calculate_cluster_base_indices output",
        })
    return petioles


def build_pre_grouping_replay(result: HeadlessResult) -> dict:
    """Pre-grouping diagnostic replay for Figure 13(a)-style analysis.

    PROBLEM: the official grouping step (group_apexes_by_inequality) is a single
    monolithic call inside get_segment_mask; it does not return the pre-grouping
    state. The currently saved `apex_grouping.json` post-hoc report only sees the
    FINAL selected tips (one per merged cluster), so it cannot tell whether two
    apexes that ended up in one cluster were SEPARATE leaves merged during grouping.

    FIX (pure diagnostic, ZERO algorithm change): replay the official pre-grouping
    stage by calling the same official functions in the SAME order and with the SAME
    inputs get_segment_mask uses:
        1. find_local_tips(k=len(sparse)//64)   -> candidate tips (pre-grouping)
        2. find_path_from_tip_to_root(k=len(sparse)//32, euclidean) -> per-tip paths
        3. group_apexes_by_inequality(...)      -> cluster_info (post-grouping)
    and record candidate tips / candidate paths / per-tip cluster assignment / which
    candidate tips share a cluster (the "merged" candidates). Because tips and paths
    are deterministic functions of the same inputs, the replay reproduces EXACTLY the
    grouping decision the official pipeline makes.

    This is a separate, additive diagnostic — it never calls get_segment_mask and
    never alters any core algorithm file.
    """
    from auto_segment import find_local_tips, find_path_from_tip_to_root
    from apex_grouping import group_apexes_by_inequality

    corrected = result.gaussians_centered
    sparse_indices = result.sparse_indices
    mapping = result.orig_to_sparse_mapping
    solver = result.solver
    ckdtree = result.ckdtree
    temp_field = result.temperature_field
    cached_root = result.root_geodesic_multisource
    heat_source_idx = result.root_basin_indices.tolist()

    # ---- 1) candidate tips (official k = len(sparse)//64) ----
    tips = [int(t) for t in find_local_tips(corrected, sparse_indices, mapping,
                                            temp_field, ckdtree,
                                            k=len(sparse_indices) // 64)]
    # ---- 2) candidate tip->root paths (official k = len(sparse)//32, euclidean) ----
    is_path_marks = np.zeros(len(corrected.xyz), dtype=int)
    paths = []
    for t in tips:
        p = find_path_from_tip_to_root(
            corrected, temp_field, t, heat_source_idx[0],
            {"method": "euclidean", "tree": ckdtree, "dense_solver": solver},
            is_path_marks, k=len(sparse_indices) // 32)
        paths.append([int(x) for x in p])

    # ---- 3) official grouping ----
    cluster_info = group_apexes_by_inequality(
        tips, paths,
        overlap_cut=0.8,
        root_cahced_distance=cached_root,
        dense_solver=solver,
    )

    # ---- record which candidate tip belongs to which final cluster ----
    tip_to_cluster: Dict[int, int] = {}
    for ci_idx, ci in enumerate(cluster_info):
        for t in ci.get("tips", []):
            tip_to_cluster[int(t)] = ci_idx
    # candidates that were NOT in any cluster (dropped / singletons)
    ungrouped = [t for t in tips if t not in tip_to_cluster]

    # ---- candidate-level detail ----
    candidate_tips = []
    for t in tips:
        candidate_tips.append({
            "gaussian_index": t,
            "sample_index": int(build_dense_to_sparse_map(result)[t]),
            "xyz": np.asarray(corrected.xyz[t], dtype=float).tolist(),
            "cluster_index": tip_to_cluster.get(t),
            "root_geodesic": float(cached_root[t]),
        })

    # ---- which candidate tips were merged into the SAME cluster (Fig.13a scenario) ----
    merged_groups = []
    for ci_idx, ci in enumerate(cluster_info):
        member_tips = [int(t) for t in ci.get("tips", [])]
        if len(member_tips) > 1:
            merged_groups.append({
                "cluster_index": ci_idx,
                "candidate_tips": member_tips,
                "num_candidates": len(member_tips),
                "paths_sample_indices": [
                    [int(build_dense_to_sparse_map(result)[x]) for x in paths[tips.index(t)]] if t in tips else []
                    for t in member_tips],
            })

    return {
        "num_candidate_tips": len(tips),
        "num_clusters_after_grouping": len(cluster_info),
        "candidate_tips": candidate_tips,
        "candidate_paths": [
            {"tip": tips[i], "path_gaussian_indices": paths[i],
             "path_sample_indices": [int(build_dense_to_sparse_map(result)[x]) for x in paths[i]]}
            for i in range(len(tips))],
        "cluster_info_after_grouping": [
            {"cluster_index": i, "tips": [int(t) for t in ci.get("tips", [])],
             "lca": ci.get("lca")}
            for i, ci in enumerate(cluster_info)],
        "ungrouped_candidates": ungrouped,
        "merged_candidate_groups": merged_groups,
        "note": "pre-grouping diagnostic replay; reproduced EXACTLY with official "
                "find_local_tips/find_path_from_tip_to_root/group_apexes_by_inequality "
                "in the SAME order/params get_segment_mask uses; does not alter the "
                "official grouping decision.",
    }
