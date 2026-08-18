# LeafFit Headless Leaf-Instance Segmentation Baseline

This is a **CPU-only, headless** reproduction & freeze of the LeafFit (Eurographics 2026)
**automatic leaf-instance segmentation** (paper **Section 3.1**), decoupled from the
private/GPU components of the official repo. It is the frozen baseline that later
Tasks 2/3 build on.

It reproduces the official pipeline exactly (`viewer/main.py::load_gaussian_file`
minus the GUI/rasterizer), saves **every intermediate** (root geodesics, sampling,
apexes, paths, tree, grouping, petioles, labels, colored PLY), and is **deterministic**
for a fixed root index.

## Why this exists

The full LeafFit pipeline needs private components that cannot be compiled here:
`diff_gaussian_rasterization` (orphaned gitlink), `gsplat_bvh`, `my_rasterizer`,
`OpenGL/GLFW/imgui`, `template_transform`, `gen_template_leaf`. **None** of these are
needed by the segmentation algorithm — it only uses `potpourri3d`, `fpsample`,
`numpy/scipy/sklearn`.

## Fidelity to upstream (no algorithm changes)

- Core algorithm files **`core/auto_segment.py`, `core/apex_grouping.py`,
  `core/petiole_detection.py` are NOT modified.**
- `core/headless_segmentation.py` is a pure wrapper: it calls the official functions
  and captures intermediate state post-hoc (it does not build the unused
  `sparse_solver`, and `get_segment_mask` uses the official euclidean dense path).
- Method frozen to the official default `geodesic_tip_graph` (`g_segmentation_method=8`).

### Frozen baseline constants
| constant | value | meaning |
|---|---|---|
| `BASELINE_NS` | 8192 | FPS sample count (matches paper `Ns`) |
| `BASELINE_H` | 7 | FPS kdline grid factor (code-specific) |
| `BASELINE_T_COEF` | 1e8 | potpourri3d heat-method t_coef |
| `BASELINE_ROOT_BASIN` | 0.1 | root-source geodesic basin radius |
| `BASELINE_OPACITY_THRESHOLD` | 0.0 | no opacity filtering → index space preserved |
| `BASELINE_METHOD` | `geodesic_tip_graph` | upstream segmentation method |

### paper-vs-code discrepancies (recorded, not "fixed")
These are real differences between the paper and the released code. The baseline
**keeps the code behavior** (fidelity to upstream source) and documents the difference:

| param | paper | code (frozen) | decision |
|---|---|---|---|
| tips neighbor k | Nk=512 | `len(sparse)//64` ≈ 128 | keep code |
| path neighbor k | 512/128 | `len(sparse)//32` ≈ 256 | keep code |
| apex grouping tau | 0.5 | `triangle_cut=0.62` | keep code |
| petiole epsilon | 0.05 | `tolerance_percentage=0.02` @call-site | keep code |
| petiole delta | 0.01 | `min_distance_threshold=0.05` | keep code |
| petiole rho | 0.25 | `protection_period_ratio=0.25` | matches paper |

## Usage

```bash
export PYTHONPATH=<repo>/core:<repo>
PY=/home/test/biosoft/enter/envs/agri_re_py310/bin/python

# single plant, Mode A (fixed root):
$PY scripts/run_leaf_segmentation.py \
    --input data/plant1_green_pepper.ply \
    --output outputs/baseline/plant1_green_pepper \
    --root-index 47330

# single plant, Mode B (official PCA auto root) — used once to freeze the root:
$PY scripts/run_leaf_segmentation.py --input data/plant1_green_pepper.ply \
    --output /tmp/p1_auto --root auto

# batch over all 8 official plants (freezes roots, then runs formal baseline):
$PY scripts/run_all_official_plants.py

# regression tests (Test A–E):
$PY -m unittest tests.test_headless_segmentation -v

# instance-level evaluation (IoU + Hungarian; exits 0 with no GT, no fabricated metrics):
$PY scripts/evaluate_segmentation.py --pred out/labels.npy --gt /path/to/gt.npy
```

## Outputs (per plant, under `<output>/`)

| file | content |
|---|---|
| `config.json` | frozen constants, effective runtime params, **paper_vs_code** table |
| `root.json` | root Gaussian index / source / xyz / basin size |
| `sample_indices.npy` | `(Ns,)` FPS sampled dense indices |
| `root_geodesic_single.npy` | `(N,)` single-source geodesic from root |
| `root_basin_indices.npy` | `(K,)` indices with single dist ≤ 0.1 |
| `root_geodesic_multisource.npy` | `(N,)` multi-source geodesic from basin |
| `root_geodesic_stats.json` | min/max/mean/finite (multi-source min may be ~-0.05, official) |
| `temperature_field.npy` | `(N,)` inverted heat field |
| `apexes.json` | per-leaf apex (Gaussian idx, xyz, type, tips) |
| `paths.json` | per-leaf tip→root dense path |
| `tree.json` | root→apex tree (post-hoc, **provenance-tagged**) |
| `apex_grouping.json` | pairwise grouping margins (post-hoc, **provenance-tagged**) |
| `petioles.json` | per-leaf base/petiole (official output, provenance-tagged) |
| `raw_labels.npy` | `(N,)` official labels (0=unassigned, 1..K) |
| `labels.npy` | `(N,)` unified labels (0=stem, 1..K leaf) |
| `segmentation_result.ply` | full Gaussian PLY with per-instance colors in SH-DC |
| `segmentation_points.ply` | lightweight xyz+rgb PLY |
| `runtime.json` | per-phase timing |
| `metadata.json` | commits (upstream/ours), versions, counts, status |
| `status.json` | `SUCCESS` / `SEGMENTATION_FAILED_NO_LEAVES` / `FAILED` |

Root mode: **Mode A** `--root-index N` uses index N directly; **Mode B** `--root auto`
runs the official PCA-asymmetry root and reports it. The batch first freezes Mode-B
roots into `outputs/frozen_roots.json`, then runs the formal baseline in Mode A with
those frozen roots (Task 2/3 must read the same `frozen_roots.json`).

## Provenance policy

Post-hoc derived files (`tree.json`, `apex_grouping.json`, `petioles.json`) are
**purely diagnostic**: they are recomputed from official outputs without changing any
official decision, and carry a `"provenance"` field stating exactly that.

## Index preservation & determinism

- `opacity_threshold=0.0` guarantees `corrected == input` in **index space** (same N,
  same order). `assert_index_preserved` enforces this before writing any label array.
- `label`/`sample_indices`/`apexes` are all in the **centered** coordinate space the
  official algorithm actually runs in (`metadata.json` records `centered: true`).
- Verified deterministic: same fixed root + inputs → byte-identical
  `labels.npy` / `sample_indices.npy` / `root_geodesic_*.npy` (3 runs on plant1).
  The only theoretical source of run-to-run nondeterminism is `np.linalg.eig` sign
  flips in the **auto** root path — Mode A (fixed) is fully deterministic.

## Evaluation

`scripts/evaluate_segmentation.py` is **instance-level and permutation-invariant**:
it first matches predicted instances to ground-truth instances via **IoU + Hungarian
optimal assignment**, then computes Accuracy / mIoU / mF1 / PQ. Label IDs are unordered,
so raw label-to-label comparison is rejected by design. If no GT path is provided it
prints `no GT available` and exits 0 **without fabricating any metric** (the official
data has no labeled ground-truth).

## Environment

Python 3.10 at `/home/test/biosoft/enter/envs/agri_re_py310/bin/python`; deps:
numpy, scipy, scikit-learn, open3d, potpourri3d, fpsample, plyfile, e3nn, trimesh,
networkx, tqdm.