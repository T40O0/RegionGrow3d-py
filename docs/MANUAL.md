# RegionGrow3D Python — User Manual

> 🌐 **日本語版**: [MANUAL-jp.md](MANUAL-jp.md)

Last updated: 2026-05-12

---

## Contents
1. [Concepts](#1-concepts)
2. [Installation](#2-installation)
3. [Web UI](#3-web-ui)
4. [Command line](#4-command-line)
5. [Modes in detail](#5-modes-in-detail)
6. [Input files and where to put them](#6-input-files-and-where-to-put-them)
7. [Reading the output](#7-reading-the-output)
8. [Comparing with MATLAB](#8-comparing-with-matlab)
9. [Troubleshooting](#9-troubleshooting)
10. [Performance tips](#10-performance-tips)
11. [API reference](#11-api-reference)

---

## 1. Concepts

RegionGrow3D is a deterministic landslide-source analysis. It grows a
"landslide cluster" outward from each unstable cell until the 3-D Janbu
limit-equilibrium check reaches force closure. Repeated over a probabilistic
distribution of friction angle φ' and cohesion c' (typically 10 pairs), the
weighted average gives a **0–100 % susceptibility map**.

Pipeline:
```
DEM (.tif)
  ↓ pad + gradient (gradient_prince)
  ↓ soil depth (soil_depth: Roering 5000 yr or uniform)
  ↓ W, σ_s under hydrostatic / dry / (hydromechanical) moisture
  ↓ seismic PGA (off / uniform / raster) added to Q
  ↓ no-grow mask (acc-threshold ridges+valleys OR slope units)
  ↓ for each (φ', c') pair, RegionGrow:
        Q = N·sin(α) − (c·A + (N−U)·tan(φ))·cos(α) + PGA·W
        unstable = (Q > 0)  → cluster → erosion → grow until force closure
  ↓ Σ (slides_final · prob)  → susceptibility map
  ↓ sus_*.tif written to disk
```

### 1.1 What RegionGrow actually does — in detail

#### (a) "Unstable cell" definition
For each cell the 1-D Janbu infinite-slope limit-equilibrium equation is
evaluated:

```
Q_i = N_i·sin(α_i)              ← driving (gravity + seismic) component
       − (c·A_i + (N_i − U_i)·tan(φ)) · cos(α_i)  ← shear resistance
       + PGA·W_i                ← pseudo-static seismic load
```

with `N_i = W_i·cos(α_i)`, `A_i = cellsize²/cos(α_i)`,
`U_i = γ_w·hw_i·A_i` (pore-water force, `hw = mw·depth`),
and `W_i = γ·depth·cellsize²`. Cells with **`Q_i > 0` are flagged as
unstable**. This produces `slides_initial_io`, the **seed cells** that
RegionGrow grows clusters from.

#### (b) Clustering and erosion — step by step

Take the binary `Q > 0` seed mask (`slides_initial_io`) and split it into
**individual candidate failure surfaces** in four stages.

**① Mask out no-grow cells** ([region_grow.py:80-85](../python/region3d/region_grow.py))

```python
FB_assign[ngi, ngj] = False       # remove no-grow cells from the seeds
```

Prevents clusters from connecting across ridges/valleys. This is where the
topographic partition is enforced.

**② Drop tiny components (`bwareaopen`)** ([region_grow.py:88](../python/region3d/region_grow.py))

```python
FB_assign = bwareaopen(FB_assign, cluster_size_thresh)   # default 7
```

Any 8-connected component smaller than `cluster_size_thresh` (default 7
cells) is removed from the whole mask. On a 10 m DEM this is ~700 m² — too
small to represent a real landslide and usually a numerical/boundary
artefact, so we discard it before clustering.

**③ 8-connected labelling (`bwconncomp`)** ([region_grow.py:91](../python/region3d/region_grow.py))

```python
pixel_idx_list, num_objects, _ = bwconncomp_F(FB_assign)
```

The remaining mask is labelled into connected components, each returned as
a list of pixel indices. These are the initial clusters
(`slides_initial_io` content); each is processed independently downstream.

**④ Per-cluster shape clean-up (erosion)** ([region_grow.py:184-201](../python/region3d/region_grow.py))

A "polish then commit" stage. The flow for every cluster is a single
linear path — no surprise branching:

```
Cluster C  (from step ③)
   │
   ├─ Healthy?  (size ≥ 7  &  no spur  &  connected)
   │     │ NO  → drop this cluster                        ┐
   │     │ YES → continue                                 │ ← only place
   │                                                      │   a cluster
   ├─ Save a copy:  C_save = C                            │   can vanish
   │
   ├─ Apply erosion twice (peel 1 cell off the boundary, twice)
   │
   ├─ Still healthy after erosion?  (same 3 checks)
   │     │ YES → keep the eroded C       (polish successful)
   │     │ NO  → C := C_save              (polish failed → undo)
   │
   └─→ register C in slides_eroded_io and move on to the Janbu loop
```

The whole "shrink / split / un-split" business reduces to **try-and-roll-back**:

| Stage | What it looks like | What actually happens |
|---|---|---|
| Pre-check | cluster is discarded if NG | **The only place a cluster is dropped.** |
| Erosion | cluster shrinks, may split | A **trial only**. If the next continuity check fails, the split is reverted — it is never committed. |
| Post-check | "if it split, revert" | Always commit either the polished or the original shape — never a half-eroded state. |

Consequences:
- **A cluster never actually splits.** If erosion broke it apart, the
  whole erosion is undone and the original (still 1-piece) shape is used.
- **A cluster never disappears via erosion.** If it shrank below the size
  threshold, the erosion is undone.
- Every cluster moves on with **either the eroded or the original shape**
  — exactly one of the two.

Why the round-trip? It's a **best-effort polish** for **ragged boundaries**.
You can't tell in advance which cluster will benefit, so you try erosion and
discard the trial if the result is worse than the input.

> ⚠️ **Erosion is not a cluster-splitting tool.** If two blobs joined by a
> 1-cell-wide neck come apart after erosion, `continuity_check` fails and the
> erosion is reverted — the cluster goes forward as the original 1-piece
> shape ([growth.py:169](../python/region3d/growth.py),
> [region_grow.py:193-201](../python/region3d/region_grow.py)). To actually
> split such cases, use the nogrow mask or the spur test instead.

The resulting shapes are **`slides_eroded_io`** — the set of "candidate
failure surfaces clean enough for Janbu". The cluster grow loop (c) below
operates on this set.

#### (c) Cluster grow loop — one cluster at a time

Each cluster grows by adding downslope columns until **force closure** is
reached. Closure is a **vector sum** (x and y components) of three
contributions ([forces.py:138](../python/region3d/forces.py),
[region_grow.py:260](../python/region3d/region_grow.py)):

1. **Column net forces ΣQ** — each column's Q (same formula as (a)) resolved
   into x/y along that column's own basal dip direction (deflected uniformly
   by the rotation search below).
2. **Boundary prism (wedge) forces** — the ring of triangular prisms around
   the perimeter (below).
3. **Root resistance F_roots** — boundary length × S_roots, subtracted from
   the closure error.

The cluster is accepted as stable when
`err = |ΣQ_columns + ΣQ_prisms| − F_roots` falls below
`err_percent_allowable` % (default 1 %) of the total weight including the
prisms. Otherwise downslope columns are added and the test repeats, until
`max_growth_cycles` (default 120) or a sustained error increase
(`err_increase_thresh`) terminates the loop. Growth is blocked in any
direction that hits a `no_grow` cell.

#### (c-1) Boundary prisms — full perimeter, active/passive earth pressure

Prisms are built around the **entire perimeter**: the outer boundary is
traced as a closed loop ([boundary.py:494](../python/region3d/boundary.py))
and every pair of adjacent boundary vertices gets one triangular prism.
Active vs. passive behaviour is expressed by **interpolating the wedge base
angle α with the orientation ω relative to the direction of sliding**
([boundary.py:534-543](../python/region3d/boundary.py); eqs. 14-15 of the
paper):

```
ω = (centroid→boundary direction) − direction of sliding (DOS)
ω = 0°   (toe):   α = 45° − φ/2   Rankine passive — long, high-volume wedge
ω = 180° (head):  α = 45° + φ/2   Rankine active  — short, steep wedge
flanks:           linear interpolation
```

Each prism's net force is computed with the same signed Janbu-type base
equilibrium as the columns ([forces.py:94](../python/region3d/forces.py));
positive force points into the cluster, so the head prism acts as an active
driving thrust and the toe prism as a passive buttress automatically. The
seismic term is distributed as `k·(−cos ω)`, adding drive at the head and
reducing the passive resistance at the toe. DOS is the azimuth of the
resultant of the column forces, `atan2(ΣQy, ΣQx)`. The paper itself notes
this interpolation is a simplification (e.g. a T-shaped cluster can have
passive prisms away from the toe).

#### (c-2) Rotation search — sliding-direction uncertainty (±20°, 8 states)

Closure is not tested once but for **8 force fields with the sliding
direction deflected** ([forces.py:186](../python/region3d/forces.py)
`project_slope`):

- Each column's force is turned to "**its own aspect + rot**". The same rot
  is added to every column, so the azimuthal spread between columns (e.g.
  convergence in a hollow) is preserved and the whole field swings together —
  this is **not** an alignment to one common compass direction.
- The apparent dip in the deflected direction shrinks to |∇z|·cos(rot).
- rot takes the 8 values of `linspace(−20°, +20°, 8)` (±2.9°, ±8.6°, ±14.3°,
  ±20°; **0° is not among them** — the unrotated field is used only for the
  initial instability screen in (a)).
- The **minimum** closure error over the 8 states decides: balanced in any
  one orientation ⇒ stable.

This procedure is specific to RegionGrow3D: Mathews et al. (2024) introduce
it "to account for uncertainty in the cluster's direction of sliding",
without a mechanical derivation or prior-work citation, and the range is a
free parameter (`--rot_range` / `--rot_num`). It differs from classical 3-D
limit-equilibrium formulations that solve for a single unknown sliding
direction of the whole mass (e.g. Lam & Fredlund 1993).

#### (d) Role of the no-grow mask
`no_grow_io` is a binary mask that **clusters are forbidden to cross**.
Three sources (§5.4):

- **acc-threshold**: D8 flow direction → flow accumulation →
  `acc > X` (valleys) and `inverted-acc > Y` (ridges), then thinned.
- **slope units** (Alvioli 2016/2025): half-basins refined by aspect
  circular variance and merged below `areamin`.
- **load existing .mat**: re-use a MATLAB-side or previous Python run.

**Enabling the no-grow mask (mode=1)**:
- Prevents clusters from spilling across valleys → physically plausible
  single landslide sources.
- Prevents clusters from spilling across ridges → guards against
  over-merging that connects unrelated hillslopes.

**Disabling it (mode=0)** lets clusters grow without topographic
constraints, occasionally producing a single giant cluster covering an
entire catchment.

#### (e) Probabilistic averaging
Strength pairs `(φ, c)` are drawn from the
`prob_phi`, `prob_coh`, `prob` arrays in `shear_strength.mat` (typically
N = 10). RegionGrow is run once per pair, and the final susceptibility
is

```
sus_map[i,j] = Σ_k  prob[k] · slides_final_io[k][i,j]
```

bounded by `0 ≤ sus ≤ Σ prob ≈ 1.0 (=100%)`. The map therefore answers
*"how robustly does this cell fail across the strength uncertainty?"*

---

## 2. Installation

### 2.1 Docker (recommended)

No local Python install needed; every native dependency is resolved inside
the container, so this is the most reproducible option.

```bash
# build (one-off, ~5 minutes)
docker build -t region3d:latest .

# Web UI (mount lib/ and output/ from the host)
docker run --rm -p 8501:8501 \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  -v "$(pwd)/python/output_webui:/app/python/output_webui" \
  region3d:latest

# CLI
docker run --rm \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  region3d:latest \
  python python/driver.py --soil_strength_mode 2 --phi_uniform 30 --coh_uniform 5
```

`docker-compose.yml` is also bundled:
```bash
docker compose up           # Web UI
docker compose run --rm region3d python python/driver.py   # CLI
```

Base image: `python:3.13-slim` (~200 MB). Final image is ~1 GB with deps.
Without volume mounts, the container has no DEM and results aren't persisted.

### 2.2 Conda (local install)
```bash
conda env create -f environment.yml
conda activate region3d
```
`environment.yml` pins every package to `conda-forge` so the native
libraries (GDAL via `rasterio`, LLVM via `numba`, BLAS via `scipy/numpy`)
stay consistent.

#### Pure-pip fallback
If conda is unavailable:
```bash
python -m venv .venv
.venv/Scripts/activate     # Windows  (POSIX: source .venv/bin/activate)
pip install -r requirements.txt
```

#### Required libraries
| Package | Purpose |
|---|---|
| `numpy` (≥2.0) | numerics |
| `scipy` | morphology, distance transform, .mat I/O |
| `rasterio` | GeoTIFF read/write |
| `scikit-image` | imreconstruct (fillsinks), skeletonize verification |
| `numba` | JIT acceleration (soil_depth, flow_accumulation, alpha-shape) |
| `matplotlib` | plotting |
| `streamlit` | Web UI |
| `pandas` | UI summary tables |

### 2.3 Verifying the install
```bash
python python/tests/smoke_test_modules.py
python python/tests/test_alpha_shape.py
python python/tests/test_bwboundaries.py
python python/tests/test_skel.py
```

---

## 3. Web UI

```bash
streamlit run python/gui.py
```

### 3.1 Sidebar layout

```
📁 Input file placement guide (collapsed by default)
📍 DEM       Pick from lib/DEM/*.tif or upload a new .tif
⚙ Mode settings
  💧 Soil moisture   [0=dry | 1=hydrostatic]
                     - mw slider (mode=1)
  🟫 Soil depth      [1=Roering | 2=uniform]
                     - data source (.mat / Python)
                     - .mat dropdown or Roering duration / uniform value
  🪨 Shear strength parameters  [1=distribution | 2=uniform]
                     - single run-index selector (mode=1)
                     - ⚡ Run all runs in parallel (2 concurrent) checkbox (mode=1 & All runs)
                     - φ, c (mode=2)
  🚧 Growth boundary [0=off | 1=ridges+valleys]
                     - data source (.mat / Python)
                     - .mat dropdown or ridge/valley thresholds
  🌐 Seismic         [off | uniform | raster]
                     - PGA / scaling / TIFF path
  🌿 Root strength   S_roots (kPa)
📐 Basic properties (collapsed): Gs, γ_w, γ_dry, γ_sat
📤 Output destination     parent dir + run ID + overwrite confirm
▶ Start
```

### 3.2 Run lifecycle

1. Configure parameters → click **Start**.
2. **While running**:
   - Sidebar fully locks (no accidental changes).
   - Main pane shows a banner, progress bar with ETA, and live log.
   - **Stop** button is always reachable to abort.
3. **After completion**:
   - Status banner remains visible.
   - Result tabs render automatically.
   - **In parallel mode**, the tension/compression map
     `net_force_prob_<susname>.tif` is also generated after aggregate (§7.4).

### 3.3 Result tabs

| Tab | Content |
|---|---|
| 🗺 Maps | Layer selector (susceptibility / no-grow / soil depth / PGA), hillshade overlay, TIFF downloads |
| 📊 Statistics | Valid cells, share >0% / >50% / >90%, estimated areas, per-run cluster counts ((φ, c, prob) table + bar chart) |
| 📈 Histogram | Positive-cell histogram (log scale) + CDF + median / 90th percentile |

### 3.4 Closing the browser
The Streamlit session disconnects but **the subprocess keeps running** and
its outputs land in `out_dir/<susname>/`. Reconnecting opens a fresh session
with empty log; you can still read the results from the tabs.

### 3.5 Detached launch (Windows, overnight runs)

Launching Streamlit from a terminal binds it to that shell — when the
shell exits the server dies. For long sessions on Windows, spawn it
detached via `Start-Process` so it re-parents to `explorer.exe` / system
and survives shell exit:

```powershell
$py     = "$env:LOCALAPPDATA\miniconda3\envs\gis_conda\python.exe"   # or your env's python
$repo   = "C:\Users\040869\Documents\GitHub\RegionGrow3d-py"          # repo root
$logDir = Join-Path $repo "python\output_webui"
$logFile = Join-Path $logDir ".streamlit_server.log"
$errFile = Join-Path $logDir ".streamlit_server.err.log"
$pidFile = Join-Path $logDir ".streamlit_server.pid"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
"" | Set-Content -Path $logFile -Encoding utf8
"" | Set-Content -Path $errFile -Encoding utf8

$proc = Start-Process -FilePath $py `
    -ArgumentList "-m","streamlit","run",(Join-Path $repo "python\gui.py"),`
                  "--server.headless=true","--server.address=0.0.0.0" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru
$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
"detached PID=$($proc.Id); log=$logFile"
```

**Health check / stop:**

```powershell
# Is anyone listening on 8501?
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue

# Tail the log
Get-Content "$repo\python\output_webui\.streamlit_server.log" -Wait -Tail 20

# Stop
$pidVal = (Get-Content "$repo\python\output_webui\.streamlit_server.pid").Trim()
Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
Remove-Item "$repo\python\output_webui\.streamlit_server.pid" -ErrorAction SilentlyContinue
```

> ⚠️ **Sleep:** even when detached the process is killed when Windows
> enters sleep. For genuine overnight runs either set Windows power → sleep
> to *Never*, or wrap `gui.py` startup with a `SetThreadExecutionState`
> keep-awake call (the existing `runner._set_keep_awake` hook is only
> invoked during an active analysis run, not while idling).

### 3.6 Parallel runs (⚡ Run all runs in parallel)

With shear strength = distribution (mode 1) and **All runs**, ticking
**"⚡ Run all runs in parallel (2 concurrent)"** runs the 10 φ runs **2 processes
at a time** (via the `python/_sus_parallel.py` orchestrator) instead of serially,
then combines them with `--aggregate`. Each φ contribution is a commutative
probability-weighted sum, so the **result is bit-identical to the serial loop**.
It is resume-safe: each φ's contribution is banked, and re-launching finishes
only the remaining runs.

- **Memory:** ~20 GB per run, so ~40 GB for 2 concurrent. **When running under
  Docker/WSL2, raise the VM memory** — put in `%USERPROFILE%\.wslconfig`:
  ```ini
  [wsl2]
  memory=54GB
  ```
  and apply with `wsl --shutdown` (default is ~50% of host). Check free RAM for
  native runs too.
- **The bar stays at 0 % for the first few minutes** (pre-compute: DEM,
  derivatives, interslice forces). If the log shows phases advancing it IS
  running. **Don't click Start again** — a double-start guard blocks it, but
  waiting is the right move.

After aggregate, parallel mode also auto-generates the tension/compression map
`net_force_prob_<susname>.tif` (meaning and sign convention in §7.4).

---

## 4. Command line

`driver.py` works as both a subprocess (called by Streamlit) and a standalone CLI.

### 4.1 Minimal invocation
```bash
python python/driver.py --DEM_path lib/DEM/<your_DEM>.tif --test_no 1
```
`--DEM_path` and `--test_no` are always required. Additional `--*_mat` paths
are conditionally required depending on the chosen modes (see 4.2 and the
runtime validation messages).

### 4.2 Full option list
```bash
python python/driver.py --help
```

Key options:
| Option | Example | Description |
|---|---|---|
| `--DEM_path PATH` | `lib/DEM/<your_DEM>.tif` | **required**. Input DEM (absolute or repo-relative) |
| `--test_no INT` | `1` | **required**. Numeric ID for the output sub-folder name |
| `--susname_override STR` | `my_scenario` | output sub-folder name (empty = test_no zero-padded to 5) |
| `--out_dir PATH` | `python/output` | parent output dir |
| `--soil_moisture_mode 0\|1` | `1` | 0=dry / 1=hydrostatic |
| `--mw FLOAT` | `0.5` | saturation ratio (mode 1) |
| `--soil_depth_mode 1\|2` | `1` | 1=hillslope model / 2=uniform |
| `--soil_depth_uniform FLOAT` | `2.0` | uniform soil depth [m] |
| `--soil_depth_endtime FLOAT` | `5000.0` | Roering simulation duration [yr] |
| `--soil_depth_source mat\|compute` | `mat` | load .mat / compute in Python |
| `--soil_depth_mat PATH` | `lib/soil_depth/<DEM_stem>_soil_depth.mat` | source path when source=mat |
| `--soil_depth_model roering\|massbalance` | `roering` | model used when source=compute — see §5.2 |
| `--soil_depth_mb_preset oregon\|matsushi` | (empty) | massbalance: published parameter set; individual options override it |
| `--soil_depth_mb_tag STR` | (empty, defaults to the preset name) | massbalance: suffix for the saved .mat/.tif |
| `--soil_depth_mb_hollow_endtime FLOAT` | `0.0` | massbalance: with `endtime 0`, years of refill given to convergent cells (hollows) |
| `--soil_depth_mb_E0 FLOAT` | `720.0` | massbalance: bare-bedrock soil production rate E₀ [g m⁻² yr⁻¹] |
| `--soil_depth_mb_alpha FLOAT` | `3.0` | massbalance: production decay with thickness α [m⁻¹] |
| `--soil_depth_mb_K FLOAT` | `0.005` | massbalance: creep transport coefficient K [m² yr⁻¹] |
| `--soil_depth_mb_Sc FLOAT` | `1.25` | massbalance: critical gradient S_c (non-linear law only) |
| `--soil_depth_mb_rho_soil FLOAT` | `1200.0` | massbalance: soil bulk density ρ_soil [kg m⁻³] |
| `--soil_depth_mb_W FLOAT` | `0.0` | massbalance: chemical mass loss from the soil W_soil [g m⁻² yr⁻¹] |
| `--soil_depth_mb_transport nonlinear\|linear` | `nonlinear` | massbalance: transport law (eq.10 / eq.9) |
| `--soil_depth_mb_endtime FLOAT` | `5000.0` | massbalance: duration [yr]; `0` = steady state |
| `--soil_depth_mb_h_init FLOAT` | `0.0` | massbalance: initial soil thickness [m] |
| `--soil_depth_mb_h_max FLOAT` | `3.0` | massbalance: upper cap on soil thickness [m] |
| `--soil_depth_mb_smooth FLOAT` | `1.0` | massbalance: Gaussian DEM pre-smoothing σ [cells] |
| `--nogrow_mode 0\|1` | `1` | 0=off / 1=ridge+valley |
| `--nogrow_source mat\|compute\|grass` | `mat` | `grass` = original r.slopeunits (Docker) |
| `--no_grow_mat PATH` | `lib/no_grow/<DEM_stem>_no_grow.mat` | source path when source=mat |
| `--ridge_acc_thresh FLOAT` | `5` | flow-accumulation threshold for ridges (`compute`) |
| `--valley_acc_thresh FLOAT` | `100` | same for valleys (`compute`) |
| `--slope_unit_thresh/areamin/cvmin/rf/maxiter` | 500000/100000/0.3/2/50 | r.slopeunits params for `grass` — see §5.4 |
| `--soil_strength_mode 1\|2` | `1` | 1=distribution / 2=uniform |
| `--phi_uniform FLOAT` | `25` | uniform φ' [deg] |
| `--coh_uniform FLOAT` | `2` | uniform c' [kPa] |
| `--seismic_mode off\|uniform\|raster` | `off` | seismic input |
| `--uniform_PGA FLOAT` | `0.3` | uniform PGA [g] |
| `--PGA_path PATH` | `lib/seismic/PGA.tif` | TIFF path when seismic=raster |
| `--pseudo_scaling FLOAT` | `1.0` | PGA scaling factor |
| `--S_roots FLOAT` | `10` | root strength [kPa] |
| `--save_intermediates 0\|1` | `1` | also write depth/nogrow/PGA/hillshade TIFFs |
| `--run-index INT` | `None` | compute only one 0-based distribution index and bank it to `<out>/<susname>/contribs/` for later `--aggregate` (low-memory resume, see §4.4). Also writes that run's partial `sus_*.tif`. Must be `0 ≤ index < #runs`; mutually exclusive with `--aggregate` |
| `--aggregate 0\|1` | `0` | combine all `contribs/contrib_run*.npz` into the final susceptibility map and exit; needs only `--DEM_path` + `--test_no`/`--susname_override`. Fails if the contribution set is incomplete or mixes distributions (see §4.4) |
| `--tension_compression 0\|1` | `0` | build the **tension/compression map `net_force_prob_*.tif`** from `contribs/` and exit. The per-cell net force q (q>0 = tension / q<0 = compression) is **sign-flipped (`−q`)** then probability-weighted over φ, so the **output is compression-positive / tension-negative** (§7.4). Needs the same inputs as a normal run; runs no region-grow. Parallel mode runs it automatically after aggregate |
| `--max_cell_offset INT` | `400` | cap on the local-window half-size [cells] during boundary expansion. Clusters that hit it get `terminate_reason=7` and **diverge from MATLAB** (which retries unboundedly); a warning is printed. Raise it to grow very large clusters fully |

### 4.3 Examples

**Seismic + dry + uniform soil depth**
```bash
python python/driver.py \
  --soil_moisture_mode 0 \
  --soil_depth_mode 2 --soil_depth_uniform 1.5 \
  --soil_strength_mode 2 --phi_uniform 30 --coh_uniform 5 \
  --nogrow_mode 0 \
  --seismic_mode uniform --uniform_PGA 0.3 \
  --susname_override eq_dry_test
```

**Pure-Python (no MATLAB-generated `.mat`)**
```bash
python python/driver.py \
  --soil_depth_source compute \
  --nogrow_source compute \
  --soil_strength_mode 2 --phi_uniform 25 --coh_uniform 2 \
  --susname_override python_only
```
After the run, `lib/soil_depth/<DEM>_soil_depth_python.mat` and
`lib/no_grow/<DEM>_no_grow_python.mat` are written so subsequent runs can use
`--soil_depth_source mat` for instant load.

⚠️ **`shear_strength.mat` is built by the user from the target soil's
strength-test data.** For the distribution strength mode
(`soil_strength_mode=1`), construct `(prob, prob_phi, prob_coh)` and save
to `lib/soil_strength/shear_strength.mat`. If you have no data, fall back
to the uniform-strength mode (`soil_strength_mode=2`) as shown above.

### 4.4 Low-memory resume (`--run-index` + `--aggregate`)

For a distribution run (`soil_strength_mode=1`) on a large DEM, computing all
runs in one process holds the whole growth state in memory at once. The
`--run-index`/`--aggregate` split runs each distribution index in its own
process (one growth per process ⇒ low peak memory) and combines them afterward:

```bash
# 1. Compute each run in its own process. Each writes
#    <out>/<susname>/contribs/contrib_run<NN>.npz (banked LAST, so its
#    presence means "index NN is complete" — safe to resume after a crash).
for N in 0 1 2 3 4 5 6 7 8 9; do
  python python/driver.py $COMMON --run-index $N
done

# 2. Combine all banked contributions into the final sus_<susname>_python.tif.
python python/driver.py --DEM_path <dem> --susname_override <susname> \
  --out_dir <out> --aggregate 1
```

`--aggregate` refuses to write a misleading map: it errors if any index is
missing, if indices are duplicated, or if the banked contributions disagree on
the distribution size (stale files from a different `shear_strength.mat`), and
warns if the probabilities do not sum to 1. Each contribution records the run
count and a "no slides" flag so aggregation reproduces the monolithic loop's
early-stop behavior. The shared intermediate rasters (depth/nogrow/PGA/
hillshade) are written once, by `--run-index 0`.

> A resume-safe orchestrator for the multi-phi runs lives in
> `python/_sus_parallel.py` (2-wide, Windows/Docker; also backs the GUI's
> "Run all runs in parallel" checkbox).

---

## 5. Modes in detail

### 5.1 `soil_moisture_mode` — soil moisture model

| Value | Behaviour | Required parameters |
|---|---|---|
| **0 = dry** | hw=0, σ_s=0, W = γ_dry · depth · cellsize² | (none) |
| **1 = hydrostatic** | hw = mw · depth, σ_s = γ_w · hw | `mw` |
| ~~2 = hydromechanical~~ | van Genuchten + infiltration (not implemented) | SMAP, sand%, clay%, rainfall, Rosetta.csv |

In mode 1, `mw=0.5` means "the bottom half of the soil column is saturated"
— the water table sits at depth/2 above the slip surface.

### 5.2 `soil_depth_mode` / `soil_depth_model`

| Value | Behaviour |
|---|---|
| **1 = hillslope model** | compute the model selected by `--soil_depth_model` (with `--soil_depth_source compute`), or load a `.mat` |
| **2 = uniform** | depth = `soil_depth_uniform` everywhere |

Model choice when `--soil_depth_source compute`:

| `--soil_depth_model` | Description |
|---|---|
| **`roering`** (default) | Non-linear hillslope evolution (Roering 2008) for `soil_depth_endtime` years — the port of `lib/functions/soil_depth.m` (Numba-JIT). Evolves the *topography*; because it starts from a uniform 1 m soil mantle the absolute thickness stays dominated by that initial condition and only the spatial pattern is meaningful. Hours for a 60M-cell 5 m DEM. |
| **`massbalance`** | Soil-production function + mass balance as formulated in 松四雄騎 / Matsushi (2017), *Journal of Geography* 126(4), 487–511, eq. (7)–(15), and applied to a Japanese granite watershed by Matsushi et al. (2016), *Trans. Jpn. Geomorph. Union* 37, 427–453. Holds the **measured DEM fixed** and solves for the thickness `h` only. |

#### The massbalance model

Mass balance (eq. 7) `ρ_soil·∂h/∂t = E_sap − ∇·q − W_soil`, soil production function (eq. 13)
`E_sap = E₀·exp(−α·h)`, and creep transport that is either non-linear (eq. 10)
`q = −ρ_soil·K·∇z / (1−(|∇z|/S_c)²)` or linear (eq. 9) `q = −ρ_soil·K·∇z`.

With the DEM held fixed, `D ≡ ∇·q + W_soil` is a constant map, so the per-cell ODE has a closed-form
solution and no time loop is required:

- steady state (`--soil_depth_mb_endtime 0`, eq. 14/15): `h = −(1/α)·ln(D/E₀)`
- transient (after `t` years from `h₀`): `h = (1/α)·ln( [E₀ − (E₀ − D·e^{αh₀})·e^{−αDt/ρ_soil}] / D )`

Divergent (convex, ridge/nose) cells have `D > 0` and reach a steady thickness; convergent (hollow)
cells have `D ≤ 0`, never reach steady state and keep accumulating soil, capped by
`--soil_depth_mb_h_max`. That matches the behaviour reported by Matsushi et al. (2016): noses mostly
below 0.5 m at steady state, hollows growing to ~1.2 m over a few hundred years.

#### Parameter sets (`--soil_depth_mb_preset`)

| | `oregon` | `matsushi` |
|---|---|---|
| E₀ (D₀) [g m⁻² yr⁻¹] | 720 | **965.8** |
| α [m⁻¹] | 3.0 | **0.948** |
| K [m² yr⁻¹] | 0.005 | **0.005** (envelope 3.5–6.5×10⁻³) |
| transport law | non-linear (S_c=1.25) | **linear** (S_c unused) |
| ρ_soil [kg m⁻³] | 1200 | **1090** |
| W_soil [g m⁻² yr⁻¹] | 0 | **0** (explicitly neglected) |
| thickness in the SPF | vertical h | **slope-normal H = h·cosθ** |

The `matsushi` values are the ¹⁰Be calibration for a granite watershed near Kyoto published in
**松四雄騎・外山 真・松崎浩之・千木良雅弘 / Matsushi, Toyama, Matsuzaki & Chigira (2016),
*Transactions, Japanese Geomorphological Union* 37(4), 427–453**: D₀ and α from eq. (5) and Fig. 7
(p.440), K from Fig. 6B (p.441; the preset uses their arithmetic mean of the envelope), ρ_soil from
the mean dry unit weight (p.442). Element leaching W and the aeolian/organic input S are dropped
because the paper argues |S − W| ≪ |D| on steep slopes (p.429). Their own run used a 1 m grid, a
uniform 0.5 m initial mantle and 500 years, and gave 0.3–0.5 m on convex noses, ~1 m in hollows
after 300–400 yr and a 700–800 yr shallow-landslide return period. The same paper reports
c = 1.43 kPa, φ = 29.1°, saturated/dry unit weights 15.4/10.7 kN m⁻³ and Ksat = 2×10⁻⁴ m s⁻¹.

> ⚠ **These parameters do not make Noto's ridges thin.** Convex cells on the Noto 5 m DEM have a
> median curvature of 0.0195 m⁻¹, but a 0.4 m steady soil needs 0.121 m⁻¹ — reached by only
> **0.14 %** of them; at 0.02 m⁻¹ the steady solution is 2.30 m. Noto is therefore
> production-limited under this parameter set, and their exact setup (0.5 m initial, 500 yr) gives a
> nearly uniform ~0.75 m (convex noses 0.66 m, planar 0.75 m, hollows 0.86 m). This is a site
> difference, not a parameter error: their own Fig. 6B shows nose cells of the Shirakawa watershed
> at 5 m curvatures of **+0.1 to +0.25 m⁻¹** (regression −∇²z = −0.34H + 0.22) — an order of
> magnitude sharper than Noto — and the paper itself notes that α = 0.948 sits below the published
> 1.4–3.8 m⁻¹ range while D₀ is high (p.440). State explicitly which variant you report: the
> steady state gives 2–3 m, their setup gives ~0.75 m.
>
> Verification: the closed-form solver matches the paper's own numerical scheme (forward Euler of
> eq. 7, 1 yr × 500 steps) to within 0.0001 m on a 300×300-cell Noto window (2026-07-28).

The earlier paper **Matsushi, Matsuzaki & Makino (2014), *Trans. Jpn. Geomorph. Union* 35(2),
165–185** also reports Japanese granitic values (S_c ≈ 1, K/L = 7×10⁻⁵–3×10⁻⁴ m yr⁻¹,
ρ_soil = 1.9×10⁶ g m⁻³, W = 66.8 g m⁻² yr⁻¹, basin denudation 2.0×10²–1.8×10⁴ g m⁻² yr⁻¹), but those
belong to a basin-scale denudation / transport-law test, not to the soil-development simulation, and
are not used by the preset.

#### Hollows (`--soil_depth_mb_hollow_endtime`)

A steady state only exists on divergent slopes (D > 0); **35.3 %** of the Noto DEM is convergent and
would otherwise sit at `h_max`. With `--soil_depth_mb_endtime 0 --soil_depth_mb_hollow_endtime 750`
divergent cells take the steady solution while convergent cells get the transient one — i.e. 750
years of refill since the last shallow landslide (Matsushi et al. 2016 report a 700–800 yr return
period).

#### Cells steeper than S_c

Cells with |∇z| ≥ (1−0.025)·S_c are transport-unlimited (landsliding) and are pinned to bare bedrock
(h = h_min), and the non-linear divergence is evaluated with the analytic expansion used by
`soil_depth.m`, `div = K[∇²z/den + num2/(S_c²·den²)]`, where `den` is the cell's own value.
Differentiating the flux field numerically instead lets the clamped flux of over-steep cells inject a
huge artificial convergence into their neighbours — a 54° cliff then ends up with 3 m of soil. This
is the divergence Matsushi et al. (2014, p.180) describe for cells with |∇z| > S_c.

Example — the same setup Matsushi et al. (2016) used, a uniform 0.5 m mantle evolved for 500 years
(Noto 5 m DEM, 61.9M valid cells, ~67 s):

```bash
python python/precompute_inputs.py --DEM_path lib/DEM/dem_afterEQ_5m_crop.tif --soil_depth_model massbalance --soil_depth_mb_preset matsushi --soil_depth_mb_endtime 500 --soil_depth_mb_h_init 0.5 --soil_depth_mb_smooth 1
```

Output goes to `lib/soil_depth/<DEM>_soil_depth_python_massbalance_matsushi.mat` (+ QA GeoTIFF); the
Roering output (`..._soil_depth_python.mat`) is a different filename and is not overwritten. Result
over land only: mean 0.756 m, median 0.749 m, 5–95th pct 0.63–0.89 m; convex noses 0.66 m, planar
0.75 m, hollows 0.86 m (Roering 5000 yr: mean 1.14 m, median 1.06 m, 13.8 % below 0.5 m).

> ⚠ Always mask the sea before computing statistics or drawing maps. This DEM fills the sea with
> ~0–0.5 m, so no elevation threshold separates it from low coastal land; rasterize
> `lib/coastline/N03-20260101_17.shp` instead (land = 56,386,695 cells = 1,410 km²). The product
> raster itself keeps values everywhere because the driver expects a soil depth wherever the DEM is
> valid.

### 5.3 `soil_strength_mode`

| Value | Behaviour |
|---|---|
| **1 = distribution** | load `shear_strength.mat` containing `prob, prob_phi, prob_coh` (10 pairs), weighted susceptibility |
| **2 = uniform** | single (`phi_uniform`, `coh_uniform`) pair, one run only |

### 5.4 `nogrow_mode`

`nogrow_mode` is a 2-valued switch (0=off / 1=on). When on, the actual
algorithm is chosen via `nogrow_source`.

| `nogrow_mode` | Behaviour |
|---|---|
| **0 = off** | no growth constraint (clusters can grow anywhere) |
| **1 = on** | obtain the no-grow mask from one of the three sources below |

`nogrow_source` (active when mode=1):

| Value | Name | What it does |
|---|---|---|
| **`mat`** | Load existing `.mat` | reuse a MATLAB-side or previous Python run (seconds) |
| **`compute`** | Acc-threshold ridges + valleys (TopoToolbox-style) | D8 flow → accumulation; cells with `acc > valley_acc_thresh` are valleys, cells with high acc on the inverted DEM (`acc > ridge_acc_thresh`) are ridges; thin to 1-pixel lines |
| **`grass`** | Slope units — original GRASS r.slopeunits (Alvioli 2016/2020) | runs `r.slopeunits.create` (MFD by default) → `r.slopeunits.clean` (removes units below the minimum area) → complete slope-unit partition → unit boundaries become no-grow. **Requires the GRASS-enabled Docker image.** |

> The former `slopeunits` source (a pure-Python approximation of the upstream
> algorithm) has been removed: it diverged from the reference (ARI ≈ 0.35 on a
> real DEM) and suffered from over-segmentation / holes / exclaves. For faithful
> results, run the original via `grass`.

Main parameters of the `grass` mode (UI / CLI):

| Parameter | Meaning (r.slopeunits) | Typical |
|---|---|---|
| `slope_unit_thresh` | `thresh`: initial channel-defining acc threshold [m²] | 100,000–1,000,000 (default **500,000**) |
| `slope_unit_areamin` | `areamin`: minimum unit area [m²]; also used as `clean`'s `cleansize` | 50,000–200,000 (default **100,000**) |
| `slope_unit_cvmin` | `cvmin`: aspect circular-variance ceiling (0–1) | 0.25–0.5 (default **0.3**) |
| `slope_unit_rf` | `rf`: per-iteration threshold reduction factor (rounded to int) | 2–3 (default **2**) |
| `slope_unit_maxiter` | `maxiteration`: refinement-iteration cap (stops early on convergence) | 10–50 (default **50**) |

#### r.slopeunits toolset ↔ literature

The upstream r.slopeunits has four modules; this pipeline uses two:

| Module | Role | Used | Paper |
|---|---|---|---|
| **r.slopeunits.create** | delineation (half-basins + aspect-CV refinement; MFD default) | ✅ | Alvioli 2016 |
| **r.slopeunits.clean** | merge/remove units below `cleansize` | ✅ | (cleanup) |
| **r.slopeunits.metrics** | quality metric (V·I) | (via optimize) | Alvioli 2016 |
| **r.slopeunits.optimize** | search optimal `cvmin`/`areamin` | △ optional | Alvioli 2016/2020 |

- `create` produces **single-level (non-nested)** slope units — it has no
  "nested" parameter.
- **Important:** `create` alone does NOT enforce a minimum area (`areamin` only
  stops subdivision), so a few-cell fragments survive. **`clean`
  (`cleansize=areamin`)** removes them; the pipeline runs create → clean
  automatically.

##### `--slope_unit_optimize 1` (GUI "🎯 Optimize")

Runs `r.slopeunits.optimize` to **auto-tune cvmin/areamin** via the morphometric
objective **F = V·I** (Alvioli 2016) — **no landslide inventory** (`basin` is the
auto-derived DEM footprint). It searches the given cvmin/areamin ranges (defaults
cvmin∈[0.05,0.25], areamin∈[50000,200000]) to maximise F, then rebuilds the final
map with create+clean at the optimal values. `thresh`/`rf`/`maxiter` stay fixed.
Caveats: **very slow** (many create+clean+metrics runs — practical only on small
representative areas); **requires a ≥ 1 m DEM** (metrics truncates the resolution
to an integer, so a 0.5 m DEM gives `resolution=0` and fails — use create+clean
with optimize OFF instead). Landslide polygons are not used by r.slopeunits.

When to choose what:
- **Reproduce MATLAB results** → `mat` (existing file) or `compute`
- **Derive a ridge/valley network from the DEM without GRASS** → `compute`
- **Faithful r.slopeunits slope units** → `grass` (Docker)

### 5.5 `seismic_mode`

Pseudostatic analysis adds PGA · W to Q.

| Value | Behaviour |
|---|---|
| **off** | PGA = 0 |
| **uniform** | PGA = `uniform_PGA` everywhere |
| **raster** | PGA loaded from a GeoTIFF (NaN cells set to 0) |

The final PGA is multiplied by `pseudo_scaling`.

---

## 6. Input files and where to put them

| File | Location | Required? | Regenerable by Python? |
|---|---|---|---|
| **DEM (.tif)** | `lib/DEM/<name>.tif` | ✅ required | — (input data) |
| **Soil depth .mat** | `lib/soil_depth/<DEM_stem>_soil_depth.mat` | mode=1 + source=mat | ✅ `--soil_depth_source compute` |
| **No-grow .mat** | `lib/no_grow/<DEM_stem>_no_grow.mat` | mode=1 + source=mat | ✅ `--nogrow_source compute` or `grass` |
| **Shear-strength parameter distribution .mat** | `lib/soil_strength/shear_strength.mat` | strength_mode=1 | ⚠ **Built by the user from local soil-strength test data.** Schema: `prob[N], prob_phi[N], prob_coh[N]`, Σ prob = 1 |
| **PGA TIFF** | anywhere (recommended `lib/seismic/`) | seismic=raster | — (external input) |

⚠️ Everything under `lib/` is git-ignored. Either fetch the sample data
from <https://code.usgs.gov/ghsc/lhp/regiongrow3d>, or drop your own DEM
and recompute the rest via Python.

### .mat schema
**soil_depth**: key `depth` (2-D array)

**no_grow**: keys `nogrow_io`, `nogrow_idx`, `nogrow_i`, `nogrow_j`, `ridge_io`,
`valley_io`

**shear_strength**: keys `prob[N]`, `prob_phi[N]`, `prob_coh[N]`

---

## 7. Reading the output

### 7.1 Files
Output directory `<out_dir>/<susname>/`:

| File | Content |
|---|---|
| `sus_<susname>_python.tif` | susceptibility 0–100 % (main output) |
| `net_force_prob_<susname>.tif` | tension/compression map. Per-cell net force q (q>0 = tension, q<0 = compression) **sign-flipped (`−q`)** and probability-weighted over φ → **output is compression-positive / tension-negative** (parallel mode or `--tension_compression 1`; details §7.4) |
| `depth.tif` | soil depth [m] (`save_intermediates=1`) |
| `nogrow_io.tif` | no-grow mask 0/1 |
| `PGA.tif` | PGA [g] |
| `hillshade.tif` | hillshade for visualisation |
| `run_summary.json` | per-run cluster counts + φ/c/prob |

### 7.2 Interpreting susceptibility

**In one line**: susceptibility is the probability (0–100 %) that the cell
becomes a landslide source under the assumed strength distribution. It is
the sum of run probabilities for the runs that flagged the cell.

#### Formula
```
sus[i,j] = Σ_k  prob[k] · slides_final[k][i,j]
```
- `k` = 0, 1, …, 9 indexes the 10 strength runs
- each run uses a different `(φ_k, c_k)` pair (weakest → strongest soil)
- **`prob[k]`** = prior probability that the soil strength equals
  `(φ_k, c_k)` (the discrete probability mass of the k-th bin in the
  strength distribution). Supplied externally via `shear_strength.mat`,
  with Σ prob[k] = 1
- `slides_final[k][i,j] = 1` iff the cell ended up in a landslide cluster
  in run `k`, else 0
- weighted by `prob[k]` and summed gives the cell's susceptibility

#### Monotonicity
Janbu's limit-equilibrium equation is monotone in `(φ, c)`: **a cell that
fails under a given strength keeps failing under any weaker strength**.
Each cell therefore has a threshold `k*` (the strongest run that still
fails it), and

```
slides_final[k][i,j] = 1  if k ≤ k*  (weaker soils)
                     = 0  if k >  k*  (stronger soils)
```

so susceptibility is the **cumulative probability from the weakest end**:

```
sus[i,j] = prob[0] + prob[1] + … + prob[k*]
```

#### Concrete example

With `prob = [0.01, 0.04, 0.13, 0.29, 0.29, 0.13, 0.07, 0.03, 0.02, 0.005]`
(index 0 = weakest, index 9 = strongest, median at index 4–5),
the susceptibility takes the following **discrete values** depending on
`k*`:

| k* (strongest failing run) | sus | Reading |
|---|---|---|
| **never fails** | **0 %** | stable even under the weakest soil → fully safe |
| 0 (only the weakest) | **1 %** | fails only under extreme weakness → essentially safe |
| 0–1 | **5 %** | fails under very weak soil → low risk |
| 0–2 | **18 %** | fails under somewhat weak soil → watch under weathered profile |
| 0–3 | **47 %** | fails under sub-median strength → moderate risk |
| 0–4 | **76 %** | already fails at the median → **high risk (operational hazard zone)** |
| 0–5 | **89 %** | fails even under above-median strength → high risk |
| 0–6 | **96 %** | fails under most strengths |
| 0–7 | **99 %** | fails even under strong soil → very high risk |
| 0–8 | **99.5 %** | fails under everything except the strongest |
| 0–9 (every run) | **100 %** | fails even under the strongest → certain landslide |

With a 10-run distribution the susceptibility raster only takes these
**11 discrete values** — there is no continuous gradient between them.

#### Operational thresholds

| sus | Interpretation |
|---|---|
| `0 %` | stable across the assumed strength range → no mitigation |
| `< 50 %` | only fails under weaker-than-median soils → monitor |
| `≥ 50 %` | fails at or below median strength → **hazard zone, consider mitigation** |
| `100 %` | fails regardless of strength → must mitigate |

### 7.3 Statistics
- **>0% cells**: included by at least one run (overestimation-leaning)
- **>50% cells**: above-half probability of failure (high-confidence)
- **>90% cells**: near-certain landslide source

### 7.4 Tension/compression map (`net_force_prob_<susname>.tif`)

A continuous field describing the **force state inside the slides** (auto-generated
in parallel mode, or via CLI `--tension_compression 1`; needs each φ's contrib).

**What it is:** `Interslice_Force` solves equilibrium at **every cell of a slide**,
not just its margin, returning a per-cell net force **`q = driving − resisting`**:

| sign of q | meaning | typical location |
|---|---|---|
| `q > 0` | driving dominates = **tension** (active) | slide head / upslope |
| `q < 0` | resisting dominates = **compression** (passive) | slide toe / downslope |

**Sign convention (important):** for visualisation the TIF stores the
**sign-flipped `−q`** probability-weighted over φ (`value = Σ_k prob[k]·(−q_k)`,
only for runs where the cell slides). Because of this flip, **file values are
compression-positive / tension-negative**:

| file value | state | suggested colour |
|---|---|---|
| **positive (> 0)** | compression (resisting) | red |
| **negative (< 0)** | tension (driving) | blue |

**Display:** apply a 0-centred *diverging* colour ramp (e.g. QGIS RdBu, with a
symmetric min/max, positive = red / negative = blue).

---

## 8. Comparing with MATLAB

### 8.1 Comparison scripts (developer-only)

Developer scripts `compare_with_matlab.py` / `analyze_diff.py` /
`verify_against_upstream.py` exist for pixel-level verification against
MATLAB reference outputs (typically `post_processing/tests/<test_id>/sus_*.tif`).
These scripts depend on private MATLAB-derived data and are
**excluded from the public repository** (`.gitignore`). To reproduce the
numerical verification, generate the reference TIFF in the upstream MATLAB
environment and place it at the same path.

### 8.2 Local MATLAB vs USGS upstream
The (developer-only) `verify_against_upstream.py` script diffs the local
`lib/functions/*.m` against the USGS public repo. Result: only
`lib/driver.m` differs (63 lines — parameter changes plus a
`sigma_s_wedge` initialisation patch). The 33 helper functions are identical.

### 8.3 Known Python-vs-MATLAB divergences

The port targets pixel-exact parity. The following are **intentional or
tracked** differences; regenerate the MATLAB reference and re-run §8.1 after
changing any of them:

- **Boundary-expansion cap** (`--max_cell_offset`, default 400): MATLAB retries
  window expansion unboundedly. Python caps it as a memory/hang safety valve;
  a capped cluster gets `terminate_reason=7` and its growth is truncated. A
  per-cluster warning is printed. Raise the cap (or set it very high) for a
  strict comparison.
- **Alpha-shape boundary** (`alpha_shape_boundary`): the shrink-factor→alpha
  mapping is linear in circumradius rather than MATLAB `boundary()`'s discrete
  rank selection. No MATLAB fixture currently validates this; treated as an
  open parity item.
- **Root reinforcement** (`--S_roots > 0`): the skip-slide test compares
  `F_roots` against the current cluster's `Q_mag` only, whereas MATLAB compares
  against its full preallocated matrix. Moot at the default production
  `S_roots=0` (`F_roots=0`).

`terminate_reason` codes in the diagnostics dict: 0=converged, 2=no eligible
cells, 3=error increased, 4=max growth cycles, 5=geometry failure, 6=degenerate
weight, 7=boundary-expansion cap (Python-only).

---

## 9. Troubleshooting

### 9.1 GDAL / rasterio warning
```
Warning 3: Cannot find gdalvrt.xsd (GDAL_DATA is not defined)
```
Cosmetic only. To silence it: `conda install -c conda-forge gdal`.

### 9.2 Numba compile errors
Older numpy may be incompatible with the installed numba:
```bash
pip install --upgrade numba numpy
```

### 9.3 Streamlit port conflict
If 8501 is taken:
```bash
streamlit run python/gui.py --server.port 8502
```

### 9.4 Run is slow (full distribution > 60 min)
- `--soil_strength_mode 2` reduces to one run (≤ 10 min)
- `--run-index 9` computes only one distribution index (writes a partial map + banks a contribution; combine with `--aggregate` — see §4.4)
- `--soil_depth_mode 2` skips soil-depth simulation

### 9.5 Memory pressure
A DEM larger than 5000×5000 needs ≥16 GB RAM (a full-shape `cluster_io`
buffer, etc.).

### 9.6 Stuck on "Computing..."
- Check `python/output/<susname>/` for partial outputs.
- The subprocess survives even after the Web UI is closed; you can kill it
  via Task Manager.

### 9.7 .mat output files are large
`scipy.io.savemat` writes uncompressed by default (~100 MB). To compress:
```python
from scipy.io import savemat
savemat(path, data, do_compression=True)
```

---

## 10. Performance tips

### 10.1 Indicative cost

Absolute timings depend heavily on DEM size, terrain complexity, and CPU.
The table below is a **relative-cost guide** (orders of magnitude). Measure
on your machine for real numbers.

| Stage | Relative cost |
|---|---|
| DEM load + gradient | 1× (light) |
| Soil depth (mat) | 1× |
| Soil depth (compute, Roering 5000 yr) | ~100× (heavy, JIT speeds it up) |
| No-grow (mat) | 1× |
| No-grow (compute, acc-threshold) | ~30× |
| No-grow (compute, slope units) | ~50× |
| Hydrostatic + PGA | 1× |
| Force fields (1 + 8 rotations) | ~3× |
| RegionGrow (1 run) | ~100–300× (scales with cluster count) |
| Full distribution (10 runs) | ~10× a single RegionGrow run |

> ⚡ The table is a **compute-cost** guide. Using "Run all runs in parallel"
> (2-wide) roughly halves the full-distribution **wall-clock** (same compute; §10.2).

### 10.2 Tips
1. **Numba is essential**: without it `soil_depth` is 50× slower.
2. **`mat` source**: if the DEM doesn't change, loading `.mat` is seconds.
3. **φ parallelism (2-wide)**: the "⚡ Run all runs in parallel" checkbox runs the
   10 runs 2 processes at a time (`_sus_parallel.py`, §3.6) for ~2× throughput;
   result is bit-identical to serial. ~20 GB per run (raise the VM via
   `.wslconfig` under Docker).
4. **Qhull speedup (Windows, automatic)**: `region3d/_fastqhull.py` removes
   scipy Delaunay's per-call temp file (Defender-scanned on Windows — the
   dominant cost of the cluster-boundary geometry) by reusing one file → ~2×
   per run. Output bit-identical; a no-op on Linux.
5. **CPU parallelism (within a run)**: the cluster loop of a single run is mostly
   single-threaded; only the `prange` JIT'd functions parallelise automatically.

### 10.3 Algorithmic options
- `cluster_size_thresh`: 7 (default). Smaller = more candidate clusters →
  more compute.
- `max_growth_cycles`: 120 (default). Maximum growth iterations per cluster.
- `rot_num`: 8 (default). Rotation count for force-closure check. Lower =
  faster but less accurate.

---

## 11. API reference

### 11.1 driver.py
```python
from driver import run, DEFAULTS
from types import SimpleNamespace

args = SimpleNamespace(**DEFAULTS)
args.DEM_path = 'lib/DEM/<your_DEM>.tif'  # required
args.test_no = 1                           # required (drives output sub-folder)
args.susname_override = 'my_run'           # optional free-form override
run(args)
```

### 11.2 region3d.region_grow
```python
from region3d.region_grow import region_grow_fxn

result = region_grow_fxn(Z, coh, phi, gam_w, gam_dry, gam_sat, Gs,
                         W, sigma_s, sigma_s_wedge, PGA, ...)
# result fields:
#   slides_initial_io, slides_eroded_io, slides_final_io  (bool 2D)
#   cluster_idx_initial / eroded / final  (list of 1D arrays)
#   diagnostics (dict: terminate_reason, growth_cycles)
```

### 11.3 region3d.preprocessing
```python
from region3d.preprocessing import (soil_depth, ridges_valleys,
                                    fillsinks, identify_flats,
                                    flow_direction, flow_accumulation)

depth = soil_depth(Z, cellsize, endtime=5000.0, use_numba=True)
rv = ridges_valleys(Z, cellsize, ridge_acc_thresh=5, valley_acc_thresh=100)
# rv.nogrow_io, rv.ridge_io, rv.valley_io
```

### 11.4 region3d.io
```python
from region3d.io import read_dem, write_raster, load_soil_depth, load_no_grow

Z, georef = read_dem('DEM.tif')
write_raster('out.tif', sus_map, georef)
depth = load_soil_depth('soil_depth.mat')   # ndarray
nogrow = load_no_grow('no_grow.mat')        # dict
```

---

## References

- Mathews, N. W., Leshchinsky, B. A., Olsen, M. J., & Booth, A. M. (2024).
  RegionGrow3D: A Deterministic Analysis for Characterizing Discrete
  Three-Dimensional Landslide Source Areas on a Regional Scale.
  *JGR Earth Surface*, 129.
- Roering, J. J. (2008). How well can hillslope evolution models "explain"
  topography? *GSA Bulletin*, 120(9-10), 1248–1262.
- Hungr, O., Salgado, F. M., & Byrne, P. M. (1989). Evaluation of a
  three-dimensional method of slope stability analysis. *Canadian Geotechnical
  Journal*, 26(4), 679–686.
- Schwanghart, W., & Scherler, D. (2014). TopoToolbox 2 — MATLAB-based
  software for topographic analysis and modeling. *Earth Surface Dynamics*,
  2, 1–7.

### Soil-depth massbalance model (§5.2)

Primary sources of the equations (the model itself):

- Dietrich, W. E., Reiss, R., Hsu, M.-L., & Montgomery, D. R. (1995). A
  process-based model for colluvial soil depth and shallow landsliding using
  digital elevation data. *Hydrological Processes*, 9, 383–400.
  → soil mass balance, eq. (7) `ρ_soil ∂h/∂t = E_sap − ∇·q − W_soil`.
- Heimsath, A. M., Dietrich, W. E., Nishiizumi, K., & Finkel, R. C. (1997). The
  soil production function and landscape equilibrium. *Nature*, 388, 358–361.
  → soil production function, eq. (13) `E_sap = E₀ exp(−α h)`.
- Roering, J. J., Kirchner, J. W., & Dietrich, W. E. (1999). Evidence for
  nonlinear, diffusive sediment transport on hillslopes and implications for
  landscape morphology. *Water Resources Research*, 35, 853–870.
  → non-linear soil creep, eq. (10) `q = −ρ_soil K ∇z / {1 − (|∇z|/S_c)²}`.
  The linear law, eq. (9), is McKean et al. (1993) / Small et al. (1999).
- Larsen, I. J., Almond, P. C., Eger, A., Stone, J. O., Montgomery, D. R., &
  Malcolm, B. (2014). Rapid soil production and weathering in the Southern Alps,
  New Zealand. *Science*, 343, 637–640.
  → source of the published ranges E₀ = 10¹–10³ g m⁻² yr⁻¹ and α = 1–5 m⁻¹.

Formulation followed here, and the Japanese parameters:

- Matsushi, Y. / 松四雄騎 (2017). Quantification of long-term rates of bedrock
  weathering and soil production using terrestrial cosmogenic nuclides:
  principles, methodology, current research status, and perspectives.
  *Journal of Geography (Chigaku Zasshi)*, 126(4), 487–511.
  doi:10.5026/jgeography.126.487 (in Japanese with English abstract)
  → the formulation: mass balance eq. (7), transport laws eq. (9)/(10), soil
  production function eq. (13), steady-state forms eq. (14)/(15).
- Matsushi, Y., Matsuzaki, H., & Makino, H. / 松四雄騎・松崎浩之・牧野久識
  (2014). Testing models of landform evolution by determining the denudation
  rates of mountainous watersheds using terrestrial cosmogenic nuclides.
  *Transactions, Japanese Geomorphological Union*, 35(2), 165–185.
  (in Japanese with English abstract)
  → basin-scale values for Japanese granitic watersheds (S_c ≈ 1,
  K/L = 7×10⁻⁵–3×10⁻⁴ m yr⁻¹, W = 66.8 g m⁻² yr⁻¹, basin denudation
  2.0×10²–1.8×10⁴ g m⁻² yr⁻¹). A transport-law test, NOT used by the
  `matsushi` preset. Open access on J-STAGE.
- Matsushi, Y., Toyama, M., Matsuzaki, H., & Chigira, M. / 松四雄騎・外山 真・
  松崎浩之・千木良雅弘 (2016). Simulation of soil production and transport for
  prediction of location and magnitude of shallow landslides. *Transactions,
  Japanese Geomorphological Union*, 37(4), 427–453.
  (in Japanese with English abstract)
  → **the source of the `matsushi` preset** (Shirakawa watershed, Cretaceous
  granite east of the Kyoto basin): D₀ = 965.8 g m⁻² yr⁻¹ and α = 0.948 m⁻¹
  (p.440, eq. 5 + Fig. 7), K = 5×10⁻³ m² yr⁻¹ (mean of the 3.5–6.5×10⁻³
  envelope, Fig. 6B), ρ_soil = 1.09×10⁶ g m⁻³ (p.442), update rule
  ∂h/∂t = K∇²z + (D₀/ρ_soil)e^(−αh|cosθ|) (eq. 7). In-situ values c = 1.43 kPa,
  φ = 29.1°, γ_sat/γ_dry = 15.4/10.7 kN m⁻³, K_sat ≈ 2×10⁻⁴ m s⁻¹ (Table 3,
  Fig. 8). Every parameter cross-checked against the paper's pages and figures
  (2026-07-28).
