# RegionGrow3D — Python Port + Web UI

> 🌐 **日本語版**: [README-jp.md](README-jp.md)

A Python port of USGS's MATLAB
[RegionGrow3D](https://code.usgs.gov/ghsc/lhp/regiongrow3d) (Mathews et al.,
2024, *JGR Earth Surface*). Computes **shallow-landslide susceptibility maps**
under varying earthquake, water-table, and rainfall conditions. Includes a
Streamlit-based Web UI so you can run the analysis without a MATLAB licence.

Original paper: Mathews et al. (2024). *RegionGrow3D: A Deterministic Analysis
for Characterizing Discrete Three-Dimensional Landslide Source Areas on a
Regional Scale*. JGR Earth Surface.

> ⚠️ **Research code — community re-implementation.** This repository is
> **not** affiliated with or endorsed by the USGS. The software is provided
> "AS IS" and **there is no guarantee of exact numerical equivalence with
> the MATLAB upstream**. See [`DISCLAIMER.md`](DISCLAIMER.md) before any
> operational use.

---

## 1. Quick start

### Requirements
- Windows / macOS / Linux
- Docker (recommended) or Python 3.11+ (tested with 3.13)

### Setup (Docker — recommended)
```bash
docker build -t region3d:latest .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  region3d:latest
```
Or `docker compose up`. No local Python install needed; every native dependency
is resolved inside the container, so this is the most reproducible option.
See [docs/MANUAL.md](docs/MANUAL.md#22-docker) for details.

### Setup (conda)
```bash
conda env create -f environment.yml
conda activate region3d
```
`environment.yml` pins everything to `conda-forge` so `rasterio` (GDAL),
`numba` (llvmlite) and `scipy`/`numpy` (BLAS) all use compatible native
libraries.

### Setup (pip only — fallback)
If you cannot use conda, a pure-pip install also works on most platforms:
```bash
python -m venv .venv
.venv/Scripts/activate     # Windows  (POSIX: source .venv/bin/activate)
pip install -r requirements.txt
```
On Windows, `rasterio`/`numba` install from wheels; on Linux/macOS you may
need system-level GDAL / LLVM if pre-built wheels are unavailable for your
Python version.

### Launch the Web UI (local Python install)
```bash
streamlit run python/gui.py
```
Open <http://localhost:8501>, configure the parameters, and click **Start**.

For long sessions on Windows, launch detached so the server survives shell
exit — see [docs/MANUAL.md §3.5](docs/MANUAL.md#35-detached-launch-windows-overnight-runs).

### Command-line run
```bash
python python/driver.py --DEM_path lib/DEM/<your_DEM>.tif --test_no 1
```
Pass additional `--*_mat` paths as required by the chosen modes
(see [`docs/MANUAL.md`](docs/MANUAL.md) for the full argument list).
Output is written under `python/output/<susname>/sus_<susname>_python.tif`
where `<susname>` is the zero-padded `--test_no` or the value of
`--susname_override`.

### Where to put the sample data
> ⚠️ **Important**: DEMs (`.tif`) and pre-processed `.mat` files are **not
> bundled** with this repository (they are large; see `.gitignore`).
> Obtain them in either of two ways:
>
> - **From the USGS upstream**: <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
>   - `lib/DEM/<DEM>.tif`
>   - `lib/soil_depth/<DEM>_soil_depth.mat` (optional — Python can compute)
>   - `lib/no_grow/<DEM>_no_grow.mat` (optional — Python can compute)
>   - `lib/soil_strength/shear_strength.mat` (required for the distribution
>     strength mode — **built by the user from the target soil's strength
>     data.** Schema: `prob[N], prob_phi[N], prob_coh[N]`, Σ prob = 1)
>   - `lib/hydro_interp/*.mat` (mode 2 only — not used by this Python port)
> - Drop any GeoTIFF into `lib/DEM/`; it will appear in the Web-UI dropdown.
>
> Soil-depth and no-grow `.mat` files can be computed by Python
> (`--soil_depth_source compute --nogrow_source compute` → saved as
> `<DEM stem>_..._python.mat`).
> **`shear_strength.mat` is a file the user builds from the target soil's
> strength data** (`prob[N], prob_phi[N], prob_coh[N]` arrays, Σ prob = 1).
> If you have no strength-distribution data, use the uniform-strength
> mode instead, e.g. `--soil_strength_mode 2 --phi_uniform 25 --coh_uniform 2`.
> A companion repository
> [`Simplified_Janbu_Method_3D_2D`](https://github.com/T40O0/Simplified_Janbu_Method_3D_2D)
> can also build `shear_strength.mat`.

---

## 2. Directory layout

```
RegionGrow3d-py/
├ lib/                       # input-data drop-zone (contents are git-ignored)
│  ├ DEM/                    # place sample DEM(s) (.tif) here
│  ├ soil_depth/             # soil-depth .mat (optional — Python can recompute)
│  ├ no_grow/                # no-grow zone .mat (optional — Python can recompute)
│  ├ soil_strength/          # shear-strength parameter distribution .mat (user-built per site)
│  └ seismic/                # PGA raster drop-zone
├ python/
│  ├ gui.py                  # Streamlit Web UI
│  ├ driver.py               # CLI driver (used by both CLI and UI)
│  └ region3d/               # core algorithm (see "Port lineage" below)
│     ├ region_grow.py
│     ├ forces.py
│     ├ boundary.py
│     ├ growth.py
│     ├ localize.py
│     ├ derivatives.py
│     ├ bwmorph.py
│     ├ matlab_compat.py
│     ├ preprocessing.py
│     ├ grass_slopeunits.py   # GRASS r.slopeunits wrapper
│     ├ vectorize.py          # slope-unit polygons / coastline clip
│     ├ runner.py
│     └ io.py
├ docs/MANUAL.md             # detailed manual (English)
├ docs/MANUAL-jp.md          # detailed manual (Japanese)
├ README.md / README-jp.md   # this file
└ .gitignore
```

> **The original USGS MATLAB sources** (`lib/driver.m`, `lib/functions/*.m`,
> `lib/hydro_interp/*.m`, `post_processing/susceptibility_map.m`) are
> **not bundled** with this repository — they are excluded via `.gitignore`.
> Obtain them from the upstream <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
> if you need to cross-reference. The Python port below stands on its own.

### Port lineage (MATLAB → Python)

| MATLAB source (USGS, Mathews et al. 2024) | Python port (this repo) |
|---|---|
| `driver.m` | `python/driver.py` |
| `RegionGrowFxn.m` | `python/region3d/region_grow.py` |
| `Interslice_Force.m`, `Interslice_Force_Prism.m`, `force_closure_interslice.m` | `python/region3d/forces.py` |
| `boundary_geometry_interslice.m`, `polygeom.m`, `root_force_boundary.m`, `nogrow_not_eligible.m` | `python/region3d/boundary.py` |
| `downhill_dilate.m`, `spur_test.m`, `continuity_check.m`, `update_cluster_interslice.m` | `python/region3d/growth.py` |
| `create_localized_rasters_interslice.m` | `python/region3d/localize.py` |
| `pad_DEM.m`, `gradient_prince.m`, `hillshade.m` | `python/region3d/derivatives.py` |
| `soil_depth.m`, `fillsinks.m`, `identifyflats.m`, `flowacc.m`, `ridgelines.m`, `valleys.m`, `FLOWobj.m`, `FLOWobjInv.m`, `GRIDobj.m`, `copy2GRIDobj.m` | `python/region3d/preprocessing.py` *(some TopoToolbox-derived; see `LICENSE`)* |
| `saveraster.m` + MATLAB `geotiffread` | `python/region3d/io.py` |
| MATLAB `bwmorph`, `bwconncomp`, `bwboundaries` (Image-Processing Toolbox) | `python/region3d/bwmorph.py`, `python/region3d/matlab_compat.py` |
| *(new — no MATLAB counterpart)* slope-unit segmentation via the original GRASS r.slopeunits (Alvioli 2016/2020) | `python/region3d/grass_slopeunits.py` (Docker) |
| *(new — no MATLAB counterpart)* persistent run-state for the Streamlit UI | `python/region3d/runner.py` |

The Python sources keep the MATLAB function name in their module docstring so
the lineage is traceable from the code as well.

---

## 3. Feature overview

### Supported modes (same names as MATLAB driver.m)
| Mode | Values | Python support |
|---|---|---|
| `soil_moisture_mode` | 0=dry / 1=hydrostatic (mw) / 2=hydromechanical | ✅ / ✅ / ❌ |
| `soil_depth_mode` | 1=Roering / 2=uniform | ✅ / ✅ |
| `soil_strength_mode` | 1=distribution / 2=uniform | ✅ / ✅ |
| `nogrow_mode` | 0=off / 1=on | ✅ / ✅ |
| `nogrow_source` (when mode=1) | mat / compute (acc-threshold) / grass (original r.slopeunits, Docker) | ✅ / ✅ / ✅ |
| `seismic_mode` | off / uniform / raster | ✅ / ✅ / ✅ |
| `root_mode` | uniform | ✅ |

### Preprocessing pipeline
Soil depth and the no-grow mask can be:
- **`mat`**: load existing `.mat` files (MATLAB-generated or previously
  Python-generated), seconds.
- **`compute`**: regenerate from the DEM (Numba JIT; soil depth ≈ 3 min for
  5000 yr, ridges/valleys ≈ 1 min). Results are saved as
  `<DEM stem>_..._python.mat` and automatically appear in the dropdown next
  time.

### Web UI highlights
- Six mode toggles, each with its own parameters directly underneath.
- File-placement guide at the top of the sidebar.
- DEM upload — dropped files are saved to `lib/DEM/` and re-listed.
- Output destination: parent directory + sub-folder name (free naming) +
  overwrite warning when the folder already exists.
- Live progress bar with ETA + log stream.
- During a run the sidebar locks, a banner is shown on the main pane, and a
  Stop button is always available.
- Result tabs:
  - 🗺 Maps (susceptibility / no-grow / soil depth / PGA, with hillshade
    overlay)
  - 📊 Statistics (valid cells, share above 50%, per-run cluster counts)
  - 📈 Histogram (positive-cell distribution + CDF)
  - GeoTIFF download buttons

---

## 4. License

This repository is **dual-licensed**:

| Scope | Licence | Reference |
|---|---|---|
| Most files (default) | **CC0 1.0 Universal** (public domain) | [`LICENSE-CC0`](LICENSE-CC0) |
| TopoToolbox-derived helpers in `python/region3d/preprocessing.py` (`fillsinks`, `identify_flats`, `flow_direction`, `flow_accumulation`, etc.) | **GPL-3.0-or-later** | [`LICENSE-GPL`](LICENSE-GPL) |

Per-file SPDX identifiers mark which licence applies. See [`LICENSE`](LICENSE)
for the function-level breakdown and reuse rules.

- Original MATLAB (USGS RegionGrow3D): CC0 1.0 — <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
- TopoToolbox (Schwanghart & Scherler 2014): GPL-3.0 — <https://github.com/wschwanghart/topotoolbox>

> ⚠️ Redistributing a combined work that includes the GPL-3.0 functions
> (e.g. a container image or PyPI package) triggers the GPL source-provision
> obligations. Extracting only the CC0 portions for downstream reuse keeps
> them under CC0.

---

## 5. Credits
- Original algorithm: Nicolas W. Mathews et al. (USGS)
- TopoToolbox: Wolfgang Schwanghart
- Python port / Web UI: this repository

---

## 6. References

- Mathews, N. W., Leshchinsky, B. A., Olsen, M. J., & Booth, A. M. (2024).
  *RegionGrow3D: A Deterministic Analysis for Characterizing Discrete
  Three-Dimensional Landslide Source Areas on a Regional Scale.*
  Journal of Geophysical Research: Earth Surface, 129.
  USGS upstream code: <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
- Schwanghart, W., & Scherler, D. (2014). *TopoToolbox 2 — MATLAB-based
  software for topographic analysis and modeling in Earth surface sciences.*
  Earth Surface Dynamics, 2, 1–7. <https://github.com/wschwanghart/topotoolbox>
- Alvioli, M., Marchesini, I., Reichenbach, P., Rossi, M., Ardizzone, F.,
  Fiorucci, F., & Guzzetti, F. (2016). *Automatic delineation of
  geomorphological slope units with `r.slopeunits` v1.0 and their
  optimization for landslide susceptibility modeling.* Geoscientific
  Model Development, 9, 3975–3991. <https://doi.org/10.5194/gmd-9-3975-2016>
  (the slope-unit delineation [`create`] and optimisation [`metrics`/`optimize`]).
- Alvioli, M., Guzzetti, F., & Marchesini, I. (2020). *Parameter-free
  delineation of slope units and terrain subdivision of Italy.*
  Geomorphology, 358, 107124. <https://doi.org/10.1016/j.geomorph.2020.107124>
  (basis for the automatic parameter optimisation, `r.slopeunits.optimize`).
- GRASS GIS addon **`r.slopeunits`** (Marchesini, Alvioli, Metz, Tawalika, et al.).
  Run directly via `nogrow_source=grass`; bundled in the Docker image with
  `g.extension r.slopeunits` (create / clean / metrics / optimize).
  <https://grass.osgeo.org/grass-stable/manuals/addons/r.slopeunits.html>
- Hungr, O. (1989). *An extension of Bishop's simplified method of slope
  stability analysis to three dimensions.* Géotechnique, 39(4), 559–562.
- Companion repository for building `shear_strength.mat`:
  <https://github.com/T40O0/Simplified_Janbu_Method_3D_2D>
