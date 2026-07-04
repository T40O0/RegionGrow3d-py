"""Streamlit UI for the RegionGrow3D Python driver.

Run from the repo root:
    streamlit run python/gui.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / 'python' / 'driver.py'
SUS_PARALLEL = REPO / 'python' / '_sus_parallel.py'
PYTHON_EXE = sys.executable

# Allow imports from the repo
sys.path.insert(0, str(REPO / 'python'))

from region3d.runner import (read_manifest, write_manifest, clear_manifest,
                              proc_create_time,
                              read_last_completed, write_last_completed,
                              pid_alive, kill_pid, tail_log)


# Sentinel used by the run-index selector.
ALL_RUNS = "All runs"


# =============================================================================
#  Sidebar (parameter form)
# =============================================================================

st.set_page_config(page_title="RegionGrow3D", layout="wide",
                   initial_sidebar_state="expanded")

# Widen the sidebar (default ~21rem). Adjust SIDEBAR_PX as needed.
SIDEBAR_PX = 520
st.markdown(
    f"""
    <style>
      [data-testid="stSidebar"] {{
        min-width: {SIDEBAR_PX}px !important;
        max-width: {SIDEBAR_PX}px !important;
        width: {SIDEBAR_PX}px !important;
      }}
      [data-testid="stSidebar"] > div:first-child {{
        width: {SIDEBAR_PX}px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Running flag (computed FIRST so we can disable widgets) ----------------
# Source of truth is the on-disk manifest, NOT session_state. That way a fresh
# browser session (or a tomorrow-morning re-open) still sees the running run.
MANIFEST_ROOT = REPO / 'python' / 'output_webui'
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
_manifest = read_manifest(MANIFEST_ROOT)
IS_RUNNING = _manifest is not None and pid_alive(
    _manifest.get('pid'), _manifest.get('create_time'))
DIS = IS_RUNNING  # convenience alias

st.sidebar.title("RegionGrow3D")
st.sidebar.caption("Slope-stability analysis parameters")

if IS_RUNNING:
    st.sidebar.caption("🔒 Locked while running")

# ---- Input file placement guide --------------------------------------------
DEM_DIR = REPO / 'lib' / 'DEM'
SOIL_DEPTH_DIR = REPO / 'lib' / 'soil_depth'
NO_GROW_DIR = REPO / 'lib' / 'no_grow'
SOIL_STRENGTH_DIR = REPO / 'lib' / 'soil_strength'
SEISMIC_DIR = REPO / 'lib' / 'seismic'
SEISMIC_DIR.mkdir(parents=True, exist_ok=True)
COASTLINE_DIR = REPO / 'lib' / 'coastline'
COASTLINE_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar.expander("📁 Input file placement guide", expanded=False):
    st.markdown(
        f"""
| File | Location |
|---|---|
| **DEM (.tif)** | `{DEM_DIR.relative_to(REPO).as_posix()}/` |
| **Soil depth .mat** *(optional / used when `soil_depth_mode=1` and source=`mat`)* | `{SOIL_DEPTH_DIR.relative_to(REPO).as_posix()}/<DEM stem>_soil_depth.mat` |
| **No-grow .mat** *(optional / used when `nogrow_mode=1` and source=`mat`)* | `{NO_GROW_DIR.relative_to(REPO).as_posix()}/<DEM stem>_no_grow.mat` |
| **Shear strength parameters .mat** *(used when `soil_strength_mode=1`)* | `{SOIL_STRENGTH_DIR.relative_to(REPO).as_posix()}/shear_strength.mat` |
| **PGA raster (.tif)** *(used when `seismic_mode=raster`)* | `{SEISMIC_DIR.relative_to(REPO).as_posix()}/` (recommended; any path is OK) |

If you choose `compute` as the source, Python regenerates soil-depth /
no-grow from the DEM, so the `.mat` files are not strictly required.

DEMs can also be added via the file uploader below.
""")

# ---- DEM selection / upload -------------------------------------------------
st.sidebar.subheader("📍 DEM")
dem_choices = sorted([p.name for p in DEM_DIR.glob('*.tif')]) if DEM_DIR.exists() else []
dem_name = st.sidebar.selectbox("Existing DEM (lib/DEM/)", dem_choices,
                                 index=0 if dem_choices else None,
                                 disabled=DIS, key='dem_name')
uploaded = st.sidebar.file_uploader(
    "Upload a new DEM (.tif → saved to lib/DEM/)",
    type=['tif', 'tiff'], accept_multiple_files=False, disabled=DIS,
    key='dem_upload')
if uploaded is not None and not DIS:
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    target = DEM_DIR / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    st.sidebar.success(f"Saved: {target.relative_to(REPO).as_posix()}")
    dem_name = uploaded.name

# =============================================================================
#  Mode + parameters (each mode immediately followed by its inputs)
# =============================================================================

st.sidebar.subheader("⚙ Mode settings")
st.sidebar.caption("Each mode's parameters appear directly below its toggle")

# ---- 1. Soil moisture -------------------------------------------------------
soil_moisture_mode = st.sidebar.radio(
    "💧 Soil moisture", ["0 = Dry", "1 = Hydrostatic (mw)"], index=1,
    help="2 = Hydromechanical model is not implemented (would require SMAP/"
         "rainfall/sand-clay rasters).",
    key='soil_moisture_mode_radio', disabled=DIS)
soil_moisture_mode = int(soil_moisture_mode.split(' ')[0])
if soil_moisture_mode == 1:
    mw = st.sidebar.slider(
        "  mw (saturation ratio = hw/depth)", 0.0, 1.0, 0.5, 0.05, key='mw',
        help="0=dry, 1=fully saturated. 0.5 sets the water table at half "
             "the soil depth.",
        disabled=DIS)
else:
    mw = 0.5
    st.sidebar.caption("  Dry mode: hw=σ_s=0, W=γ_dry·depth·cellsize²")

# ---- 2. Soil depth ----------------------------------------------------------
soil_depth_mode = st.sidebar.radio(
    "🟫 Soil depth", ["1 = Roering evolution model", "2 = Uniform"], index=0,
    key='soil_depth_mode_radio', disabled=DIS)
soil_depth_mode = int(soil_depth_mode.split(' ')[0])
soil_depth_source = 'mat'
soil_depth_mat = ''
if soil_depth_mode == 2:
    soil_depth_uniform = st.sidebar.number_input(
        "  Uniform soil depth (m)", 0.1, 10.0, 2.0, 0.1, key='soil_depth_uniform',
        disabled=DIS)
    soil_depth_endtime = 5000.0
else:
    soil_depth_uniform = 2.0
    # Source under the toggle: load .mat or run Python's Roering model.
    soil_depth_source = st.sidebar.radio(
        "  Soil-depth data source",
        ["💾 Load existing .mat (fast)",
         "🐍 Compute with Python (saves result to .mat)"],
        index=0, key='soil_depth_source_radio', disabled=DIS,
        help="Either load an existing .mat file (MATLAB or Python output) or "
             "recompute with Python's Numba JIT.")
    soil_depth_source = 'mat' if soil_depth_source.startswith('💾') else 'compute'
    if soil_depth_source == 'mat':
        sd_files = sorted(SOIL_DEPTH_DIR.glob('*.mat')) \
            if SOIL_DEPTH_DIR.exists() else []
        sd_choices = [p.name for p in sd_files]
        if not sd_choices:
            st.sidebar.error(
                f"  No .mat under `{SOIL_DEPTH_DIR.relative_to(REPO).as_posix()}/` "
                "— switch to Python compute.")
            soil_depth_endtime = 5000.0
            soil_depth_mat = ''
        else:
            # Default to <DEM stem>_soil_depth.mat if available
            stem = Path(dem_name).stem if dem_name else ''
            preferred = f"{stem}_soil_depth.mat"
            default_idx = sd_choices.index(preferred) \
                if preferred in sd_choices else 0
            sel = st.sidebar.selectbox(
                "  .mat file to load", sd_choices, index=default_idx,
                key='soil_depth_mat_select', disabled=DIS)
            soil_depth_mat = str(SOIL_DEPTH_DIR / sel)
            soil_depth_endtime = 5000.0
    else:
        soil_depth_endtime = st.sidebar.number_input(
            "  Roering simulation duration (yr)",
            1000.0, 20000.0, 5000.0, 1000.0, key='soil_depth_endtime',
            help="Saved to `lib/soil_depth/<DEM>_soil_depth_python.mat` "
                 "after the run.",
            disabled=DIS)

# ---- 3. Shear strength parameters -------------------------------------------
soil_strength_mode = st.sidebar.radio(
    "🪨 Shear strength parameters", ["1 = Distribution (.mat)", "2 = Uniform (φ, c)"],
    index=0, key='soil_strength_mode_radio', disabled=DIS)
soil_strength_mode = int(soil_strength_mode.split(' ')[0])
if soil_strength_mode == 2:
    phi_uniform = st.sidebar.number_input(
        "  φ' friction angle (deg)", 0.0, 90.0, 25.0, 1.0, key='phi_uniform',
        disabled=DIS)
    coh_uniform = st.sidebar.number_input(
        "  c' cohesion (kPa)", 0.0, 100.0, 2.0, 0.5, key='coh_uniform',
        disabled=DIS)
    run_only = ALL_RUNS  # mode=2 yields a single run anyway
    parallel_runs = False  # single run — nothing to parallelize
else:
    phi_uniform = 25.0
    coh_uniform = 2.0
    ss_path = SOIL_STRENGTH_DIR / 'shear_strength.mat'
    st.sidebar.caption(
        f"  Distribution: load `{ss_path.relative_to(REPO).as_posix()}` "
        "(10 runs)")
    run_only = st.sidebar.selectbox(
        "  Run a single run-index (quick test)",
        [ALL_RUNS, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        index=0, key='run_only', disabled=DIS)
    # Fan the 10 phi runs out across processes (each banks a contribution;
    # combined by --aggregate → bit-identical to the serial loop). Only
    # meaningful for the full distribution (not a single run-index).
    parallel_runs = st.sidebar.checkbox(
        "  ⚡ Run all runs in parallel (2 concurrent)",
        value=False, key='parallel_runs',
        disabled=DIS or (run_only != ALL_RUNS),
        help="Runs the 10 φ runs 2-at-a-time instead of serially, then "
             "aggregates. Result is identical to serial. Needs enough RAM for "
             "2 runs (~20 GB each) — in Docker raise the WSL VM "
             "(%USERPROFILE%\\.wslconfig → [wsl2] memory=48GB).")

# ---- 4. Growth boundary -----------------------------------------------------
nogrow_mode = st.sidebar.radio(
    "🚧 Growth boundary", ["0 = Off", "1 = Ridge + Valley"], index=1,
    key='nogrow_mode_radio', disabled=DIS)
nogrow_mode = int(nogrow_mode.split(' ')[0])
nogrow_source = 'mat'
no_grow_mat = ''
if nogrow_mode == 1:
    nogrow_source_label = st.sidebar.radio(
        "  No-grow algorithm",
        ["💾 Load pre-computed .mat",
         "🟰 Acc-threshold ridges + valleys (TopoToolbox-style)",
         "🟢 Slope units — GRASS r.slopeunits (reference)"],
        index=0, key='nogrow_source_radio', disabled=DIS,
        help="All compute paths save outputs to "
             "`lib/no_grow/<DEM>_no_grow*.mat` and (with save_intermediates) "
             "to GeoTIFF.\n\n"
             "• Acc-threshold: ridge/valley by flow-accumulation thresholds.\n"
             "• GRASS r.slopeunits: runs the original Alvioli algorithm via "
             "GRASS (bundled in the Docker image). Output is a complete, "
             "connected slope-unit partition; needs the GRASS-enabled "
             "container (docker compose).")
    if nogrow_source_label.startswith('💾'):
        nogrow_source = 'mat'
    elif nogrow_source_label.startswith('🟰'):
        nogrow_source = 'compute'
    else:
        nogrow_source = 'grass'

    # Defaults so the variables exist on every path
    ridge_acc_thresh = 5.0
    valley_acc_thresh = 100.0
    slope_unit_thresh = 500000.0
    slope_unit_areamin = 100000.0
    slope_unit_cvmin = 0.3
    slope_unit_rf = 2.0
    slope_unit_maxiter = 50
    slope_unit_save_shp = 1
    slope_unit_shp_format = '.shp'
    slope_unit_shp_clip = ''
    slope_unit_shp_clip_invert = 0
    slope_unit_optimize = 0
    slope_unit_cvmin_min, slope_unit_cvmin_max = 0.05, 0.25
    slope_unit_areamin_min, slope_unit_areamin_max = 50000.0, 200000.0

    if nogrow_source == 'mat':
        ng_files = sorted(NO_GROW_DIR.glob('*.mat')) \
            if NO_GROW_DIR.exists() else []
        ng_choices = [p.name for p in ng_files]
        if not ng_choices:
            st.sidebar.error(
                f"  No .mat under `{NO_GROW_DIR.relative_to(REPO).as_posix()}/` "
                "— switch to Python compute.")
            no_grow_mat = ''
        else:
            stem = Path(dem_name).stem if dem_name else ''
            preferred = f"{stem}_no_grow.mat"
            default_idx = ng_choices.index(preferred) \
                if preferred in ng_choices else 0
            sel = st.sidebar.selectbox(
                "  .mat file to load", ng_choices, index=default_idx,
                key='no_grow_mat_select', disabled=DIS)
            no_grow_mat = str(NO_GROW_DIR / sel)
    elif nogrow_source == 'compute':
        ridge_acc_thresh = st.sidebar.number_input(
            "  Ridge flow-accumulation threshold", 1.0, 100.0, 5.0, 1.0,
            key='ridge_acc_thresh', disabled=DIS)
        valley_acc_thresh = st.sidebar.number_input(
            "  Valley flow-accumulation threshold", 1.0, 1000.0, 100.0, 10.0,
            key='valley_acc_thresh', disabled=DIS)
        st.sidebar.caption(
            "  Saved to `lib/no_grow/<DEM>_no_grow_python.mat` after the run.")
    else:  # grass (GRASS r.slopeunits)
        slope_unit_thresh = st.sidebar.number_input(
            "  Initial channel threshold [m²]", 1000.0, 5_000_000.0, 500000.0,
            10000.0, key='slope_unit_thresh', disabled=DIS,
            help="r.slopeunits `thresh`. Channel-defining flow-accumulation "
                 "area; lower = denser channel network = smaller units.")
        slope_unit_rf = st.sidebar.number_input(
            "  Reduction factor (rf)", 2, 10, 2, 1,
            key='slope_unit_rf', disabled=DIS,
            help="r.slopeunits `rf` (integer). Per iteration: "
                 "thresh ← thresh − thresh/rf. Larger = gentler steps. "
                 "Manual example uses 2.")
        slope_unit_maxiter = st.sidebar.number_input(
            "  Max refinement iterations", 1, 100, 50, 1,
            key='slope_unit_maxiter', disabled=DIS,
            help="Upper bound on the aspect-CV refinement loop. The loop stops "
                 "early once no unit needs splitting, so a high value is safe. "
                 "r.slopeunits' manual example uses 50.")
        # Optimize toggle FIRST, so cvmin/areamin show as single values (off)
        # or as search ranges (on) — never both.
        slope_unit_optimize = 1 if st.sidebar.checkbox(
            "  🎯 Optimize cvmin/areamin (r.slopeunits.optimize — slow)",
            value=False, key='slope_unit_optimize', disabled=DIS,
            help="Morphometric auto-tuning (Alvioli 2016 objective F = V·I; "
                 "NO landslide inventory). Searches the ranges below to maximise "
                 "F; thresh/rf/maxiter stay fixed. SLOW — runs "
                 "create+clean+metrics many times (minutes on small DEMs, much "
                 "longer on large ones).") else 0
        if not slope_unit_optimize:
            slope_unit_cvmin = st.sidebar.number_input(
                "  Aspect CV ceiling (cvmin)", 0.0, 1.0, 0.3, 0.01,
                format="%.2f", key='slope_unit_cvmin', disabled=DIS,
                help="Aspect circular variance threshold: units with CV ≤ cvmin "
                     "stop being subdivided. Lower = more refinement.")
            slope_unit_areamin = st.sidebar.number_input(
                "  Minimum unit area [m²]", 100.0, 5_000_000.0, 100000.0, 5000.0,
                key='slope_unit_areamin', disabled=DIS,
                help="r.slopeunits `areamin` / clean `cleansize`. Units below "
                     "this are merged away.")
        else:
            st.sidebar.caption("  cvmin search range (literature 0.05–0.25)")
            _cc1, _cc2 = st.sidebar.columns(2)
            slope_unit_cvmin_min = _cc1.number_input(
                "cvmin min", 0.0, 1.0, 0.05, 0.01, format="%.2f",
                key='slope_unit_cvmin_min', disabled=DIS)
            slope_unit_cvmin_max = _cc2.number_input(
                "cvmin max", 0.0, 1.0, 0.25, 0.01, format="%.2f",
                key='slope_unit_cvmin_max', disabled=DIS)
            st.sidebar.caption("  areamin search range [m²] (literature 50,000–200,000)")
            _ac1, _ac2 = st.sidebar.columns(2)
            slope_unit_areamin_min = _ac1.number_input(
                "areamin min", 1000.0, 5_000_000.0, 50000.0, 1000.0,
                key='slope_unit_areamin_min', disabled=DIS)
            slope_unit_areamin_max = _ac2.number_input(
                "areamin max", 1000.0, 5_000_000.0, 200000.0, 1000.0,
                key='slope_unit_areamin_max', disabled=DIS)
        slope_unit_save_shp_bool = st.sidebar.checkbox(
            "  🧩 Export slope units as polygons (vector)",
            value=True, key='slope_unit_save_shp', disabled=DIS,
            help="Also write each slope unit as a polygon (attributes: "
                 "unit_id, n_cells, area_m2) to "
                 "`lib/no_grow/<DEM>_no_grow_slopeunits_grass_units.<fmt>`. "
                 "Downloadable from the Maps tab after the run.")
        slope_unit_save_shp = 1 if slope_unit_save_shp_bool else 0
        slope_unit_shp_format = st.sidebar.selectbox(
            "  Vector format", ['.shp', '.geojson', '.gpkg'], index=0,
            key='slope_unit_shp_format', disabled=DIS or not slope_unit_save_shp,
            help="Shapefile (zipped on download) keeps EPSG:6675 natively. "
                 "GeoJSON is single-file (use for web). GeoPackage is a single "
                 "modern container.")
        # Coastline clip: stop polygons at the shoreline using a land/sea vector
        _clip_files = []
        for _cext in ('*.shp', '*.geojson', '*.gpkg'):
            _clip_files += sorted(COASTLINE_DIR.glob(_cext))
        _clip_names = ['(none)'] + [f.name for f in _clip_files]
        _clip_sel = st.sidebar.selectbox(
            "  ✂ Clip to coastline (land/sea vector)", _clip_names, index=0,
            key='slope_unit_shp_clip', disabled=DIS or not slope_unit_save_shp,
            help="Optional. Place a land (or sea) polygon vector in "
                 f"`{COASTLINE_DIR.relative_to(REPO).as_posix()}/` "
                 "(.shp/.geojson/.gpkg). Slope-unit polygons are clipped to it "
                 "so they stop at the shoreline. e.g. 国土数値情報 海岸線/行政区域.")
        if _clip_sel == '(none)':
            slope_unit_shp_clip = ''
        else:
            slope_unit_shp_clip = str(COASTLINE_DIR / _clip_sel)
        slope_unit_shp_clip_invert = 1 if st.sidebar.checkbox(
            "  ↪ the clip vector is SEA (subtract it)", value=False,
            key='slope_unit_shp_clip_invert',
            disabled=DIS or not slope_unit_save_shp or _clip_sel == '(none)',
            help="OFF: the vector is LAND → keep the part inside it. "
                 "ON: the vector is SEA → remove that part.") else 0
else:
    ridge_acc_thresh = 5.0
    valley_acc_thresh = 100.0
    slope_unit_thresh = 500000.0
    slope_unit_areamin = 100000.0
    slope_unit_cvmin = 0.3
    slope_unit_rf = 2.0
    slope_unit_maxiter = 50
    slope_unit_save_shp = 1
    slope_unit_shp_format = '.shp'
    slope_unit_shp_clip = ''
    slope_unit_shp_clip_invert = 0
    slope_unit_optimize = 0
    slope_unit_cvmin_min, slope_unit_cvmin_max = 0.05, 0.25
    slope_unit_areamin_min, slope_unit_areamin_max = 50000.0, 200000.0

# ---- 5. Seismic -------------------------------------------------------------
seismic_mode = st.sidebar.radio(
    "🌐 Seismic (pseudostatic)", ["off", "uniform", "raster"], index=0,
    key='seismic_mode_radio', disabled=DIS)
if seismic_mode == 'uniform':
    uniform_PGA = st.sidebar.slider(
        "  PGA (g)", 0.0, 1.0, 0.3, 0.05, key='uniform_PGA', disabled=DIS)
    pseudo_scaling = st.sidebar.slider(
        "  PGA scaling factor", 0.0, 2.0, 1.0, 0.1, key='pseudo_scaling',
        disabled=DIS)
    PGA_path = ''
elif seismic_mode == 'raster':
    PGA_path = st.sidebar.text_input(
        "  PGA TIFF path (relative or absolute)",
        value=str(SEISMIC_DIR.relative_to(REPO).as_posix()) + "/PGA.tif",
        key='PGA_path',
        help=f"Recommended location: {SEISMIC_DIR.relative_to(REPO).as_posix()}/",
        disabled=DIS)
    pseudo_scaling = st.sidebar.slider(
        "  PGA scaling factor", 0.0, 2.0, 1.0, 0.1, key='pseudo_scaling',
        disabled=DIS)
    uniform_PGA = 0.3
else:
    uniform_PGA = 0.3
    pseudo_scaling = 1.0
    PGA_path = ''

# ---- 6. Roots ---------------------------------------------------------------
S_roots = st.sidebar.number_input(
    "🌿 Root strength S_roots (kPa)", 0.0, 100.0, 10.0, 1.0, key='S_roots',
    disabled=DIS)

# (Source selection is now under each mode toggle above.)

# ---- Basic properties (always shown but collapsed) -------------------------
with st.sidebar.expander("📐 Basic properties (specific gravity, unit weights)",
                           expanded=False):
    Gs = st.number_input("Gs", 2.0, 3.5, 2.65, 0.05, key='Gs', disabled=DIS)
    gam_w = st.number_input("γ_w water unit weight (kN/m³)", 9.0, 10.5, 9.8, 0.1,
                             key='gam_w', disabled=DIS)
    gam_dry = st.number_input("γ_dry dry unit weight (kN/m³)", 12.0, 22.0, 16.0,
                               0.5, key='gam_dry', disabled=DIS)
    gam_sat = st.number_input("γ_sat saturated unit weight (kN/m³)", 15.0, 24.0,
                               20.0, 0.5, key='gam_sat', disabled=DIS)

# ---- Output destination -----------------------------------------------------
st.sidebar.subheader("📤 Output destination")
default_out_root = str(REPO / 'python' / 'output_webui')
out_root = st.sidebar.text_input(
    "Parent directory", value=default_out_root, key='out_root',
    help="Parent folder where rasters and statistics are written.",
    disabled=DIS)
test_no = st.sidebar.number_input(
    "test_no (driver-internal integer ID)", 1, 99999, 999, 1, key='test_no',
    disabled=DIS)
custom_run_name = st.sidebar.text_input(
    "Run ID / sub-folder name (empty = test_no zero-padded to 5)",
    value="", key='custom_run_name',
    help="Use a free-form name (e.g. 'my_scenario') or leave empty to "
         "use test_no zero-padded to five digits.",
    disabled=DIS)
susname = custom_run_name.strip() if custom_run_name.strip() \
    else f"{int(test_no):05d}"
out_path_preview = Path(out_root) / susname
st.sidebar.caption(f"Output: `{Path(out_root).name}/{susname}/`")

# ---- Existing-folder check & overwrite confirmation ------------------------
folder_exists = (out_path_preview.exists() and out_path_preview.is_dir()
                 and any(out_path_preview.iterdir()))
if folder_exists:
    n_files = len([p for p in out_path_preview.iterdir() if p.is_file()])
    st.sidebar.warning(
        f"⚠️ Folder already exists: `{susname}/` ({n_files} files)\n\n"
        "**Starting will overwrite existing files.**",
        icon="⚠️")
    overwrite_ok = st.sidebar.checkbox(
        f"Allow overwrite of `{susname}/`", value=False,
        key='overwrite_ok', disabled=DIS)
else:
    overwrite_ok = True

st.sidebar.subheader("▶ Run")
_block_start = DIS or (folder_exists and not overwrite_ok)
if folder_exists and not overwrite_ok and not DIS:
    _btn_label = "Start (overwrite must be allowed)"
elif IS_RUNNING:
    _btn_label = "Running..."
else:
    _btn_label = "Start analysis"
start = st.sidebar.button(
    _btn_label, type="primary", use_container_width=True,
    disabled=_block_start, key='start_button')


# =============================================================================
#  Main panel
# =============================================================================

st.title("RegionGrow3D — Web UI")

if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None
if 'log_lines' not in st.session_state:
    st.session_state.log_lines = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False


def _build_cmd():
    cmd = [PYTHON_EXE, '-u', str(DRIVER),
           '--DEM_path', str(REPO / 'lib' / 'DEM' / dem_name),
           '--test_no', str(test_no),
           '--susname_override', susname,
           '--soil_moisture_mode', str(soil_moisture_mode),
           '--soil_depth_mode', str(soil_depth_mode),
           '--soil_strength_mode', str(soil_strength_mode),
           '--nogrow_mode', str(nogrow_mode),
           '--seismic_mode', seismic_mode,
           '--soil_depth_source', soil_depth_source,
           '--nogrow_source', nogrow_source,
           '--mw', str(mw),
           '--Gs', str(Gs), '--gam_w', str(gam_w),
           '--gam_dry', str(gam_dry), '--gam_sat', str(gam_sat),
           '--S_roots', str(S_roots),
           '--soil_depth_uniform', str(soil_depth_uniform),
           '--soil_depth_endtime', str(soil_depth_endtime),
           '--phi_uniform', str(phi_uniform), '--coh_uniform', str(coh_uniform),
           '--uniform_PGA', str(uniform_PGA),
           '--pseudo_scaling', str(pseudo_scaling),
           '--ridge_acc_thresh', str(ridge_acc_thresh),
           '--valley_acc_thresh', str(valley_acc_thresh),
           '--slope_unit_thresh', str(slope_unit_thresh),
           '--slope_unit_areamin', str(slope_unit_areamin),
           '--slope_unit_cvmin', str(slope_unit_cvmin),
           '--slope_unit_rf', str(slope_unit_rf),
           '--slope_unit_maxiter', str(slope_unit_maxiter),
           '--slope_unit_save_shp', str(slope_unit_save_shp),
           '--slope_unit_shp_format', str(slope_unit_shp_format),
           '--slope_unit_shp_clip_invert', str(slope_unit_shp_clip_invert),
           '--slope_unit_optimize', str(slope_unit_optimize),
           '--slope_unit_cvmin_min', str(slope_unit_cvmin_min),
           '--slope_unit_cvmin_max', str(slope_unit_cvmin_max),
           '--slope_unit_areamin_min', str(slope_unit_areamin_min),
           '--slope_unit_areamin_max', str(slope_unit_areamin_max),
           '--save_intermediates', '1',
           '--out_dir', out_root,
           ]
    if soil_depth_mat:
        cmd += ['--soil_depth_mat', soil_depth_mat]
    if no_grow_mat:
        cmd += ['--no_grow_mat', no_grow_mat]
    if soil_strength_mode == 1:
        cmd += ['--shear_strength_mat',
                str(SOIL_STRENGTH_DIR / 'shear_strength.mat')]
    if PGA_path:
        cmd += ['--PGA_path', PGA_path]
    if slope_unit_shp_clip:
        cmd += ['--slope_unit_shp_clip', slope_unit_shp_clip]
    if run_only != ALL_RUNS:
        cmd += ['--run-index', str(run_only)]
    return cmd


def _parse_progress(line: str, state: dict):
    """Update a progress dict from a driver log line."""
    # The parallel orchestrator prefixes each line with a "YYYY-MM-DD HH:MM:SS "
    # timestamp; the plain driver does not. Strip a leading timestamp so the
    # anchored re.match patterns below fire for both (otherwise parallel runs
    # never parse "PARALLEL progress"/"Total elapsed" and look stuck at 0 %).
    line = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+", "", line)
    m = re.match(r"Found (\d+) candidate clusters", line)
    if m:
        state['n_clusters_total'] = int(m.group(1))
        return
    m = re.match(r"\[run (\d+)/(\d+)\]", line)
    if m:
        state['run_idx'] = int(m.group(1))
        state['n_runs'] = int(m.group(2))
        state['cluster_done'] = 0
        return
    m = re.match(r"LS Cluster (\d+)/(\d+)", line)
    if m:
        state['cluster_done'] = int(m.group(1))
        state['n_clusters_total'] = int(m.group(2))
        return
    m = re.match(r"PARALLEL progress: ([\d.]+)/(\d+)", line)
    if m:
        state['parallel'] = True
        state['parallel_frac'] = float(m.group(1))
        state['parallel_total'] = int(m.group(2))
        return
    m = re.match(r"Total elapsed: ([\d.]+) min", line)
    if m:
        state['elapsed_min'] = float(m.group(1))
        state['done'] = True


def _progress_fraction(state: dict) -> float:
    if state.get('done'):
        return 1.0
    if state.get('parallel'):
        pt = max(1, state.get('parallel_total', 1))
        return min(0.99, state.get('parallel_frac', 0.0) / pt)
    n_runs = state.get('n_runs', 1)
    run_idx = state.get('run_idx', 1)
    n_clusters = state.get('n_clusters_total', 0)
    cluster_done = state.get('cluster_done', 0)
    cluster_frac = (cluster_done / n_clusters) if n_clusters else 0.0
    return min(0.99, ((run_idx - 1) + cluster_frac) / max(1, n_runs))


# ---- Per-session helper state (just for ephemeral UI affordances) ----------
for k, default in [
        ('output_dir', None),
        ('last_status', None)]:
    if k not in st.session_state:
        st.session_state[k] = default


def _parse_progress_lines(lines):
    """Build a fresh progress dict from a list of log lines (cheap)."""
    state = {'n_runs': 1, 'run_idx': 1, 'n_clusters_total': 0,
             'cluster_done': 0, 'done': False}
    for line in lines:
        _parse_progress(line, state)
    return state


def _compute_already_running():
    """Return the PID of an already-running driver/orchestrator process, or None.

    Backstop against the "stale manifest" cascade: if a previous run's manifest
    was cleared (e.g. the orchestrator was OOM-killed) but its child driver runs
    are still alive, launching again would pile up duplicate ~20 GB runs and
    exhaust memory. We refuse to start while any compute process is live.
    """
    try:
        import psutil
    except ImportError:
        return None
    me = os.getpid()
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if p.info['pid'] == me:
                continue
            cl = ' '.join(p.info.get('cmdline') or [])
            if 'driver.py' in cl or '_sus_parallel.py' in cl:
                return p.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _validate_inputs():
    """Return a list of human-readable problems that would make the run fail
    immediately. Checked BEFORE launching so we never spawn a doomed detached
    subprocess (which would truncate the log and write a manifest for a run
    that dies in <1 s)."""
    problems = []
    if not dem_name:
        problems.append("No DEM selected — pick one in lib/DEM/ or upload a .tif.")
    if soil_depth_mode == 1 and soil_depth_source == 'mat' and not soil_depth_mat:
        problems.append("Soil-depth source is 'mat' but no soil-depth .mat is "
                        "available — add one to lib/soil_depth/ or switch to "
                        "'compute'/uniform.")
    if nogrow_mode == 1 and nogrow_source == 'mat' and not no_grow_mat:
        problems.append("No-grow source is 'mat' but no no-grow .mat is "
                        "available — add one to lib/no_grow/ or switch to "
                        "'compute'/'grass'.")
    if soil_strength_mode == 1 and not (SOIL_STRENGTH_DIR / 'shear_strength.mat').exists():
        problems.append("soil_strength_mode=1 needs "
                        f"{(SOIL_STRENGTH_DIR / 'shear_strength.mat').as_posix()} "
                        "— add it or use uniform (mode 2).")
    return problems


# ---- Start: spawn a detached subprocess and persist a manifest -------------
if start and not IS_RUNNING:
    _problems = _validate_inputs()
    if _problems:
        st.error("Cannot start — fix these first:\n\n"
                 + "\n".join(f"- {p}" for p in _problems))
        st.stop()
    _busy_pid = _compute_already_running()
    if _busy_pid:
        st.error(
            f"⚠️ A computation is already running (PID {_busy_pid}). "
            "It may be a previous run whose status card disappeared — it is "
            "still working in the background. **Do not start another** (that "
            "piles up duplicate ~20 GB runs and exhausts memory). Wait for it "
            "to finish, or use the Stop control if shown. Reload the page in a "
            "minute to reconnect to it.")
        st.stop()
    cmd = _build_cmd()
    out_dir = Path(out_root) / susname
    out_dir.mkdir(parents=True, exist_ok=True)
    if parallel_runs:
        # Fan out via the parallel orchestrator instead of one serial driver.
        # It runs the same driver command 2-at-a-time (--run-index) and
        # aggregates; its stdout (progress) is captured into _run.log below,
        # and killing its PID tears down the child runs (kill_pid kills the
        # process tree), so the Stop button still works.
        spec_path = out_dir / '_parallel_spec.json'
        spec_path.write_text(json.dumps({
            'driver_cmd': cmd,
            'out_root': str(out_root),
            'susname': susname,
            'n_runs': 10,
            'maxconc': int(os.environ.get('SUS_MAXCONC', '2')),
            # Stagger launches so two runs never hit their ~24 GB pre-compute
            # peak simultaneously (that OOM-killed the pool). See _sus_parallel.py.
            'stagger': int(os.environ.get('SUS_STAGGER', '240')),
        }), encoding='utf-8')
        cmd = [PYTHON_EXE, '-u', str(SUS_PARALLEL), '--gui-args', str(spec_path)]
    log_path = out_dir / '_run.log'
    # Truncate any previous log for this susname.
    log_path.write_text('', encoding='utf-8')
    # Open in append mode so the subprocess streams continuously; line buffered
    # so the Streamlit UI can tail it in (near) real time.
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)

    # Fully detach on Windows so the subprocess survives Streamlit being
    # restarted, the browser being closed, etc. Without this flag,
    # subprocess.Popen() inherits Streamlit's process group and Windows
    # terminates the child when the parent exits, which is what stranded the
    # overnight runs.
    popen_kwargs = dict(stdout=log_file, stderr=subprocess.STDOUT,
                         text=True, cwd=str(REPO), close_fds=True)
    if os.name == 'nt':
        # CREATE_NEW_PROCESS_GROUP makes the child its own process group;
        # DETACHED_PROCESS hides the console window. Together they make the
        # subprocess truly independent of the parent.
        popen_kwargs['creationflags'] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS)
    else:
        # POSIX: create a new session so signal propagation from Streamlit
        # doesn't reach the child.
        popen_kwargs['start_new_session'] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    write_manifest(MANIFEST_ROOT, {
        'pid': proc.pid,
        'create_time': proc_create_time(proc.pid),  # pins the PID to THIS process
        'susname': susname,
        'out_root': str(out_root),
        'out_dir': str(out_dir),
        'log_path': str(log_path),
        'start_time': time.time(),
        'cmd': cmd,
    })
    st.session_state.output_dir = None
    st.session_state.last_status = None
    st.rerun()


# ---- Running display (driven entirely by the on-disk manifest) -------------
if _manifest is not None:
    log_path = Path(_manifest['log_path'])
    susname_active = _manifest.get('susname', '')
    out_dir_active = Path(_manifest.get('out_dir', ''))
    start_time = float(_manifest.get('start_time', time.time()))
    elapsed = max(0.0, time.time() - start_time)
    # Check liveness FIRST, then read the log: if we tailed the log first and
    # the process finished in the gap, we'd miss the final "Total elapsed" line
    # and misreport a successful run as a crash.
    is_alive = pid_alive(_manifest.get('pid'), _manifest.get('create_time'))
    recent = tail_log(log_path, max_lines=25)
    state = _parse_progress_lines(recent)

    if is_alive:
        # Big warning banner
        st.error(
            "### ⚠️ Computing — sidebar is locked\n\n"
            "Click **[⏹ Stop]** below to abort. The analysis subprocess "
            "keeps running even if you close the browser; reopening this "
            "page later will reconnect to it.",
            icon="⚠️")

        # Header / control row
        header_cols = st.columns([4, 1])
        with header_cols[0]:
            st.info(f"⏳ Running ({elapsed:.0f} s elapsed) — "
                    f"PID {_manifest['pid']}, output `{susname_active}/`")
        with header_cols[1]:
            if st.button("⏹ Stop", type="secondary",
                          use_container_width=True, key='stop_btn'):
                kill_pid(_manifest.get('pid'), create_time=_manifest.get('create_time'))
                clear_manifest(MANIFEST_ROOT)
                st.session_state.last_status = ('error',
                                                  "⏹ Stopped by user")
                st.rerun()

        st.progress(_progress_fraction(state))
        # Use fixed placeholders so the element tree never changes shape between
        # reruns. Conditionally adding/removing widgets (e.g. the pre-compute
        # note when it appears/disappears) shifts everything below it and leaves
        # Streamlit rendering an orphaned, stale copy of the log panel.
        _info_ph = st.empty()
        _caption_ph = st.empty()
        _log_ph = st.empty()

        if state.get('parallel') and state.get('parallel_frac', 0.0) < 0.05:
            _info_ph.info(
                "⏳ **Pre-compute phase** (loading DEM, derivatives, interslice "
                "forces). The bar stays at 0 % until clusters start — this takes "
                "a few minutes and IS running (see the log below). "
                "**Don't press Start again** — the parallel runs advance on their "
                "own; re-starting piles up duplicate runs.")
        else:
            _info_ph.empty()

        parts = []
        if state.get('parallel'):
            parts.append(f"⚡ Parallel {state.get('parallel_frac', 0.0):.1f}/"
                         f"{state.get('parallel_total', 10)} runs")
        elif state.get('run_idx'):
            # Only meaningful for a serial run; hidden in parallel mode (where
            # the per-phi index lives inside each child log, not this one).
            parts.append(f"Run {state['run_idx']}/{state.get('n_runs', 1)}")
        if state.get('n_clusters_total'):
            parts.append(
                f"Cluster {state.get('cluster_done', 0)}/"
                f"{state['n_clusters_total']}")
        _caption_ph.caption(" | ".join(parts) if parts else " ")

        # Hide the machine-readable "PARALLEL progress:" lines (they drive the
        # bar above; showing them too just clutters the human log).
        _disp = [l for l in recent if 'PARALLEL progress:' not in l]
        _log_ph.code('\n'.join(_disp) if _disp else '(waiting for output...)',
                     language='text')

        # Auto-refresh while running so the log/progress update without
        # user interaction.
        time.sleep(1.0)
        st.rerun()

    else:
        # PID is gone — the run finished (or crashed) since the last check.
        # Determine status from the last log lines, persist as the last
        # completed run, clear the manifest, and rerun so the idle UI shows.
        rc_line = next((l for l in recent if l.startswith('Total elapsed:')),
                       None)
        if rc_line:
            kind = 'success'
            msg = f"✅ Done ({state.get('elapsed_min', elapsed/60):.2f} min) " \
                  f"→ {out_dir_active}"
            write_last_completed(MANIFEST_ROOT, {
                'susname': susname_active,
                'out_dir': str(out_dir_active),
                'elapsed_min': state.get('elapsed_min', elapsed / 60),
                'finished_at': time.time(),
            })
        else:
            kind = 'error'
            msg = (f"❌ Run ended unexpectedly (no 'Total elapsed' line). "
                   f"See `{log_path.relative_to(REPO).as_posix()}`.")
        st.session_state.last_status = (kind, msg)
        st.session_state.output_dir = out_dir_active
        clear_manifest(MANIFEST_ROOT)
        st.rerun()


# ---- Show last run status (preserved across reruns) ------------------------
if st.session_state.last_status is not None:
    kind, msg = st.session_state.last_status
    if kind == 'success':
        st.success(msg)
    else:
        st.error(msg)


# ---- Result tabs ------------------------------------------------------------
# If session lost its output_dir (e.g. new browser session), fall back to the
# on-disk pointer to the most recently completed run.
out_dir = st.session_state.output_dir
if out_dir is None:
    lc = read_last_completed(MANIFEST_ROOT)
    if lc:
        out_dir = Path(lc['out_dir'])
        st.session_state.output_dir = out_dir
if out_dir and Path(out_dir).exists():
    susname = out_dir.name
    sus_path = out_dir / f"sus_{susname}_python.tif"
    nogrow_path = out_dir / 'nogrow_io.tif'
    depth_path = out_dir / 'depth.tif'
    pga_path = out_dir / 'PGA.tif'
    hs_path = out_dir / 'hillshade.tif'
    summary_path = out_dir / 'run_summary.json'

    tab_map, tab_stats, tab_hist = st.tabs(["🗺 Maps", "📊 Statistics",
                                              "📈 Histogram"])

    # ---- Lazy-import display deps -----------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import rasterio

    def _read_tif(p: Path) -> np.ndarray:
        with rasterio.open(p) as src:
            return src.read(1)

    with tab_map:
        layer = st.radio("Layer",
                          ["Susceptibility (main output)", "No-grow mask",
                           "Soil depth", "PGA map"], horizontal=True)
        overlay_hs = st.checkbox("Hillshade overlay", value=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        if overlay_hs and hs_path.exists():
            hs = _read_tif(hs_path)
            ax.imshow(hs, cmap='gray', alpha=0.7)

        if layer.startswith("Susceptibility") and sus_path.exists():
            data = _read_tif(sus_path)
            im = ax.imshow(data, cmap='inferno', vmin=0, vmax=100,
                           alpha=0.6 if overlay_hs else 1.0)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04)
            cbar.set_label("Susceptibility (%)")
        elif layer == "No-grow mask" and nogrow_path.exists():
            data = _read_tif(nogrow_path)
            im = ax.imshow(data, cmap='Reds',
                           alpha=0.5 if overlay_hs else 1.0)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04)
            cbar.set_label("No-grow (1=boundary)")
        elif layer == "Soil depth" and depth_path.exists():
            data = _read_tif(depth_path)
            # Soil depth spans several orders of magnitude (most cells ~1 m,
            # a few outliers reach 20-30 m), so display on a log colour
            # scale. Cells <= 0 are masked out so LogNorm stays well-defined.
            disp = np.where(np.isfinite(data) & (data > 0), data, np.nan)
            valid = disp[np.isfinite(disp)]
            if valid.size:
                vmin = max(float(valid.min()), 0.01)
                vmax = float(valid.max())
                if vmax <= vmin:
                    vmax = vmin * 10.0
            else:
                vmin, vmax = 0.01, 10.0
            im = ax.imshow(disp, cmap='viridis',
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           alpha=0.6 if overlay_hs else 1.0)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04)
            cbar.set_label(
                f"Soil depth (m, log scale) — {vmin:.2f}–{vmax:.2f} m")
        elif layer == "PGA map" and pga_path.exists():
            data = _read_tif(pga_path)
            im = ax.imshow(data, cmap='magma',
                           alpha=0.6 if overlay_hs else 1.0)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04)
            cbar.set_label("PGA (g)")
        else:
            ax.text(0.5, 0.5, "(file not available)", ha='center', va='center',
                    transform=ax.transAxes)
        ax.axis('off')
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        # downloads
        cols = st.columns(4)
        for col, p, label in zip(
            cols, [sus_path, nogrow_path, depth_path, pga_path],
            ["sus.tif", "nogrow.tif", "depth.tif", "PGA.tif"]):
            if p.exists():
                with col:
                    st.download_button(
                        label, data=p.read_bytes(), file_name=p.name,
                        mime='image/tiff', use_container_width=True)

        # ---- Slope-unit polygons (vector) download ---------------------------
        # The slope-unit vector is written to lib/no_grow/ next to the .mat,
        # named "<DEM stem>_no_grow_slopeunits*_units.<ext>". Shapefiles are a
        # multi-file set, so they are zipped on the fly before download.
        import io
        import zipfile
        _SHP_SIDECARS = ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj')
        dem_stem = Path(dem_name).stem if dem_name else ''
        vec_files = []
        if dem_stem:
            for vext in ('.shp', '.geojson', '.gpkg'):
                vec_files += sorted(NO_GROW_DIR.glob(
                    f"{dem_stem}_no_grow_slopeunits*_units{vext}"))
        if vec_files:
            st.markdown("**🧩 Slope-unit polygons**")
            vcols = st.columns(min(len(vec_files), 3) or 1)
            for col, vp in zip(vcols, vec_files):
                with col:
                    if vp.suffix.lower() == '.shp':
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, 'w',
                                             zipfile.ZIP_DEFLATED) as zf:
                            for side in vp.parent.glob(vp.stem + '.*'):
                                if side.suffix.lower() in _SHP_SIDECARS:
                                    zf.write(side, arcname=side.name)
                        st.download_button(
                            f"⬇ {vp.stem}.zip (SHP)", data=buf.getvalue(),
                            file_name=vp.stem + '.zip',
                            mime='application/zip',
                            use_container_width=True, key=f"dl_{vp.name}")
                    else:
                        mime = ('application/geo+json'
                                if vp.suffix.lower() == '.geojson'
                                else 'application/octet-stream')
                        st.download_button(
                            f"⬇ {vp.name}", data=vp.read_bytes(),
                            file_name=vp.name, mime=mime,
                            use_container_width=True, key=f"dl_{vp.name}")

    with tab_stats:
        if sus_path.exists():
            sus = _read_tif(sus_path)
            valid = ~np.isnan(sus)
            n_total = int(valid.sum())
            n_pos = int(((sus > 0) & valid).sum())
            n_50 = int(((sus >= 50) & valid).sum())
            n_90 = int(((sus >= 90) & valid).sum())
            cs = 10.0  # cellsize hint
            if n_total == 0:
                st.warning("Susceptibility raster has no valid (non-NaN) cells "
                           "— nothing to summarise.")
            else:
                mean_sus = float(np.nanmean(sus))
                def _pct(n):
                    return f"{100*n/n_total:.2f}%"
                st.metric("Valid cells", f"{n_total:,}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("susceptibility > 0%", f"{n_pos:,}", _pct(n_pos))
                c2.metric("susceptibility ≥ 50%", f"{n_50:,}", _pct(n_50))
                c3.metric("susceptibility ≥ 90%", f"{n_90:,}", _pct(n_90))
                c4.metric("Mean susceptibility", f"{mean_sus:.2f}%")
            st.caption(
                f"Estimated area: > 50% = {n_50 * cs * cs / 1e6:.3f} km², "
                f"> 0% = {n_pos * cs * cs / 1e6:.3f} km² (assuming cs=10m)")

        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as fh:
                summary = json.load(fh)
            st.subheader("Per-run cluster counts ((φ, c, prob) → cluster count)")
            import pandas as pd
            df = pd.DataFrame(summary)
            st.dataframe(df, hide_index=True, use_container_width=True)
            if not df.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.bar(df['run'].astype(str), df['n_clusters'],
                        color='steelblue')
                ax2.set_xlabel("run")
                ax2.set_ylabel("# clusters")
                ax2.set_title("Per-run cluster counts")
                for i, row in df.iterrows():
                    ax2.annotate(f"φ={row['phi']:.1f}\nc={row['coh']:.1f}",
                                 (str(row['run']), row['n_clusters']),
                                 ha='center', va='bottom', fontsize=7)
                st.pyplot(fig2, clear_figure=True, use_container_width=True)

    with tab_hist:
        if sus_path.exists():
            sus = _read_tif(sus_path)
            valid_pos = sus[(sus > 0) & ~np.isnan(sus)]
            if valid_pos.size == 0:
                st.warning("No cells with susceptibility > 0")
            else:
                fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(valid_pos, bins=50, color='darkorange',
                              edgecolor='k', alpha=0.7)
                axes[0].set_xlabel("Susceptibility (%)")
                axes[0].set_ylabel("Number of cells")
                axes[0].set_title("Positive-cell distribution (>0%)")
                axes[0].set_yscale('log')
                # CDF
                vals_sorted = np.sort(valid_pos)
                cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
                axes[1].plot(vals_sorted, cdf * 100, color='steelblue')
                axes[1].set_xlabel("Susceptibility (%)")
                axes[1].set_ylabel("Cumulative (%)")
                axes[1].set_title("CDF")
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig3, clear_figure=True, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive cells", f"{valid_pos.size:,}")
                c2.metric("Median", f"{np.median(valid_pos):.2f}%")
                c3.metric("90th percentile",
                           f"{np.percentile(valid_pos, 90):.2f}%")
elif not start:
    st.info("← Configure parameters in the left sidebar and click "
            "[Start analysis].")
