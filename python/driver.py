"""Mode-A+ driver: hydrostatic moisture, seismic optional, distribution-based
shear-strength parameters.

Mirrors the `lib/driver.m` parameterisation. Default modes:
- soil_moisture_mode = 1 (hydrostatic)
- mw = 0.5
- seismic_mode = 'off' (override with --seismic_mode uniform/raster)
- soil_strength_mode = 1 (distribution)
- nogrow_mode = 1 (load existing .mat)
- S_roots = 10 kPa
- gam_w = 9.8, gam_dry = 16, gam_sat = 20
- soil_depth_mode = 1 (load existing .mat)

The `sigma_s_wedge` patch (`% damy 20250111`) is included.

Required arguments (no defaults, user must supply):
  --DEM_path  <tif>              input DEM (GeoTIFF)
  --test_no   <int>              numeric ID used for the output sub-folder name
                                  (e.g. 1 → "00001"); override entirely with
                                  --susname_override.

Conditionally-required arguments (validated post-parse):
  --soil_depth_mat    <.mat>     required when --soil_depth_source=mat
                                  (and --soil_depth_mode=1)
  --no_grow_mat       <.mat>     required when --nogrow_mode=1 and
                                  --nogrow_source=mat
  --shear_strength_mat <.mat>    required when --soil_strength_mode=1

Seismic options (driver.m lines 84-92, 295-313):
  --seismic_mode off|uniform|raster   ('off' = PGA = 0 everywhere)
  --uniform_PGA <g>                   (used when seismic_mode='uniform')
  --PGA_path  <tif>                   (used when seismic_mode='raster')
  --pseudo_scaling <s>                (multiplier applied to PGA, default 1.0)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr so non-ASCII characters in log lines (e.g. m^2,
# φ, °) don't crash on Japanese-locale Windows consoles where the default
# is cp932. Streamlit reads this stdout to render the live log, so the
# subprocess and the parent both need UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
except AttributeError:
    pass  # Python <3.7 — extremely unlikely here.

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'python'))

from region3d.io import (read_dem, write_raster, load_soil_depth, load_no_grow,
                         load_shear_strength)
from region3d.derivatives import pad_DEM, gradient_prince
from region3d.matlab_compat import find_F
from region3d.region_grow import region_grow_fxn
from region3d.preprocessing import (soil_depth as compute_soil_depth,
                                    ridges_valleys as compute_ridges_valleys)


# Sample-name-free defaults. DEM_path / test_no are required (see main()).
# *_mat paths are conditionally required at runtime depending on mode.
DEFAULTS = dict(
    soil_depth_mat='',     # required when soil_depth_mode=1 and soil_depth_source='mat'
    no_grow_mat='',        # required when nogrow_mode=1 and nogrow_source='mat'
    shear_strength_mat='', # required when soil_strength_mode=1
    out_dir=REPO_ROOT / 'python' / 'output',
    soil_moisture_mode=1,
    mw=0.5,
    Gs=2.65,
    gam_w=9.8,
    gam_dry=16.0,
    gam_sat=20.0,
    S_roots=10.0,
    err_percent_allowable=1.0,
    max_growth_cycles=120,
    err_increase_thresh=20,
    cluster_size_thresh=7,
    erosion_rounds=2,
    cleanup_rounds_initial=5,
    cleanup_rounds_grow=1,
    rot_range=(-20.0, 20.0),
    rot_num=8,
    nogrow_mode=1,
    sigma_s_wedge_mode='matlab_local',  # 'matlab_local' (=sigma_s) or 'physical' (=sigma_s/2)
    reg_grow_on=1,
    soil_depth_endtime=5000.0,  # years for the Roering hillslope-evolution model
    ridge_acc_thresh=5.0,
    valley_acc_thresh=100.0,
    # ---- Slope-unit option (nogrow_source='grass' → GRASS r.slopeunits) ------
    slope_unit_thresh=500000.0,  # r.slopeunits thresh: channel-defining acc area [m^2]
    slope_unit_areamin=100000.0, # r.slopeunits areamin: minimum unit area [m^2]
    slope_unit_cvmin=0.3,        # aspect circular-variance ceiling
    slope_unit_rf=2.0,           # threshold reduction factor (cast to int for r.slopeunits)
    slope_unit_maxiter=50,       # max refinement iterations (r.slopeunits example uses 50)
    slope_unit_save_shp=1,       # 1 = also export slope units as vector (SHP/GeoJSON/GPKG) to lib/no_grow/
    slope_unit_shp_format='.shp',# .shp | .geojson | .gpkg
    slope_unit_shp_min_cells=1,  # drop units smaller than this many cells from the vector
    slope_unit_shp_wgs84=0,      # 1 = reproject vector to EPSG:4326 (for GeoJSON/web)
    slope_unit_shp_clip='',      # clip slope-unit polygons to this land (or sea) coastline vector
    slope_unit_shp_clip_invert=0,# 1 = treat --slope_unit_shp_clip as SEA and subtract it (else LAND)
    # ---- GRASS r.slopeunits.optimize (morphometric F=V*I, no inventory; slow) -
    slope_unit_optimize=0,       # 1 = auto-optimise cvmin/areamin via r.slopeunits.optimize
    slope_unit_cvmin_min=0.05, slope_unit_cvmin_max=0.25,      # cvmin search range
    slope_unit_areamin_min=50000.0, slope_unit_areamin_max=200000.0,  # areamin search range [m^2]
    slope_unit_opt_epsx=0.01,    # optimize: stop when cvmin range < this
    slope_unit_opt_epsy=50000.0, # optimize: stop when areamin range < this [m^2]
    # ---- Per-component data source (replaces legacy preprocess_source) -------
    soil_depth_source='mat',    # 'mat' (load soil_depth_mat) | 'compute' (Python)
    nogrow_source='mat',        # 'mat' | 'compute' (D8 ridges/valleys) | 'grass' (GRASS r.slopeunits, Docker)
    preprocess_source='',       # legacy: if non-empty, sets both above (back-compat)
    save_intermediates=0,       # 1 = also write depth/nogrow/PGA/hillshade TIFFs to out_dir
    susname_override='',        # if non-empty, use this string as the output subdir name (else "{test_no:05d}")
    # ---- Seismic (driver.m lines 84-92) -----------------------------------------
    seismic_mode='off',          # 'off' | 'uniform' | 'raster'
    uniform_PGA=0.3,             # used when seismic_mode='uniform' (units: g)
    PGA_path='',                 # GeoTIFF path; used when seismic_mode='raster'
    pseudo_scaling=1.0,          # scaling factor applied to PGA
    # ---- Soil depth (driver.m line 103-104) -------------------------------------
    soil_depth_mode=1,           # 1 = Roering hillslope evolution | 2 = uniform
    soil_depth_uniform=2.0,      # used when soil_depth_mode=2 (units: m)
    # ---- Soil strength (driver.m line 107-112) ----------------------------------
    soil_strength_mode=1,        # 1 = distribution (.mat) | 2 = uniform single (phi,c)
    phi_uniform=25.0,            # friction angle for uniform mode (deg)
    coh_uniform=2.0,             # cohesion for uniform mode (kPa)
    # ---- Safety valve (Python-only; MATLAB retries unboundedly) ---------------
    max_cell_offset=400,         # cap on the local-window half-size during boundary
                                 # expansion (cells); capped clusters get
                                 # terminate_reason=7 and diverge from MATLAB
)


def _build_pga(Z, georef, args):
    """Construct the peak-ground-acceleration raster.

    Mirrors the three branches of `lib/driver.m` lines 295-313:
      - 'off':     PGA = 0 everywhere (no seismic body force).
      - 'uniform': PGA = uniform_PGA at every non-NaN cell, then * pseudo_scaling.
      - 'raster':  PGA = geotiffread(PGA_path); negative values -> NaN; cells
                   where Z is NaN -> NaN; then * pseudo_scaling.
    Output is float32 and aligned to Z.
    """
    f32 = np.float32
    mode = (args.seismic_mode or 'off').lower()
    if mode == 'off':
        return np.zeros(Z.shape, dtype=f32)

    if mode == 'uniform':
        PGA = np.zeros(Z.shape, dtype=f32)
        valid = ~np.isnan(Z)
        PGA[valid] = f32(args.uniform_PGA)
        PGA = PGA * f32(args.pseudo_scaling)
        return PGA

    if mode == 'raster':
        if not args.PGA_path:
            raise SystemExit("[missing input] seismic_mode='raster' requires "
                             "--PGA_path (a PGA GeoTIFF on the DEM grid).")
        path = Path(args.PGA_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise SystemExit(f"[missing input] PGA raster not found:\n  {path}")
        Z_pga, _geo_pga = read_dem(path)  # NaN handling matches driver.m: <0 -> NaN
        if Z_pga.shape != Z.shape:
            raise SystemExit(
                f"[shape mismatch] PGA raster has shape {Z_pga.shape} but the "
                f"DEM is {Z.shape}.\n  source: {path}\n"
                f"  -> Resample/clip the PGA raster to the same grid as the DEM.")
        PGA = Z_pga.astype(f32, copy=True)
        PGA[np.isnan(Z)] = np.nan
        PGA = PGA * f32(args.pseudo_scaling)
        return PGA

    raise ValueError(f"unknown seismic_mode: {args.seismic_mode!r}")


def _set_keep_awake(enable: bool) -> None:
    """Tell the OS not to put the machine to sleep while a long run is active.

    On Windows this calls SetThreadExecutionState with ES_CONTINUOUS +
    ES_SYSTEM_REQUIRED + ES_AWAYMODE_REQUIRED so the system stays awake (the
    display is still allowed to turn off) until the flag is cleared.
    No-op on non-Windows platforms — modern Linux desktops typically expose
    org.freedesktop.PowerManagement.Inhibit via dbus, but most server boxes
    don't auto-sleep, so we simply skip.
    """
    if os.name != 'nt':
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        flags = ES_CONTINUOUS
        if enable:
            flags |= ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        # Silently ignore — the user can still complete the run manually.
        pass


def _require_file(path, *, what):
    """Fail with a clear message (no traceback) when an input file is missing."""
    p = Path(path) if path else None
    if not path or not p.exists():
        raise SystemExit(
            f"[missing input] {what} not found:\n"
            f"  {path}\n"
            f"  -> Check the path, or generate this input for the SELECTED "
            f"DEM (it must match the DEM you are running).")
    return p


def _require_shape(arr, ref_shape, *, what, src):
    """Fail with a clear message when a loaded array does not match the DEM.

    The most common cause is loading a soil-depth / no-grow / PGA file that was
    produced for a DIFFERENT DEM (e.g. the old 5 m grid) — the raw NumPy
    broadcast error ("operands could not be broadcast together") is replaced by
    an actionable message that names both shapes and the offending file.
    """
    if tuple(arr.shape) != tuple(ref_shape):
        raise SystemExit(
            f"[shape mismatch] {what} has shape {tuple(arr.shape)} but the "
            f"DEM is {tuple(ref_shape)}.\n"
            f"  source: {src}\n"
            f"  -> This file belongs to a DIFFERENT DEM. For THIS DEM either "
            f"select the matching file, or regenerate it:\n"
            f"       soil depth : soil_depth_source='compute' (or uniform mode)\n"
            f"       no-grow    : nogrow_source='compute' or 'grass'\n"
            f"       PGA raster : provide a PGA GeoTIFF on the same grid as the DEM")


def _write_sus_raster(sus, Z, georef, out_dir, susname):
    """NaN-mask, scale to percent and write the susceptibility GeoTIFF.

    Shared by the normal run path and --aggregate so masking/scaling/naming
    live in one place and the two outputs cannot drift.
    """
    sus[np.isnan(Z)] = np.nan
    out_path = out_dir / f"sus_{susname}_python.tif"
    print(f"\nWriting susceptibility raster: {out_path}")
    write_raster(out_path, sus * 100.0, georef)
    return out_path


def _tension_compression(args, Z, georef, subdx, subdy, W, sigma_s, PGA,
                         prob, prob_phi, prob_coh, susname, out_dir):
    """Probability-weighted per-cell net driving force q (tension/compression).

    Interslice_Force gives, for EVERY cell of a slide (not just its margin), a
    net force  q = driving - resisting  (n·sinα + PGA·W minus cohesion+friction;
    Interslice_Force.m). q > 0 => net driving = active/**tension** state (typically
    the head); q < 0 => net resisting = passive/**compression** (the toe). This is
    the interior force balance the boundary earth-pressure wedges sit on top of.

    For each soil-strength run i we evaluate q on that run's final slide cells
    (read from contribs/), weight by prob[i], and accumulate — mirroring how the
    susceptibility map weights slide occurrence. subdx/subdy/W/sigma_s/PGA are
    phi-independent and already computed, so this needs no region-grow.

    Writes ONE signed GeoTIFF with the convention **compression positive,
    tension negative** (stores Σ prob·(−q)), so a diverging colour ramp shows
    compression red / tension blue:
      net_force_prob_<s>.tif   Σ prob·(−q)   ( >0 compression … <0 tension )
    """
    from region3d.forces import interslice_force
    cdir = out_dir / 'contribs'
    if not sorted(cdir.glob('contrib_run*.npz')):
        sys.exit(f"[tension_compression] no contributions in {cdir} — run the "
                 "susceptibility analysis first (--run-index runs or the UI "
                 "parallel mode), then re-run with --tension_compression 1.")
    shape = Z.shape
    m = shape[0]
    field = np.zeros(shape, dtype=np.float64)   # + compression … − tension
    n_used = 0
    for i in range(prob.size):
        cf = cdir / f"contrib_run{i:02d}.npz"
        if not cf.exists():
            print(f"  run {i+1}/{prob.size}: contrib missing, skipping")
            continue
        with np.load(cf) as d:
            idx0 = d['idx'].astype(np.int64)          # C-order flat slide indices
            pr = float(d['prob'])
            no_slides = bool(d['no_slides']) if 'no_slides' in d.files else False
        if pr == 0.0 or idx0.size == 0:
            if no_slides:
                print(f"  run {i+1}: no slides — stopping (matches SUS break)")
                break
            continue
        ii, jj = np.unravel_index(idx0, shape)                    # C-order row,col
        idx_1based = (jj.astype(np.int64) * m + ii + 1)           # F-order 1-based
        coh_full = np.full(shape, np.float32(prob_coh[i]), dtype=np.float32)
        phi_full = np.full(shape, np.float32(prob_phi[i]), dtype=np.float32)
        Q, *_ = interslice_force(subdx, subdy, georef.x_cellsize,
                                 georef.y_cellsize, coh_full, phi_full, W,
                                 sigma_s, idx_1based, PGA)
        q = Q[ii, jj].astype(np.float64)
        # Store −q so compression (q<0) is POSITIVE and tension (q>0) NEGATIVE.
        # A few degenerate near-vertical subsurface cells yield NaN q (cos→0 in
        # the column-area term); zero their contribution so a single NaN can't
        # poison a cell that other runs cover.
        n_nan = int(np.isnan(q).sum())
        neg_q = np.where(np.isnan(q), 0.0, -q)
        field[ii, jj] += pr * neg_q
        n_used += 1
        qlo = float(np.nanmin(q)) if n_nan < q.size else float('nan')
        qhi = float(np.nanmax(q)) if n_nan < q.size else float('nan')
        print(f"  run {i+1}/{prob.size}: phi={prob_phi[i]:.2f} "
              f"cells={idx0.size} prob={pr:.4f} "
              f"q=[{qlo:.1f},{qhi:.1f}] nan={n_nan}", flush=True)
        if no_slides:
            print(f"  run {i+1}: no-slides flag — stopping (matches SUS break)")
            break
    field[np.isnan(Z)] = np.nan
    p = out_dir / f"net_force_prob_{susname}.tif"
    write_raster(p, field.astype(np.float32), georef)
    print(f"[tension_compression] combined {n_used} run(s) -> "
          f"{p.relative_to(REPO_ROOT).as_posix()}  (+compression / −tension)")


def run(args):
    _set_keep_awake(True)
    try:
        _run_impl(args)
    finally:
        _set_keep_awake(False)


def _run_impl(args):
    t0 = time.time()
    # Back-compat: legacy --preprocess_source overrides per-component flags.
    if args.preprocess_source:
        args.soil_depth_source = args.preprocess_source
        args.nogrow_source = args.preprocess_source
    print(f"Loading DEM: {args.DEM_path}")
    Z, georef = read_dem(args.DEM_path)
    Z = pad_DEM(Z)
    m, n = Z.shape
    print(f"  shape={Z.shape}, cellsize=({georef.x_cellsize}, {georef.y_cellsize})")

    # Output naming — shared by the normal path, --run-index and --aggregate
    # (deriving it once keeps the three modes pointing at the same directory).
    if args.susname_override:
        susname = str(args.susname_override).strip()
    else:
        susname = f"{int(args.test_no):05d}"
    out_dir = Path(args.out_dir) / susname

    # ---- Aggregate mode: combine saved per-run contributions, then exit ----
    # (low-memory resume: each run is computed in its own process via
    #  --run-index and saved to contribs/; here we sum them weighted by prob.)
    if int(args.aggregate):
        cdir = out_dir / 'contribs'
        files = sorted(cdir.glob('contrib_run*.npz'))
        if not files:
            sys.exit(f"[aggregate] no contribution files found in {cdir}")
        contribs = []
        for f in files:
            with np.load(f) as d:
                if d['shape'].tolist() != list(Z.shape):
                    sys.exit(f"[aggregate] shape mismatch in {f.name}: "
                             f"{d['shape'].tolist()} vs {list(Z.shape)} — "
                             f"contribution from a different DEM?")
                contribs.append({
                    'run': int(d['run']),
                    'prob': float(d['prob']),
                    'idx': d['idx'],
                    'n_runs': int(d['n_runs']) if 'n_runs' in d.files else None,
                    'no_slides': (bool(d['no_slides'])
                                  if 'no_slides' in d.files else False),
                })
        contribs.sort(key=lambda c: c['run'])
        # -- Completeness / staleness checks: a partial or mixed contribution
        #    set would silently produce a wrong map under the canonical name.
        runs = [c['run'] for c in contribs]
        if len(set(runs)) != len(runs):
            sys.exit(f"[aggregate] duplicate run indices in {cdir}: {runs}")
        n_runs_set = {c['n_runs'] for c in contribs if c['n_runs'] is not None}
        if len(n_runs_set) > 1:
            sys.exit(f"[aggregate] contributions disagree on the distribution "
                     f"size ({sorted(n_runs_set)}) — stale files from a "
                     f"different shear-strength distribution in {cdir}?")
        expected = n_runs_set.pop() if n_runs_set else max(runs) + 1
        missing = sorted(set(range(expected)) - set(runs))
        extra = sorted(set(runs) - set(range(expected)))
        if missing or extra:
            sys.exit(f"[aggregate] contribution set does not cover runs "
                     f"0..{expected - 1}: missing={missing} extra={extra}\n"
                     f"  -> compute the missing --run-index runs (or remove "
                     f"stale files) before aggregating")
        total_prob = sum(c['prob'] for c in contribs)
        if abs(total_prob - 1.0) > 1e-6:
            print(f"[aggregate] WARNING: probabilities sum to "
                  f"{total_prob:.6f}, not 1.0 — stale or mismatched "
                  f"contribution set?", flush=True)
        sus = np.zeros(Z.shape, dtype=np.float32)
        flat = sus.reshape(-1)              # C-order view (matches saved idx)
        summary = []
        n_used = 0
        for c in contribs:
            flat[c['idx']] += np.float32(c['prob'])
            n_used += 1
            sp = cdir / f"summary_run{c['run']:02d}.json"
            if sp.exists():
                with open(sp, encoding='utf-8') as fh:
                    summary.append(json.load(fh))
            print(f"  + run {c['run']+1}: prob={c['prob']:.4f} "
                  f"cells={int(c['idx'].size)}")
            if c['no_slides']:
                # Mirror the monolithic loop: a run with no initial slides
                # breaks the loop, so later runs contribute nothing.
                print(f"  run {c['run']+1} had no slides — stopping here "
                      f"(matches the monolithic loop's early break)")
                break
        out_path = _write_sus_raster(sus, Z, georef, out_dir, susname)
        print(f"[aggregate] combined {n_used}/{len(files)} runs -> {out_path}")
        if summary:
            summary.sort(key=lambda r: r['run'])
            with open(out_dir / 'run_summary.json', 'w', encoding='utf-8') as fh:
                json.dump(summary, fh, indent=2)
        print(f"Total elapsed: {(time.time()-t0)/60.0:.2f} min")
        return

    print("Computing surface derivatives...")
    slope, aspect, dx, dy = gradient_prince(Z, georef.x_cellsize, georef.y_cellsize)

    notnan_mask = ~np.isnan(Z)
    notnanidx = find_F(notnan_mask)
    print(f"  {notnanidx.size} non-NaN cells")

    # ---- Soil depth (driver.m lines 175-209) -------------------------------
    if args.soil_depth_mode == 2:
        # Uniform soil depth: depth = soil_depth_uniform * ones(Z.shape)
        print(f"Assigning uniform soil depth = {args.soil_depth_uniform} m ...")
        depth = np.full(Z.shape, np.float32(args.soil_depth_uniform), dtype=np.float32)
    elif args.soil_depth_source == 'compute':
        print(f"Computing soil depth (Roering, {args.soil_depth_endtime} yr) ...",
              flush=True)
        t_sd = time.time()
        depth = compute_soil_depth(Z, georef.x_cellsize,
                                   endtime=args.soil_depth_endtime,
                                   verbose=True)
        depth = depth.astype(np.float32)
        print(f"  done in {time.time()-t_sd:.1f}s "
              f"(mean={float(np.nanmean(depth)):.3f} m)", flush=True)
        # Save the result back to .mat so the user can later choose `mat` mode.
        from scipy.io import savemat
        sd_save = REPO_ROOT / 'lib' / 'soil_depth' / \
            f"{Path(args.DEM_path).stem}_soil_depth_python.mat"
        sd_save.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(sd_save), {'depth': depth.astype(np.float64)})
        print(f"  saved: {sd_save.relative_to(REPO_ROOT).as_posix()}", flush=True)
    else:
        print(f"Loading soil depth: {args.soil_depth_mat}")
        _require_file(args.soil_depth_mat, what='soil-depth .mat')
        depth = load_soil_depth(args.soil_depth_mat).astype(np.float32)
        _require_shape(depth, Z.shape, what='soil depth',
                       src=args.soil_depth_mat)

    print("Computing subsurface derivatives (Z - depth)...")
    Z_sub = Z - depth
    subslope, subaspect, subdx, subdy = gradient_prince(
        Z_sub, georef.x_cellsize, georef.y_cellsize)

    # ---- Hydrology & soil weight (driver.m lines 274-290) ------------------
    f32 = np.float32
    if args.soil_moisture_mode == 0:
        # Completely dry. driver.m lines 275-281 (incl. `% damy 20250111`).
        print("Soil moisture mode 0: dry (hw = sigma_s = 0)")
        hw = np.zeros(Z.shape, dtype=np.float32)
        sigma_s = np.zeros(Z.shape, dtype=np.float32)
        sigma_s_wedge = np.zeros(Z.shape, dtype=np.float32)
        W = (f32(args.gam_dry) * depth
             * f32(georef.x_cellsize) * f32(georef.y_cellsize)).astype(np.float32)
    elif args.soil_moisture_mode == 1:
        # Hydrostatic with saturation ratio mw. driver.m lines 283-290.
        # Match MATLAB's single-precision arithmetic exactly.
        print(f"Building hydrostatic stress (mw={args.mw})...")
        hw = f32(args.mw) * depth
        sigma_s = f32(args.gam_w) * hw
        if args.sigma_s_wedge_mode == 'matlab_local':
            sigma_s_wedge = sigma_s  # `% damy 20250111` patch
        else:
            sigma_s_wedge = sigma_s * f32(0.5)
        W = (hw * f32(args.gam_sat) + (depth - hw) * f32(args.gam_dry)) \
            * f32(georef.x_cellsize) * f32(georef.y_cellsize)
        # Defensive cast in case numpy widened any intermediate
        W = W.astype(np.float32)
        sigma_s = sigma_s.astype(np.float32)
        sigma_s_wedge = sigma_s_wedge.astype(np.float32)
    else:
        raise NotImplementedError(
            f"soil_moisture_mode={args.soil_moisture_mode}: hydromechanical "
            "model (mode 2) requires SMAP/sand/clay/rainfall rasters and the "
            "`load_PWP_GW_soilweight` port — not yet implemented in Python.")

    # ---- Seismic (driver.m lines 295-313) ----------------------------------
    PGA = _build_pga(Z, georef, args)
    print(f"PGA: mode={args.seismic_mode!r}, "
          f"max={float(np.nanmax(PGA)) if PGA.size else 0.0:.3f} g, "
          f"mean={float(np.nanmean(PGA[~np.isnan(Z)])) if (~np.isnan(Z)).any() else 0.0:.3f} g, "
          f"scaling={args.pseudo_scaling}")

    # ---- No-grow zones (driver.m lines 325-395) ----------------------------
    if args.nogrow_mode == 0:
        print("Nogrow mode 0: no growth boundaries (ridges/valleys disabled)")
        nogrow_io = np.zeros(Z.shape, dtype=bool)
        nogrow_idx = np.zeros(0, dtype=np.int64)
        nogrow_i = np.zeros(0, dtype=np.int64)
        nogrow_j = np.zeros(0, dtype=np.int64)
        ridge_io = np.zeros(Z.shape, dtype=bool)
        valley_io = np.zeros(Z.shape, dtype=bool)
    elif args.nogrow_source == 'compute':
        print(f"Computing ridges/valleys (acc_thresh ridge={args.ridge_acc_thresh}, "
              f"valley={args.valley_acc_thresh}) ...")
        t_rv = time.time()
        # ridges_valleys uses the UNPADDED DEM (MATLAB re-loads from disk).
        rv = compute_ridges_valleys(Z, georef.x_cellsize,
                                    ridge_acc_thresh=args.ridge_acc_thresh,
                                    valley_acc_thresh=args.valley_acc_thresh)
        nogrow_io = rv.nogrow_io
        nogrow_idx = rv.nogrow_idx
        nogrow_i = rv.nogrow_i
        nogrow_j = rv.nogrow_j
        ridge_io = rv.ridge_io
        valley_io = rv.valley_io
        print(f"  done in {time.time()-t_rv:.1f}s "
              f"({int(nogrow_io.sum())} no-grow cells)")
        # Save to .mat so the user can later choose `mat` mode.
        from scipy.io import savemat
        ng_save = REPO_ROOT / 'lib' / 'no_grow' / \
            f"{Path(args.DEM_path).stem}_no_grow_python.mat"
        ng_save.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(ng_save), {
            'nogrow_io': nogrow_io.astype(np.uint8),
            'nogrow_idx': nogrow_idx.astype(np.float64),
            'nogrow_i': nogrow_i.astype(np.float64),
            'nogrow_j': nogrow_j.astype(np.float64),
            'ridge_io': ridge_io.astype(np.uint8),
            'valley_io': valley_io.astype(np.uint8),
        })
        print(f"  saved: {ng_save.relative_to(REPO_ROOT).as_posix()}", flush=True)
    elif args.nogrow_source == 'grass':
        # Reference slope units from GRASS r.slopeunits.create (Alvioli et al.).
        # Requires GRASS + the r.slopeunits addon (baked into the Docker image).
        from region3d.grass_slopeunits import (run_grass_slopeunits,
                                               read_units, nogrow_from_units)
        print(f"Computing slope units via GRASS r.slopeunits.create "
              f"(thresh={args.slope_unit_thresh:.0f} m^2, "
              f"areamin={args.slope_unit_areamin:.0f} m^2, "
              f"cvmin={args.slope_unit_cvmin:.2f}, rf={args.slope_unit_rf:.2f}, "
              f"maxiter={args.slope_unit_maxiter}) ...", flush=True)
        t_rv = time.time()
        stem = Path(args.DEM_path).stem
        slu_tif = REPO_ROOT / 'lib' / 'no_grow' / f"{stem}_slopeunits_grass.tif"
        run_grass_slopeunits(
            args.DEM_path, slu_tif,
            thresh=args.slope_unit_thresh, areamin=args.slope_unit_areamin,
            cvmin=args.slope_unit_cvmin, rf=args.slope_unit_rf,
            maxiter=args.slope_unit_maxiter,
            optimize=bool(int(args.slope_unit_optimize)),
            cvmin_range=f"{args.slope_unit_cvmin_min},{args.slope_unit_cvmin_max}",
            areamin_range=f"{args.slope_unit_areamin_min},{args.slope_unit_areamin_max}",
            epsilonx=args.slope_unit_opt_epsx, epsilony=args.slope_unit_opt_epsy)
        units_grass = read_units(slu_tif)
        if units_grass.shape != Z.shape:
            raise SystemExit(
                f"[shape mismatch] GRASS slope-unit raster {units_grass.shape} "
                f"!= DEM {Z.shape}")
        rv = nogrow_from_units(units_grass, Z)
        nogrow_io = rv.nogrow_io
        nogrow_idx = rv.nogrow_idx
        nogrow_i = rv.nogrow_i
        nogrow_j = rv.nogrow_j
        ridge_io = rv.ridge_io
        valley_io = rv.valley_io
        n_units = int(np.unique(units_grass[units_grass > 0]).size)
        print(f"  done in {time.time()-t_rv:.1f}s "
              f"({int(nogrow_io.sum())} no-grow cells, {n_units} units)",
              flush=True)
        from scipy.io import savemat
        ng_save = REPO_ROOT / 'lib' / 'no_grow' / \
            f"{stem}_no_grow_slopeunits_grass.mat"
        ng_save.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(ng_save), {
            'nogrow_io': nogrow_io.astype(np.uint8),
            'nogrow_idx': nogrow_idx.astype(np.float64),
            'nogrow_i': nogrow_i.astype(np.float64),
            'nogrow_j': nogrow_j.astype(np.float64),
            'ridge_io': ridge_io.astype(np.uint8),
            'valley_io': valley_io.astype(np.uint8),
        })
        print(f"  saved: {ng_save.relative_to(REPO_ROOT).as_posix()}", flush=True)
        # GRASS output is already a complete, connected partition -> polygonise
        # directly (no clean / complete-partition pass needed).
        if bool(int(args.slope_unit_save_shp)):
            from region3d.vectorize import write_units_vector
            ext = args.slope_unit_shp_format
            if not ext.startswith('.'):
                ext = '.' + ext
            su_save = REPO_ROOT / 'lib' / 'no_grow' / \
                f"{stem}_no_grow_slopeunits_grass_units{ext}"
            write_units_vector(
                units_grass, georef.transform, georef.crs, su_save,
                min_cells=int(args.slope_unit_shp_min_cells),
                clip_path=(args.slope_unit_shp_clip or None),
                clip_invert=bool(int(args.slope_unit_shp_clip_invert)),
                to_wgs84=bool(int(args.slope_unit_shp_wgs84)))
            print(f"  vector: {su_save.relative_to(REPO_ROOT).as_posix()}",
                  flush=True)
    else:
        print(f"Loading no-grow zones: {args.no_grow_mat}")
        _require_file(args.no_grow_mat, what='no-grow .mat')
        ng = load_no_grow(args.no_grow_mat)
        nogrow_io = ng['nogrow_io']
        nogrow_idx = ng['nogrow_idx']
        nogrow_i = ng['nogrow_i']
        nogrow_j = ng['nogrow_j']
        ridge_io = ng['ridge_io']
        valley_io = ng['valley_io']
        _require_shape(nogrow_io, Z.shape, what='no-grow mask',
                       src=args.no_grow_mat)

    # ---- Shear strength parameters (driver.m lines 414-421) ----------------
    if args.soil_strength_mode == 2:
        # Uniform single-pair (phi, c). MATLAB:
        #   prob=1; prob_phi=phi_uniform; prob_coh=coh_uniform
        print(f"Soil strength mode 2: uniform "
              f"phi={args.phi_uniform} deg, coh={args.coh_uniform} kPa")
        prob = np.array([1.0], dtype=np.float64)
        prob_phi = np.array([float(args.phi_uniform)], dtype=np.float64)
        prob_coh = np.array([float(args.coh_uniform)], dtype=np.float64)
    else:
        print(f"Loading shear strength parameter distribution: {args.shear_strength_mat}")
        _require_file(args.shear_strength_mat, what='shear-strength .mat')
        ss = load_shear_strength(args.shear_strength_mat)
        prob = ss['prob']
        prob_phi = ss['prob_phi']
        prob_coh = ss['prob_coh']
        print(f"  {prob.size} runs, phi={prob_phi}, coh={prob_coh}, prob={prob}")

    # Validate --run-index against the actual distribution BEFORE anything is
    # written: an out-of-range index would otherwise bank a bogus empty
    # contribution that a later --aggregate would silently ingest.
    if args.run_index is not None and args.run_index >= prob.size:
        sys.exit(f"[run-index] --run-index {args.run_index} is out of range: "
                 f"the distribution has {prob.size} runs "
                 f"(valid: 0..{prob.size - 1})")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- Tension/compression post-process (reuses the inputs above) --------
    if int(getattr(args, 'tension_compression', 0)):
        print("Tension/compression mode: per-cell net force q, prob-weighted "
              "over runs (no region-grow).")
        _tension_compression(args, Z, georef, subdx, subdy, W, sigma_s, PGA,
                             prob, prob_phi, prob_coh, susname, out_dir)
        print(f"Total elapsed: {(time.time()-t0)/60.0:.2f} min")
        return

    # ---- Susceptibility runs ------------------------------------------------
    sus_map = np.zeros(Z.shape, dtype=np.float32)
    diagnostics_per_run = []
    run_had_no_slides = False  # mirrors the MATLAB loop's early-break condition

    for sus_i in range(prob.size):
        if args.run_index is not None and sus_i != args.run_index:
            continue
        if prob[sus_i] == 0:
            print(f"\n[run {sus_i+1}/{prob.size}] zero probability, skipping")
            continue
        print(f"\n[run {sus_i+1}/{prob.size}] phi={prob_phi[sus_i]:.2f} coh={prob_coh[sus_i]:.2f} prob={prob[sus_i]:.4f}")

        result = region_grow_fxn(
            Z=Z, coh=float(prob_coh[sus_i]), phi=float(prob_phi[sus_i]),
            gam_w=args.gam_w, gam_dry=args.gam_dry, gam_sat=args.gam_sat,
            Gs=args.Gs, W=W, sigma_s=sigma_s, sigma_s_wedge=sigma_s_wedge,
            PGA=PGA, reg_grow_on=args.reg_grow_on,
            err_percent_allowable=args.err_percent_allowable,
            max_growth_cycles=args.max_growth_cycles,
            err_increase_thresh=args.err_increase_thresh,
            cluster_size_thresh=args.cluster_size_thresh,
            erosion_rounds=args.erosion_rounds,
            rot_range=args.rot_range, rot_num=args.rot_num,
            cleanup_rounds_initial=args.cleanup_rounds_initial,
            cleanup_rounds_grow=args.cleanup_rounds_grow,
            x_cellsize=georef.x_cellsize, y_cellsize=georef.y_cellsize,
            x_ext=georef.x_ext, y_ext=georef.y_ext,
            subslope=subslope, subaspect=subaspect, subdx=subdx, subdy=subdy,
            nogrow_idx=nogrow_idx, depth=depth, mw=args.mw,
            nogrow_mode=args.nogrow_mode, slope=slope, aspect=aspect, dx=dx, dy=dy,
            nogrow_io=nogrow_io, ridge_io=ridge_io, valley_io=valley_io,
            sus_i=sus_i + 1, susname=susname, notnanidx=notnanidx,
            DEM_name=Path(args.DEM_path).name, nogrow_i=nogrow_i, nogrow_j=nogrow_j,
            root_mode='uniform', S_roots=args.S_roots, S_roots_healthy=0.0,
            bs=None, verbose=True, max_cell_offset=int(args.max_cell_offset),
        )
        diagnostics_per_run.append((sus_i, result.diagnostics))
        sus_map[result.slides_final_io] += float(prob[sus_i])

        if not result.cluster_idx_initial:
            print("  no slides for this run, breaking")
            run_had_no_slides = True
            break

    out_path = _write_sus_raster(sus_map, Z, georef, out_dir, susname)
    if args.run_index is not None:
        print(f"  NOTE: single-run mode — this raster holds ONLY run "
              f"{int(args.run_index) + 1}/{prob.size}; combine all runs with "
              f"--aggregate 1")

    # Optional intermediate rasters for the web UI / diagnostics. They are
    # identical across runs, so in single-run fan-out mode only index 0 writes
    # them (avoids N redundant hillshade computations).
    if int(args.save_intermediates) and args.run_index in (None, 0):
        from region3d.derivatives import hillshade as _hillshade
        print("Writing intermediate rasters (depth/nogrow/PGA/hillshade)...")
        write_raster(out_dir / 'depth.tif', depth.astype(np.float32), georef)
        write_raster(out_dir / 'nogrow_io.tif',
                     nogrow_io.astype(np.float32), georef)
        write_raster(out_dir / 'PGA.tif', PGA.astype(np.float32), georef)
        try:
            hs = _hillshade(Z, georef.x_ext, georef.y_ext)
            write_raster(out_dir / 'hillshade.tif',
                         hs.astype(np.float32), georef)
        except Exception as exc:
            print(f"  hillshade failed: {exc}")

    # Per-run cluster counts (read from diagnostics)
    if diagnostics_per_run:
        run_summary = []
        for sus_i, diag in diagnostics_per_run:
            n_clusters = int(diag['terminate_reason'].size)
            run_summary.append({
                'run': sus_i + 1, 'phi': float(prob_phi[sus_i]),
                'coh': float(prob_coh[sus_i]), 'prob': float(prob[sus_i]),
                'n_clusters': n_clusters,
            })
        with open(out_dir / 'run_summary.json', 'w', encoding='utf-8') as fh:
            json.dump(run_summary, fh, indent=2)

    # ---- Single-run mode: bank this run's contribution for --aggregate ----
    # Written LAST so the .npz appears only after every other artifact of this
    # run — the resume loop treats its presence as "this index is complete".
    if args.run_index is not None:
        ri = int(args.run_index)  # validated above: 0 <= ri < prob.size
        cdir = out_dir / 'contribs'
        cdir.mkdir(parents=True, exist_ok=True)
        if diagnostics_per_run:
            idx = np.flatnonzero(result.slides_final_io).astype(np.int64)
            n_clusters = int(result.diagnostics['terminate_reason'].size)
        else:                                  # zero-probability run
            idx = np.zeros(0, dtype=np.int64)
            n_clusters = 0
        np.savez_compressed(cdir / f"contrib_run{ri:02d}.npz",
                            idx=idx, prob=np.float64(prob[ri]),
                            run=np.int64(ri),
                            shape=np.array(Z.shape, dtype=np.int64),
                            n_runs=np.int64(prob.size),
                            no_slides=np.bool_(run_had_no_slides))
        with open(cdir / f"summary_run{ri:02d}.json", 'w',
                  encoding='utf-8') as fh:
            json.dump({'run': ri + 1, 'phi': float(prob_phi[ri]),
                       'coh': float(prob_coh[ri]), 'prob': float(prob[ri]),
                       'n_clusters': n_clusters, 'cells': int(idx.size)},
                      fh, indent=2)
        print(f"[run {ri+1}/{prob.size}] contribution banked -> "
              f"contribs/contrib_run{ri:02d}.npz "
              f"(cells={idx.size}, prob={float(prob[ri]):.4f})")

    print(f"Total elapsed: {(time.time()-t0)/60.0:.2f} min")


def main():
    p = argparse.ArgumentParser()
    # ---- Required arguments (no defaults) -----------------------------------
    p.add_argument('--DEM_path', type=Path, required=True,
                   help='Input DEM (GeoTIFF). Required.')
    p.add_argument('--test_no', type=int, required=True,
                   help='Numeric ID used for the output sub-folder name '
                        '(zero-padded to 5 digits). Override entirely with '
                        '--susname_override.')
    # ---- Optional arguments from DEFAULTS -----------------------------------
    for k, v in DEFAULTS.items():
        if isinstance(v, tuple):
            p.add_argument(f'--{k}', nargs='+', type=float, default=list(v))
        elif isinstance(v, bool):
            p.add_argument(f'--{k}', type=int, default=int(v))
        elif isinstance(v, (int, float)):
            p.add_argument(f'--{k}', type=type(v), default=v)
        else:
            p.add_argument(f'--{k}', type=type(v), default=v)
    p.add_argument('--run-index', type=int, default=None,
                   help='Run only a single index from the shear-strength parameter distribution (0-based). '
                        'Saves that run to contribs/ for later --aggregate (low-memory resume).')
    p.add_argument('--aggregate', type=int, default=0,
                   help='Combine saved per-run contributions (contribs/*.npz) into the final '
                        'susceptibility map and exit. Needs only --DEM_path + --test_no/--susname_override.')
    p.add_argument('--tension_compression', type=int, default=0,
                   help='Post-process the banked contribs into probability-weighted '
                        'per-cell net-force (tension/compression) GeoTIFFs, then exit. '
                        'Same inputs as a normal run; runs no region-grow.')
    args = p.parse_args()
    if isinstance(args.rot_range, list):
        args.rot_range = tuple(args.rot_range)
    if args.run_index is not None and args.run_index < 0:
        p.error('--run-index must be >= 0 (0-based index into the '
                'shear-strength distribution); negative values would silently '
                'bank an empty contribution')
    if int(args.aggregate) and args.run_index is not None:
        p.error('--aggregate and --run-index are mutually exclusive: compute '
                'each run first, then aggregate')
    # ---- Conditional-required validation ------------------------------------
    # Aggregate mode only needs the DEM (for shape/georef) + the run name.
    missing = []
    if not int(args.aggregate):     # aggregate needs none of the input .mat files
        if args.soil_depth_mode == 1 and args.soil_depth_source == 'mat' \
                and not args.soil_depth_mat:
            missing.append('--soil_depth_mat is required when --soil_depth_mode=1 '
                           'and --soil_depth_source=mat')
        if args.nogrow_mode == 1 and args.nogrow_source == 'mat' \
                and not args.no_grow_mat:
            missing.append('--no_grow_mat is required when --nogrow_mode=1 and '
                           '--nogrow_source=mat')
        if args.soil_strength_mode == 1 and not args.shear_strength_mat:
            missing.append('--shear_strength_mat is required when '
                           '--soil_strength_mode=1')
    if missing:
        p.error('missing conditionally-required arguments:\n  '
                + '\n  '.join(missing))
    try:
        run(args)
    except SystemExit:
        # Clean, already-formatted guard messages (_require_file/_require_shape
        # etc.) — let them through without a confusing traceback.
        raise
    except Exception as e:
        import traceback
        print("\n" + "=" * 72, flush=True)
        print(f"[error] {type(e).__name__}: {e}", flush=True)
        print("=" * 72, flush=True)
        print("Full traceback below (for debugging):", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
