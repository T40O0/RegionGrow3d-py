"""Precompute the RegionGrow3D soil-depth input from a DEM WITHOUT running the
susceptibility region-growing step.

Mirrors driver.py's `soil_depth_source='compute'` branch so the saved `.mat` is
byte-for-byte what the driver would auto-produce (consumable later via
`--soil_depth_source mat`).

  * --soil_depth_model roering (default)
        Roering (2008) hillslope evolution, 5000 yr
        (constants currently Oregon-calibrated)
  * --soil_depth_model massbalance
        soil-production function + mass balance, 松四 (2017) 地学雑誌 126(4)
        eq.(7)-(15) / 松四ほか (2016) 地形 37, 427-453; measured topography is
        held fixed and the thickness is solved in closed form (~1 min for a
        60M-cell DEM). Saved as `<stem>_soil_depth_python_massbalance.mat`.

Note: the no-grow / slope-unit precompute now lives in the GRASS r.slopeunits
path (`driver.py --nogrow_source grass`, bundled in the Docker image); the old
Python-port slope-unit precompute has been removed.

Usage:
  python precompute_inputs.py --DEM_path lib/DEM/dem_afterEQ_5m.tif
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8')   # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'python'))

from region3d.io import read_dem, write_raster
from region3d.derivatives import pad_DEM
from region3d.preprocessing import soil_depth as compute_soil_depth
from region3d.soil_production import (MASSBALANCE_ALIASES,
                                      params_from_namespace,
                                      soil_depth_massbalance)


def _set_keep_awake(enable: bool) -> None:
    """Keep Windows awake during the long run (mirrors driver._set_keep_awake)."""
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
        pass


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def precompute_soil_depth(Z, georef, stem, args):
    """Mirror driver.py's soil_depth_source='compute' branch."""
    t0 = time.time()
    model = str(args.soil_depth_model or 'roering').lower()
    if model in MASSBALANCE_ALIASES:
        mb_endtime = float(args.soil_depth_mb_endtime)
        mb_hollow = float(args.soil_depth_mb_hollow_endtime)
        when = 'steady state' if mb_endtime <= 0 else f'{mb_endtime:.0f} yr'
        if mb_endtime <= 0 and mb_hollow > 0:
            when += f' + hollows refilled for {mb_hollow:.0f} yr'
        mb_params = params_from_namespace(args)
        print(f"Computing soil depth (soil-production function + mass balance, "
              f"{when}) ...", flush=True)
        print(f"  params: E0={mb_params.E0:.0f} g/m2/yr alpha={mb_params.alpha:g} /m "
              f"K={mb_params.K:g} m2/yr Sc={mb_params.Sc:g} "
              f"rho_soil={mb_params.rho_soil:.0f} kg/m3 W={mb_params.W_soil:g} "
              f"g/m2/yr ({mb_params.transport})", flush=True)
        depth = soil_depth_massbalance(
            Z, georef.x_cellsize, georef.y_cellsize,
            params=mb_params,
            endtime=None if mb_endtime <= 0 else mb_endtime,
            hollow_endtime=mb_hollow if mb_hollow > 0 else None,
            h_init=float(args.soil_depth_mb_h_init),
            h_max=float(args.soil_depth_mb_h_max),
            smooth_sigma=float(args.soil_depth_mb_smooth),
            verbose=True)
        extra = str(getattr(args, 'soil_depth_mb_tag', '') or
                    getattr(args, 'soil_depth_mb_preset', '') or '').strip()
        tag = '_massbalance' + (f'_{extra}' if extra else '')
    else:
        print(f"Computing soil depth (Roering, {args.soil_depth_endtime:.0f} yr) ...",
              flush=True)
        depth = compute_soil_depth(Z, georef.x_cellsize,
                                   endtime=args.soil_depth_endtime, verbose=True)
        tag = ''
    depth = depth.astype(np.float32)
    print(f"  done in {time.time()-t0:.1f}s "
          f"(mean={float(np.nanmean(depth)):.3f} m)", flush=True)

    from scipy.io import savemat
    sd_save = REPO_ROOT / 'lib' / 'soil_depth' / \
        f"{stem}_soil_depth_python{tag}.mat"
    sd_save.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(sd_save), {'depth': depth.astype(np.float64)})
    print(f"  saved: {sd_save.relative_to(REPO_ROOT).as_posix()}", flush=True)

    if int(args.save_tif):
        qa = REPO_ROOT / 'lib' / 'soil_depth' / \
            f"{stem}_soil_depth_python{tag}.tif"
        write_raster(qa, depth.astype(np.float32), georef)
        print(f"  QA TIFF: {qa.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--DEM_path', required=True)
    p.add_argument('--save_tif', type=int, default=1)
    p.add_argument('--soil_depth_endtime', type=float, default=5000.0)
    p.add_argument('--soil_depth_model', default='roering',
                   help="roering | massbalance (soil-production function + "
                        "mass balance, 松四 2017)")
    # --- massbalance model parameters (mirror driver.py's DEFAULTS) ----------
    p.add_argument('--soil_depth_mb_preset', default='',
                   choices=['', 'oregon', 'matsushi'],
                   help="published parameter set; individual "
                        "--soil_depth_mb_* options still override it")
    p.add_argument('--soil_depth_mb_tag', default='',
                   help='suffix for the saved .mat/.tif (defaults to the preset '
                        'name)')
    p.add_argument('--soil_depth_mb_E0', type=float, default=720.0,
                   help='bare-bedrock soil production rate [g m-2 yr-1]')
    p.add_argument('--soil_depth_mb_alpha', type=float, default=3.0,
                   help='production decay with soil thickness [m-1]')
    p.add_argument('--soil_depth_mb_K', type=float, default=0.005,
                   help='soil-creep transport coefficient [m2 yr-1]')
    p.add_argument('--soil_depth_mb_Sc', type=float, default=1.25,
                   help='critical gradient of the non-linear creep law')
    p.add_argument('--soil_depth_mb_rho_soil', type=float, default=1200.0,
                   help='soil bulk density [kg m-3]')
    p.add_argument('--soil_depth_mb_W', type=float, default=0.0,
                   help='chemical mass loss from the soil [g m-2 yr-1]')
    p.add_argument('--soil_depth_mb_transport', default='nonlinear',
                   choices=['nonlinear', 'linear'])
    p.add_argument('--soil_depth_mb_endtime', type=float, default=5000.0,
                   help='0 = steady state, else transient duration [yr]')
    p.add_argument('--soil_depth_mb_hollow_endtime', type=float, default=0.0,
                   help='with --soil_depth_mb_endtime 0: years of refill given '
                        'to convergent cells (hollows) instead of h_max; set to '
                        'the shallow-landslide return period')
    p.add_argument('--soil_depth_mb_h_init', type=float, default=0.0,
                   help='initial soil thickness of the transient run [m]')
    p.add_argument('--soil_depth_mb_h_max', type=float, default=3.0,
                   help='upper cap on soil thickness [m]')
    p.add_argument('--soil_depth_mb_smooth', type=float, default=1.0,
                   help='Gaussian DEM pre-smoothing before derivatives [cells]')
    args = p.parse_args()

    dem = _resolve(args.DEM_path)
    stem = dem.stem
    t_all = time.time()
    _set_keep_awake(True)
    try:
        print(f"Loading DEM: {dem}", flush=True)
        Z, georef = read_dem(dem)
        Z = pad_DEM(Z)
        notnan = int(np.isfinite(Z).sum())
        print(f"  shape={Z.shape}, cellsize=({georef.x_cellsize},"
              f"{georef.y_cellsize}), valid={notnan:,}", flush=True)

        precompute_soil_depth(Z, georef, stem, args)

        print(f"Total elapsed: {(time.time()-t_all)/60.0:.2f} min", flush=True)
    finally:
        _set_keep_awake(False)


if __name__ == '__main__':
    main()
