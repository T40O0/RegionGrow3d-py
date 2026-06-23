"""Precompute the RegionGrow3D soil-depth input from a DEM WITHOUT running the
susceptibility region-growing step.

Mirrors driver.py's `soil_depth_source='compute'` branch so the saved `.mat` is
byte-for-byte what the driver would auto-produce (consumable later via
`--soil_depth_source mat`).

  * soil depth = Roering (2008) hillslope evolution, 5000 yr
        (constants currently Oregon-calibrated)

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
    print(f"Computing soil depth (Roering, {args.soil_depth_endtime:.0f} yr) ...",
          flush=True)
    t0 = time.time()
    depth = compute_soil_depth(Z, georef.x_cellsize,
                               endtime=args.soil_depth_endtime, verbose=True)
    depth = depth.astype(np.float32)
    print(f"  done in {time.time()-t0:.1f}s "
          f"(mean={float(np.nanmean(depth)):.3f} m)", flush=True)

    from scipy.io import savemat
    sd_save = REPO_ROOT / 'lib' / 'soil_depth' / f"{stem}_soil_depth_python.mat"
    sd_save.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(sd_save), {'depth': depth.astype(np.float64)})
    print(f"  saved: {sd_save.relative_to(REPO_ROOT).as_posix()}", flush=True)

    if int(args.save_tif):
        qa = REPO_ROOT / 'lib' / 'soil_depth' / f"{stem}_soil_depth_python.tif"
        write_raster(qa, depth.astype(np.float32), georef)
        print(f"  QA TIFF: {qa.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--DEM_path', required=True)
    p.add_argument('--save_tif', type=int, default=1)
    p.add_argument('--soil_depth_endtime', type=float, default=5000.0)
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
