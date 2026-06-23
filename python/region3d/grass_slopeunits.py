"""Run the reference GRASS r.slopeunits.create and adapt its output to the
RegionGrow3D pipeline.

This lets the pipeline use the *original* slope-unit algorithm (Alvioli et al.,
GRASS GIS) instead of the in-tree Python re-implementation. It shells out to
the ``grass`` executable, so it needs GRASS + the r.slopeunits addon available
— which the project Docker image bakes in (``Dockerfile``). Everything is then
reproducible inside the container.

Functions
---------
run_grass_slopeunits(dem, out_raster, ...) -> Path
    Run r.slopeunits.create on ``dem`` and write the slope-unit raster.
nogrow_from_units(units, Z) -> NoGrowResult
    Build the no-grow boundary mask (for region growing) from a complete
    slope-unit partition.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .preprocessing import NoGrowResult
from .bwmorph import bwmorph

_HERE = Path(__file__).resolve().parent
_PY_DIR = _HERE.parent                       # .../python
_EXEC = _PY_DIR / "grass_runner_exec.py"
# Optional standalone r.slopeunits.create.py (Windows fallback when the addon
# is not installed); shipped if present.
_RSU_SCRIPT = _PY_DIR / "_rsu_create.py"


def _is_ascii(s) -> bool:
    return all(ord(c) < 128 for c in str(s))


def _find_grass() -> str | None:
    """Locate a GRASS launcher. PATH first (Docker: `grass`), then common
    Windows conda / OSGeo4W / Program Files install locations."""
    for cand in (os.environ.get("GRASS_BIN"), "grass", "grass.bat",
                 "grass85", "grass84", "grass83"):
        if cand:
            hit = shutil.which(cand)
            if hit:
                return hit
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, "AppData", "Local", "miniconda3", "envs", "*",
                     "Library", "bin", "grass.bat"),
        os.path.join(home, "AppData", "Local", "anaconda3", "envs", "*",
                     "Library", "bin", "grass.bat"),
        r"C:\OSGeo4W\bin\grass*.bat",
        r"C:\Program Files\GRASS GIS*\grass*.bat",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _grass_python_for(grass_bin: str) -> str:
    """Pick the python interpreter for `grass --exec`. Env override wins; for a
    conda grass.bat use that env's python.exe; otherwise `python3` (Docker)."""
    env = os.environ.get("GRASS_PYTHON")
    if env:
        return env
    p = Path(grass_bin)
    # .../envs/<name>/Library/bin/grass.bat -> .../envs/<name>/python.exe
    if p.name.lower().startswith("grass") and p.suffix.lower() == ".bat":
        cand = p.parents[2] / "python.exe"
        if cand.exists():
            return str(cand)
    return "python3"


def run_grass_slopeunits(dem, out_raster, *,
                         thresh: float, areamin: float, cvmin: float,
                         rf: float, maxiter: int,
                         optimize: bool = False,
                         cvmin_range: str = "0.05,0.25",
                         areamin_range: str = "50000,200000",
                         epsilonx: float = 0.01, epsilony: float = 50000.0,
                         grass_bin: str | None = None,
                         grass_python: str | None = None,
                         verbose: bool = True) -> Path:
    """Run GRASS r.slopeunits on ``dem`` -> ``out_raster`` (Int32 GeoTIFF).

    Default: r.slopeunits.create + r.slopeunits.clean with the given params.
    ``optimize=True``: r.slopeunits.optimize searches ``cvmin_range`` /
    ``areamin_range`` (``"min,max"``) to maximise the morphometric F=V·I
    objective (no landslide inventory; slow). ``thresh``/``rf``/``maxiter`` stay
    fixed; the chosen cvmin/areamin are printed.

    ``grass_bin``/``grass_python`` default to ``GRASS_BIN``/``GRASS_PYTHON`` or
    ``grass``/``python3`` on PATH (the Docker image provides both). Non-ASCII
    DEM paths (e.g. Japanese filenames) are copied to an ASCII temp file first.
    """
    grass_bin = grass_bin or _find_grass()
    if not grass_bin:
        raise RuntimeError(
            "GRASS executable not found. The 'grass' no-grow source needs GRASS "
            "+ the r.slopeunits addon.\n"
            "  -> Recommended: run in the Docker image (it bundles both):\n"
            "       docker compose run --rm region3d python python/driver.py "
            "... --nogrow_source grass\n"
            "     or use the Dockerised Web UI (docker compose up).\n"
            "  -> Or set the GRASS_BIN env var to a local grass launcher.")
    grass_python = grass_python or _grass_python_for(grass_bin)
    dem = Path(dem).resolve()
    out_raster = Path(out_raster).resolve()
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    if optimize:
        # r.slopeunits.metrics (called by optimize) truncates the cell size to
        # an integer -> a sub-metre DEM gives resolution=0 and metrics fails.
        # Slope-unit optimisation is a meso-scale operation; require >= 1 m.
        import rasterio
        with rasterio.open(dem) as _s:
            _res = min(abs(_s.res[0]), abs(_s.res[1]))
        if _res < 1.0:
            raise RuntimeError(
                f"r.slopeunits.optimize does not support a sub-metre DEM "
                f"(cell size = {_res} m): its metrics step truncates the "
                f"resolution to an integer (0), which fails.\n"
                f"  -> Run optimize on a >= 1 m DEM (e.g. the 5 m crop), or "
                f"resample this DEM to >= 1 m. Plain create+clean (optimize "
                f"OFF) works at any resolution.")

    tmpdir = Path(tempfile.mkdtemp(prefix="rg3d_grass_"))
    try:
        dem_use = dem
        if not _is_ascii(dem):
            dem_use = tmpdir / "dem_ascii.tif"
            shutil.copy(dem, dem_use)
        loc = tmpdir / "loc"
        args = [str(grass_bin), "-c", str(dem_use), str(loc), "--exec",
                str(grass_python), str(_EXEC),
                "--dem", str(dem_use), "--out", str(out_raster),
                "--thresh", str(thresh), "--areamin", str(areamin),
                "--cvmin", str(cvmin), "--rf", str(rf),
                "--maxiter", str(maxiter)]
        if _RSU_SCRIPT.exists():
            args += ["--rsu_script", str(_RSU_SCRIPT)]
        if optimize:
            args += ["--optimize", "--cvmin_range", str(cvmin_range),
                     "--areamin_range", str(areamin_range),
                     "--epsilonx", str(epsilonx), "--epsilony", str(epsilony)]
        if verbose:
            if optimize:
                print(f"GRASS r.slopeunits.optimize: cvmin∈[{cvmin_range}] "
                      f"areamin∈[{areamin_range}] thresh={thresh} (slow) ...",
                      flush=True)
            else:
                print("GRASS r.slopeunits.create+clean: "
                      f"thresh={thresh} areamin={areamin} cvmin={cvmin} "
                      f"rf={rf} maxiter={maxiter}", flush=True)
        proc = subprocess.run(args, capture_output=True, text=True)
        if verbose and proc.stdout:
            for line in proc.stdout.splitlines():
                if any(k in line for k in ("SLOPEUNITS_DONE", "SLOPEUNITS_CLEANED",
                                           "SLOPEUNITS_OPTIMIZED", "RSU via",
                                           "unavailable")):
                    print("  " + line.strip(), flush=True)
        if proc.returncode != 0 or not out_raster.exists():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise RuntimeError(
                "GRASS r.slopeunits.create failed "
                f"(rc={proc.returncode}).\n{tail}")
        return out_raster
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def read_units(raster) -> np.ndarray:
    """Read a slope-unit GeoTIFF as an int32 array (0 = nodata)."""
    import rasterio
    with rasterio.open(raster) as src:
        u = src.read(1)
        nd = src.nodata
    u = u.astype(np.int64)
    if nd is not None and not np.isnan(nd):
        u[u == int(nd)] = 0
    u[u < 0] = 0
    return u.astype(np.int32)


def _idx_F(mask: np.ndarray) -> np.ndarray:
    flat = mask.ravel(order='F')
    return np.flatnonzero(flat).astype(np.int64) + 1


def _ij_F(idx_1based: np.ndarray, shape: tuple):
    m, _ = shape
    idx0 = idx_1based - 1
    j = idx0 // m
    i = idx0 - j * m
    return (i + 1).astype(np.int64), (j + 1).astype(np.int64)


def nogrow_from_units(units: np.ndarray, Z: np.ndarray) -> NoGrowResult:
    """Derive the no-grow boundary mask from a complete slope-unit partition.

    The no-grow line = unit boundaries (a 4-neighbour difference in unit id),
    thinned with the same bwmorph chain used elsewhere. ``valley_io`` is left
    empty (the GRASS partition encodes channels as unit boundaries rather than a
    separate channel mask); ``ridge_io`` carries the boundary.
    """
    units = np.ascontiguousarray(units, dtype=np.int32)
    nan_mask = ~np.isfinite(Z)
    boundary = np.zeros(units.shape, dtype=bool)
    diff_v = units[:-1, :] != units[1:, :]
    boundary[:-1, :] |= diff_v
    boundary[1:, :] |= diff_v
    diff_h = units[:, :-1] != units[:, 1:]
    boundary[:, :-1] |= diff_h
    boundary[:, 1:] |= diff_h
    boundary[nan_mask] = False
    # only boundaries between two real units (not the nodata rim)
    boundary &= (units > 0)

    nogrow_io = bwmorph(boundary, 'skel', 200)
    nogrow_io = bwmorph(nogrow_io, 'bridge', 1)
    nogrow_io = bwmorph(nogrow_io, 'diag', 1)

    nogrow_idx = _idx_F(nogrow_io)
    ni, nj = _ij_F(nogrow_idx, nogrow_io.shape)
    return NoGrowResult(
        nogrow_io=nogrow_io,
        nogrow_idx=nogrow_idx,
        nogrow_i=ni,
        nogrow_j=nj,
        ridge_io=boundary.astype(bool),
        valley_io=np.zeros(units.shape, dtype=bool),
    )
