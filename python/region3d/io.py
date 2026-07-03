"""I/O helpers: read DEM and reference rasters, write outputs.

Mirrors `geotiffread` / `saveraster.m` semantics: NoData values become NaN,
single (float32) precision is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class GeoRef:
    """Holds the metadata needed to write rasters in the same grid as the input."""
    crs: object
    transform: Affine
    nodata: float
    height: int
    width: int
    x_cellsize: float
    y_cellsize: float

    @property
    def x_ext(self) -> np.ndarray:
        """MATLAB driver: x_ext = x_cellsize : x_cellsize : RasterExtentInWorldX."""
        return (np.arange(1, self.width + 1, dtype=np.float64) * self.x_cellsize)

    @property
    def y_ext(self) -> np.ndarray:
        return (np.arange(1, self.height + 1, dtype=np.float64) * self.y_cellsize)


def read_dem(path) -> tuple[np.ndarray, GeoRef]:
    """Load a DEM GeoTIFF and convert NoData / negative values to NaN (per driver.m)."""
    with rasterio.open(path) as src:
        Z = src.read(1).astype(np.float32, copy=True)
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            Z[Z == nodata] = np.nan
        # driver.m: Z(Z<0)=NaN
        Z[Z < 0] = np.nan
        georef = GeoRef(
            crs=src.crs,
            transform=src.transform,
            nodata=float(nodata) if nodata is not None else float('nan'),
            height=src.height,
            width=src.width,
            x_cellsize=float(abs(src.transform.a)),
            y_cellsize=float(abs(src.transform.e)),
        )
    return Z, georef


def write_raster(path, arr: np.ndarray, georef: GeoRef, dtype: str = 'float32'):
    """Write a single-band GeoTIFF mirroring the input grid.

    Masked cells in this pipeline are NaN, so for float outputs the nodata tag
    is set to NaN (legal for float GeoTIFFs) — NOT inherited from the source
    DEM. Previously the file advertised the source's nodata (e.g. -9999) while
    the masked pixels were actually NaN, so tag-honouring readers (GDAL stats,
    QGIS zonal tools) counted every masked cell as valid data. The pixels
    themselves stay NaN, which is what gui.py's raw reads expect.
    """
    arr = np.asarray(arr, dtype=dtype)
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,
        'height': georef.height,
        'width': georef.width,
        'crs': georef.crs,
        'transform': georef.transform,
        'compress': 'lzw',
    }
    if np.issubdtype(arr.dtype, np.floating):
        profile['nodata'] = float('nan')
    elif not np.isnan(georef.nodata):
        profile['nodata'] = georef.nodata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def load_soil_depth(path) -> np.ndarray:
    """Load soil-depth raster from a MATLAB .mat file (variable: 'depth')."""
    from scipy.io import loadmat
    d = loadmat(path)
    return d['depth'].astype(np.float64)


def load_no_grow(path) -> dict:
    """Load no-grow zone .mat: returns dict with nogrow_io/idx/i/j/ridge_io/valley_io."""
    from scipy.io import loadmat
    d = loadmat(path)
    return {
        'nogrow_io': d['nogrow_io'].astype(bool),
        'nogrow_idx': d['nogrow_idx'].ravel().astype(np.int64),
        'nogrow_i': d['nogrow_i'].ravel().astype(np.int64),
        'nogrow_j': d['nogrow_j'].ravel().astype(np.int64),
        'ridge_io': d.get('ridge_io', np.zeros_like(d['nogrow_io'])).astype(bool),
        'valley_io': d.get('valley_io', np.zeros_like(d['nogrow_io'])).astype(bool),
    }


def load_shear_strength(path) -> dict:
    """Load shear-strength parameter distribution: prob, prob_phi, prob_coh."""
    from scipy.io import loadmat
    d = loadmat(path)
    return {
        'prob': d['prob'].ravel().astype(np.float64),
        'prob_phi': d['prob_phi'].ravel().astype(np.float64),
        'prob_coh': d['prob_coh'].ravel().astype(np.float64),
    }
