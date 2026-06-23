"""Reconstruct per-slope-unit POLYGONS from a no-grow `.mat` and export them
as vector data (Shapefile / GeoJSON / GeoPackage).

Route B (faithful): the no-grow `.mat` produced by ``precompute_inputs.py``
does NOT store the slope-unit label raster (``su.units``), but it DOES store
the raw, un-skeletonised boundary masks ``ridge_io`` (unit boundaries) and
``valley_io`` (channels). Because ``ridge_io`` is the 4-neighbour boundary of
``su.units`` (both sides of every seam are flagged), the connected components
of ``~(ridge_io | valley_io)`` reproduce the individual slope units exactly
(each unit eroded by its 1-px boundary ring). We then optionally regrow the
ring so neighbouring polygons abut, and polygonise.

The grid (and CRS) is taken from the source DEM GeoTIFF so the output
coordinates are correct (EPSG:6675 for the Noto workflow).

What CANNOT be recovered from the mask: the v2.0 nested hierarchy
(parent / depth). Those attributes require re-running slope_units(); they are
omitted here.

Usage
-----
  # Shapefile in native EPSG:6675 (recommended for desktop GIS)
  python nogrow_to_polygons.py \
      --mat lib/no_grow/dem_afterEQ_5m_crop_no_grow_slopeunits_nested.mat \
      --dem lib/DEM/dem_afterEQ_5m_crop.tif \
      --out lib/no_grow/slope_units_nested.shp

  # GeoJSON reprojected to WGS84 (for web maps)
  python nogrow_to_polygons.py --mat ... --dem ... \
      --out lib/no_grow/slope_units.geojson --to-wgs84
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8')   # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np

# Ensure `region3d` (this script's sibling package) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_masks(mat_path: Path):
    """Return (ridge_io, valley_io) as bool arrays from the no-grow .mat."""
    from scipy.io import loadmat
    d = loadmat(str(mat_path))
    if 'ridge_io' not in d or 'valley_io' not in d:
        raise SystemExit(
            f"{mat_path} has keys {[k for k in d if not k.startswith('__')]}; "
            "expected 'ridge_io' and 'valley_io'.")
    ridge = np.asarray(d['ridge_io']).astype(bool)
    valley = np.asarray(d['valley_io']).astype(bool)
    return ridge, valley


def _load_dem_georef(dem_path: Path):
    """Return (finite_mask, transform, crs) from the source DEM GeoTIFF."""
    import rasterio
    with rasterio.open(dem_path) as src:
        Z = src.read(1).astype(np.float32)
        nodata = src.nodata
        finite = np.isfinite(Z)
        if nodata is not None and not np.isnan(nodata):
            finite &= (Z != nodata)
        finite &= (Z >= 0)   # driver.m convention: Z<0 -> NaN
        return finite, src.transform, src.crs


def reconstruct_units(ridge: np.ndarray, valley: np.ndarray,
                      finite: np.ndarray, *, fill_ring: bool = True,
                      fill_distance: int = 2,
                      min_cells: int = 4, verbose: bool = True) -> np.ndarray:
    """Reconstruct the slope-unit label raster from boundary masks.

    Returns int32 array, 0 = boundary/channel/nodata, >0 = unit id.
    """
    from scipy import ndimage as ndi

    if not (ridge.shape == valley.shape == finite.shape):
        raise SystemExit(f"shape mismatch: ridge{ridge.shape} "
                         f"valley{valley.shape} dem{finite.shape}")

    boundary = (ridge | valley)
    seed = finite & ~boundary
    # 4-connectivity matches how the boundary was defined (4-neighbour diffs)
    structure = ndi.generate_binary_structure(2, 1)
    labels, n = ndi.label(seed, structure=structure)
    if verbose:
        print(f"  reconstructed {n} raw components", flush=True)

    # Drop tiny specks (boundary noise) by merging them back to 0
    if min_cells > 1:
        counts = np.bincount(labels.ravel())
        small_ids = np.where(counts < min_cells)[0]
        small_ids = small_ids[small_ids > 0]
        if small_ids.size:
            labels[np.isin(labels, small_ids)] = 0
            if verbose:
                print(f"  dropped {small_ids.size} specks (<{min_cells} cells)",
                      flush=True)

    if fill_ring:
        # Regrow labels into the (ridge) boundary ring so neighbouring units
        # abut. Channels (valley) and nodata stay as gaps.
        try:
            from skimage.segmentation import expand_labels
            grown = expand_labels(labels, distance=fill_distance)
        except Exception:
            # Fallback: nearest-label fill via distance transform indices
            idx = ndi.distance_transform_edt(labels == 0,
                                             return_distances=False,
                                             return_indices=True)
            grown = labels[tuple(idx)]
        # Only keep growth onto former ridge cells inside the valid area;
        # never fill channels or nodata.
        grown[valley] = 0
        grown[~finite] = 0
        labels = grown.astype(np.int32)

    # Renumber to a compact 1..K
    ids = np.unique(labels)
    ids = ids[ids > 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    labels = remap[labels]
    if verbose:
        print(f"  final {ids.size} slope-unit polygons", flush=True)
    return labels.astype(np.int32)


def polygonize(labels: np.ndarray, transform, crs, cellsize_area: float,
               *, verbose: bool = True):
    """Vectorise the label raster into a GeoDataFrame (one row per unit)."""
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union

    # Per-unit cell counts in ONE pass (NOT a scan per polygon).
    counts = np.bincount(labels.ravel())

    mask = labels > 0
    geoms_by_id: dict[int, list] = {}
    for geom, val in features.shapes(labels, mask=mask, transform=transform,
                                     connectivity=4):
        v = int(val)
        geoms_by_id.setdefault(v, []).append(shp_shape(geom))

    rows = []
    for uid, gl in geoms_by_id.items():
        g = gl[0] if len(gl) == 1 else unary_union(gl)
        rows.append({'unit_id': uid,
                     'n_cells': int(counts[uid]) if uid < counts.size else 0,
                     'area_m2': float(g.area),
                     'geometry': g})
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)
    gdf = gdf.sort_values('unit_id').reset_index(drop=True)
    if verbose:
        print(f"  polygonised {len(gdf)} features, "
              f"total area {gdf['area_m2'].sum()/1e6:.1f} km^2", flush=True)
    return gdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mat', required=True, help='no-grow .mat path')
    p.add_argument('--dem', required=True,
                   help='source DEM GeoTIFF (for grid + CRS)')
    p.add_argument('--out', required=True,
                   help='output vector path; .shp/.geojson/.gpkg by extension')
    p.add_argument('--no-fill', action='store_true',
                   help='do NOT regrow the boundary ring (leave 1-px gaps)')
    p.add_argument('--fill-distance', type=int, default=2)
    p.add_argument('--min-cells', type=int, default=4,
                   help='drop reconstructed components smaller than this')
    p.add_argument('--to-wgs84', action='store_true',
                   help='reproject output to EPSG:4326 (for GeoJSON/web)')
    p.add_argument('--clip', default='',
                   help='clip polygons to this land (or sea) coastline vector')
    p.add_argument('--clip-invert', action='store_true',
                   help='treat --clip as SEA and subtract it (else LAND)')
    args = p.parse_args()

    mat = Path(args.mat).resolve()
    dem = Path(args.dem).resolve()
    out = Path(args.out).resolve()
    t0 = time.time()

    print(f"Loading masks: {mat.name}", flush=True)
    ridge, valley = _load_masks(mat)
    print(f"  ridge={int(ridge.sum()):,}  valley={int(valley.sum()):,}  "
          f"shape={ridge.shape}", flush=True)

    print(f"Loading DEM georef: {dem.name}", flush=True)
    finite, transform, crs = _load_dem_georef(dem)
    cellsize_area = abs(transform.a) * abs(transform.e)
    print(f"  crs={crs}  cell_area={cellsize_area:.1f} m^2  "
          f"valid={int(finite.sum()):,}", flush=True)

    labels = reconstruct_units(ridge, valley, finite,
                               fill_ring=not args.no_fill,
                               fill_distance=args.fill_distance,
                               min_cells=args.min_cells)

    gdf = polygonize(labels, transform, crs, cellsize_area)

    if args.clip:
        from region3d.vectorize import clip_gdf
        gdf = clip_gdf(gdf, args.clip, invert=bool(args.clip_invert))

    if args.to_wgs84:
        print("  reprojecting to EPSG:4326", flush=True)
        gdf = gdf.to_crs(4326)

    ext = out.suffix.lower()
    driver = {'.shp': 'ESRI Shapefile',
              '.geojson': 'GeoJSON',
              '.json': 'GeoJSON',
              '.gpkg': 'GPKG'}.get(ext)
    if driver is None:
        raise SystemExit(f"unsupported extension {ext}; use .shp/.geojson/.gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out), driver=driver, encoding='utf-8')
    print(f"Wrote {len(gdf)} polygons -> {out}  ({driver})", flush=True)
    print(f"Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
