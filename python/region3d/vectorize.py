"""Vectorise a slope-unit (or any integer-label) raster into polygons, with
optional coastline clipping.

Used by the GRASS r.slopeunits path (``nogrow_source='grass'``) and by
``nogrow_to_polygons.py`` to turn a label raster into a Shapefile / GeoJSON /
GeoPackage. Label convention: 0 = nodata / background, >0 = unit id. Each
CONNECTED component becomes one feature (no union across disconnected pieces,
so no multipolygon "exclaves"). GRASS r.slopeunits output is already a
complete, connected partition, so no extra cleaning is applied here.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def polygonize_units(units: np.ndarray, transform, crs, *,
                     min_cells: int = 1, verbose: bool = True):
    """Polygonise an integer label raster into a GeoDataFrame.

    Emits ONE feature per connected component (no union by unit_id), tagged
    with ``unit_id`` and a per-unit ``part`` index. Columns: ``unit_id, part,
    n_cells, area_m2``.
    """
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape as shp_shape

    units = np.ascontiguousarray(units, dtype=np.int32)
    cell_area = abs(transform.a) * abs(transform.e)

    mask = units > 0
    part_counter: dict[int, int] = {}
    rows = []
    for geom, val in features.shapes(units, mask=mask, transform=transform,
                                     connectivity=4):
        uid = int(val)
        g = shp_shape(geom)
        ncell = int(round(g.area / cell_area)) if cell_area else 0
        if min_cells > 1 and ncell < min_cells:
            continue
        part = part_counter.get(uid, 0) + 1
        part_counter[uid] = part
        rows.append({'unit_id': uid, 'part': part,
                     'n_cells': ncell, 'area_m2': float(g.area),
                     'geometry': g})

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)
    gdf = gdf.sort_values(['unit_id', 'part']).reset_index(drop=True)
    if verbose:
        print(f"  polygonize_units: {len(gdf)} polygons, "
              f"total area {gdf['area_m2'].sum()/1e6:.1f} km^2", flush=True)
    return gdf


def clip_gdf(gdf, clip_path, *, invert: bool = False,
             simplify: Optional[float] = None, verbose: bool = True):
    """Clip a polygon GeoDataFrame to a land/sea mask vector (the coastline).

    ``clip_path`` is a polygon layer (Shapefile / GeoJSON / GeoPackage). By
    default it is treated as LAND and the units are intersected with it
    (everything outside the land — i.e. the sea — is removed, so polygons stop
    exactly at the coastline). With ``invert=True`` it is treated as SEA and
    subtracted instead. The mask is reprojected to ``gdf``'s CRS first, and
    ``area_m2`` is recomputed from the clipped geometry.
    """
    import geopandas as gpd
    import pandas as pd

    mask = gpd.read_file(str(clip_path))
    if mask.crs is None:
        raise ValueError(f"{clip_path} has no CRS; cannot clip safely")
    if gdf.crs is not None and mask.crs != gdf.crs:
        mask = mask.to_crs(gdf.crs)

    # Dissolve the mask to a single (multi)polygon (the land or sea region).
    try:
        mask_geom = mask.geometry.union_all()        # geopandas >= 0.14
    except AttributeError:                            # older geopandas
        mask_geom = mask.geometry.unary_union
    if simplify and simplify > 0:
        mask_geom = mask_geom.simplify(float(simplify), preserve_topology=True)

    # Intersection against a detailed coastline is expensive, so only the
    # polygons that actually straddle the shoreline are intersected; fully-
    # inland units are kept untouched, fully-offshore units dropped.
    n0 = len(gdf)
    keep_geom = (gdf.geometry.intersects(mask_geom) if invert
                 else gdf.geometry.within(mask_geom))
    if invert:
        untouched = gdf[~keep_geom]
        rest = gdf[keep_geom].copy()
        if len(rest):
            rest['geometry'] = rest.geometry.difference(mask_geom)
        clipped = pd.concat([untouched, rest])
    else:
        inside = gdf[keep_geom]
        edge = gdf[~keep_geom]
        edge = edge[edge.geometry.intersects(mask_geom)].copy()
        if len(edge):
            edge['geometry'] = edge.geometry.intersection(mask_geom)
        clipped = pd.concat([inside, edge])

    out = gpd.GeoDataFrame(clipped, geometry='geometry', crs=gdf.crs)
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out = out[out.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    out['area_m2'] = out.geometry.area
    out = out.sort_values('unit_id').reset_index(drop=True)
    if verbose:
        kind = 'sea-difference' if invert else 'land-intersection'
        print(f"  clip ({kind}): {n0} -> {len(out)} polygons", flush=True)
    return out


_DRIVER_BY_EXT = {'.shp': 'ESRI Shapefile',
                  '.geojson': 'GeoJSON',
                  '.json': 'GeoJSON',
                  '.gpkg': 'GPKG'}


def write_units_vector(units: np.ndarray, transform, crs, out_path, *,
                       min_cells: int = 1,
                       clip_path=None,
                       clip_invert: bool = False,
                       to_wgs84: bool = False,
                       verbose: bool = True):
    """Polygonise ``units`` and write to ``out_path`` (driver by extension).

    If ``clip_path`` is given the polygons are clipped to that coastline/land
    (or sea, with ``clip_invert``) mask before writing.
    """
    out = Path(out_path)
    driver = _DRIVER_BY_EXT.get(out.suffix.lower())
    if driver is None:
        raise ValueError(f"unsupported extension {out.suffix}; "
                         "use .shp/.geojson/.gpkg")
    gdf = polygonize_units(units, transform, crs, min_cells=min_cells,
                           verbose=verbose)
    if clip_path:
        gdf = clip_gdf(gdf, clip_path, invert=clip_invert, verbose=verbose)
    if to_wgs84:
        gdf = gdf.to_crs(4326)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out), driver=driver, encoding='utf-8')
    if verbose:
        print(f"  wrote {len(gdf)} polygons -> {out}  ({driver})", flush=True)
    return gdf
