"""Sub-rectangle extraction so the region-growing loop does not iterate over the
full DEM each cycle. Mirrors `create_localized_rasters_interslice.m`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LocalizedRasters:
    loc_j: np.ndarray  # 1-based column subscripts
    loc_i: np.ndarray  # 1-based row subscripts
    x_ext_loc: np.ndarray
    y_ext_loc: np.ndarray
    x_surf_loc: np.ndarray
    y_surf_loc: np.ndarray
    cluster_io_loc: np.ndarray
    Z_loc: np.ndarray
    W_loc: np.ndarray
    depth_loc: np.ndarray
    area_loc: np.ndarray
    aspect_loc: np.ndarray
    subaspect_loc: Optional[np.ndarray]  # MATLAB sets to []
    phi_loc: np.ndarray
    coh_loc: np.ndarray
    Q0_loc: np.ndarray
    Q_x0_loc: np.ndarray
    Q_y0_loc: np.ndarray
    Q_x_cell_loc: List[np.ndarray]
    Q_y_cell_loc: List[np.ndarray]
    ridge_io_loc: Optional[np.ndarray]
    valley_io_loc: Optional[np.ndarray]
    nogrow_io_loc: np.ndarray
    ridge_idx_loc: Optional[np.ndarray]
    valley_idx_loc: Optional[np.ndarray]
    nogrow_idx_loc: np.ndarray
    sigma_s_wedge_loc: np.ndarray
    PGA_loc: np.ndarray
    bs_loc: Optional[np.ndarray]


def create_localized_rasters_interslice(
        cell_offset, cl_i, cl_j, x_ext, y_ext, cluster_io, Z, W, depth, area_col,
        aspect, subaspect, phi, coh, Q0, Q_x0, Q_y0, Q_x_cell, Q_y_cell, rot,
        nogrow_mode, nogrow_io, ridge_io, valley_io, sigma_s_wedge, PGA,
        root_mode, bs) -> LocalizedRasters:
    cl_i = np.asarray(cl_i, dtype=np.int64).ravel()
    cl_j = np.asarray(cl_j, dtype=np.int64).ravel()

    loc_i = np.arange(cl_i.min() - cell_offset, cl_i.max() + cell_offset + 1, dtype=np.int64)
    loc_j = np.arange(cl_j.min() - cell_offset, cl_j.max() + cell_offset + 1, dtype=np.int64)

    loc_i = loc_i[(loc_i >= 1) & (loc_i <= len(y_ext))]
    loc_j = loc_j[(loc_j >= 1) & (loc_j <= len(x_ext))]

    i0 = loc_i - 1
    j0 = loc_j - 1
    rows = i0[:, None]
    cols = j0[None, :]

    x_ext_loc = np.asarray(x_ext)[j0]
    y_ext_loc = np.asarray(y_ext)[i0]
    y_surf_loc, x_surf_loc = np.meshgrid(y_ext_loc, x_ext_loc, indexing='ij')
    # Note: MATLAB's [x_surf, y_surf] = meshgrid(x, y) gives x_surf where rows
    # vary by x and cols by y. Match by storing the columns-of-x, rows-of-y form.
    x_surf_loc = np.asarray(x_ext_loc)[None, :].repeat(len(y_ext_loc), axis=0)
    y_surf_loc = np.asarray(y_ext_loc)[:, None].repeat(len(x_ext_loc), axis=1)

    def crop(arr):
        return np.asarray(arr)[rows, cols]

    cluster_io_loc = crop(cluster_io).astype(bool)
    Z_loc = crop(Z)
    W_loc = crop(W)
    depth_loc = crop(depth)
    area_loc = crop(area_col)
    aspect_loc = crop(aspect)
    phi_loc = crop(phi)
    coh_loc = crop(coh)
    Q0_loc = crop(Q0)
    Q_x0_loc = crop(Q_x0)
    Q_y0_loc = crop(Q_y0)
    sigma_s_wedge_loc = crop(sigma_s_wedge)
    PGA_loc = crop(PGA)

    Q_x_cell_loc = [crop(qx) for qx in Q_x_cell]
    Q_y_cell_loc = [crop(qy) for qy in Q_y_cell]

    nogrow_io_loc = crop(nogrow_io).astype(bool)
    nogrow_idx_loc_F = np.flatnonzero(nogrow_io_loc.ravel(order='F')) + 1

    bs_loc = crop(bs) if (root_mode == 'pfd' and bs is not None) else None

    return LocalizedRasters(
        loc_j=loc_j, loc_i=loc_i,
        x_ext_loc=x_ext_loc, y_ext_loc=y_ext_loc,
        x_surf_loc=x_surf_loc, y_surf_loc=y_surf_loc,
        cluster_io_loc=cluster_io_loc, Z_loc=Z_loc, W_loc=W_loc,
        depth_loc=depth_loc, area_loc=area_loc, aspect_loc=aspect_loc,
        subaspect_loc=None, phi_loc=phi_loc, coh_loc=coh_loc,
        Q0_loc=Q0_loc, Q_x0_loc=Q_x0_loc, Q_y0_loc=Q_y0_loc,
        Q_x_cell_loc=Q_x_cell_loc, Q_y_cell_loc=Q_y_cell_loc,
        ridge_io_loc=None, valley_io_loc=None, nogrow_io_loc=nogrow_io_loc,
        ridge_idx_loc=None, valley_idx_loc=None,
        nogrow_idx_loc=nogrow_idx_loc_F.astype(np.int64),
        sigma_s_wedge_loc=sigma_s_wedge_loc, PGA_loc=PGA_loc, bs_loc=bs_loc,
    )
