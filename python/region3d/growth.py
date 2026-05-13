"""Cluster growth helpers: dilate, erosion, eligibility filters, continuity tests.

Each function mirrors its MATLAB counterpart. Unless stated otherwise the
operands are localised rasters (Z_loc, cluster_io_loc, ...).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

from .matlab_compat import find_F, to_F_index

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


_KERNEL = np.ones((3, 3), dtype=bool)


# =============================================================================
#  Compact / JIT-accelerated cluster-grid builder
# =============================================================================
#
# `update_cluster_interslice` returns 7 NaN-padded full-shape arrays plus the
# total weight. In practice only `cluster_elev` and `cluster_W_sum` are read
# downstream — the other six allocations are pure overhead. With ~5-25 calls
# per cluster × 50K+ clusters the cost is significant. `update_cluster_compact`
# returns just (cluster_elev, cluster_W_sum), and is JIT-compiled when numba
# is available (drops the call cost from ~1 ms to ~50 µs on 30K-cell rasters).

if _HAVE_NUMBA:
    @_numba.njit(cache=True, fastmath=False)
    def _update_cluster_compact_jit(cluster_idx_F, m, n, Z, W):
        cluster_elev = np.full((m, n), np.nan)
        wsum = 0.0
        for k in range(cluster_idx_F.size):
            idx = cluster_idx_F[k] - 1  # 1-based F-order -> 0-based
            j = idx // m
            i = idx - j * m
            cluster_elev[i, j] = Z[i, j]
            w = W[i, j]
            if not np.isnan(w):
                wsum += w
        return cluster_elev, wsum


def update_cluster_compact(cluster_idx_1based, Z_loc, W_loc, wedge_W=0):
    """Build (cluster_elev, cluster_W_sum) only — drop-in fast path for
    `update_cluster_interslice` when callers only need those two values.

    Parameters
    ----------
    cluster_idx_1based : 1-D int array of MATLAB column-major linear indices.
    Z_loc, W_loc       : 2-D arrays (localised rasters).
    wedge_W            : scalar 0 or 1-D array; nansum is added to W_sum.
    """
    cluster_idx_F = np.asarray(cluster_idx_1based, dtype=np.int64)
    m, n = Z_loc.shape
    if cluster_idx_F.size == 0:
        cluster_elev = np.full(Z_loc.shape, np.nan, dtype=np.float64)
        wsum = 0.0
    elif _HAVE_NUMBA:
        cluster_elev, wsum = _update_cluster_compact_jit(
            cluster_idx_F, m, n,
            np.ascontiguousarray(Z_loc, dtype=np.float64),
            np.ascontiguousarray(W_loc, dtype=np.float64))
    else:
        cluster_elev = np.full(Z_loc.shape, np.nan, dtype=np.float64)
        i0, j0 = to_F_index(cluster_idx_F, Z_loc.shape)
        cluster_elev[i0, j0] = Z_loc[i0, j0]
        wsum = float(np.nansum(W_loc[i0, j0]))
    if wedge_W is not None and np.size(wedge_W) > 0:
        wsum += float(np.nansum(wedge_W))
    return cluster_elev, float(wsum)


def downhill_dilate(Z_loc, cluster_io_loc, cluster_elev, boundary_expand):
    """Mirror of `downhill_dilate.m`.

    Returns
    -------
    add_cells_idx : np.ndarray (int64, 1-based linear indices) of eligible cells
    boundary_io   : np.ndarray (bool, same shape)
    boundary_idx  : np.ndarray (int64, 1-based)
    bound_i, bound_j : 1-based subscripts
    boundary_expand : updated flag
    """
    cluster_io_loc = np.asarray(cluster_io_loc, dtype=bool)
    Z_loc = np.asarray(Z_loc)
    cluster_elev = np.asarray(cluster_elev)

    cluster_io_dilate = binary_dilation(cluster_io_loc, structure=_KERNEL)
    boundary_io = cluster_io_dilate & ~cluster_io_loc
    boundary_idx = find_F(boundary_io)
    if boundary_idx.size == 0:
        return (np.zeros(0, dtype=np.int64), boundary_io, boundary_idx,
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                int(boundary_expand))

    bi, bj = to_F_index(boundary_idx, cluster_io_loc.shape)
    bi += 1; bj += 1  # to 1-based for MATLAB parity in returned values
    m_bound, n_bound = cluster_elev.shape

    if (np.any(bi == 1) or np.any(bj == 1)
            or np.any(bi == m_bound) or np.any(bj == n_bound)):
        return (np.zeros(0, dtype=np.int64), boundary_io, boundary_idx, bi, bj, 1)

    boundary_check = np.zeros(Z_loc.shape, dtype=bool)
    bi0 = bi - 1
    bj0 = bj - 1
    boundary_elev_vals = Z_loc[bi0, bj0]

    # Build a 3x3 stack of cluster_elev around each boundary cell.
    nbh_min = np.full(boundary_idx.size, np.inf, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            vals = cluster_elev[bi0 + dr, bj0 + dc]
            # NaN must propagate as in MATLAB so that cells whose neighborhood
            # contains a NaN never compare strictly less than min(NaN, ...).
            # MATLAB's `min(min(...))` returns NaN if any cell is NaN, hence
            # `boundary_elev < NaN` is false. We replicate by mapping NaN -> inf
            # only after the running-min is complete.
            nbh_min = np.minimum(nbh_min, np.where(np.isnan(vals), np.inf, vals))
    # If all surrounding cells are NaN, MATLAB returned NaN and the < NaN test
    # was false. We replicate by leaving nbh_min as inf (always greater).
    cond = boundary_elev_vals < nbh_min
    boundary_check[bi0[cond], bj0[cond]] = True
    add_cells_idx = find_F(boundary_check)
    return add_cells_idx, boundary_io, boundary_idx, bi, bj, int(boundary_expand)


def nogrow_not_eligible(add_cells_idx, nogrow_idx):
    """Remove cells from add_cells_idx that intersect with nogrow_idx."""
    add_cells_idx = np.asarray(add_cells_idx, dtype=np.int64).ravel()
    nogrow_idx = np.asarray(nogrow_idx, dtype=np.int64).ravel()
    add_cells_idx_OG = add_cells_idx.copy()
    if add_cells_idx.size == 0 or nogrow_idx.size == 0:
        return add_cells_idx, add_cells_idx_OG
    mask = np.isin(add_cells_idx, nogrow_idx)
    return add_cells_idx[~mask], add_cells_idx_OG


def spur_test(cl_i_loc, cl_j_loc):
    """Detect spur (linear/single-row/column) clusters. Returns int 0..6."""
    ci = np.asarray(cl_i_loc).ravel()
    cj = np.asarray(cl_j_loc).ravel()
    if ci.size <= 1:
        # Single-cell cluster: MATLAB enters the loop with empty diff arrays.
        # `all(empty) == true`, so test1..test6 evaluate to 6. Match that.
        if ci.size == 0:
            return 0
        return 6
    di = np.diff(ci)
    dj = np.diff(cj)
    test1 = bool(np.all(cj == cj[0]))
    test2 = bool(np.all(ci == ci[0]))
    test3 = bool(np.all(di == 1))
    test4 = bool(np.all(di == -1))
    test5 = bool(np.all(dj == 1))
    test6 = bool(np.all(dj == -1))
    return int(test1) + int(test2) + int(test3) + int(test4) + int(test5) + int(test6)


def continuity_check(cluster_io_loc):
    """Return 1 if cluster has exactly one connected component, else 0."""
    cc, n = label(np.asarray(cluster_io_loc, dtype=bool), structure=_KERNEL)
    return 1 if n == 1 else 0


def update_cluster_interslice(cluster_idx_1based, Z, W, wedge_W, depth, area,
                              aspect, subaspect, Q_x0, Q_y0):
    """Build NaN-padded cluster rasters and W_sum (incl. wedge weights)."""
    shape = Z.shape
    cluster_elev = np.full(shape, np.nan, dtype=np.float64)
    cluster_W = np.full(shape, np.nan, dtype=np.float64)
    cluster_depth = np.full(shape, np.nan, dtype=np.float64)
    cluster_area = np.full(shape, np.nan, dtype=np.float64)
    cluster_aspect = np.full(shape, np.nan, dtype=np.float64)
    cluster_Qx0 = np.full(shape, np.nan, dtype=np.float64)
    cluster_Qy0 = np.full(shape, np.nan, dtype=np.float64)

    if len(cluster_idx_1based) == 0:
        cluster_W_sum = (np.nansum(wedge_W) if wedge_W is not None
                         and np.size(wedge_W) else 0.0)
        return (cluster_elev, cluster_W, cluster_depth, cluster_area,
                cluster_aspect, None, cluster_Qx0, cluster_Qy0,
                float(cluster_W_sum))

    i0, j0 = to_F_index(np.asarray(cluster_idx_1based, dtype=np.int64), shape)
    cluster_elev[i0, j0] = Z[i0, j0]
    cluster_W[i0, j0] = W[i0, j0]
    cluster_depth[i0, j0] = depth[i0, j0]
    cluster_area[i0, j0] = area[i0, j0]
    cluster_aspect[i0, j0] = aspect[i0, j0]
    cluster_Qx0[i0, j0] = Q_x0[i0, j0]
    cluster_Qy0[i0, j0] = Q_y0[i0, j0]

    cluster_W_sum = float(np.nansum(cluster_W))
    if wedge_W is not None and np.size(wedge_W) > 0:
        cluster_W_sum += float(np.nansum(wedge_W))

    return (cluster_elev, cluster_W, cluster_depth, cluster_area,
            cluster_aspect, None, cluster_Qx0, cluster_Qy0, cluster_W_sum)


def erode_once(cluster_io_loc, cluster_elev, Z_loc):
    """One round of cluster erosion (mirrors the inner block in RegionGrowFxn).

    Returns the eroded cluster_io_loc and the indices that were removed.
    """
    cluster_io_loc = np.asarray(cluster_io_loc, dtype=bool)
    cluster_io_er = binary_erosion(cluster_io_loc, structure=_KERNEL)
    er_io = cluster_io_loc & ~cluster_io_er
    er_idx = find_F(er_io)
    if er_idx.size == 0:
        return cluster_io_loc.copy(), np.zeros(0, dtype=np.int64)
    ei, ej = to_F_index(er_idx, cluster_io_loc.shape)
    er_elev_vals = Z_loc[ei, ej]
    nbh_min = np.full(er_idx.size, np.inf, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            i = ei + dr
            j = ej + dc
            valid = (i >= 0) & (i < cluster_elev.shape[0]) \
                & (j >= 0) & (j < cluster_elev.shape[1])
            sample = np.where(valid, cluster_elev[np.where(valid, i, 0),
                                                  np.where(valid, j, 0)], np.nan)
            sample = np.where(np.isnan(sample), np.inf, sample)
            nbh_min = np.minimum(nbh_min, sample)
    cond = er_elev_vals < nbh_min
    erode_idx = er_idx[cond]
    out = cluster_io_loc.copy()
    if erode_idx.size:
        ii, jj = to_F_index(erode_idx, cluster_io_loc.shape)
        out[ii, jj] = False
    return out, erode_idx
