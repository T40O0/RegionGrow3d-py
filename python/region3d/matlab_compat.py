"""MATLAB compatibility helpers.

MATLAB uses 1-based indexing and column-major (Fortran) memory order.
NumPy uses 0-based indexing and row-major (C) memory order. The functions
here translate between the two while preserving the linear index semantics
that the MATLAB code relies on (e.g. find/sub2ind/ind2sub on 2D arrays).
"""
from __future__ import annotations

import numpy as np


def find_F(mask):
    """MATLAB find(mask): return 1-based column-major linear indices of True cells.

    Returned as int64 1-D array in column-major scan order.
    """
    mask = np.asarray(mask)
    # Flatten in Fortran (column-major) order so positions match MATLAB
    flat = mask.ravel(order='F')
    idx0 = np.flatnonzero(flat)  # 0-based
    return idx0.astype(np.int64) + 1  # 1-based


def ind2sub_F(shape, idx_1based):
    """MATLAB ind2sub on 2D shape, accepting 1-based column-major linear indices.

    Returns (i, j) 1-based row/column subscripts.
    """
    m, n = shape
    idx0 = np.asarray(idx_1based, dtype=np.int64) - 1
    j = idx0 // m  # 0-based column
    i = idx0 - j * m  # 0-based row
    return i.astype(np.int64) + 1, j.astype(np.int64) + 1


def sub2ind_F(shape, i_1based, j_1based):
    """MATLAB sub2ind on 2D shape, accepting 1-based subscripts.

    Returns 1-based column-major linear indices (int64).
    """
    m, n = shape
    i = np.asarray(i_1based, dtype=np.int64) - 1
    j = np.asarray(j_1based, dtype=np.int64) - 1
    return (j * m + i + 1).astype(np.int64)


def to_F_index(idx_1based, shape):
    """Convert 1-based column-major linear index to numpy (i, j) 0-based subscripts."""
    i, j = ind2sub_F(shape, idx_1based)
    return i - 1, j - 1


def from_F_index(i0, j0, shape):
    """Convert 0-based numpy (i, j) subscripts to 1-based column-major linear index."""
    return sub2ind_F(shape, np.asarray(i0) + 1, np.asarray(j0) + 1)


def get_F(arr, idx_1based):
    """Index a 2D array using 1-based column-major linear indices (MATLAB style)."""
    arr = np.asarray(arr)
    i0, j0 = to_F_index(idx_1based, arr.shape)
    return arr[i0, j0]


def set_F(arr, idx_1based, value):
    """In-place: set 2D array elements at 1-based column-major linear indices."""
    i0, j0 = to_F_index(idx_1based, arr.shape)
    arr[i0, j0] = value


def bwconncomp_F(bw):
    """MATLAB bwconncomp with 8-connectivity, returning column-major-ordered cluster
    pixel index lists (1-based linear indices).

    Returns
    -------
    pixel_idx_list : list[np.ndarray]
        Each element is a 1-D int64 array of 1-based column-major linear indices for
        one connected component, in the order MATLAB would produce them.
    num_objects : int
    labels : np.ndarray (int32)
        Label image (0 = background, 1..NumObjects = labels in MATLAB order).
    """
    from scipy.ndimage import label

    bw = np.asarray(bw, dtype=bool)
    # MATLAB scans column-by-column. To get the same labeling order we transpose,
    # label (which scans row-by-row in C order — equivalent to column-by-column on
    # the original), and transpose back. scipy.ndimage.label uses raster scan, so
    # on the transposed array the first encountered foreground pixel is the one at
    # the smallest column-then-row in the original.
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    lbl_T, n = label(bw.T, structure=structure)
    labels = lbl_T.T.astype(np.int32, copy=False)

    pixel_idx_list = []
    if n == 0:
        return pixel_idx_list, 0, labels

    # For each label k=1..n, gather column-major linear indices of its pixels.
    # We need them sorted in MATLAB's natural order (ascending linear index).
    flat_F = labels.ravel(order='F')
    # bucket by label
    order = np.argsort(flat_F, kind='stable')
    sorted_labels = flat_F[order]
    # Find boundaries between consecutive labels
    # First skip background (label 0)
    nz_start = np.searchsorted(sorted_labels, 1, side='left')
    sorted_labels = sorted_labels[nz_start:]
    order = order[nz_start:]
    # For each label k=1..n, slice the contiguous range
    bounds = np.searchsorted(sorted_labels, np.arange(1, n + 1), side='right')
    prev = 0
    for b in bounds:
        idxs0 = order[prev:b]  # 0-based column-major linear indices
        pixel_idx_list.append((idxs0.astype(np.int64) + 1))
        prev = b

    return pixel_idx_list, n, labels


def omitnan_sum(a):
    """np.nansum with proper handling of all-NaN slices (returns 0 like MATLAB)."""
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return float(np.nansum(a))


def atan2d(y, x):
    """MATLAB atan2d: arctangent in degrees."""
    return np.degrees(np.arctan2(y, x))


def atand(x):
    return np.degrees(np.arctan(x))


def sind(x):
    return np.sin(np.radians(x))


def cosd(x):
    return np.cos(np.radians(x))


def tand(x):
    return np.tan(np.radians(x))
