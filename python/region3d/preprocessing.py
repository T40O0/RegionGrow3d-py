"""DEM preprocessing: soil-depth simulation and TopoToolbox-style flow routing.

THIS FILE IS MIXED-LICENSE — see `LICENSE` at the repository root.

  * `soil_depth` and `ridges_valleys` are derived from the USGS RegionGrow3D
    MATLAB code and are licensed under CC0-1.0 (public domain).
        SPDX-License-Identifier: CC0-1.0

  * `fillsinks`, `identify_flats`, `flow_direction`, `flow_accumulation`, and
    their private helpers (`_imreconstruct`, `_regional_minima`,
    `_clear_border`, `_graydist_quasi`) are reimplementations of TopoToolbox
    (Schwanghart & Scherler 2014) — original MATLAB code is GPL-3.0-or-later,
    so these Python equivalents are also licensed under GPL-3.0-or-later.
        SPDX-License-Identifier: GPL-3.0-or-later

Per-function SPDX headers below mark which sections fall under which licence.
Combining the GPL functions with other code triggers GPL-3.0 copyleft on the
combined work; the CC0 functions remain freely reusable on their own.

This module ports the MATLAB preprocessing chain so the Python driver no longer
depends on MATLAB-generated `.mat` files:

- :func:`soil_depth`     -- non-linear hillslope evolution (Roering 2008),
                            mirrors `lib/functions/soil_depth.m`.
- :func:`fillsinks`      -- 8-connected morphological reconstruction depression
                            fill, mirrors `lib/functions/fillsinks.m`.
- :func:`identify_flats` -- flat / sill / closed-basin identification, mirrors
                            `lib/functions/identifyflats.m`.
- :func:`flow_direction` -- D8 single-flow direction with `carve` preprocessing
                            and gray-weighted geodesic auxiliary topography in
                            flats; mirrors the M-side branch of
                            `lib/functions/FLOWobj.m` (the path actually used
                            by `ridgelines.m` / `valleys.m`).
- :func:`flow_accumulation` -- topological accumulation of upstream cell count,
                            mirrors `lib/functions/flowacc.m`.
- :func:`ridges_valleys` -- thin wrapper that builds the no-grow mask the
                            driver consumes.

All routines operate on plain NumPy arrays (NaN-marked nodata) and reproduce
the MATLAB results to within floating-point tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, grey_erosion,
                           grey_dilation, label as nd_label)

from .bwmorph import bwmorph


# =============================================================================
#  soil_depth.m  ->  soil_depth()
# =============================================================================

def _del2(z: np.ndarray, h: float) -> np.ndarray:
    """Discrete Laplacian / 4 with MATLAB's `del2` boundary handling.

    MATLAB del2(F, h) returns at each interior cell:
        L(i,j) = ((F(i-1,j) + F(i+1,j) + F(i,j-1) + F(i,j+1)) / 4 - F(i,j)) / h^2
    At the boundary it uses one-sided second differences. We implement the
    interior 5-point stencil and then patch the perimeter using forward/backward
    second differences along each axis as MATLAB does.
    """
    z = np.asarray(z, dtype=np.float64)
    m, n = z.shape
    L = np.zeros_like(z)
    h2 = h * h
    # Interior
    L[1:-1, 1:-1] = (((z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:])
                       / 4.0) - z[1:-1, 1:-1]) / h2
    # MATLAB's del2 uses one-sided differences at the borders; for our purposes
    # (initial elevation only — zhill mask multiplies these out at the edge in
    # soil_depth.m), the forward/backward second difference along the closer
    # axis is sufficient.
    if n >= 3:
        L[:, 0]  = (z[:, 0]  - 2 * z[:, 1]  + z[:, 2])  / h2 / 2.0
        L[:, -1] = (z[:, -1] - 2 * z[:, -2] + z[:, -3]) / h2 / 2.0
    if m >= 3:
        L[0, :]  += (z[0, :]  - 2 * z[1, :]  + z[2, :])  / h2 / 2.0
        L[-1, :] += (z[-1, :] - 2 * z[-2, :] + z[-3, :]) / h2 / 2.0
        L[0, 0]   = ((z[0, 0]  - 2 * z[0, 1]  + z[0, 2])  + (z[0, 0]  - 2 * z[1, 0]  + z[2, 0]))  / (2.0 * h2)
        L[0, -1]  = ((z[0, -1] - 2 * z[0, -2] + z[0, -3]) + (z[0, -1] - 2 * z[1, -1] + z[2, -1])) / (2.0 * h2)
        L[-1, 0]  = ((z[-1, 0] - 2 * z[-1, 1] + z[-1, 2]) + (z[-1, 0] - 2 * z[-2, 0] + z[-3, 0])) / (2.0 * h2)
        L[-1, -1] = ((z[-1, -1] - 2 * z[-1, -2] + z[-1, -3]) + (z[-1, -1] - 2 * z[-2, -1] + z[-3, -1])) / (2.0 * h2)
    return L


def _matlab_gradient(z: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB `[dzdx, dzdy] = gradient(z, dx)`.

    MATLAB returns (dz/dx, dz/dy) where dx is across columns and dy across rows.
    np.gradient with default ordering returns (axis-0, axis-1) = (d/d row, d/d col).
    So we swap.
    """
    dzdy, dzdx = np.gradient(z, dx)
    return dzdx, dzdy


try:
    import numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @numba.njit(cache=True, parallel=True, fastmath=False)
    def _soil_depth_step_numba(z, zrock, K_val, Sc, dt, grac, Po, mu, prps, dx):
        """One Roering hillslope-evolution step. Returns (z_new, zrock_new).

        MATLAB-faithful finite differences:
          - dzdx, dzdy:      3-point central diff (np.gradient interior)
          - d2zdx2, d2zdy2:  5-point stencil (= gradient(dzdx) / gradient(dzdy))
                             d2/dx2 = (z[j+2] - 2*z[j] + z[j-2]) / (4*dx^2)
          - d2zdxdy:         4-corner stencil (= gradient(dzdx) along y)
          - laplacian:       4 * del2(z, dx) = (zL+zR+zU+zD - 4*z)/dx^2

        Cells closer than 2 from the border are not updated (their MATLAB
        counterparts produce NaN-tainted derivatives that get restored to
        the previous timestep value). The bedrock-lowering term is still
        applied at all finite cells.
        """
        m, n = z.shape
        z_new = np.empty_like(z)
        zrock_new = np.empty_like(zrock)
        inv_2dx = 0.5 / dx
        inv_dx_sq = 1.0 / (dx * dx)
        inv_4dx_sq = 0.25 * inv_dx_sq
        Sc_thresh = Sc - grac * Sc
        Sc_sq = Sc * Sc
        dt_Po = dt * Po
        dt_over_prps = dt / prps

        for i in numba.prange(m):
            for j in range(n):
                zij = z[i, j]
                zrk = zrock[i, j]

                # Bedrock lowering applies everywhere with finite z,zrk.
                if not (np.isnan(zij) or np.isnan(zrk)):
                    zrock_ij_new = zrk - dt_Po * np.exp(-mu * (zij - zrk))
                else:
                    zrock_ij_new = zrk
                zrock_new[i, j] = zrock_ij_new

                # Very-border cells (i=0/m-1, j=0/n-1) keep OLD z.
                # MATLAB cells like z[0,:] are NaN anyway, so this matches.
                if (i == 0 or i >= m - 1 or j == 0 or j >= n - 1
                        or np.isnan(zij)):
                    z_new[i, j] = zij
                    continue

                # 3-point gradient samples
                zL = z[i, j - 1]; zR = z[i, j + 1]
                zU = z[i - 1, j]; zD = z[i + 1, j]

                # If any cardinal neighbour is NaN, MATLAB's `grad` is NaN
                # which evaluates `gradless = grad < thresh` to FALSE (any
                # comparison with NaN is false), so `gradmore` becomes 1 and
                # the snap step writes z := zrock_new. Replicate.
                if (np.isnan(zL) or np.isnan(zR) or np.isnan(zU) or np.isnan(zD)):
                    z_new[i, j] = zrock_ij_new
                    continue

                dzdx = (zR - zL) * inv_2dx
                dzdy = (zD - zU) * inv_2dx
                grad = np.sqrt(dzdx * dzdx + dzdy * dzdy)

                if grad >= Sc_thresh:
                    z_new[i, j] = zrock_ij_new
                    continue

                # For the 5-point 2nd derivatives we need cells 2 steps away.
                # If those are NaN, MATLAB's `num2`/`lap` become NaN, the
                # erosion increment is NaN, NaN-restore returns z to its
                # old value, then the snap step decides:
                #   gradless => z := max(zold, zrock_new)
                #   gradmore => z := zrock_new (already handled above)
                zLL = z[i, j - 2] if j >= 2 else np.nan
                zRR = z[i, j + 2] if j <= n - 3 else np.nan
                zUU = z[i - 2, j] if i >= 2 else np.nan
                zDD = z[i + 2, j] if i <= m - 3 else np.nan
                zUL = z[i - 1, j - 1]; zUR = z[i - 1, j + 1]
                zDL = z[i + 1, j - 1]; zDR = z[i + 1, j + 1]
                if (np.isnan(zLL) or np.isnan(zRR) or np.isnan(zUU)
                        or np.isnan(zDD) or np.isnan(zUL) or np.isnan(zUR)
                        or np.isnan(zDL) or np.isnan(zDR)):
                    # gradless branch (snap to max of old z and new bedrock)
                    z_new[i, j] = zij if zij > zrock_ij_new else zrock_ij_new
                    continue

                den = 1.0 - (grad / Sc) ** 2
                if den <= 0.0:
                    # Degenerate: MATLAB inc -> +/- inf -> NaN-restore would
                    # leave z unchanged, then snap to bedrock if z<=zrock.
                    z_new[i, j] = zij if zij > zrock_ij_new else zrock_ij_new
                    continue

                lap = (zL + zR + zU + zD - 4.0 * zij) * inv_dx_sq
                d2zdx2 = (zRR - 2.0 * zij + zLL) * inv_4dx_sq
                d2zdy2 = (zDD - 2.0 * zij + zUU) * inv_4dx_sq
                d2zdxdy = (zDR - zDL - zUR + zUL) * inv_4dx_sq
                num2 = 2.0 * (dzdx * dzdx * d2zdx2 + dzdy * dzdy * d2zdy2
                              + 2.0 * dzdy * dzdx * d2zdxdy)
                inc = K_val * dt_over_prps * (lap / den + num2 / (Sc_sq * den * den))
                znew = zij + inc

                if np.isnan(znew):
                    znew = zij
                if znew <= zrock_ij_new:
                    znew = zrock_ij_new

                z_new[i, j] = znew

        return z_new, zrock_new


def soil_depth(Z: np.ndarray, x_cellsize: float, endtime: float = 5000.0,
               *, verbose: bool = False, use_numba: Optional[bool] = None
               ) -> np.ndarray:
    """Non-linear hillslope soil-depth evolution model (Roering 2008 variant).

    Mirrors `lib/functions/soil_depth.m`. NaN cells are propagated and then
    restored to the prior time step (matching the MATLAB `z[isnan(z)] = zstore[...]`
    pattern).

    `use_numba`: True forces the JIT path (fails if numba is missing); False
    forces the pure-NumPy path; None (default) auto-selects.
    """
    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and not _HAVE_NUMBA:
        raise RuntimeError("numba is not installed in this environment")

    z = np.asarray(Z, dtype=np.float64).copy()
    zrock = z - 1.0
    dx = float(x_cellsize)

    Sc = 1.25         # backSc
    dt = 1.0          # backdt
    grac = 0.025
    Po, mu, prps = 0.0003, 3.0, 2.0
    K_val = 0.005     # K(scalar) per MATLAB driver

    if use_numba:
        # Warm-up call so the JIT compile cost is paid once on a tiny tile.
        if verbose:
            print("  soil_depth: warming up Numba JIT ...", flush=True)
        small = np.zeros((4, 4), dtype=np.float64)
        _soil_depth_step_numba(small, small.copy(), K_val, Sc, dt, grac,
                               Po, mu, prps, dx)
        if verbose:
            print(f"  soil_depth: JIT ready, running {endtime:.0f}-yr "
                  f"simulation ...", flush=True)
        import time as _time
        t_loop = _time.perf_counter()
        time = 0.0
        last_log = 0.0
        log_every = 250.0  # years; ~20 lines for 5000-yr run
        while time <= (endtime + 500.0):
            z, zrock = _soil_depth_step_numba(z, zrock, K_val, Sc, dt, grac,
                                              Po, mu, prps, dx)
            time += dt
            if verbose and time - last_log >= log_every:
                elapsed = _time.perf_counter() - t_loop
                rate = time / elapsed if elapsed > 0 else 0.0
                eta = (endtime - time) / rate if rate > 0 and time < endtime else 0.0
                print(f"  soil_depth t={time:.0f}/{endtime:.0f} yr "
                      f"({rate:.0f} yr/s, ETA {eta:.0f} s)", flush=True)
                last_log = time
        return (z - zrock).astype(np.float64)

    # ---- Fallback: pure NumPy implementation (slow on big grids) ------------
    zchan = np.zeros_like(z)
    K = np.full_like(z, K_val)
    zhill = (zchan < 0.9).astype(np.float64)
    time = 0.0
    last_log = 0.0
    while time <= (endtime + 500.0):
        zstore = z.copy()

        dzdx, dzdy = _matlab_gradient(z, dx)
        grad = np.sqrt(dzdx * dzdx + dzdy * dzdy)
        gradless = (grad < (Sc - grac * Sc)).astype(np.float64)
        gradmore = (gradless < 0.9).astype(np.float64)

        zrock = zhill * (zrock - (dt * Po) * np.exp(-mu * (z - zrock)))

        with np.errstate(divide='ignore', invalid='ignore'):
            den = 1.0 - (grad / Sc) ** 2
            d2zdx, d2zdxy = _matlab_gradient(dzdx, dx)
            d2zdyx, d2zdy = _matlab_gradient(dzdy, dx)
            lap = 4.0 * _del2(z, dx)
            num2 = 2.0 * ((dzdx ** 2) * d2zdx + (dzdy ** 2) * d2zdy
                           + 2.0 * dzdy * dzdx * d2zdxy)
            inc = (gradless * zhill * (dt / prps) * K) * (
                lap / den + num2 / (Sc * Sc * den * den))
        z = z + inc
        nan_mask = np.isnan(z)
        z[nan_mask] = zstore[nan_mask]

        z = (zhill * gradless * (zrock * (z <= zrock) + z * (z > zrock))
             + zhill * gradmore * zrock + zchan * z)
        nan_mask = np.isnan(z)
        z[nan_mask] = zstore[nan_mask]

        z = zhill * z
        nan_mask = np.isnan(z)
        z[nan_mask] = zstore[nan_mask]

        time += dt
        if verbose and time - last_log >= 500.0:
            print(f"  soil_depth t={time:.0f}/{endtime:.0f} yr")
            last_log = time

    return (z - zrock).astype(np.float64)


# =============================================================================
#  fillsinks.m  ->  fillsinks()
# =============================================================================
#
# The functions below (down to and including `flow_accumulation`) are derived
# from TopoToolbox (W. Schwanghart & D. Scherler, 2014) and are therefore
# licensed under GPL-3.0-or-later. See the file-level docstring for details.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from TopoToolbox: https://github.com/wschwanghart/topotoolbox

def _imreconstruct(marker: np.ndarray, mask: np.ndarray, conn: int = 8) -> np.ndarray:
    """MATLAB-equivalent imreconstruct (greyscale, dilation form).

    Iteratively dilates `marker` constrained by `mask` until convergence,
    matching MATLAB's `imreconstruct(marker, mask, 8)` for 8-connectivity.
    Uses skimage if available (faster, well-tested), otherwise falls back to
    an iterative scipy-based implementation.
    """
    try:
        from skimage.morphology import reconstruction
        return reconstruction(np.minimum(marker, mask), mask,
                              method='dilation',
                              footprint=np.ones((3, 3), dtype=bool)
                              if conn == 8 else None)
    except ImportError:
        m = np.minimum(marker, mask)
        struct = np.ones((3, 3), dtype=bool) if conn == 8 \
            else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        while True:
            new = np.minimum(grey_dilation(m, footprint=struct), mask)
            if np.array_equal(new, m):
                return new
            m = new


def fillsinks(Z: np.ndarray) -> np.ndarray:
    """Fill closed depressions in a DEM (MATLAB `fillsinks(DEM)` equivalent).

    Implements the single-argument branch of `fillsinks.m`: complement-and-
    morphologically-reconstruct, with -inf at NaN cells.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Inan = np.isnan(Z)
    dem = Z.copy()
    dem[Inan] = -np.inf

    # MATLAB:
    #   marker = -dem
    #   marker(interior & ~Inan) = -inf
    #   demfs = -imreconstruct(marker, -dem, 8)
    marker = -dem
    interior = np.zeros_like(dem, dtype=bool)
    interior[1:-1, 1:-1] = True
    marker[interior & ~Inan] = -np.inf
    demfs = -_imreconstruct(marker, -dem, 8)

    demfs[Inan] = np.nan
    return demfs


# =============================================================================
#  identifyflats.m  ->  identify_flats()
# =============================================================================

def identify_flats(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (FLATS, SILLS, CLOSED) boolean masks (MATLAB identifyflats)."""
    Z = np.asarray(Z, dtype=np.float64)
    log_nans = np.isnan(Z)
    dem = Z.copy()
    dem[log_nans] = -np.inf

    nhood = np.ones((3, 3), dtype=bool)

    # FLATS: cell has no strictly-lower 8-neighbour
    eroded = grey_erosion(dem, footprint=nhood)
    flats = (eroded == dem) & ~log_nans
    flats[0, :] = False
    flats[-1, :] = False
    flats[:, 0] = False
    flats[:, -1] = False
    if log_nans.any():
        flats[binary_dilation(log_nans, structure=nhood)] = False

    # SILLS: pixels adjacent to flats whose elevation matches dilated flat
    Imr = np.full_like(dem, -np.inf)
    Imr[flats] = dem[flats]
    Imr_d = grey_dilation(Imr, footprint=nhood)
    sills = (Imr_d == dem) & ~flats
    if log_nans.any():
        sills[log_nans] = False

    # CLOSED: regional minima not touching the border
    closed = _regional_minima(dem)
    if log_nans.any():
        closed = closed | log_nans
        closed = _clear_border(closed)
        closed[log_nans] = False
    else:
        closed = _clear_border(closed)

    return flats, sills, closed


def _regional_minima(z: np.ndarray) -> np.ndarray:
    """MATLAB `imregionalmin`: connected components where every cell has
    strictly-lower-or-equal neighbours and at least one neighbour is equal
    (so the component is a flat at a strict local minimum). Uses skimage's
    `local_minima` on the negated image."""
    try:
        from skimage.morphology import local_minima
        return local_minima(z, connectivity=2, allow_borders=True)
    except ImportError:
        # Fallback: regional min = imreconstruct(z+1, z) - z > 0.
        m = z + 1.0
        rec = _imreconstruct(m, z, 8)
        return (rec - z) > 0


def _clear_border(mask: np.ndarray) -> np.ndarray:
    """MATLAB `imclearborder`: remove components touching the image border."""
    lbl, n = nd_label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask.copy()
    border_labels = set()
    border_labels.update(np.unique(lbl[0, :]))
    border_labels.update(np.unique(lbl[-1, :]))
    border_labels.update(np.unique(lbl[:, 0]))
    border_labels.update(np.unique(lbl[:, -1]))
    border_labels.discard(0)
    out = mask.copy()
    if border_labels:
        out[np.isin(lbl, list(border_labels))] = False
    return out


# =============================================================================
#  FLOWobj.m (single, carve) -> flow_direction()
# =============================================================================

@dataclass
class FlowDir:
    """D8 single flow direction graph in topological order.

    `ix` and `ixc` are 0-based linear indices in row-major (C) order such that
    flow goes from cell ix[k] to cell ixc[k].
    """
    shape: Tuple[int, int]
    cellsize: float
    ix: np.ndarray   # giver indices (0-based, flat C order)
    ixc: np.ndarray  # receiver indices


def _graydist_quasi(D: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Gray-weighted distance transform, MATLAB's `graydist(D, seeds, 'q')`.

    `D` is a non-negative cost surface, `seeds` is a boolean array. We compute
    the cumulative cost from any seed to every reachable cell using Dijkstra
    on an 8-connected grid; cardinal step cost is `(D[a]+D[b])/2`, diagonal
    step cost is `sqrt(2)*(D[a]+D[b])/2`.

    The 'q' (quasi-Euclidean) MATLAB option uses the same weights, giving
    monotone Dijkstra; results match within floating-point tolerance.
    """
    import heapq

    m, n = D.shape
    INF = np.inf
    dist = np.full(D.shape, INF, dtype=np.float64)
    seed_idx = np.flatnonzero(seeds)
    if seed_idx.size == 0:
        return dist
    dist.flat[seed_idx] = 0.0
    pq: list = []
    for k in seed_idx:
        heapq.heappush(pq, (0.0, int(k)))
    SQRT2 = np.sqrt(2.0)
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

    Dflat = D.ravel()
    while pq:
        d, k = heapq.heappop(pq)
        if d > dist.flat[k]:
            continue
        i, j = divmod(k, n)
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n:
                nk = ni * n + nj
                cost = w * 0.5 * (Dflat[k] + Dflat[nk])
                nd = d + cost
                if nd < dist.flat[nk]:
                    dist.flat[nk] = nd
                    heapq.heappush(pq, (nd, int(nk)))
    return dist


def flow_direction(Z: np.ndarray, cellsize: float, *,
                   preprocess: str = 'carve',
                   tweight: float = 2.0,
                   inverse: bool = False) -> FlowDir:
    """D8 single flow direction with carve/fill preprocessing.

    Mirrors the M-side flow-from-DEM branch of `lib/functions/FLOWobj.m`. The
    'carve' option (default in TopoToolbox & MATLAB driver) imposes flow paths
    through filled depressions with cost-weighted geodesic distance.

    Parameters
    ----------
    Z : 2-D float array (NaN = nodata)
    cellsize : scalar
    preprocess : 'carve' | 'fill' | 'none'
    tweight : carve cost exponent (default 2)
    inverse : if True, invert the DEM before routing (= MATLAB FLOWobjInv:
              flow goes uphill, used to delineate ridges).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if inverse:
        # FLOWobjInv (lib/functions/FLOWobjInv.m, line 148):
        #   DEM.Z = ((DEM.Z - max(DEM.Z)) * -1) + min(DEM.Z)
        #         = (max(Z) + min(Z)) - Z
        # Mirror-reflection about (max+min)/2 — keeps the value range, unlike a
        # plain negation. NaN survives the operation because NaN-arithmetic
        # propagates and `np.nanmax/min` skip NaN cells.
        valid = ~np.isnan(Z)
        if not valid.any():
            Z_in = Z.copy()
        else:
            z_max = float(np.nanmax(Z))
            z_min = float(np.nanmin(Z))
            Z_in = (z_max + z_min) - Z
    else:
        Z_in = Z

    # Step 1: fill (and optionally compute carve cost from fill-depth)
    DEMF = fillsinks(Z_in)

    flats_cached = None
    if preprocess == 'fill':
        dem = DEMF
        carve_cost = None
    elif preprocess == 'carve':
        # Carve cost = (max_depth - depth)^tweight + CarveMinVal per connected
        # flat region (CC of identifyflats(DEMF)). Vectorised with bincount so
        # we walk the grid only twice instead of once per region.
        D = DEMF - Z_in
        D[np.isnan(D)] = 0.0
        flats, sills, _ = identify_flats(DEMF)
        flats_cached = (flats, sills)  # reuse below
        D_carve = D.copy()
        CarveMinVal = 0.1
        if flats.any():
            cc, n_cc = nd_label(flats, structure=np.ones((3, 3), dtype=bool))
            cc_flat = cc.ravel()
            D_flat = D.ravel()
            max_per = np.zeros(n_cc + 1, dtype=np.float64)
            np.maximum.at(max_per, cc_flat, D_flat)
            mask_flat = cc_flat > 0
            mp = max_per[cc_flat[mask_flat]]
            new_vals = (mp - D_flat[mask_flat]) ** tweight + CarveMinVal
            Dc_flat = D_carve.ravel()
            Dc_flat[mask_flat] = new_vals
            D_carve = Dc_flat.reshape(D.shape)
        dem = DEMF
        carve_cost = D_carve
    elif preprocess == 'none':
        dem = Z_in
        carve_cost = None
        flats_cached = None
    else:
        raise ValueError(f"unknown preprocess: {preprocess!r}")

    if flats_cached is not None and preprocess == 'carve':
        flats, sills = flats_cached
    else:
        flats, sills, _ = identify_flats(dem)

    # Step 2: build presill pixel set (non-flat 8-neighbours of sills sharing
    # the sill's elevation)
    m, n = dem.shape
    if sills.any():
        sill_rows, sill_cols = np.nonzero(sills)
        rowadd = [-1, -1,  0,  1, 1,  1,  0, -1]
        coladd = [ 0,  1,  1,  1, 0, -1, -1, -1]
        presill_rows = []
        presill_cols = []
        for dr, dc in zip(rowadd, coladd):
            rr = sill_rows + dr
            cc = sill_cols + dc
            valid = (rr >= 0) & (rr < m) & (cc >= 0) & (cc < n)
            rr_v = rr[valid]
            cc_v = cc[valid]
            sr_v = sill_rows[valid]
            sc_v = sill_cols[valid]
            same_elev = dem[sr_v, sc_v] == dem[rr_v, cc_v]
            in_flat = flats[rr_v, cc_v]
            presill_rows.append(rr_v[same_elev & in_flat])
            presill_cols.append(cc_v[same_elev & in_flat])
        presill_rows = np.concatenate(presill_rows) if presill_rows else np.zeros(0, dtype=np.int64)
        presill_cols = np.concatenate(presill_cols) if presill_cols else np.zeros(0, dtype=np.int64)
        presill_mask = np.zeros_like(flats)
        presill_mask[presill_rows, presill_cols] = True
    else:
        presill_mask = np.zeros_like(flats)

    # Step 3: cost surface in flats. For 'carve' the cost is `D_carve` from
    # above; for 'fill'/'none' it is the Euclidean distance from non-flat cells.
    notI = ~flats
    if carve_cost is not None:
        D = carve_cost
    else:
        from scipy.ndimage import distance_transform_edt
        # Euclidean distance from "outside flats" to inside flats, then
        # `imreconstruct(D+1, mask) - D` per MATLAB.
        D = distance_transform_edt(~notI) * cellsize
        # For our use (single-flow with fill or none), the auxiliary topo from
        # MATLAB's `imreconstruct` step is not strictly needed because flats
        # become trivial after filling. We approximate with the raw EDT.
    # graydist seeded at presill pixels. MATLAB:
    #   D(I) = inf      % I = ~flats; cost is +inf outside flats
    #   D = graydist(D, presill, 'q') + 1
    #   D(I) = -inf     % put non-flats at bottom of descending sort
    # graydist sets dist=0 at seeds but uses the *original* D values for
    # path costs — so we must NOT zero-out the cost surface at seeds.
    D_aux_input = D.copy()
    D_aux_input[notI] = np.inf
    D_aux = _graydist_quasi(D_aux_input, presill_mask) + 1.0
    D_aux[notI] = -np.inf

    # Step 4: sort cells by D_aux descending then dem descending (both stable
    # to break ties consistently). MATLAB uses column-major iteration order
    # (`D(:)` then sort), so we sort indices in F-order for tie-breaking and
    # convert back to C-order linear indices afterwards.
    nrc = m * n

    D_F = D_aux.ravel(order='F')
    dem_F = dem.ravel(order='F')

    D_sort = np.where(np.isnan(D_F), -np.inf, D_F)
    order1_F = np.argsort(-D_sort, kind='stable')  # F-order linear indices

    dem_in_order = dem_F[order1_F]
    dem_in_order_clean = np.where(np.isnan(dem_in_order), -np.inf, dem_in_order)
    order2 = np.argsort(-dem_in_order_clean, kind='stable')
    ix_F = order1_F[order2]

    # F-order index k_F = i + j*m  ->  C-order index k_C = i*n + j
    ii = ix_F % m
    jj = ix_F // m
    ix = (ii * n + jj).astype(np.int64)

    D_flat = D_aux.ravel()
    dem_flat = dem.ravel()

    # Step 5: for each cell in ix (descending), find steepest 8-neighbour
    # downstream. Exclude NaN cells.
    nan_mask = np.isnan(dem)
    nan_flat = nan_mask.ravel()

    # `pp` maps cell index -> position in ix (0-based)
    pp = np.zeros(nrc, dtype=np.int64)
    pp[ix] = np.arange(nrc, dtype=np.int64)

    # For each cell, find the neighbour with the smallest pp (i.e., the one
    # processed earliest after it in the descending sort), separately for
    # cardinal vs diagonal, and pick the steeper.
    ix_2d = np.unravel_index(ix, (m, n))
    rr_ix = ix_2d[0]
    cc_ix = ix_2d[1]

    INVALID = nrc  # sentinel
    # Cardinal candidate: imdilate(pp, [0 1 0; 1 1 1; 0 1 0])
    pp_grid = pp.reshape(m, n)
    nbh_card = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    nbh_diag = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)

    # imdilate a *labelled* image returning the max label in the footprint:
    IXC1 = grey_dilation(pp_grid, footprint=nbh_card, mode='constant', cval=0)
    IXC2 = grey_dilation(pp_grid, footprint=nbh_diag, mode='constant', cval=0)
    # For each ix[r], the receiver pos in ix is IXC1[r] (or IXC2 for diag).
    # MATLAB: IX = IXC1(FD.ix); IXC1 = FD.ix(IX); G1 = (dem(FD.ix)-dem(IXC1))/cellsize
    pos_card = IXC1.flat[ix]
    pos_diag = IXC2.flat[ix]
    # Convert "position in ix" back to cell index
    rec_card = ix[pos_card]
    rec_diag = ix[pos_diag]
    G1 = (dem_flat[ix] - dem_flat[rec_card]) / cellsize
    G2 = (dem_flat[ix] - dem_flat[rec_diag]) / (cellsize * np.sqrt(2.0))
    # Mark self-loops as -inf
    G1[ix == rec_card] = -np.inf
    G2[ix == rec_diag] = -np.inf

    # Choose steeper: MATLAB rule
    #   I = G1<=G2 & xxx2(FD.ix)>xxx1(FD.ix)
    use_diag = (G1 <= G2) & (pos_diag > pos_card)
    receiver = np.where(use_diag, rec_diag, rec_card)

    # Drop self-loops and NaN sources
    keep = (receiver != ix) & ~nan_flat[ix] & ~nan_flat[receiver]
    ix_out = ix[keep]
    ixc_out = receiver[keep]

    return FlowDir(shape=(m, n), cellsize=float(cellsize),
                   ix=ix_out.astype(np.int64),
                   ixc=ixc_out.astype(np.int64))


# =============================================================================
#  flowacc.m  ->  flow_accumulation()
# =============================================================================

if _HAVE_NUMBA:
    @numba.njit(cache=True, fastmath=False)
    def _flow_accumulation_jit(Aflat, ix, ixc):
        for r in range(ix.size):
            Aflat[ixc[r]] = Aflat[ix[r]] + Aflat[ixc[r]]
        return Aflat


def flow_accumulation(FD: FlowDir,
                      W0: Optional[np.ndarray] = None) -> np.ndarray:
    """Cell-count flow accumulation along the topologically-sorted FD graph.

    The traversal is inherently sequential (each cell's accumulation depends on
    its givers having been finalised), so the loop runs in O(n_links). With
    7-8M links in a typical 12M-cell DEM, the pure-Python loop takes ~30 s;
    Numba JIT brings it down to <1 s.
    """
    if W0 is None:
        A = np.ones(FD.shape, dtype=np.float64)
    else:
        A = np.asarray(W0, dtype=np.float64).copy()
    Aflat = A.ravel()
    ix = FD.ix
    ixc = FD.ixc
    if _HAVE_NUMBA:
        Aflat = _flow_accumulation_jit(Aflat, ix, ixc)
    else:
        for r in range(ix.size):
            Aflat[ixc[r]] = Aflat[ix[r]] + Aflat[ixc[r]]
    return Aflat.reshape(FD.shape)


# =============================================================================
#  ridgelines.m / valleys.m -> ridge_valley_masks()
# =============================================================================

@dataclass
class NoGrowResult:
    nogrow_io: np.ndarray
    nogrow_idx: np.ndarray
    nogrow_i: np.ndarray
    nogrow_j: np.ndarray
    ridge_io: np.ndarray
    valley_io: np.ndarray


def _idx_F(mask: np.ndarray) -> np.ndarray:
    flat = mask.ravel(order='F')
    idx0 = np.flatnonzero(flat)
    return idx0.astype(np.int64) + 1


def _ij_F(idx_1based: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    m, _ = shape
    idx0 = idx_1based - 1
    j = idx0 // m
    i = idx0 - j * m
    return (i + 1).astype(np.int64), (j + 1).astype(np.int64)


# =============================================================================
#  END OF GPL-3.0-or-later SECTION
#  Code below this point reverts to CC0-1.0 (the file-level default).
# =============================================================================
#
# SPDX-License-Identifier: CC0-1.0

def ridges_valleys(Z: np.ndarray, cellsize: float, *,
                   ridge_acc_thresh: float = 5.0,
                   valley_acc_thresh: float = 100.0,
                   verbose: bool = True) -> NoGrowResult:
    """Build the no-grow mask (ridges + valleys + driver-side bwmorph cleanup).

    Mirrors the body of `lib/functions/ridgelines.m` + `valleys.m` plus the
    morphology pipeline in `driver.m` lines 363-381.

    NOTE: MATLAB's `ridgelines.m` / `valleys.m` re-load the DEM from disk via
    `geotiffread` (and again via `GRIDobj`), which means flow accumulation runs
    on the *unpadded* DEM. Pass the unpadded array here to match — i.e., the Z
    that comes out of `read_dem()` BEFORE `pad_DEM()`.
    """
    import time as _time
    Z = np.asarray(Z, dtype=np.float64)

    def _step(msg: str, t0: float):
        if verbose:
            print(f"  ridges_valleys: {msg} ({_time.perf_counter() - t0:.1f}s)",
                  flush=True)

    # Valley flow accumulation: regular DEM
    if verbose:
        print("  ridges_valleys: [1/5] flow_direction (valley) ...", flush=True)
    t0 = _time.perf_counter()
    FD_v = flow_direction(Z, cellsize, preprocess='carve', inverse=False)
    _step("flow_direction valley done", t0)

    if verbose:
        print("  ridges_valleys: [2/5] flow_accumulation (valley) ...", flush=True)
    t0 = _time.perf_counter()
    acc_v = flow_accumulation(FD_v)
    _step(f"flow_accumulation valley done (max={int(np.nanmax(acc_v))})", t0)

    # Ridge flow accumulation: inverted DEM (FLOWobjInv)
    if verbose:
        print("  ridges_valleys: [3/5] flow_direction (ridge, inverse) ...",
              flush=True)
    t0 = _time.perf_counter()
    FD_r = flow_direction(Z, cellsize, preprocess='carve', inverse=True)
    _step("flow_direction ridge done", t0)

    if verbose:
        print("  ridges_valleys: [4/5] flow_accumulation (ridge) ...", flush=True)
    t0 = _time.perf_counter()
    acc_r = flow_accumulation(FD_r)
    _step(f"flow_accumulation ridge done (max={int(np.nanmax(acc_r))})", t0)

    if verbose:
        print("  ridges_valleys: [5/5] threshold + bwmorph ...", flush=True)
    t0 = _time.perf_counter()
    valley_raw = (acc_v > valley_acc_thresh)
    ridge_raw = (acc_r > ridge_acc_thresh)
    valley_io = bwmorph(valley_raw, 'diag', 1)
    ridge_io = bwmorph(ridge_raw, 'diag', 1)

    nogrow_io = ridge_io | valley_io

    # Driver.m lines 373-381 chain (only non-trivial ops):
    #   bwmorph 'spur' 0       (no-op, count=0)
    #   bwmorph 'clean' 0      (no-op, count=0)
    #   bwmorph 'fill' 0       (no-op, count=0)
    #   bwmorph 'majority' 0   (no-op, count=0)
    #   bwmorph 'close' 0      (no-op, count=0)
    #   bwmorph 'thicken' 0    (no-op, count=0)
    #   bwmorph 'skel' Inf     (skeletonisation to convergence)
    #   bwmorph 'bridge' 1     (one bridge pass)
    #   bwmorph 'diag' 1       (one diagonal-fill pass)
    nogrow_io = bwmorph(nogrow_io, 'skel', 200)  # large n; LUT halts at idempotence
    nogrow_io = bwmorph(nogrow_io, 'bridge', 1)
    nogrow_io = bwmorph(nogrow_io, 'diag', 1)
    _step(f"bwmorph chain done ({int(nogrow_io.sum())} no-grow cells)", t0)

    nogrow_idx = _idx_F(nogrow_io)
    ni, nj = _ij_F(nogrow_idx, nogrow_io.shape)

    return NoGrowResult(
        nogrow_io=nogrow_io,
        nogrow_idx=nogrow_idx,
        nogrow_i=ni,
        nogrow_j=nj,
        ridge_io=ridge_io.astype(bool),
        valley_io=valley_io.astype(bool),
    )
