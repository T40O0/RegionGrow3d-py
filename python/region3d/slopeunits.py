"""Slope-unit delineation (Option 3 for no-grow mask).

Python approximation of Alvioli et al. r.slopeunits (GRASS GIS add-on):

  Alvioli, M., Marchesini, I., Reichenbach, P., Rossi, M., Ardizzone, F.,
  Fiorucci, F., & Guzzetti, F. (2016) "Automatic delineation of geomorpho-
  logical slope units with r.slopeunits v1.0 and their optimization for
  landslide susceptibility modeling" Geosci. Model Dev., 9, 3975-3991.

  Alvioli, M. et al. (2025) "Automatic optimization and delineation of nested
  slope units with r.slopeunits v2.0" Zenodo 15274445.

The original r.slopeunits is GPL-3.0 (GRASS GIS). This Python implementation
is an independent re-implementation of the published algorithm; per the
project-wide dual-license policy, it is licensed GPL-3.0-or-later to stay
compatible with the GRASS-derived flow-routing helpers it depends on.

  SPDX-License-Identifier: GPL-3.0-or-later

Pipeline
--------
1. fillsinks → D8 flow_direction → flow_accumulation
2. Channel mask: acc > thresh_cells
3. Half-basin propagation: each non-channel cell drains to a channel cell;
   the FIRST channel cell reached (via D8) defines the unit. The cell's
   "side" is determined by its 8-direction offset from that channel cell —
   so left bank and right bank get distinct unit IDs naturally.
4. Iterative refinement: per unit, compute the circular variance of aspect.
   If CV > cvmin AND area > 2*areamin, lower the channel threshold by
   `1/rf` and resegment within that unit. Stop after `maxiteration` passes
   or when no unit needs splitting.
5. Boundary extraction: cells where any 4-neighbor has a different unit ID
   are no-grow cells. Channel cells themselves are also no-grow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .preprocessing import (FlowDir, NoGrowResult, flow_accumulation,
                             flow_direction)
from .bwmorph import bwmorph


# =============================================================================
#  Aspect & circular variance
# =============================================================================

def _aspect(Z: np.ndarray, dx: float) -> np.ndarray:
    """Slope aspect in radians, MATLAB convention (atan2(-dz/dy, dz/dx)).

    NaN cells stay NaN. Returns values in [-π, π].
    """
    dzdy, dzdx = np.gradient(Z.astype(np.float64), dx)
    return np.arctan2(-dzdy, dzdx)


def _circular_variance_per_unit(aspect: np.ndarray, units: np.ndarray,
                                  n_units: int) -> np.ndarray:
    """1 − |mean(e^(i·θ))| per unit. Returns array of length n_units+1.

    Index 0 is reserved for "no unit" (channel / nodata) — its CV is 0.
    """
    valid = (units > 0) & np.isfinite(aspect)
    u = units[valid].ravel()
    a = aspect[valid].ravel()
    sin_sum = np.bincount(u, weights=np.sin(a), minlength=n_units + 1)
    cos_sum = np.bincount(u, weights=np.cos(a), minlength=n_units + 1)
    cnt = np.bincount(u, minlength=n_units + 1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / np.where(cnt > 0, cnt, 1.0)
    cv = 1.0 - R
    cv[cnt == 0] = 0.0
    return cv


def _area_per_unit(units: np.ndarray, cellsize: float,
                    n_units: int) -> np.ndarray:
    cnt = np.bincount(units[units > 0].ravel(),
                       minlength=n_units + 1).astype(np.float64)
    return cnt * (cellsize * cellsize)


# =============================================================================
#  Half-basin propagation
# =============================================================================

try:
    import numba
    _HAVE_NUMBA = True

    @numba.njit(cache=True, fastmath=False)
    def _propagate_upstream_jit(sub, ix, ixc, channel_flat):
        for r in range(ix.size - 1, -1, -1):
            g = ix[r]
            if sub[g] == 0 and not channel_flat[g]:
                sub[g] = sub[ixc[r]]
        return sub
except ImportError:
    _HAVE_NUMBA = False


def _assign_subbasins(FD: FlowDir, channel_flat: np.ndarray) -> np.ndarray:
    """Propagate subbasin IDs upstream from channel banks.

      * Channel cells stay ID 0 (treated as boundaries later).
      * Each non-channel cell whose immediate downstream neighbour is a
        channel cell becomes a "seed" with a fresh unique ID.
      * All other non-channel cells inherit the ID of their downstream
        neighbour via a reverse topological-order pass. Because FD is
        single-flow and ixc[r] is processed before ix[r] in reverse order,
        the inherited value is always already finalised.
    """
    n_cells = FD.shape[0] * FD.shape[1]
    sub = np.zeros(n_cells, dtype=np.int32)
    ix = FD.ix
    ixc = FD.ixc

    seed_mask = (~channel_flat[ix]) & channel_flat[ixc]
    seed_givers = ix[seed_mask]
    sub[seed_givers] = np.arange(1, seed_givers.size + 1, dtype=np.int32)

    if _HAVE_NUMBA:
        sub = _propagate_upstream_jit(sub, ix, ixc, channel_flat)
    else:
        for r in range(ix.size - 1, -1, -1):
            g = ix[r]
            if sub[g] == 0 and not channel_flat[g]:
                sub[g] = sub[ixc[r]]
    return sub.reshape(FD.shape)


# =============================================================================
#  Iterative refinement
# =============================================================================

def _renumber(units: np.ndarray) -> tuple[np.ndarray, int]:
    """Compact unit IDs to 1..K. Returns (renumbered, K)."""
    ids = np.unique(units)
    ids = ids[ids > 0]
    if ids.size == 0:
        return units.copy(), 0
    remap = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return remap[units], int(ids.size)


def _renumber_with_meta(units: np.ndarray,
                         parent: np.ndarray,
                         depth: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compact unit IDs and remap the parent / depth metadata arrays.

    Returns (renumbered_units, new_parent, new_depth, K).
    Parent IDs are translated through the same remap so the hierarchy stays
    consistent. Top-level units retain parent ID 0.
    """
    ids = np.unique(units)
    ids = ids[ids > 0]
    if ids.size == 0:
        return (units.copy(),
                np.zeros(1, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                0)
    remap = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    new_units = remap[units]
    K = int(ids.size)
    new_parent = np.zeros(K + 1, dtype=np.int32)
    new_depth = np.zeros(K + 1, dtype=np.int32)
    for old_id in ids:
        new_id = int(remap[old_id])
        if old_id < parent.size:
            old_parent = int(parent[old_id])
            # Map parent id through the same remap if it is still alive
            if 0 < old_parent < remap.size:
                new_parent[new_id] = int(remap[old_parent])
            new_depth[new_id] = (int(depth[old_id])
                                  if old_id < depth.size else 0)
    return new_units, new_parent, new_depth, K


def _merge_small_units(units: np.ndarray, cellsize: float, areamin: float,
                        max_passes: int = 8, verbose: bool = True,
                        parent: Optional[np.ndarray] = None,
                        depth: Optional[np.ndarray] = None,
                        ):
    """`r.slopeunits.clean` equivalent: merge sub-areamin units into the
    largest adjacent (positive-ID) neighbour, iterating until stable or
    `max_passes` is hit.

    Isolated small units (surrounded only by ID 0 / nodata) are left
    untouched because no merge target exists.

    If ``parent``/``depth`` are provided, the same per-iteration remap is
    applied to them so the hierarchy metadata stays consistent with the
    surviving unit IDs. The metadata of merged-out IDs is discarded; the
    receiving unit keeps its own ``parent`` and ``depth`` (its identity
    survives the merge, not the small unit's).

    Returns ``(units, n_units)`` when called without metadata (backward
    compatible) or ``(units, n_units, parent, depth)`` when metadata is
    provided.
    """
    units = units.astype(np.int32, copy=True)
    areamin_cells = max(1, int(np.ceil(float(areamin) / (cellsize * cellsize))))
    with_meta = parent is not None and depth is not None
    if with_meta:
        parent = parent.astype(np.int32, copy=True)
        depth = depth.astype(np.int32, copy=True)

    for it in range(int(max_passes)):
        max_id = int(units.max())
        if max_id == 0:
            break
        sizes = np.bincount(units.ravel(), minlength=max_id + 1)
        # Small = positive ID with cell count below threshold
        is_small = (sizes < areamin_cells)
        is_small[0] = False
        if not is_small.any():
            if verbose and it == 0:
                print("  merge_small_units: no small units present",
                      flush=True)
            break

        # Collect (small_id, neighbour_id) pairs from 4-edges where the
        # neighbour is a different POSITIVE unit. Including the neighbour's
        # size in the tiebreaker means we prefer the LARGEST adjacent unit.
        a, b = units[:-1, :], units[1:, :]
        diff_v = (a != b) & (a > 0) & (b > 0)
        ap, bp = a[diff_v], b[diff_v]
        a, b = units[:, :-1], units[:, 1:]
        diff_h = (a != b) & (a > 0) & (b > 0)
        ah, bh = a[diff_h], b[diff_h]

        # Symmetric: each pair contributes both (small, big) directions.
        src = np.concatenate([ap, bp, ah, bh]).astype(np.int64)
        dst = np.concatenate([bp, ap, bh, ah]).astype(np.int64)
        small_src = is_small[src]
        src = src[small_src]
        dst = dst[small_src]
        if src.size == 0:
            if verbose:
                print(f"  merge_small_units: pass {it+1}: "
                      "no merge candidates, stop", flush=True)
            break

        # Pick, for each small src, the destination with the LARGEST size.
        # Implemented by sorting (src asc, dst-size desc) and taking the
        # first row per src — this avoids building an explicit pair-count
        # table.
        dst_sizes = sizes[dst]
        order = np.lexsort((-dst_sizes, src))
        src_s = src[order]
        dst_s = dst[order]
        _, first = np.unique(src_s, return_index=True)
        merge_from = src_s[first]
        merge_to = dst_s[first]

        # Build a remap; identity for unaffected IDs.
        remap = np.arange(max_id + 1, dtype=np.int32)
        remap[merge_from] = merge_to.astype(np.int32)

        # Resolve transitive small→small→big chains in a single pass via
        # path compression on the remap array.
        for _ in range(int(max_passes)):
            new_remap = remap[remap]
            if np.array_equal(new_remap, remap):
                break
            remap = new_remap

        units_new = remap[units]
        n_merged = int(merge_from.size)
        if (units_new == units).all():
            if verbose:
                print(f"  merge_small_units: pass {it+1}: "
                      "fixed point reached", flush=True)
            break
        units = units_new
        if with_meta:
            # Children whose recorded parent was merged-out should now point
            # to the merge target. Other parent IDs unchanged. depth is the
            # receiver's depth, which is already what we keep.
            if parent.size <= max_id:
                # Grow parent/depth to cover all IDs we might index.
                grow = max_id + 1 - parent.size
                parent = np.concatenate(
                    [parent, np.zeros(grow, dtype=np.int32)])
                depth = np.concatenate(
                    [depth, np.zeros(grow, dtype=np.int32)])
            parent[1:max_id + 1] = remap[parent[1:max_id + 1]]
        if verbose:
            remaining_ids = int(np.unique(units[units > 0]).size)
            print(f"  merge_small_units: pass {it+1}: merged {n_merged} "
                  f"units, {remaining_ids} remain", flush=True)

    if with_meta:
        # Compact both units and metadata together so parent IDs stay valid.
        units, parent, depth, n_units = \
            _renumber_with_meta(units, parent, depth)
        return units, n_units, parent, depth
    units, n_units = _renumber(units)
    return units, n_units


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (i0, i1, j0, j1) slice bounds tight to True cells. (i1/j1 exclusive)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    i = np.where(rows)[0]
    j = np.where(cols)[0]
    return (int(i.min()), int(i.max()) + 1,
            int(j.min()), int(j.max()) + 1)


def _subdivide_unit_local(Z: np.ndarray, cellsize: float,
                            unit_mask: np.ndarray,
                            thresh_cells: float
                            ) -> tuple[np.ndarray, np.ndarray,
                                        tuple[int, int, int, int], int]:
    """Run a local slope-unit segmentation inside one parent unit.

    The DEM is sliced to the parent's bounding box and any cell outside
    ``unit_mask`` is set to NaN so flow routing cannot leak across the
    parent boundary. Flow accumulation, channel mask, and half-basin
    propagation are then computed on that local raster.

    Returns ``(sub_units_local, channel_local, bbox, n_sub_units)``:
      * ``sub_units_local``: int32, shape ``bbox`` slice. 0 = channel/outside.
      * ``channel_local``: bool, shape ``bbox`` slice.
      * ``bbox``: ``(i0, i1, j0, j1)`` slice bounds in the global array.
      * ``n_sub_units``: number of distinct positive IDs in ``sub_units_local``.

    If the parent unit has too few non-channel cells to seed any subbasin
    (e.g. it is mostly channel), returns 0 sub-units; caller should then
    skip this unit.
    """
    i0, i1, j0, j1 = _bbox_of_mask(unit_mask)
    if i1 <= i0 or j1 <= j0:
        return (np.zeros((0, 0), dtype=np.int32),
                np.zeros((0, 0), dtype=bool),
                (i0, i1, j0, j1), 0)

    Z_loc = Z[i0:i1, j0:j1].astype(np.float64).copy()
    mask_loc = unit_mask[i0:i1, j0:j1]
    Z_loc[~mask_loc] = np.nan

    # If the local patch is mostly NaN we still need at least one valid cell
    if not np.any(np.isfinite(Z_loc)):
        return (np.zeros(Z_loc.shape, dtype=np.int32),
                np.zeros(Z_loc.shape, dtype=bool),
                (i0, i1, j0, j1), 0)

    FD_loc = flow_direction(Z_loc, cellsize, preprocess='fill', inverse=False)
    acc_loc = flow_accumulation(FD_loc)
    channel_loc = (acc_loc > thresh_cells) & mask_loc
    sub_loc = _assign_subbasins(FD_loc, channel_loc.ravel())
    # Cells outside the parent mask must not receive an ID even if they
    # were transiently filled by flow_direction's padding behaviour.
    sub_loc[~mask_loc] = 0
    sub_loc, n_sub = _renumber(sub_loc)
    return sub_loc.astype(np.int32), channel_loc, (i0, i1, j0, j1), n_sub


@dataclass
class SlopeUnitResult:
    units: np.ndarray            # int32, 0 = channel/nodata
    n_units: int
    channel: np.ndarray          # bool
    iterations_used: int
    final_thresh_cells: float
    # ---- v2.0 nested-hierarchy metadata (length n_units+1, index 0 unused) --
    parent: Optional[np.ndarray] = None  # int32, 0 = top-level (no parent)
    depth: Optional[np.ndarray] = None   # int32, refinement depth per unit
    nested: bool = False                 # True when produced by v2.0 path


def slope_units(Z: np.ndarray, cellsize: float, *,
                 thresh: float = 500000.0,
                 areamin: float = 100000.0,
                 cvmin: float = 0.3,
                 rf: float = 2.0,
                 maxiteration: int = 10,
                 nested: bool = True,
                 verbose: bool = True) -> SlopeUnitResult:
    """Delineate slope units (half-basins) from a DEM.

    Parameters mirror r.slopeunits.create where reasonable:

    thresh : float
        Initial channel-defining flow-accumulation threshold, in m^2.
    areamin : float
        Minimum unit area, in m^2. Units smaller than 2*areamin are not
        subdivided further.
    cvmin : float in [0,1]
        Aspect circular-variance ceiling. Units with CV ≤ cvmin are
        considered aspect-homogeneous and stop subdividing.
    rf : float > 1
        Threshold reduction factor. Each iteration sets
        thresh ← thresh * (1 − 1/rf), so larger rf = gentler refinement.
    maxiteration : int
        Safety cap on the refinement loop.
    nested : bool
        ``True`` (default) selects the v2.0 "nested" behaviour: each
        split-target unit is re-segmented **locally** (flow routing
        recomputed inside the parent's bounding box, with cells outside the
        parent masked out) so sub-units cannot leak across the parent
        boundary. Parent and depth metadata is recorded in the returned
        :class:`SlopeUnitResult`.
        ``False`` reverts to the v1.0 behaviour: each iteration re-segments
        the whole DEM with the lowered threshold and adopts the new IDs only
        inside split-target units (faster, no hierarchy info).
    """
    import time as _time

    Z = np.asarray(Z, dtype=np.float64)
    cell_area = cellsize * cellsize
    thresh_cells = float(thresh) / cell_area
    areamin_cells = float(areamin) / cell_area

    if verbose:
        print(f"  slope_units: thresh={thresh:.0f} m^2 ({thresh_cells:.0f} cells), "
              f"areamin={areamin:.0f} m^2, cvmin={cvmin:.2f}, rf={rf:.2f}",
              flush=True)

    t0 = _time.perf_counter()
    if verbose:
        print("  slope_units: [1/5] flow_direction (fill preprocess) ...",
              flush=True)
    FD = flow_direction(Z, cellsize, preprocess='fill', inverse=False)
    if verbose:
        print(f"  slope_units: [2/5] flow_accumulation ... "
              f"(elapsed {_time.perf_counter() - t0:.1f}s)", flush=True)
    acc = flow_accumulation(FD)
    if verbose:
        print(f"  slope_units: flow_accumulation max={int(np.nanmax(acc))} "
              f"(elapsed {_time.perf_counter() - t0:.1f}s)", flush=True)

    aspect = _aspect(Z, cellsize)

    # Initial segmentation
    if verbose:
        print("  slope_units: [3/5] initial half-basin segmentation ...",
              flush=True)
    channel = acc > thresh_cells
    units = _assign_subbasins(FD, channel.ravel())
    units, n_units = _renumber(units)
    if verbose:
        print(f"  slope_units: initial pass → {n_units} units "
              f"(channel cells={int(channel.sum())})", flush=True)

    # Track parent (0 = top-level) and refinement depth per unit. Only used
    # when nested=True, but kept consistent always so renumbering stays cheap.
    parent = np.zeros(n_units + 1, dtype=np.int32)
    depth = np.zeros(n_units + 1, dtype=np.int32)

    # Iterative refinement
    label = "[4/5] iterative aspect-CV refinement"
    if nested:
        label += " (nested / v2.0)"
    if verbose:
        print(f"  slope_units: {label} ...", flush=True)
    it_used = 0
    for it in range(int(maxiteration)):
        cv = _circular_variance_per_unit(aspect, units, n_units)
        area = _area_per_unit(units, cellsize, n_units)
        to_split = (cv > cvmin) & (area > 2.0 * areamin)
        to_split[0] = False  # never split the "background" id
        n_split = int(to_split.sum())
        if n_split == 0:
            if verbose:
                print(f"  slope_units: iter {it+1}: no units to split, stop",
                      flush=True)
            break

        new_thresh_cells = thresh_cells * (1.0 - 1.0 / rf)
        if new_thresh_cells < 1.0:
            if verbose:
                print(f"  slope_units: iter {it+1}: thresh_cells "
                      f"{new_thresh_cells:.2f} too small, stop", flush=True)
            break
        thresh_cells = new_thresh_cells

        if nested:
            # ---- v2.0: subdivide each split-target unit *locally* -----------
            # IMPORTANT: We deliberately do NOT renumber unit IDs inside the
            # iteration loop. Renumbering collapses parent → child links
            # whenever the parent unit's cells get fully replaced by its
            # sub-units (the parent becomes an "internal node" with no cells,
            # which renumber treats as "dead" and remaps to 0). Keeping the
            # raw monotonically-increasing IDs preserves the hierarchy until
            # the very end of the function.
            max_old = int(units.max())
            next_id = max_old + 1
            new_parents: list[int] = []
            new_depths: list[int] = []
            n_actually_split = 0
            for uid in np.where(to_split)[0]:
                uid = int(uid)
                if uid == 0:
                    continue
                unit_mask = (units == uid)
                sub_loc, ch_loc, (i0, i1, j0, j1), n_sub = \
                    _subdivide_unit_local(Z, cellsize, unit_mask, thresh_cells)
                # Need at least 2 sub-units for a real split — otherwise the
                # local lowered threshold did not actually split this unit.
                if n_sub < 2:
                    continue
                target = units[i0:i1, j0:j1]
                channel_slice = channel[i0:i1, j0:j1]
                mask_slice = unit_mask[i0:i1, j0:j1]
                # Channel cells inside the parent become boundary (id 0)
                target[mask_slice & (sub_loc == 0)] = 0
                channel_slice |= ch_loc
                # Non-channel sub-cells get new global IDs (offset by next_id)
                positive = (sub_loc > 0) & mask_slice
                if positive.any():
                    target[positive] = sub_loc[positive] + (next_id - 1)
                # Write back (slices share memory but be explicit)
                units[i0:i1, j0:j1] = target
                channel[i0:i1, j0:j1] = channel_slice
                for _ in range(n_sub):
                    new_parents.append(uid)
                    new_depths.append(int(depth[uid]) + 1)
                next_id += n_sub
                n_actually_split += 1
            if new_parents:
                parent = np.concatenate(
                    [parent, np.array(new_parents, dtype=np.int32)])
                depth = np.concatenate(
                    [depth, np.array(new_depths, dtype=np.int32)])
            # Update n_units as the max ID (used as bincount minlength next
            # iteration). The actual leaf count is recomputed at the end.
            n_units = int(units.max())

            if verbose:
                leaf_count = int(np.unique(units[units > 0]).size)
                print(f"  slope_units: iter {it+1}: locally split "
                      f"{n_actually_split}/{n_split} units, "
                      f"new thresh_cells={thresh_cells:.1f}, "
                      f"leaf units={leaf_count}, max depth={int(depth.max())}",
                      flush=True)
            if n_actually_split == 0:
                if verbose:
                    print("  slope_units: no units actually split locally, "
                          "stop", flush=True)
                break
        else:
            # ---- v1.0: global resegment + selective adopt (unchanged) -------
            split_mask = to_split[units]      # bool, shape == Z
            channel_global = acc > thresh_cells
            sub_global = _assign_subbasins(FD, channel_global.ravel())

            max_old = int(units.max())
            merged = units.copy()
            inside = split_mask & (sub_global > 0)
            merged[inside] = sub_global[inside] + max_old
            merged[split_mask & (sub_global == 0)] = 0
            units, n_units = _renumber(merged)
            channel = channel_global | channel
            # v1 path produces no nesting metadata; keep parent/depth = 0
            parent = np.zeros(n_units + 1, dtype=np.int32)
            depth = np.zeros(n_units + 1, dtype=np.int32)

            if verbose:
                print(f"  slope_units: iter {it+1}: split {n_split} units, "
                      f"new thresh_cells={thresh_cells:.1f}, "
                      f"total units={n_units}", flush=True)
        it_used = it + 1

    # Final clean-up: merge sub-areamin units into largest neighbour, the
    # Python equivalent of GRASS's `r.slopeunits.clean`. This removes the
    # narrow units that would otherwise produce thick boundary blocks when
    # `ridges_valleys_slopeunits` extracts the no-grow mask.
    if verbose:
        print(f"  slope_units: [5/5] merge sub-areamin units (areamin="
              f"{areamin:.0f} m^2) ...", flush=True)
    if nested:
        units, n_units, parent, depth = _merge_small_units(
            units, cellsize, areamin, max_passes=8, verbose=verbose,
            parent=parent, depth=depth)
    else:
        units, n_units = _merge_small_units(units, cellsize, areamin,
                                              max_passes=8, verbose=verbose)

    if verbose:
        print(f"  slope_units: done in {_time.perf_counter()-t0:.1f}s, "
              f"{n_units} final units, {it_used} refinement iters"
              + (f", max depth={int(depth.max())}" if nested else ""),
              flush=True)

    return SlopeUnitResult(
        units=units, n_units=n_units, channel=channel,
        iterations_used=it_used, final_thresh_cells=thresh_cells,
        parent=parent if nested else None,
        depth=depth if nested else None,
        nested=nested,
    )


# =============================================================================
#  Boundary extraction → no-grow mask
# =============================================================================

def _idx_F(mask: np.ndarray) -> np.ndarray:
    flat = mask.ravel(order='F')
    return np.flatnonzero(flat).astype(np.int64) + 1


def _ij_F(idx_1based: np.ndarray, shape: tuple) -> tuple:
    m, _ = shape
    idx0 = idx_1based - 1
    j = idx0 // m
    i = idx0 - j * m
    return (i + 1).astype(np.int64), (j + 1).astype(np.int64)


def ridges_valleys_slopeunits(Z: np.ndarray, cellsize: float, *,
                                thresh: float = 25000.0,
                                areamin: float = 10000.0,
                                cvmin: float = 0.25,
                                rf: float = 2.0,
                                maxiteration: int = 10,
                                nested: bool = True,
                                verbose: bool = True) -> NoGrowResult:
    """Slope-unit boundary mask as a drop-in replacement for `ridges_valleys`.

    Returns the standard `NoGrowResult` so the rest of the driver pipeline
    needs no changes. `ridge_io` is the unit-boundary mask, `valley_io` is
    the channel mask (high-flow-accumulation cells) — together they form
    the no-grow region. Defaults to ``nested=True`` (v2.0 per-unit local
    subdivision); pass ``nested=False`` to fall back to the v1.0
    global-resegment variant.
    """
    su = slope_units(Z, cellsize, thresh=thresh, areamin=areamin,
                      cvmin=cvmin, rf=rf, maxiteration=maxiteration,
                      nested=nested, verbose=verbose)

    units = su.units
    # Boundary = any 4-neighbour has a different (positive) unit ID
    boundary = np.zeros_like(units, dtype=bool)
    diff_v = units[:-1, :] != units[1:, :]
    boundary[:-1, :] |= diff_v
    boundary[1:, :] |= diff_v
    diff_h = units[:, :-1] != units[:, 1:]
    boundary[:, :-1] |= diff_h
    boundary[:, 1:] |= diff_h
    nan_mask = ~np.isfinite(Z)
    boundary[nan_mask] = False

    valley_io = su.channel & ~nan_mask
    ridge_io = boundary & ~su.channel

    nogrow_io = ridge_io | valley_io
    # Apply the same bwmorph chain that option 1 uses for D8 ridges+valleys.
    # The merge-small-units pass inside slope_units() prevents narrow units
    # from creating thick boundary blocks, so skel+bridge+diag here produces
    # clean 1-pixel lines.
    nogrow_io = bwmorph(nogrow_io, 'skel', 200)
    nogrow_io = bwmorph(nogrow_io, 'bridge', 1)
    nogrow_io = bwmorph(nogrow_io, 'diag', 1)

    nogrow_idx = _idx_F(nogrow_io)
    ni, nj = _ij_F(nogrow_idx, nogrow_io.shape)

    if verbose:
        print(f"  ridges_valleys_slopeunits: {int(nogrow_io.sum())} no-grow "
              f"cells ({int(ridge_io.sum())} ridge + {int(valley_io.sum())} "
              "channel)", flush=True)

    return NoGrowResult(
        nogrow_io=nogrow_io,
        nogrow_idx=nogrow_idx,
        nogrow_i=ni,
        nogrow_j=nj,
        ridge_io=ridge_io.astype(bool),
        valley_io=valley_io.astype(bool),
    )
