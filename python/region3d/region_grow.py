"""Top-level RegionGrow3D function (port of `RegionGrowFxn.m`).

Mode A only: hydrostatic moisture, no seismic, a single shear-strength parameter pair (φ, c).
The signature mirrors the MATLAB function for ease of cross-checking, but only
the inputs the algorithm actually consumes are required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .matlab_compat import (find_F, to_F_index, sub2ind_F, ind2sub_F,
                            bwconncomp_F, get_F, set_F)
from .bwmorph import bwmorph, bwareaopen
from .forces import (interslice_force, interslice_force_prism,
                    force_closure_interslice, project_slope)
from .boundary import boundary_geometry_interslice, root_force_boundary
from .growth import (downhill_dilate, nogrow_not_eligible, spur_test,
                    continuity_check, update_cluster_interslice, erode_once,
                    update_cluster_compact)
from .localize import create_localized_rasters_interslice


@dataclass
class RegionGrowResult:
    slides_idx_initial: np.ndarray
    slides_idx_eroded: np.ndarray
    slides_idx_final: np.ndarray
    slides_initial_io: np.ndarray
    slides_eroded_io: np.ndarray
    slides_final_io: np.ndarray
    cluster_idx_initial: list
    cluster_idx_eroded: list
    cluster_idx_final: list
    diagnostics: dict


def region_grow_fxn(Z, coh, phi, gam_w, gam_dry, gam_sat, Gs, W, sigma_s,
                    sigma_s_wedge, PGA, reg_grow_on, err_percent_allowable,
                    max_growth_cycles, err_increase_thresh, cluster_size_thresh,
                    erosion_rounds, rot_range, rot_num, cleanup_rounds_initial,
                    cleanup_rounds_grow, x_cellsize, y_cellsize, x_ext, y_ext,
                    subslope, subaspect, subdx, subdy, nogrow_idx, depth, mw,
                    nogrow_mode, slope, aspect, dx, dy, nogrow_io, ridge_io,
                    valley_io, sus_i, susname, notnanidx, DEM_name, nogrow_i,
                    nogrow_j, root_mode, S_roots, S_roots_healthy, bs,
                    *, verbose=True, max_cell_offset=400) -> RegionGrowResult:
    """Run the RegionGrow3D algorithm. See `RegionGrowFxn.m` for the meanings of
    inputs/outputs. All raster inputs are 2D NumPy arrays with the same shape
    as Z; `*_idx` arguments are 1-based MATLAB column-major linear indices.
    """
    Z = np.asarray(Z, dtype=np.float32)
    shape = Z.shape

    # Per-cell scalar parameter rasters
    phi_full = np.full(shape, np.float32(phi), dtype=np.float32)
    coh_full = np.full(shape, np.float32(coh), dtype=np.float32)

    # ---- Zero-rotation forces ------------------------------------------------
    if verbose:
        print("Computing zero-rotation interslice forces...", end=' ', flush=True)
    Q0, Q_x0, Q_y0, _, area_col, _, _, _, _, _, _ = interslice_force(
        subdx, subdy, x_cellsize, y_cellsize, coh_full, phi_full, W, sigma_s,
        notnanidx, PGA)
    if verbose:
        print("done.", flush=True)

    # ---- Binary stability and cleanup ---------------------------------------
    FB_assign = (Q0 > 0)
    FB_assign = bwmorph(FB_assign, 'spur', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'bridge', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'fill', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'majority', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'spur', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'clean', cleanup_rounds_initial)
    FB_assign = bwmorph(FB_assign, 'spur', cleanup_rounds_initial)

    # Remove cells inside no-grow zones
    if nogrow_idx is not None and np.size(nogrow_idx) > 0:
        ng_arr = np.asarray(nogrow_idx, dtype=np.int64).ravel()
        if ng_arr.size > 0 and ng_arr[0] != 0:  # MATLAB sometimes uses 0 as sentinel
            ngi, ngj = to_F_index(ng_arr, shape)
            FB_assign[ngi, ngj] = False

    # Drop tiny clusters
    FB_assign = bwareaopen(FB_assign, cluster_size_thresh)

    # ---- Connected components -----------------------------------------------
    pixel_idx_list, num_objects, _labels = bwconncomp_F(FB_assign)
    if verbose:
        print(f"Found {num_objects} candidate clusters")

    # ---- Rotation forces ----------------------------------------------------
    rot = np.linspace(rot_range[0], rot_range[1], rot_num)
    Q_x_cell = []
    Q_y_cell = []
    if num_objects > 0:
        if verbose:
            print(f"Computing rotation forces (1..{len(rot)})...",
                  end=' ', flush=True)
        for i_rot, r in enumerate(rot):
            _, sx_rot, sy_rot = project_slope(subdx, subdy, r)
            _, qx, qy, *_ = interslice_force(
                sx_rot, sy_rot, x_cellsize, y_cellsize, coh_full, phi_full, W,
                sigma_s, notnanidx, PGA)
            Q_x_cell.append(np.asarray(qx, dtype=np.float32))
            Q_y_cell.append(np.asarray(qy, dtype=np.float32))
        if verbose:
            print("done.", flush=True)

    # ---- Allocate trackers --------------------------------------------------
    cluster_idx_initial = list(pixel_idx_list)  # 1-based linear indices in DEM
    cluster_idx_eroded = [None] * num_objects
    cluster_idx_final = [None] * num_objects

    # Width is max_growth_cycles+1: slot 0 is the pre-growth checkpoint and the
    # final growth cycle (when growth_cycles reaches max_growth_cycles) writes
    # slot max_growth_cycles, mirroring MATLAB's it_count(i)=max_growth_cycles+1
    # (1-based) implicit-grow write at RegionGrowFxn.m:662. A row is NEVER
    # cleared once a cluster starts — it persists across that cluster's
    # boundary-expansion retries exactly like MATLAB's preallocated
    # err_percent_store (allocated once at m:178, indexed per-cluster).
    err_percent_store = np.full((num_objects, max_growth_cycles + 1), np.nan,
                                dtype=np.float64)

    slides_initial_io = np.zeros(shape, dtype=bool)
    slides_eroded_io = np.zeros(shape, dtype=bool)
    slides_final_io = np.zeros(shape, dtype=bool)

    # ---- Main cluster loop --------------------------------------------------
    i = 0  # 0-based loop variable (MATLAB used 1..NumObjects)
    boundary_expand = 0
    cell_offset = 20
    # `max_cell_offset` caps the local-window half-size (cells). Boundary
    # expansion grows cell_offset by 20 each retry; without a ceiling a huge
    # cluster can balloon the localized rasters indefinitely (memory blow-up /
    # apparent hang). NOTE: MATLAB (RegionGrowFxn.m:697) retries unconditionally,
    # so a capped cluster (terminate_reason=7) diverges from MATLAB output.
    capped_clusters = 0
    diagnostics = {'terminate_reason': np.zeros(num_objects, dtype=np.int8),
                   'growth_cycles': np.zeros(num_objects, dtype=np.int32)}

    # Print roughly 100 progress lines per run regardless of cluster count.
    log_every = max(20, num_objects // 100)
    t_loop_start = time.perf_counter() if verbose else 0.0

    # Re-use a single full-DEM bool buffer for cluster_io (avoid per-iter alloc)
    cluster_io_full = np.zeros(shape, dtype=bool)

    while i < num_objects:
        if verbose and ((i + 1) % log_every == 0 or i + 1 == num_objects):
            elapsed = time.perf_counter() - t_loop_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta_s = (num_objects - (i + 1)) / rate if rate > 0 else 0.0
            print(f"LS Cluster {i+1}/{num_objects}  "
                  f"({rate:.1f} cl/s, ETA {eta_s/60:.1f} min)",
                  flush=True)

        cluster_idx_global = cluster_idx_initial[i]  # 1-based DEM linear indices
        cl_i_glob, cl_j_glob = ind2sub_F(shape, cluster_idx_global)
        # Toggle cluster_io_full in-place: set, use, then clear when done.
        set_F(cluster_io_full, cluster_idx_global, True)
        cluster_io = cluster_io_full

        # Set boundary offset. is_retry: we re-entered the SAME cluster after a
        # boundary-expansion (the checkpoint stores below must persist across
        # retries, matching MATLAB's once-allocated per-cluster rows).
        is_retry = (boundary_expand == 1)
        if boundary_expand == 1:
            cell_offset += 20
            boundary_expand = 0
        else:
            cell_offset = 20

        loc = create_localized_rasters_interslice(
            cell_offset, cl_i_glob, cl_j_glob, x_ext, y_ext, cluster_io, Z, W,
            depth, area_col, aspect, subaspect, phi_full, coh_full, Q0, Q_x0,
            Q_y0, Q_x_cell, Q_y_cell, rot, nogrow_mode, nogrow_io, ridge_io,
            valley_io, sigma_s_wedge, PGA, root_mode, bs)

        cluster_io_loc = loc.cluster_io_loc.copy()
        cluster_idx_loc = find_F(cluster_io_loc)
        cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)

        cluster_idx_loc_initial_i = cluster_idx_loc.copy()

        # Update cluster parameters (compact: only cluster_elev + W_sum used)
        cluster_elev, cluster_W_sum = update_cluster_compact(
            cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)

        # ---- Skip spur / discontinuous initial clusters ----------------------
        if spur_test(cl_i_loc, cl_j_loc) > 0 or continuity_check(cluster_io_loc) == 0:
            cluster_idx_eroded[i] = cluster_idx_global
            cluster_idx_final[i] = cluster_idx_global
            set_F(cluster_io_full, cluster_idx_global, False)
            i += 1
            continue

        # ---- Cluster erosion -------------------------------------------------
        for _ in range(erosion_rounds):
            cluster_io_loc, _ = erode_once(cluster_io_loc, cluster_elev, loc.Z_loc)
            cluster_idx_loc = find_F(cluster_io_loc)
            cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)
            cluster_elev, cluster_W_sum = update_cluster_compact(
                cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)

        # Reverse erosion if degenerate
        if (len(cluster_idx_loc) <= cluster_size_thresh
                or spur_test(cl_i_loc, cl_j_loc) > 0
                or continuity_check(cluster_io_loc) == 0):
            cluster_idx_loc = cluster_idx_loc_initial_i.copy()
            cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)
            cluster_io_loc = np.zeros(loc.Z_loc.shape, dtype=bool)
            set_F(cluster_io_loc, cluster_idx_loc, True)
            cluster_elev, cluster_W_sum = update_cluster_compact(
                cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)

        cluster_idx_loc_eroded_i = cluster_idx_loc.copy()

        # ---- Pre-growth force closure ---------------------------------------
        sum_W = float(np.nansum(loc.W_loc.ravel()[(cluster_idx_loc - 1)
                                                   .astype(np.int64)]
                                if False else
                                loc.W_loc[to_F_index(cluster_idx_loc, loc.Z_loc.shape)]))
        sum_depth_area = float(np.nansum(
            loc.depth_loc[to_F_index(cluster_idx_loc, loc.Z_loc.shape)]
            * x_cellsize * y_cellsize))
        gam = sum_W / sum_depth_area if sum_depth_area > 0 else 0.0

        try:
            (wedge_dir, wedge_subdr, wedge_subdx, wedge_subdy, wedge_width_avg,
             wedge_r_avg, wedge_V, wedge_W, wedge_U_pressure, wedge_depth,
             wedge_phi, wedge_coh, cluster_cenx, cluster_ceny, wedge_x_loc,
             wedge_y_loc, slide_dir, bound_dir_diff, wedge_k, wedge_kW,
             prism_dir, boundary_depth, wedge_bs) = boundary_geometry_interslice(
                loc.Q_x0_loc, loc.Q_y0_loc, cluster_idx_loc, cl_i_loc, cl_j_loc,
                cluster_io_loc, loc.x_ext_loc, loc.y_ext_loc, loc.Z_loc,
                loc.phi_loc, loc.coh_loc, loc.depth_loc, gam,
                loc.sigma_s_wedge_loc, loc.PGA_loc, i + 1, loc.bs_loc)
        except Exception as exc:
            if verbose:
                print(f"  cluster {i+1}: boundary geometry failed ({exc}); skipping",
                      flush=True)
            cluster_idx_eroded[i] = cluster_idx_global
            cluster_idx_final[i] = cluster_idx_global
            set_F(cluster_io_full, cluster_idx_global, False)
            i += 1
            continue

        wedge_Q, wedge_Q_x, wedge_Q_y = interslice_force_prism(
            wedge_subdx, wedge_subdy, wedge_width_avg, wedge_r_avg, wedge_W,
            wedge_phi, wedge_coh, wedge_U_pressure, wedge_kW)

        F_roots, Frx, Fry = root_force_boundary(
            slide_dir, prism_dir, boundary_depth, wedge_width_avg, S_roots,
            S_roots_healthy, wedge_bs)

        QX, QY, Q_mag, err_x, err_y, err_mag = force_closure_interslice(
            rot, loc.Q_x_cell_loc, loc.Q_y_cell_loc, wedge_Q_x, wedge_Q_y,
            cluster_idx_loc)

        skip_slide = False
        if F_roots > Q_mag.max():
            skip_slide = True
            cluster_idx_loc = np.zeros(0, dtype=np.int64)
            cluster_io_loc[:, :] = False
        else:
            QX = QX - Frx
            QY = QY + Fry
            err_mag = err_mag - F_roots
            err_x = err_x - Frx
            err_y = err_y + Fry

        # Pre-size store as a list of None so that, like MATLAB's cell array,
        # we can write to a specific slot (and overwrite when the algorithm
        # rolls back without incrementing it_count). It is allocated fresh only
        # for a NEW cluster; on a boundary-expansion retry it persists (matching
        # MATLAB's cluster_idx_loc_store, allocated once at m:177) so the
        # end-of-cluster checkpoint pick can still see the previous attempt's
        # stored clusters. Slot 0 is always overwritten with this attempt's
        # eroded cluster (mirrors the m:436 write at it_count=1 each attempt).
        if not is_retry:
            cluster_idx_loc_store = [None] * (max_growth_cycles + 1)
        cluster_idx_loc_store[0] = cluster_idx_loc.copy()
        err_allow = (err_percent_allowable / 100.0) * cluster_W_sum
        cluster_W_sum_wedges = cluster_W_sum + float(np.nansum(wedge_W))
        # Degenerate cluster (all weights NaN/zero) — skip rather than crash.
        if cluster_W_sum_wedges <= 0.0:
            cluster_idx_eroded[i] = cluster_idx_global
            cluster_idx_final[i] = cluster_idx_global
            set_F(cluster_io_full, cluster_idx_global, False)
            i += 1
            continue
        err_percent_store[i, 0] = float(err_mag.min()) / cluster_W_sum_wedges * 100.0

        # ---- Region grow loop ----------------------------------------------
        terminate_growth = False
        term_reason = 0
        growth_cycles = 0
        growth_cycles_at_expand = 0  # cycles completed when a window-expand fired
        it_count = 0  # 0-based slot already used

        if reg_grow_on:
            while (err_mag.min() > err_allow and not skip_slide
                   and not terminate_growth and boundary_expand == 0):
                add_cells_idx, _bio, _bidx, _bi, _bj, boundary_expand = \
                    downhill_dilate(loc.Z_loc, cluster_io_loc, cluster_elev, boundary_expand)
                if boundary_expand == 1:
                    # Remember the cycle count so the capped fall-through below
                    # doesn't record growth_cycles=0 for a well-grown cluster.
                    growth_cycles_at_expand = growth_cycles
                    growth_cycles = 0
                    continue

                add_cells_idx, _ = nogrow_not_eligible(add_cells_idx, loc.nogrow_idx_loc)
                # Drop NaN-force cells
                if add_cells_idx.size:
                    Q0_vals = loc.Q0_loc[to_F_index(add_cells_idx, loc.Z_loc.shape)]
                    add_cells_idx = add_cells_idx[~np.isnan(Q0_vals)]

                if add_cells_idx.size == 0:
                    terminate_growth = True
                    term_reason = 2
                    err_finite = err_percent_store[i, :]
                    cp_idx = int(np.nanargmin(err_finite))
                    stored = cluster_idx_loc_store[cp_idx]
                    cluster_idx_loc = stored.copy() if stored is not None else cluster_idx_loc
                    cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)
                    cluster_io_loc = np.zeros(loc.Z_loc.shape, dtype=bool)
                    set_F(cluster_io_loc, cluster_idx_loc, True)
                    cluster_elev, cluster_W_sum = update_cluster_compact(
                        cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)
                else:
                    it_count += 1
                    growth_cycles += 1

                    # Multi-cell growth selection
                    err_x_grow = np.zeros(add_cells_idx.size + 1)
                    err_y_grow = np.zeros(add_cells_idx.size + 1)
                    err_mag_grow = np.zeros(add_cells_idx.size + 1)
                    err_W_grow = np.zeros(add_cells_idx.size + 1)
                    err_percent_grow = np.zeros(add_cells_idx.size + 1)

                    minerr = int(np.argmin(err_mag))

                    err_x_grow[0] = err_x[minerr]
                    err_y_grow[0] = err_y[minerr]
                    err_mag_grow[0] = err_mag[minerr]
                    err_W_grow[0] = cluster_W_sum + float(np.nansum(wedge_W))
                    err_percent_grow[0] = err_mag.min() / err_W_grow[0] * 100.0

                    elig_i0, elig_j0 = to_F_index(add_cells_idx, loc.Z_loc.shape)
                    Qx_elig = loc.Q_x_cell_loc[minerr][elig_i0, elig_j0].astype(np.float64).copy()
                    Qy_elig = loc.Q_y_cell_loc[minerr][elig_i0, elig_j0].astype(np.float64).copy()

                    growth_list = [None]  # MATLAB's growth_list(1)=0 sentinel
                    add_cells_idx_remaining = list(add_cells_idx)
                    Qx_elig_list = list(Qx_elig)
                    Qy_elig_list = list(Qy_elig)

                    gac = 0
                    while add_cells_idx_remaining:
                        gac += 1
                        ex_diff = err_x_grow[gac - 1] - np.asarray(Qx_elig_list)
                        ey_diff = err_y_grow[gac - 1] - np.asarray(Qy_elig_list)
                        em_diff = np.hypot(ex_diff, ey_diff)
                        best = int(np.argmin(em_diff))

                        err_x_grow[gac] = err_x_grow[gac - 1] + Qx_elig_list[best]
                        err_y_grow[gac] = err_y_grow[gac - 1] + Qy_elig_list[best]
                        err_mag_grow[gac] = np.hypot(err_x_grow[gac], err_y_grow[gac])
                        bi0, bj0 = to_F_index(np.array([add_cells_idx_remaining[best]]),
                                              loc.Z_loc.shape)
                        err_W_grow[gac] = err_W_grow[gac - 1] + float(loc.W_loc[bi0[0], bj0[0]])
                        err_percent_grow[gac] = err_mag_grow[gac] / err_W_grow[gac] * 100.0

                        growth_list.append(add_cells_idx_remaining[best])
                        del add_cells_idx_remaining[best]
                        del Qx_elig_list[best]
                        del Qy_elig_list[best]

                    # Find lowest-error point during this growth cycle
                    minerr_grow_idx = int(np.argmin(err_percent_grow[:gac + 1]))

                    # Clip growth list: keep [1 .. minerr_grow_idx], drop the leading
                    # MATLAB sentinel (index 0) and anything after the optimum.
                    final_growth_list = growth_list[1:minerr_grow_idx + 1]

                    # MATLAB RegionGrowFxn.m:578-579:
                    #   isempty(growth_list) || err_percent_store(i,it_count-1)
                    #                            < err_percent_store(i,it_count)
                    # The current slot (it_count) is written LATER this cycle
                    # (below), so on a cluster's FIRST attempt it is still NaN
                    # and the `<` is False. On a boundary-expansion RETRY the
                    # row is not cleared, so slot it_count holds a stale finite
                    # value from the previous (smaller-window) attempt and the
                    # comparison can fire term_reason=3 (error increased vs the
                    # prior attempt). NaN comparisons are False in NumPy too, so
                    # this expression reproduces MATLAB on both first pass and
                    # retry.
                    err_increased = (err_percent_store[i, it_count - 1]
                                     < err_percent_store[i, it_count])
                    if len(final_growth_list) == 0 or err_increased:
                        terminate_growth = True
                        term_reason = 3

                    cluster_idx_loc = np.concatenate([cluster_idx_loc,
                                                      np.asarray(final_growth_list, dtype=np.int64)])
                    cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)
                    cluster_io_loc = np.zeros(loc.Z_loc.shape, dtype=bool)
                    set_F(cluster_io_loc, cluster_idx_loc, True)

                    cluster_elev, cluster_W_sum = update_cluster_compact(
                        cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)

                    sum_W = float(np.nansum(loc.W_loc[to_F_index(cluster_idx_loc, loc.Z_loc.shape)]))
                    sum_depth_area = float(np.nansum(
                        loc.depth_loc[to_F_index(cluster_idx_loc, loc.Z_loc.shape)]
                        * x_cellsize * y_cellsize))
                    gam = sum_W / sum_depth_area if sum_depth_area > 0 else 0.0

                    try:
                        (wedge_dir, wedge_subdr, wedge_subdx, wedge_subdy,
                         wedge_width_avg, wedge_r_avg, wedge_V, wedge_W,
                         wedge_U_pressure, wedge_depth, wedge_phi, wedge_coh,
                         cluster_cenx, cluster_ceny, wedge_x_loc, wedge_y_loc,
                         slide_dir, bound_dir_diff, wedge_k, wedge_kW, prism_dir,
                         boundary_depth, wedge_bs) = boundary_geometry_interslice(
                            loc.Q_x0_loc, loc.Q_y0_loc, cluster_idx_loc, cl_i_loc,
                            cl_j_loc, cluster_io_loc, loc.x_ext_loc, loc.y_ext_loc,
                            loc.Z_loc, loc.phi_loc, loc.coh_loc, loc.depth_loc, gam,
                            loc.sigma_s_wedge_loc, loc.PGA_loc, i + 1, loc.bs_loc)
                    except Exception:
                        terminate_growth = True
                        term_reason = 5
                        break

                    wedge_Q, wedge_Q_x, wedge_Q_y = interslice_force_prism(
                        wedge_subdx, wedge_subdy, wedge_width_avg, wedge_r_avg,
                        wedge_W, wedge_phi, wedge_coh, wedge_U_pressure, wedge_kW)
                    F_roots, Frx, Fry = root_force_boundary(
                        slide_dir, prism_dir, boundary_depth, wedge_width_avg,
                        S_roots, S_roots_healthy, wedge_bs)

                # Stability check
                QX, QY, Q_mag, err_x, err_y, err_mag = force_closure_interslice(
                    rot, loc.Q_x_cell_loc, loc.Q_y_cell_loc, wedge_Q_x, wedge_Q_y,
                    cluster_idx_loc)
                QX = QX - Frx
                QY = QY + Fry
                err_mag = err_mag - F_roots
                err_x = err_x - Frx
                err_y = err_y + Fry

                if it_count <= max_growth_cycles:
                    # Slot assignment mirrors MATLAB cell-array indexing. The
                    # bound is inclusive so the final growth cycle (it_count ==
                    # max_growth_cycles, i.e. MATLAB's it_count+1 slot) is
                    # stored and can win the end-of-cluster nanargmin — matching
                    # RegionGrowFxn.m:658-664.
                    cluster_idx_loc_store[it_count] = cluster_idx_loc.copy()
                    cluster_W_sum_wedges = cluster_W_sum + float(np.nansum(wedge_W))
                    if cluster_W_sum_wedges <= 0.0:
                        # Cluster degenerated mid-growth — stop growing it
                        terminate_growth = True
                        term_reason = 6
                    else:
                        err_percent_store[i, it_count] = (
                            float(err_mag.min()) / cluster_W_sum_wedges * 100.0)

                err_allow = (err_percent_allowable / 100.0) * cluster_W_sum

                if growth_cycles >= max_growth_cycles:
                    terminate_growth = True
                    term_reason = 4
                    cp_idx = int(np.nanargmin(err_percent_store[i, :]))
                    stored = cluster_idx_loc_store[cp_idx]
                    if stored is not None:
                        cluster_idx_loc = stored.copy()
                        cl_i_loc, cl_j_loc = ind2sub_F(cluster_io_loc.shape, cluster_idx_loc)
                        cluster_io_loc = np.zeros(loc.Z_loc.shape, dtype=bool)
                        set_F(cluster_io_loc, cluster_idx_loc, True)
                        cluster_elev, cluster_W_sum = update_cluster_compact(
                            cluster_idx_loc, loc.Z_loc, loc.W_loc, 0)

            if boundary_expand == 1:
                # Re-do same cluster with a bigger local window — but cap the
                # window so a pathological cluster can't expand without bound
                # (runaway memory / apparent freeze). Beyond the cap, accept the
                # current growth state instead of expanding further.
                if cell_offset < max_cell_offset:
                    # Do NOT clear err_percent_store[i, :] here: MATLAB keeps
                    # the per-cluster row (and cluster_idx_loc_store row, kept
                    # via `is_retry` above) across boundary-expansion retries,
                    # so stale slots from the previous attempt remain visible to
                    # the term_reason=3 comparison and the end-of-cluster
                    # nanargmin. Clearing would diverge from RegionGrowFxn.m.
                    continue
                boundary_expand = 0
                growth_cycles = growth_cycles_at_expand
                term_reason = 7          # boundary-expansion capped
                capped_clusters += 1
                if verbose:
                    print(f"  WARNING: cluster {i+1} hit the boundary-expansion "
                          f"cap (cell_offset={cell_offset}); accepting truncated "
                          f"growth (terminate_reason=7). Raise --max_cell_offset "
                          f"to grow it fully.", flush=True)

        diagnostics['terminate_reason'][i] = term_reason
        diagnostics['growth_cycles'][i] = growth_cycles

        # MATLAB (RegionGrowFxn.m:708-709) takes the CURRENT cluster_idx_loc as
        # the final cluster; the lowest-percent-error checkpoint is only
        # restored INSIDE the loop for term_reason 2 (no eligible cells) and 4
        # (max cycles) — both handled above. For normal convergence and
        # term_reason 3, MATLAB keeps the last-grown state, so we must NOT do a
        # post-loop restore here (an earlier unconditional restore diverged from
        # MATLAB on every normally-converged cluster whose row minimum was not
        # the last slot).

        # ---- Map back to global coordinates --------------------------------
        loc_i_min = int(loc.loc_i.min())
        loc_j_min = int(loc.loc_j.min())

        cl_i_loc_eroded, cl_j_loc_eroded = ind2sub_F(loc.Z_loc.shape, cluster_idx_loc_eroded_i)
        cl_i_eroded_glob = cl_i_loc_eroded + loc_i_min - 1
        cl_j_eroded_glob = cl_j_loc_eroded + loc_j_min - 1
        cluster_idx_eroded[i] = sub2ind_F(shape, cl_i_eroded_glob, cl_j_eroded_glob)

        cl_i_loc_final, cl_j_loc_final = ind2sub_F(loc.Z_loc.shape, cluster_idx_loc)
        cl_i_final_glob = cl_i_loc_final + loc_i_min - 1
        cl_j_final_glob = cl_j_loc_final + loc_j_min - 1
        cluster_idx_final[i] = sub2ind_F(shape, cl_i_final_glob, cl_j_final_glob)

        set_F(slides_initial_io, cluster_idx_initial[i], True)
        set_F(slides_eroded_io, cluster_idx_eroded[i], True)
        set_F(slides_final_io, cluster_idx_final[i], True)

        # Clear the full-DEM bool buffer for the next iteration. We touch only
        # the cells we just set so this is O(cluster size) not O(DEM).
        set_F(cluster_io_full, cluster_idx_global, False)

        i += 1

    if capped_clusters and verbose:
        print(f"  WARNING: {capped_clusters} cluster(s) hit the boundary-"
              f"expansion cap (max_cell_offset={max_cell_offset}); their growth "
              f"was truncated (terminate_reason=7) and diverges from MATLAB "
              f"there.", flush=True)

    return RegionGrowResult(
        slides_idx_initial=find_F(slides_initial_io),
        slides_idx_eroded=find_F(slides_eroded_io),
        slides_idx_final=find_F(slides_final_io),
        slides_initial_io=slides_initial_io,
        slides_eroded_io=slides_eroded_io,
        slides_final_io=slides_final_io,
        cluster_idx_initial=cluster_idx_initial,
        cluster_idx_eroded=cluster_idx_eroded,
        cluster_idx_final=cluster_idx_final,
        diagnostics=diagnostics,
    )
