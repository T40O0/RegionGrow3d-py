"""Cluster boundary geometry.

Implements `polygeom`, `boundary_geometry_interslice`, `root_force_boundary`
plus the `bwboundaries(BW, 'N')` and MATLAB `boundary(X, Y)` (alpha-shape)
helpers that the geometry functions depend on.

Design choice for `cluster_centroid`: we follow the MATLAB code in computing
the centroid as the centroid of the boundary polygon (`polygeom(X(BD), Y(BD))`)
returned by MATLAB's alpha-shape `boundary`. We replicate this as faithfully
as possible using `scipy.spatial.Delaunay` with the same default shrink factor
(0.5).
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import atan2d, cosd, sind

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @_numba.njit(cache=True, fastmath=False)
    def _alpha_shape_critical_alpha(order, simplices, neighbors, R, finite,
                                     n_points):
        """Find the critical alpha radius via union-find.

        Returns the smallest R such that the union of triangles with
        circumradius <= R covers all points AND forms a single connected
        component (sharing Delaunay edges). If no such R exists within the
        finite triangles, returns -1.0 (caller falls back to R_max).
        """
        n_tri = simplices.shape[0]
        parent = np.arange(n_tri, dtype=np.int64)
        in_set = np.zeros(n_tri, dtype=np.bool_)
        point_in = np.zeros(n_points, dtype=np.bool_)
        n_in = 0
        n_components = 0

        for k in range(order.size):
            ti = order[k]
            if not finite[ti]:
                break
            in_set[ti] = True
            for vk in range(3):
                v = simplices[ti, vk]
                if not point_in[v]:
                    point_in[v] = True
                    n_in += 1
            n_components += 1
            for kk in range(3):
                nb = neighbors[ti, kk]
                if nb >= 0 and in_set[nb]:
                    # Inline find with path-halving
                    ra = ti
                    while parent[ra] != ra:
                        parent[ra] = parent[parent[ra]]
                        ra = parent[ra]
                    rb = nb
                    while parent[rb] != rb:
                        parent[rb] = parent[parent[rb]]
                        rb = parent[rb]
                    if ra != rb:
                        parent[ra] = rb
                        n_components -= 1
            if n_in == n_points and n_components == 1:
                return R[ti]
        return -1.0

    @_numba.njit(cache=True, fastmath=False)
    def _alpha_shape_circumradii(simplices, pts):
        """Vectorisable inner loop: per-triangle circumradius."""
        n_tri = simplices.shape[0]
        R = np.empty(n_tri, dtype=np.float64)
        for ti in range(n_tri):
            i0 = simplices[ti, 0]
            i1 = simplices[ti, 1]
            i2 = simplices[ti, 2]
            ax = pts[i0, 0]; ay = pts[i0, 1]
            bx = pts[i1, 0]; by = pts[i1, 1]
            cx = pts[i2, 0]; cy = pts[i2, 1]
            a = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
            b = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
            c = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            sp = 0.5 * (a + b + c)
            v = sp * (sp - a) * (sp - b) * (sp - c)
            if v <= 0.0:
                R[ti] = np.inf
            else:
                area = np.sqrt(v)
                R[ti] = a * b * c / (4.0 * area)
        return R


# ---- bwboundaries -------------------------------------------------------------
# Moore-neighbor tracing with Jacob's stopping criterion, mirroring MATLAB's
# bwboundaries(BW, 'N'). Returns 1-based (i, j) coordinates that form a closed
# loop; the first and last points coincide (MATLAB convention).

# Clockwise neighbor offsets starting at "left" of the previous direction.
_MOORE_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]  # N, NE, E, SE, S, SW, W, NW


def _trace_outer_boundary(bw):
    """Trace the outer boundary of a single connected blob in `bw`.

    Returns
    -------
    boundary : np.ndarray of shape (K, 2), 1-based (i, j) MATLAB-style coords.
        First and last rows are the same (closed loop), matching MATLAB.
    """
    bw = np.asarray(bw, dtype=bool)
    if not bw.any():
        return np.zeros((0, 2), dtype=np.int64)

    # Find starting pixel: scan in column-major order, top-to-bottom within
    # each column, mirroring MATLAB's bwboundaries which begins with the first
    # column-major foreground pixel.
    flat_F = bw.ravel(order='F')
    first_F = int(np.argmax(flat_F))
    m = bw.shape[0]
    start_j = first_F // m
    start_i = first_F - start_j * m
    start = (start_i, start_j)

    # Initial direction: arrived from "above" (i-1). The first neighbor to
    # check is the one to the right of the entry direction.
    prev = (start_i - 1, start_j)  # virtual previous pixel
    # Determine which Moore offset corresponds to `prev - start`
    prev_offset = (prev[0] - start[0], prev[1] - start[1])  # (-1, 0)
    start_dir = _MOORE_OFFSETS.index(prev_offset)

    boundary = [start]
    current = start
    direction = start_dir

    # Jacob's stopping criterion: stop when we are about to exit the start pixel
    # via the same direction as the first move out of it.
    first_next = None  # (next_pixel, exit_direction) for stopping check

    while True:
        # Examine the 8 neighbors clockwise starting from `direction + 1` (next
        # neighbor after the one we came from).
        found = False
        for k in range(1, 9):
            d = (direction + k) % 8
            di, dj = _MOORE_OFFSETS[d]
            ni, nj = current[0] + di, current[1] + dj
            if 0 <= ni < bw.shape[0] and 0 <= nj < bw.shape[1] and bw[ni, nj]:
                next_pixel = (ni, nj)
                # The new "came-from" direction is opposite of how we left
                next_direction = (d + 4) % 8
                if first_next is None:
                    first_next = (next_pixel, d)
                elif current == start and (next_pixel, d) == first_next:
                    # Returned to start and would exit the same way: done.
                    return _close_loop(boundary)
                boundary.append(next_pixel)
                current = next_pixel
                direction = next_direction
                found = True
                break
        if not found:
            # Isolated pixel
            return _close_loop(boundary)


def _close_loop(boundary):
    """Append the start point so the loop is closed (matches MATLAB convention)."""
    arr = np.asarray(boundary, dtype=np.int64)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if not (arr[0, 0] == arr[-1, 0] and arr[0, 1] == arr[-1, 1]):
        arr = np.vstack([arr, arr[0:1]])
    return arr + 1  # to MATLAB 1-based


def bwboundaries_outer(bw):
    """Trace the outer boundaries of every 8-connected component, like
    MATLAB bwboundaries(bw, 'N'). Returns a list of (K, 2) int arrays of 1-based
    (i, j) coordinates.
    """
    from scipy.ndimage import label

    bw = np.asarray(bw, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    lbl, n = label(bw, structure=structure)
    out = []
    for k in range(1, n + 1):
        comp = (lbl == k)
        out.append(_trace_outer_boundary(comp))
    return out


# ---- polygeom -----------------------------------------------------------------
def polygeom(x, y):
    """MATLAB polygeom: returns geom = [A, x_cen, y_cen, P]. Computes the
    boundary integral form for area + centroid.

    Inputs are sequences of polygon vertex coordinates (open loop, n points).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        # Degenerate: fall back to mean
        x_cen = float(np.mean(x)) if n else 0.0
        y_cen = float(np.mean(y)) if n else 0.0
        return np.array([0.0, x_cen, y_cen, 0.0], dtype=np.float64)

    xm = x.mean()
    ym = y.mean()
    xs = x - xm
    ys = y - ym

    nxt = np.arange(1, n + 1) % n
    dx = xs[nxt] - xs
    dy = ys[nxt] - ys

    A = np.sum(ys * dx - xs * dy) / 2.0
    Axc = np.sum(6.0 * xs * ys * dx - 3.0 * xs * xs * dy + 3.0 * ys * dx * dx + dx * dx * dy) / 12.0
    Ayc = np.sum(3.0 * ys * ys * dx - 6.0 * xs * ys * dy - 3.0 * xs * dy * dy - dx * dy * dy) / 12.0
    P = np.sum(np.sqrt(dx * dx + dy * dy))

    if A < 0:
        A = -A
        Axc = -Axc
        Ayc = -Ayc

    if A == 0:
        x_cen = float(xm)
        y_cen = float(ym)
    else:
        xc = Axc / A
        yc = Ayc / A
        x_cen = float(xc + xm)
        y_cen = float(yc + ym)

    return np.array([A, x_cen, y_cen, float(P)], dtype=np.float64)


# ---- alpha-shape boundary (MATLAB boundary(X, Y, 0.5)) ------------------------
def alpha_shape_boundary(X, Y, shrink_factor: float = 0.5):
    """Replicate MATLAB's `boundary(X, Y, s)` for 2-D point sets.

    Reverse-engineered algorithm (matches MATLAB to within floating-point tol.):

      1. 2-D Delaunay triangulation of all input points.
      2. Compute the circumradius R of every triangle.
      3. Find the *critical alpha radius* R_crit: the smallest R such that the
         union of triangles with circumradius <= R covers every input point and
         forms a single connected component (sharing Delaunay edges).
      4. Map shrink factor s in [0, 1] linearly to an alpha radius:
            alpha = R_crit + (1 - s) * (R_max - R_crit)
         - s = 0 -> alpha = R_max (keep all triangles, i.e., convex hull)
         - s = 1 -> alpha = R_crit (most compact valid alpha shape)
      5. Keep triangles with R <= alpha; the boundary is the set of edges that
         have exactly one kept neighbour (= outer face).
      6. Walk the outer boundary CCW (positive signed area) and return the
         vertex indices forming a closed loop (first index repeats at the end,
         matching MATLAB's `boundary()` convention).

    The connectivity check uses union-find over triangle indices via
    scipy.spatial.Delaunay's `neighbors` adjacency, giving O(n_tri * alpha(n))
    overall. For 2-D Delaunay, scipy guarantees CCW vertex orientation, so the
    boundary edges (vertex k+1 -> vertex k+2 of CCW triangles) trace the alpha
    shape's outer ring CCW.
    """
    from scipy.spatial import Delaunay, ConvexHull

    pts = np.column_stack([np.asarray(X, dtype=np.float64).ravel(),
                           np.asarray(Y, dtype=np.float64).ravel()])
    n = pts.shape[0]
    if n < 3:
        return np.arange(n, dtype=np.int64)

    def _hull_loop():
        ch = ConvexHull(pts)
        v = ch.vertices.astype(np.int64)
        return np.append(v, v[0])

    if shrink_factor <= 0:
        return _hull_loop()

    try:
        tri = Delaunay(pts)
    except Exception:
        return _hull_loop()

    simplices = tri.simplices  # (n_tri, 3), CCW for 2-D
    neighbors = tri.neighbors  # (n_tri, 3); neighbor opposite vertex k
    n_tri = simplices.shape[0]
    if n_tri == 0:
        return _hull_loop()

    A = pts[simplices[:, 0]]
    B = pts[simplices[:, 1]]
    C = pts[simplices[:, 2]]
    a_len = np.linalg.norm(B - C, axis=1)
    b_len = np.linalg.norm(A - C, axis=1)
    c_len = np.linalg.norm(A - B, axis=1)
    sp = 0.5 * (a_len + b_len + c_len)
    area = np.sqrt(np.maximum(sp * (sp - a_len) * (sp - b_len) * (sp - c_len), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(area > 0, a_len * b_len * c_len / (4.0 * area), np.inf)

    finite = np.isfinite(R)
    if not finite.any():
        return _hull_loop()
    R_max_finite = float(R[finite].max())

    # Add triangles in ascending-R order, tracking #components and point coverage.
    # JIT only kicks in for non-trivial n_tri to amortise the dispatch cost.
    order = np.argsort(R, kind='stable').astype(np.int64)
    if _HAVE_NUMBA and n_tri >= 200:
        R_crit_jit = _alpha_shape_critical_alpha(
            order,
            np.ascontiguousarray(simplices, dtype=np.int64),
            np.ascontiguousarray(neighbors, dtype=np.int64),
            R, finite, n)
        R_crit = R_max_finite if R_crit_jit < 0 else float(R_crit_jit)
    else:
        parent = np.arange(n_tri, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        in_set = np.zeros(n_tri, dtype=bool)
        point_in = np.zeros(n, dtype=bool)
        n_components = 0
        R_crit = R_max_finite

        for ti in order:
            if not finite[ti]:
                break
            in_set[ti] = True
            point_in[simplices[ti]] = True
            n_components += 1
            for nb in neighbors[ti]:
                if nb >= 0 and in_set[nb]:
                    ra, rb = find(ti), find(nb)
                    if ra != rb:
                        parent[ra] = rb
                        n_components -= 1
            if point_in.all() and n_components == 1:
                R_crit = float(R[ti])
                break

    # Linear shrink: s=0 -> all, s=1 -> minimal connected.
    alpha_radius = R_crit + (1.0 - float(shrink_factor)) * (R_max_finite - R_crit)
    keep = R <= alpha_radius
    if not keep.any():
        return _hull_loop()

    # Collect outward boundary half-edges (kept triangle -> non-kept neighbour).
    # scipy: neighbors[i, k] is the neighbour opposite vertex k; the edge
    # opposite vertex k of CCW triangle (t0, t1, t2) is (t[(k+1)%3], t[(k+2)%3]).
    out_edges = []
    keep_idx = np.flatnonzero(keep)
    for ti in keep_idx:
        t = simplices[ti]
        for k in range(3):
            nb = neighbors[ti, k]
            if nb < 0 or not keep[nb]:
                u = int(t[(k + 1) % 3])
                v = int(t[(k + 2) % 3])
                out_edges.append((u, v))

    if not out_edges:
        return _hull_loop()

    follow = {}
    for u, v in out_edges:
        # In a manifold alpha shape each vertex has at most one outgoing
        # boundary edge per loop. If a "pinch point" exists where two loops
        # share a vertex, fall back to picking any successor; loop selection
        # below picks the one with the largest signed area.
        follow.setdefault(u, []).append(v)

    visited_edges = set()
    loops = []
    for start_v in list(follow.keys()):
        for first_to in follow[start_v]:
            edge0 = (start_v, first_to)
            if edge0 in visited_edges:
                continue
            loop = [start_v]
            cur = first_to
            visited_edges.add(edge0)
            while cur != start_v:
                loop.append(cur)
                outs = follow.get(cur, [])
                # Prefer an unvisited successor.
                nxt = None
                for cand in outs:
                    if (cur, cand) not in visited_edges:
                        nxt = cand
                        break
                if nxt is None:
                    break
                visited_edges.add((cur, nxt))
                cur = nxt
            if cur == start_v and len(loop) >= 3:
                loop.append(start_v)
                loops.append(loop)

    if not loops:
        return _hull_loop()

    def signed_area(loop):
        L = len(loop) - 1  # last == first
        s2 = 0.0
        for i in range(L):
            x1, y1 = pts[loop[i]]
            x2, y2 = pts[loop[i + 1]]
            s2 += x1 * y2 - x2 * y1
        return 0.5 * s2

    # Outer ring: largest positive signed area (CCW).
    pos = [(L, signed_area(L)) for L in loops]
    pos = [(L, A) for (L, A) in pos if A > 0]
    if pos:
        outer = max(pos, key=lambda LA: LA[1])[0]
    else:
        # All loops were CW (shouldn't happen with CCW Delaunay) — pick longest.
        outer = max(loops, key=len)
    return np.asarray(outer, dtype=np.int64)


# ---- boundary_geometry_interslice --------------------------------------------
def boundary_geometry_interslice(Q_x, Q_y, cluster_idx_1based, cluster_i_1based,
                                 cluster_j_1based, cluster_io, x_ext, y_ext, Z,
                                 phi, coh, depth, gam, sigma_s_wedge, PGA, ls_index, bs):
    """Mirror of `boundary_geometry_interslice.m`. Returns a tuple of 23 arrays
    in the same order as the MATLAB function.
    """
    from .matlab_compat import to_F_index

    cluster_io = np.asarray(cluster_io, dtype=bool)
    shape = cluster_io.shape

    # Trace outer boundary
    boundaries = bwboundaries_outer(cluster_io)
    if not boundaries:
        raise ValueError("Cluster has no boundary")
    bd = boundaries[0]  # (K+1, 2) closed loop, 1-based
    bdi = bd[:-1, 0].astype(np.int64)
    bdj = bd[:-1, 1].astype(np.int64)

    # 1-based column-major linear indices of boundary cells
    m, n = shape
    bidx = ((bdj - 1) * m + bdi).astype(np.int64)

    # Cluster direction of sliding
    ci0, cj0 = to_F_index(np.asarray(cluster_idx_1based, dtype=np.int64), shape)
    QX_ep = Q_x[ci0, cj0]
    QY_ep = Q_y[ci0, cj0]
    err_ep_x = float(np.sum(QX_ep))
    err_ep_y = float(np.sum(QY_ep))
    slide_dir = atan2d(err_ep_y, err_ep_x)
    if slide_dir < 0:
        slide_dir += 360.0

    # Centroid of cluster (alpha-shape boundary of all cluster cells)
    X_all = np.asarray(x_ext)[cluster_j_1based - 1].ravel()
    Y_all = np.asarray(y_ext)[cluster_i_1based - 1].ravel()
    BD = alpha_shape_boundary(X_all, Y_all, 0.5)
    cluster_geom = polygeom(X_all[BD], Y_all[BD])
    cluster_cenx = float(cluster_geom[1])
    cluster_ceny = float(cluster_geom[2])

    # Direction from centroid to each boundary cell
    bx = np.asarray(x_ext)[bdj - 1]
    by = np.asarray(y_ext)[bdi - 1]
    vec_x = bx - cluster_cenx
    vec_y = by - cluster_ceny
    boundary_dir = atan2d(vec_y, vec_x)
    boundary_dir = np.where(boundary_dir < 0, boundary_dir + 360.0, boundary_dir)

    bound_dir_diff = boundary_dir - slide_dir
    bound_dir_diff = np.where(bound_dir_diff < 0, bound_dir_diff + 360.0, bound_dir_diff)

    # Earth-pressure alpha lookup (interp1 between active/passive)
    phi_first = float(phi[0, 0])
    EP_alpha_active = 45.0 + phi_first / 2.0
    EP_alpha_passive = 45.0 - phi_first / 2.0
    th_full = np.arange(0, 361, dtype=np.float64)
    EP_alpha = np.zeros(361, dtype=np.float64)
    th1 = np.arange(0, 181, dtype=np.float64)
    EP_alpha[:181] = np.linspace(EP_alpha_passive, EP_alpha_active, th1.size)
    th2 = np.arange(181, 361, dtype=np.float64)
    EP_alpha[181:] = np.linspace(EP_alpha_active, EP_alpha_passive, th2.size)
    boundary_alpha = np.interp(bound_dir_diff, th_full, EP_alpha)

    # Boundary cell parameters
    bdi0 = bdi - 1
    bdj0 = bdj - 1
    boundary_depth = depth[bdi0, bdj0].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        boundary_r = boundary_depth * cosd(boundary_alpha) / sind(boundary_alpha)
        boundary_dr = boundary_depth / boundary_r
        boundary_dx = boundary_dr * cosd(boundary_dir)
        boundary_dy = boundary_dr * sind(boundary_dir)

    # Wedge projection direction (ray_dir) — vectorised over all vertices.
    # The original loop ran 2 × atan2d per vertex × n_b vertices per cluster.
    # We compute prev/curr/next neighbour coordinates by index roll instead.
    n_b = bidx.size
    xv = np.asarray(x_ext)[bdj - 1].astype(np.float64)
    yv = np.asarray(y_ext)[bdi - 1].astype(np.float64)
    xv_prev = np.roll(xv, 1)
    yv_prev = np.roll(yv, 1)
    xv_next = np.roll(xv, -1)
    yv_next = np.roll(yv, -1)
    x1 = xv_prev - xv
    y1 = yv_prev - yv
    x2 = xv_next - xv
    y2 = yv_next - yv
    angdir = -atan2d(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    prism_dir = atan2d(yv_next - yv, xv_next - xv)

    angdir = np.where(angdir < 0, angdir + 360.0, angdir)
    prism_dir = np.where(prism_dir < 0, prism_dir + 360.0, prism_dir)

    ray_dir = prism_dir + angdir / 2.0 + 180.0

    # Wedge outer geometry
    wedge_x = boundary_r * cosd(ray_dir)
    wedge_y = boundary_r * sind(ray_dir)
    wedge_x_loc = bx + wedge_x
    wedge_y_loc = by + wedge_y
    wedge_A = 0.5 * boundary_depth * boundary_r

    # Burn severity (unused in mode A but preserved for fidelity)
    if bs is not None and np.size(bs) > 0:
        bs_boundary = np.asarray(bs)[bdi0, bdj0]
        wedge_bs = np.zeros(n_b, dtype=np.float64)
        wedge_bs[:-1] = np.maximum(bs_boundary[:-1], bs_boundary[1:])
        wedge_bs[-1] = max(bs_boundary[-1], bs_boundary[0])
    else:
        wedge_bs = np.array([], dtype=np.float64)

    # Prism geometry (per wedge edge between consecutive boundary points)
    # Vectorised per-edge wedge geometry (using np.roll for the cyclic
    # next-neighbour reference). `roll(-1)` shifts each entry to its successor:
    # entry [w] of the rolled array equals the original [w+1] (with [-1] -> [0]).
    bd_next = np.roll(boundary_dir, -1)
    br_next = np.roll(boundary_dr, -1)
    bx_next = np.roll(bx, -1)
    by_next = np.roll(by, -1)
    bR_next = np.roll(boundary_r, -1)
    bA_next = np.roll(wedge_A, -1)
    wxl_next = np.roll(wedge_x_loc, -1)
    wyl_next = np.roll(wedge_y_loc, -1)

    wedge_dir = 0.5 * (boundary_dir + bd_next)
    wedge_subdr = 0.5 * (boundary_dr + br_next)
    wedge_pt_dist = np.hypot(bx_next - bx, by_next - by)
    wedge_outer_dist = np.hypot(wxl_next - wedge_x_loc, wyl_next - wedge_y_loc)
    wedge_width_avg = 0.5 * (wedge_pt_dist + wedge_outer_dist)
    wedge_r_avg = 0.5 * (boundary_r + bR_next)
    wedge_V = 0.5 * (wedge_A + bA_next) * wedge_width_avg

    wedge_W = wedge_V * gam
    wedge_subdx = wedge_subdr * cosd(wedge_dir)
    wedge_subdy = wedge_subdr * sind(wedge_dir)
    wedge_depth = boundary_depth / 2.0
    wedge_phi = phi[bdi0, bdj0].astype(np.float64)
    wedge_coh = coh[bdi0, bdj0].astype(np.float64)
    wedge_U_pressure = np.asarray(sigma_s_wedge)[bdi0, bdj0].astype(np.float64)

    # Seismic
    wedge_dir_diff = wedge_dir - slide_dir
    wedge_dir_diff = np.where(wedge_dir_diff < 0, wedge_dir_diff + 360.0, wedge_dir_diff)
    k_mean = float(np.mean(np.asarray(PGA)[ci0, cj0]))
    wedge_k = k_mean * -cosd(wedge_dir_diff)
    wedge_kW = wedge_k * wedge_W

    return (wedge_dir, wedge_subdr, wedge_subdx, wedge_subdy, wedge_width_avg,
            wedge_r_avg, wedge_V, wedge_W, wedge_U_pressure, wedge_depth,
            wedge_phi, wedge_coh, cluster_cenx, cluster_ceny, wedge_x_loc,
            wedge_y_loc, slide_dir, bound_dir_diff, wedge_k, wedge_kW,
            prism_dir, boundary_depth, wedge_bs)


# ---- root force ---------------------------------------------------------------
def root_force_boundary(slide_dir, prism_dir, boundary_depth, wedge_width_avg,
                        S_roots, S_roots_healthy, wedge_bs):
    """Mirror of `root_force_boundary.m`."""
    prism_dir = np.asarray(prism_dir, dtype=np.float64)
    boundary_depth = np.asarray(boundary_depth, dtype=np.float64)
    wedge_width_avg = np.asarray(wedge_width_avg, dtype=np.float64)
    n = prism_dir.size
    if n == 0:
        return 0.0, 0.0, 0.0

    wall_dir = prism_dir + 270.0
    angdiff = wall_dir - slide_dir

    root_io = np.zeros(n, dtype=bool)
    root_io[(angdiff > -270) & (angdiff < -90)] = True
    root_io[(angdiff > 90) & (angdiff < 270)] = True
    root_io[(angdiff > 450) & (angdiff < 630)] = True

    depth_roots = np.zeros(n, dtype=np.float64)
    depth_roots[:-1] = (boundary_depth[:-1] + boundary_depth[1:]) / 2.0
    depth_roots[-1] = (boundary_depth[-1] + boundary_depth[0]) / 2.0

    area_roots = root_io.astype(np.float64) * depth_roots * wedge_width_avg * np.abs(cosd(angdiff))

    if wedge_bs is None or np.size(wedge_bs) == 0:
        F_roots_wall = float(S_roots) * area_roots
    else:
        wbs = np.asarray(wedge_bs)
        F_roots_wall = np.zeros_like(area_roots)
        mask_burn = wbs >= 3
        F_roots_wall[mask_burn] = float(S_roots) * area_roots[mask_burn]
        F_roots_wall[~mask_burn] = float(S_roots_healthy) * area_roots[~mask_burn]

    F_roots = float(np.sum(F_roots_wall))
    Frx = F_roots * cosd(slide_dir)
    Fry = F_roots * sind(slide_dir)
    return F_roots, Frx, Fry
