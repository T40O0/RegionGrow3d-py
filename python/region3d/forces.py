"""3D Janbu / infinite-slope force equilibrium.

Mirrors `Interslice_Force.m`, `Interslice_Force_Prism.m`,
`force_closure_interslice.m`. All math is performed in float32 to match the
MATLAB `single(...)` casts.
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import to_F_index, atan2d, atand, sind, cosd, tand


def interslice_force(subdx, subdy, x_cellsize, y_cellsize, coh, phi, W, sigma_s,
                     idx_1based, PGA):
    """3D Janbu method (Hungr et al., 1989). Vectorised over `idx_1based`.

    Outputs are full-size float32 arrays initialised to zero, with values written
    only at the cells named by `idx_1based`. theta_Q is full-size float32 too
    (the MATLAB code grows it sparsely with `theta_Q(idx(i))=...`; cells outside
    idx remain at the implicit zero MATLAB initialises with first assignment).
    """
    shape = subdx.shape
    z32 = np.float32

    mag = np.zeros(shape, dtype=z32)
    area_col = np.zeros(shape, dtype=z32)
    alpha = np.zeros(shape, dtype=z32)
    alphax = np.zeros(shape, dtype=z32)
    alphay = np.zeros(shape, dtype=z32)
    N = np.zeros(shape, dtype=z32)
    U = np.zeros(shape, dtype=z32)
    Q = np.zeros(shape, dtype=z32)
    Q_x = np.zeros(shape, dtype=z32)
    Q_y = np.zeros(shape, dtype=z32)
    theta_Q = np.zeros(shape, dtype=z32)

    if idx_1based is None or len(idx_1based) == 0:
        return Q, Q_x, Q_y, mag, area_col, N, U, theta_Q, alpha, alphax, alphay

    i0, j0 = to_F_index(np.asarray(idx_1based, dtype=np.int64), shape)

    sx = np.asarray(subdx[i0, j0], dtype=z32)
    sy = np.asarray(subdy[i0, j0], dtype=z32)
    c = np.asarray(coh[i0, j0], dtype=z32)
    p = np.asarray(phi[i0, j0], dtype=z32)
    w = np.asarray(W[i0, j0], dtype=z32)
    ss = np.asarray(sigma_s[i0, j0], dtype=z32)
    pga = np.asarray(PGA[i0, j0], dtype=z32)

    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.sqrt(sx * sx + sy * sy).astype(z32)
        a = atand(m).astype(z32)
        ax = atand(sx).astype(z32)
        ay = atand(sy).astype(z32)

        sax = sind(ax).astype(z32)
        say = sind(ay).astype(z32)
        cax = cosd(ax).astype(z32)
        cay = cosd(ay).astype(z32)
        sa = sind(a).astype(z32)
        ca = cosd(a).astype(z32)
        tp = tand(p).astype(z32)

        ac = (np.float32(x_cellsize) * np.float32(y_cellsize)
              * np.sqrt(np.float32(1.0) - (sax * sax) * (say * say))
              / (cax * cay)).astype(z32)

        u = (ss * ac).astype(z32)

        n = ((w + u * sa * tp - c * ac * sa) / (ca + sa * tp)).astype(z32)

        q = (n * sa - (c * ac + (n - u) * tp) * ca + pga * w).astype(z32)

        qx = ((-sx / m) * q).astype(z32)
        qy = ((-sy / m) * q).astype(z32)
        th = atan2d(qy, qx).astype(z32)

    mag[i0, j0] = m
    area_col[i0, j0] = ac
    alpha[i0, j0] = a
    alphax[i0, j0] = ax
    alphay[i0, j0] = ay
    N[i0, j0] = n
    U[i0, j0] = u
    Q[i0, j0] = q
    Q_x[i0, j0] = qx
    Q_y[i0, j0] = qy
    theta_Q[i0, j0] = th

    return Q, Q_x, Q_y, mag, area_col, N, U, theta_Q, alpha, alphax, alphay


def interslice_force_prism(wedge_subdx, wedge_subdy, wedge_width, wedge_r_avg,
                           wedge_W, wedge_phi, wedge_coh, wedge_U_pressure,
                           wedge_kW):
    """Infinite-slope equilibrium per boundary prism (Taylor, 1948)."""
    sx = np.asarray(wedge_subdx, dtype=np.float64)
    sy = np.asarray(wedge_subdy, dtype=np.float64)
    ww = np.asarray(wedge_width, dtype=np.float64)
    wr = np.asarray(wedge_r_avg, dtype=np.float64)
    W = np.asarray(wedge_W, dtype=np.float64)
    phi = np.asarray(wedge_phi, dtype=np.float64)
    coh = np.asarray(wedge_coh, dtype=np.float64)
    Up = np.asarray(wedge_U_pressure, dtype=np.float64)
    kW = np.asarray(wedge_kW, dtype=np.float64)

    mag = np.sqrt(sx * sx + sy * sy)
    ax = atand(sx)
    ay = atand(sy)
    a = atand(mag)
    base_area = (ww * wr * np.sqrt(1.0 - (sind(ax) ** 2) * (sind(ay) ** 2))
                 / (cosd(ax) * cosd(ay)))

    U = Up * base_area
    N = (W + U * sind(a) * tand(phi) - coh * base_area * sind(a)) \
        / (cosd(a) + sind(a) * tand(phi))
    Q = N * sind(a) - (coh * base_area + (N - U) * tand(phi)) * cosd(a) + kW
    Q_x = (-sx / mag) * Q
    Q_y = (-sy / mag) * Q
    return Q, Q_x, Q_y


def force_closure_interslice(rot, Q_x_cell, Q_y_cell, wedge_Q_x, wedge_Q_y,
                             cluster_idx_1based):
    """Per-rotation force closure check.

    Returns
    -------
    QX, QY, Q_mag, err_x, err_y, err_mag : 1-D float64 arrays, len = len(rot)
    """
    n_rot = len(rot)
    QX = np.zeros(n_rot, dtype=np.float64)
    QY = np.zeros(n_rot, dtype=np.float64)
    Q_mag = np.zeros(n_rot, dtype=np.float64)
    err_x = np.zeros(n_rot, dtype=np.float64)
    err_y = np.zeros(n_rot, dtype=np.float64)
    err_mag = np.zeros(n_rot, dtype=np.float64)

    if len(cluster_idx_1based) == 0:
        wsumx = np.nansum(wedge_Q_x) if wedge_Q_x is not None and len(wedge_Q_x) else 0.0
        wsumy = np.nansum(wedge_Q_y) if wedge_Q_y is not None and len(wedge_Q_y) else 0.0
        QX[:] = wsumx
        QY[:] = wsumy
        Q_mag[:] = np.hypot(QX, QY)
        err_x[:] = QX
        err_y[:] = QY
        err_mag[:] = Q_mag
        return QX, QY, Q_mag, err_x, err_y, err_mag

    shape = Q_x_cell[0].shape
    i0, j0 = to_F_index(np.asarray(cluster_idx_1based, dtype=np.int64), shape)

    wsumx = np.nansum(wedge_Q_x) if wedge_Q_x is not None and np.size(wedge_Q_x) else 0.0
    wsumy = np.nansum(wedge_Q_y) if wedge_Q_y is not None and np.size(wedge_Q_y) else 0.0

    for j in range(n_rot):
        qx_vals = Q_x_cell[j][i0, j0]
        qy_vals = Q_y_cell[j][i0, j0]
        sx = float(np.nansum(qx_vals)) + float(wsumx)
        sy = float(np.nansum(qy_vals)) + float(wsumy)
        QX[j] = sx
        QY[j] = sy
        Q_mag[j] = np.hypot(sx, sy)
        err_x[j] = sx
        err_y[j] = sy
        err_mag[j] = Q_mag[j]

    return QX, QY, Q_mag, err_x, err_y, err_mag


def project_slope(dx, dy, rot):
    """Rotate (dx, dy) clockwise by `rot` degrees and return (slope, dx1, dy1)."""
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    theta0 = atan2d(dy, dx)
    theta1 = rot + theta0
    x1 = cosd(theta1)
    y1 = sind(theta1)
    a = np.sqrt(dx * dx + dy * dy)
    b = a * cosd(rot)
    dx1 = b * x1
    dy1 = b * y1
    slope = np.sqrt(dx1 * dx1 + dy1 * dy1)
    return slope.astype(np.float32), dx1.astype(np.float32), dy1.astype(np.float32)
