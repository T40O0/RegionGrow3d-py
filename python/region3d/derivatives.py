"""DEM-derived rasters: NaN padding, gradient_prince, hillshade.

Each function mirrors the corresponding MATLAB routine.
"""
from __future__ import annotations

import numpy as np


def pad_DEM(Z: np.ndarray) -> np.ndarray:
    """Set border cells to NaN (in-place equivalent; returns a new array)."""
    Z = np.asarray(Z, dtype=np.float32).copy()
    Z[0, :] = np.nan
    Z[-1, :] = np.nan
    Z[:, 0] = np.nan
    Z[:, -1] = np.nan
    return Z


def gradient_prince(Z: np.ndarray, x_cellsize: float, y_cellsize: float):
    """Slope/aspect/gradient computed by the 3x3 Horn kernel used in MATLAB.

    Vectorised version of the inner loop. The behavior mirrors the MATLAB code:
    - Output arrays are zero on the 1-cell perimeter.
    - dz/dx = ((Z[j-1,k+1] + 2 Z[j,k+1] + Z[j+1,k+1]) -
               (Z[j-1,k-1] + 2 Z[j,k-1] + Z[j+1,k-1])) / (8 * x_cellsize)
    - dz/dy follows the same pattern in y.
    - Aspect is rounded and rotated so that 0 = North, increasing clockwise.

    All outputs cast to float32 to match the MATLAB `single(...)` calls.
    """
    # Mirror MATLAB: Z is single (float32) and all intermediate math stays in
    # single precision. Using float64 internally and casting at the end would
    # produce different ulps and could flip the `Q0 > 0` boolean for cells
    # very close to zero, propagating through bwmorph.
    Z = np.asarray(Z, dtype=np.float32)
    z32 = np.float32
    m, n = Z.shape
    slope = np.zeros((m, n), dtype=z32)
    aspect = np.zeros((m, n), dtype=z32)
    dx = np.zeros((m, n), dtype=z32)
    dy = np.zeros((m, n), dtype=z32)

    if m < 3 or n < 3:
        return slope, aspect, dx, dy

    # Slices for the 3x3 stencil over the interior (j=1..m-2, k=1..n-2)
    Zm1_kp1 = Z[0:m - 2, 2:n]      # Z(j-1, k+1)
    Zj_kp1 = Z[1:m - 1, 2:n]       # Z(j,   k+1)
    Zp1_kp1 = Z[2:m, 2:n]          # Z(j+1, k+1)

    Zm1_km1 = Z[0:m - 2, 0:n - 2]
    Zj_km1 = Z[1:m - 1, 0:n - 2]
    Zp1_km1 = Z[2:m, 0:n - 2]

    Zp1_k = Z[2:m, 1:n - 1]
    Zm1_k = Z[0:m - 2, 1:n - 1]

    two = z32(2.0)
    eight_dx = z32(8.0 * x_cellsize)
    eight_dy = z32(8.0 * y_cellsize)
    deg_per_rad = z32(57.29578)

    dz_dx = ((Zm1_kp1 + two * Zj_kp1 + Zp1_kp1)
             - (Zm1_km1 + two * Zj_km1 + Zp1_km1)) / eight_dx
    dz_dy = ((Zp1_km1 + two * Zp1_k + Zp1_kp1)
             - (Zm1_km1 + two * Zm1_k + Zm1_kp1)) / eight_dy

    rise_run = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
    slope_deg = (np.arctan(rise_run) * deg_per_rad).astype(z32)

    # Aspect (clockwise from North) — match the per-cell branch in MATLAB
    asp = (deg_per_rad * np.arctan2(dz_dy, -dz_dx)).astype(z32)
    ninety = z32(90.0)
    three_sixty = z32(360.0)
    cell = np.where(asp < 0, ninety - asp,
                    np.where(asp > ninety, three_sixty - asp + ninety,
                             ninety - asp)).astype(z32)
    # MATLAB `round` rounds halves away from zero; np.round uses banker's
    # rounding (half-to-even), which disagrees on exact x.5 values (e.g.
    # 22.5 -> MATLAB 23 vs np.round 22). `cell` is aspect in [0, 360] (always
    # non-negative), so floor(x + 0.5) reproduces MATLAB round exactly. NaN
    # propagates through unchanged, matching np.round's NaN handling.
    aspect_int = np.floor(cell + z32(0.5)).astype(z32)

    slope[1:m - 1, 1:n - 1] = slope_deg
    aspect[1:m - 1, 1:n - 1] = aspect_int
    dx[1:m - 1, 1:n - 1] = dz_dx
    dy[1:m - 1, 1:n - 1] = dz_dy

    return slope, aspect, dx, dy


def hillshade(Z: np.ndarray, x_ext: np.ndarray, y_ext: np.ndarray,
              azimuth: float = 315.0, altitude: float = 45.0,
              zfactor: float = 1.0) -> np.ndarray:
    """ESRI hillshade matching `hillshade.m`. Border cells become NaN."""
    Z = np.asarray(Z, dtype=np.float64)
    dx = abs(x_ext[1] - x_ext[0])
    dy = abs(y_ext[1] - y_ext[0])

    az = 360.0 - azimuth + 90.0
    if az >= 360.0:
        az -= 360.0
    az_rad = np.deg2rad(az)
    alt_rad = np.deg2rad(90.0 - altitude)

    fy, fx = np.gradient(Z, dy, dx)
    asp = np.arctan2(fx, fy)
    grad = np.arctan(zfactor * np.hypot(fx, fy))
    asp = np.where(asp < np.pi, asp + np.pi / 2.0, asp)
    asp = np.where(asp < 0, asp + 2.0 * np.pi, asp)

    h = 256.0 * (np.cos(alt_rad) * np.cos(grad)
                 + np.sin(alt_rad) * np.sin(grad) * np.cos(az_rad - asp))
    h = np.where(h < 0, 0, h)
    h[0, :] = np.nan
    h[-1, :] = np.nan
    h[:, 0] = np.nan
    h[:, -1] = np.nan
    return h.astype(np.float32)
