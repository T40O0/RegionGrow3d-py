"""Soil-thickness model based on the soil-production function + mass balance
(土層生成関数＋物質収支モデル), following the formulation reviewed in

    松四雄騎 (2017): 宇宙線生成核種を用いた岩盤の風化と土層の生成に関する速度論
    ─手法の原理，適用法，研究の現状と課題─. 地学雑誌 126(4), 487-511.
    doi:10.5026/jgeography.126.487

and applied to Japanese hillslopes in

    松四雄騎・外山 真・松崎浩之・千木良雅弘 (2016): 土層の生成および輸送速度の
    決定と土層発達シミュレーションに基づく表層崩壊の発生場および崩土量の予測.
    地形 37, 427-453.

SPDX-License-Identifier: CC0-1.0
(Original implementation — not derived from the USGS MATLAB code.)

Model
-----
Mass balance of the soil column (review eq. 7)::

    rho_soil * dh/dt = E_sap - E_soil - W_soil

with

  * soil production function (eq. 13, Heimsath et al. 1997)::

        E_sap = E0 * exp(-alpha * h)                       [g m-2 yr-1]

  * soil transport by creep, either linear (eq. 9) or non-linear (eq. 10,
    Roering et al. 1999)::

        q = -rho_soil * K_L * grad(z)                       (linear)
        q = -rho_soil * K_N * grad(z) / (1 - (|grad z|/Sc)^2)   (non-linear)

    and  E_soil = div(q)  (eq. 8).

Because the DEM is taken as the (fixed) soil surface, ``div(q)`` is a constant
map computed once from the topography.  Writing ``D = div(q) + W_soil`` the
mass balance becomes a per-cell ODE with a closed-form solution, so no time
loop is needed:

  * steady state (dh/dt = 0, review eq. 14 / 15)::

        h = -(1/alpha) * ln(D / E0)                        (requires 0 < D < E0)

  * transient from an initial thickness ``h0`` after ``t`` years::

        h = (1/alpha) * ln( [E0 - (E0 - D*exp(alpha*h0)) * exp(-alpha*D*t/rho_soil)] / D )

    (and ``h = (1/alpha) * ln(exp(alpha*h0) + alpha*E0*t/rho_soil)`` when D = 0).

Divergent (convex, ridge/nose) cells have D > 0 and reach a steady thickness;
convergent (concave, hollow) cells have D <= 0, never reach steady state and
keep accumulating soil — matching the behaviour reported by 松四ほか (2016)
(noses mostly < 0.5 m at steady state, hollows growing to ~1.2 m over a few
hundred years).  ``h_max`` caps that accumulation.

Difference from :func:`region3d.preprocessing.soil_depth` (Roering 2008 model)
-----------------------------------------------------------------------------
The Roering routine evolves the *topography* for `endtime` years starting from
a uniform 1 m soil mantle, so absolute thicknesses stay dominated by that
initial condition.  Here the measured topography is held fixed and thickness is
solved from the mass balance, so the result is set by the parameters
(E0, alpha, K, Sc, rho_soil) and the local curvature — not by an arbitrary
initial guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


__all__ = ['SoilProductionParams', 'soil_depth_massbalance',
           'params_from_namespace', 'MASSBALANCE_ALIASES', 'PRESETS']

#: values of ``--soil_depth_model`` that select this model
MASSBALANCE_ALIASES = ('massbalance', 'mass_balance', 'mb', 'jp', 'matsushi')


@dataclass
class SoilProductionParams:
    """Parameters of the soil-production / transport model.

    Attributes
    ----------
    E0 : float
        Soil production rate for a bare surface, ``g m-2 yr-1`` (review eq. 13).
        Published soil-production functions scatter over 1e1-1e3 g m-2 yr-1
        (Larsen et al. 2014, quoted in 松四 2017).
    alpha : float
        Decay of soil production with thickness, ``m-1``.  Published range
        1-5 m-1 (同上).
    K : float
        Soil-transport (creep) coefficient, ``m2 yr-1``.  ``K_L`` for
        ``transport='linear'``, ``K_N`` for ``'nonlinear'``.
    Sc : float
        Critical gradient (dimensionless, tan of the slope angle) at which the
        non-linear creep flux diverges.  Ignored when ``transport='linear'``.
    rho_soil : float
        Soil bulk density, ``kg m-3``.
    W_soil : float
        Chemical mass-loss rate from the soil column, ``g m-2 yr-1``
        (review eq. 7).  0 = element leaching in the soil is neglected.
    transport : str
        ``'nonlinear'`` (review eq. 10) or ``'linear'`` (review eq. 9).
    slope_normal : bool
        If True the soil production function is driven by the slope-normal
        thickness ``H = h cos(theta)`` rather than the vertical thickness ``h``
        (松四ほか 2016 eq. 5 and Fig. 1).  Since ``cos(theta)`` is fixed by the
        topography this is just a per-cell rescaling of ``alpha``.

    Notes
    -----
    The defaults are *not* a Japanese calibration: they are the mass-unit
    equivalents of the Oregon Coast Range constants hard-wired in
    `lib/functions/soil_depth.m` (Po=3e-4 m/yr with rho_sap=2400 kg m-3
    -> E0=720 g m-2 yr-1; mu=3 -> alpha=3; K=0.005; Sc=1.25; prps=2 ->
    rho_soil=1200 kg m-3), so the two models can be compared directly.
    Replace them with site-calibrated values (e.g. 松四ほか 2016) when
    available.
    """

    E0: float = 720.0          # g m-2 yr-1
    alpha: float = 3.0         # m-1
    K: float = 0.005           # m2 yr-1
    Sc: float = 1.25           # -
    rho_soil: float = 1200.0   # kg m-3
    W_soil: float = 0.0        # g m-2 yr-1
    transport: str = 'nonlinear'
    slope_normal: bool = False  # measure h along the slope normal, H = h cos(theta)

    def validate(self) -> None:
        if self.E0 <= 0:
            raise ValueError("E0 must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.K <= 0:
            raise ValueError("K must be > 0")
        if self.rho_soil <= 0:
            raise ValueError("rho_soil must be > 0")
        if self.transport not in ('linear', 'nonlinear'):
            raise ValueError("transport must be 'linear' or 'nonlinear'")
        if self.transport == 'nonlinear' and self.Sc <= 0:
            raise ValueError("Sc must be > 0 for the non-linear transport law")


def params_from_namespace(args) -> SoilProductionParams:
    """Build :class:`SoilProductionParams` from the driver's ``--soil_depth_mb_*``
    command-line options (shared by `driver.py` and `precompute_inputs.py`).

    ``--soil_depth_mb_preset`` selects a published parameter set; any explicitly
    given ``--soil_depth_mb_<name>`` value still wins over the preset.
    """
    preset = str(getattr(args, 'soil_depth_mb_preset', '') or '').lower()
    if preset:
        if preset not in PRESETS:
            raise SystemExit(f"unknown --soil_depth_mb_preset {preset!r}; "
                             f"choose from {sorted(PRESETS)}")
        base = dict(PRESETS[preset])
        # command line overrides: anything that differs from the argparse default
        for key, cli in (('E0', 'soil_depth_mb_E0'), ('alpha', 'soil_depth_mb_alpha'),
                         ('K', 'soil_depth_mb_K'), ('Sc', 'soil_depth_mb_Sc'),
                         ('rho_soil', 'soil_depth_mb_rho_soil'),
                         ('W_soil', 'soil_depth_mb_W')):
            val = float(getattr(args, cli))
            if val != PRESETS['oregon'][key]:     # argparse defaults = oregon
                base[key] = val
        cli_transport = str(args.soil_depth_mb_transport).lower()
        if cli_transport != PRESETS['oregon']['transport']:
            base['transport'] = cli_transport
        return SoilProductionParams(**base)
    return SoilProductionParams(
        E0=float(args.soil_depth_mb_E0),
        alpha=float(args.soil_depth_mb_alpha),
        K=float(args.soil_depth_mb_K),
        Sc=float(args.soil_depth_mb_Sc),
        rho_soil=float(args.soil_depth_mb_rho_soil),
        W_soil=float(args.soil_depth_mb_W),
        transport=str(args.soil_depth_mb_transport).lower(),
    )


#: parameter sets that can be selected with ``--soil_depth_mb_preset``
PRESETS = {
    # mass-unit equivalent of the constants hard-wired in soil_depth.m
    # (Roering/Heimsath, Oregon Coast Range: Po=3e-4 m/yr, mu=3, K=0.005,
    #  Sc=1.25, prps=rho_rock/rho_soil=2)
    'oregon': dict(E0=720.0, alpha=3.0, K=0.005, Sc=1.25, rho_soil=1200.0,
                   W_soil=0.0, transport='nonlinear', slope_normal=False),
    # Calibrated values for a granite watershed near Kyoto, measured with
    # cosmogenic 10Be by 松四雄騎・外山 真・松崎浩之・千木良雅弘 (2016)
    # 「土層の生成および輸送速度の決定と土層発達シミュレーションに基づく
    #   表層崩壊の発生場および崩土量の予測」地形 37(4), 427-453:
    #   D0 = 965.8 g m-2 yr-1, alpha = 0.948 m-1   (p.440, eq.5 + Fig.7)
    #   K  = 5e-3 m2 yr-1 (envelope 3.5e-3 - 6.5e-3)   (p.441, Fig.6B)
    #   rho_soil = 1.09e6 g m-3, from the mean dry unit weight (p.442)
    #   transport: LINEAR, q = -rho_soil K grad z     (eq.2)
    #   W (and the aeolian/organic input S) are explicitly neglected: the paper
    #   argues |S - W| << |D| on steep slopes (p.429)
    #   the production function uses the SLOPE-NORMAL thickness H = h cos(theta)
    # Their own simulation ran 500 yr from a uniform 0.5 m mantle on a 1 m grid
    # and gave 0.3-0.5 m on convex noses, ~1 m in hollows after 300-400 yr, and
    # a 700-800 yr shallow-landslide return period.
    # NOTE: their noses have 5 m curvatures of +0.1..+0.25 1/m (Fig.6B). On
    # terrain an order of magnitude gentler (e.g. the Noto 5 m DEM, median
    # convex curvature ~0.02) this set is production-limited: thickness is set
    # by h_init + elapsed time, not by the topography. See docs/MANUAL §5.2.
    'matsushi': dict(E0=965.8, alpha=0.948, K=0.005, Sc=1.25, rho_soil=1090.0,
                     W_soil=0.0, transport='linear', slope_normal=True),
}


def _fill_nan_nearest(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z with NaN replaced by the nearest finite value, valid mask).

    Finite differences would otherwise spread the NaN border (and the NaN sea)
    two cells inward.  Filling with the nearest valid elevation keeps the
    gradient defined right up to the coastline / DEM edge; the mask is applied
    again at the end so no value is invented outside the DEM.
    """
    valid = np.isfinite(Z)
    if valid.all():
        return Z.astype(np.float64, copy=True), valid
    if not valid.any():
        raise ValueError("DEM has no finite cells")
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~valid, return_distances=True,
                                    return_indices=True)
    return np.asarray(Z, dtype=np.float64)[tuple(idx)], valid


def _flux_divergence(z: np.ndarray, dx: float, dy: float,
                     params: SoilProductionParams, grac: float = 0.025
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """div(q) / rho_soil  =  -div(K grad z / denom)   [m yr-1].

    The non-linear divergence is evaluated with the analytic expansion used by
    `lib/functions/soil_depth.m` (Roering et al. 1999, 2007)::

        div(K grad z / den) = K * [ lap(z)/den + num2/(Sc^2 den^2) ]
        num2 = 2*(zx^2 zxx + zy^2 zyy + 2 zx zy zxy),  den = 1 - (S/Sc)^2

    Each cell therefore uses its OWN ``den``.  Differentiating the flux field
    numerically instead would let cells whose gradient exceeds ``Sc`` — where
    the flux is singular and has to be clamped — inject a huge artificial
    convergence into their neighbours (the divergence that 松四ほか 2014, p.180
    describe as "│∇z│>Sc のセルによって値が無限大へと発散する").

    Returns ``(div_over_rho, oversteep_mask)``; ``oversteep_mask`` flags cells
    at or above ``(1-grac)*Sc``, which `soil_depth.m` snaps to bare bedrock.
    """
    # np.gradient(axis0=rows=y, axis1=cols=x)
    dzdy, dzdx = np.gradient(z, dy, dx)
    slope = np.hypot(dzdx, dzdy)
    zp = np.pad(z, 1, mode='edge')          # edge-replicate, never wrap around
    lap = ((zp[:-2, 1:-1] + zp[2:, 1:-1] - 2.0 * z) / (dy * dy)
           + (zp[1:-1, :-2] + zp[1:-1, 2:] - 2.0 * z) / (dx * dx))

    if params.transport == 'linear':
        return -params.K * lap, np.zeros(z.shape, dtype=bool), slope

    Sc = params.Sc
    oversteep = slope >= (1.0 - grac) * Sc
    # den -> 0 as S -> Sc; clamp so the arithmetic stays finite. Over-steep
    # cells are forced to h_min by the caller, so their own value is unused.
    s = np.minimum(slope, (1.0 - grac) * Sc)
    den = 1.0 - (s / Sc) ** 2
    zyy, zyx = np.gradient(dzdy, dy, dx)
    zxy, zxx = np.gradient(dzdx, dy, dx)
    num2 = 2.0 * (dzdx ** 2 * zxx + dzdy ** 2 * zyy
                  + 2.0 * dzdx * dzdy * zxy)
    div_F = params.K * (lap / den + num2 / (Sc * Sc * den * den))
    # q = -rho_soil * F  ->  div(q) = -rho_soil * div(F);  divide by rho_soil.
    return -div_F, oversteep, slope


def soil_depth_massbalance(Z: np.ndarray, x_cellsize: float,
                           y_cellsize: Optional[float] = None,
                           *,
                           params: Optional[SoilProductionParams] = None,
                           endtime: Optional[float] = None,
                           hollow_endtime: Optional[float] = None,
                           h_init: float = 0.0,
                           h_max: float = 3.0,
                           h_min: float = 0.0,
                           smooth_sigma: float = 0.0,
                           verbose: bool = False) -> np.ndarray:
    """Soil thickness (m) from the soil-production function + mass balance.

    Parameters
    ----------
    Z : ndarray
        Elevation raster (m), NaN = nodata.  Taken as the soil surface and held
        fixed (LiDAR/photogrammetric DEM), as in 松四ほか (2016).
    x_cellsize, y_cellsize : float
        Cell size (m).  ``y_cellsize`` defaults to ``x_cellsize``.
    params : SoilProductionParams
        Model parameters; see that class (defaults are NOT Japan-calibrated).
    endtime : float or None
        ``None`` or <= 0 -> steady-state solution (review eq. 14/15).
        Otherwise the transient solution after ``endtime`` years starting from
        ``h_init``.  Divergent cells converge to the steady state within a few
        hundred years; convergent cells (hollows) keep accumulating, so the
        transient mode is what reproduces the hollow-filling behaviour.
    hollow_endtime : float or None
        Only used together with the steady-state mode (``endtime <= 0``).  When
        set, convergent cells (D <= 0, which have no steady state and would
        otherwise sit at ``h_max``) instead get the transient solution after
        ``hollow_endtime`` years from ``h_init``.  Physically: divergent slopes
        have had far longer than their response time to equilibrate, while
        hollows have only been refilling since the last shallow landslide — set
        this to the local landslide return period (松四ほか 2016 report 700-800
        yr for a Japanese granite watershed).
    h_init : float
        Initial soil thickness (m) for the transient solution.
    h_max : float
        Upper cap (m).  Applied to convergent cells, which have no steady
        state, and to the transient solution.
    h_min : float
        Lower cap (m).  0 means bare bedrock is allowed where soil production
        cannot keep up with the export rate (D >= E0).
    smooth_sigma : float
        Optional Gaussian pre-smoothing of the DEM, in cells, before the
        derivatives are taken.  Curvature from a 5 m DEM is noisy; 1-2 cells is
        a reasonable choice.  0 = no smoothing.
    verbose : bool
        Print a short summary of the solution.

    Returns
    -------
    ndarray of float64 with NaN wherever ``Z`` is NaN.
    """
    if params is None:
        params = SoilProductionParams()
    params.validate()

    Z = np.asarray(Z, dtype=np.float64)
    dx = float(x_cellsize)
    dy = float(y_cellsize) if y_cellsize else dx

    z_filled, valid = _fill_nan_nearest(Z)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        z_filled = gaussian_filter(z_filled, sigma=float(smooth_sigma),
                                   mode='nearest')

    div_over_rho, oversteep, slope = _flux_divergence(z_filled, dx, dy, params)

    rho_g = params.rho_soil * 1000.0          # kg m-3 -> g m-3
    # D = div(q) + W_soil : the mass export the soil column has to sustain.
    D = div_over_rho * rho_g + params.W_soil  # g m-2 yr-1
    # Production driven by the slope-normal thickness H = h cos(theta) is the
    # same algebra with a per-cell alpha (松四ほか 2016 eq.5).
    a = (params.alpha * np.cos(np.arctan(slope)) if params.slope_normal
         else params.alpha)
    E0 = params.E0

    # `a` is a scalar, or a per-cell array when slope_normal is on
    a_full = np.broadcast_to(np.asarray(a, dtype=np.float64), D.shape)

    def _solve(t: Optional[float]) -> np.ndarray:
        """Closed-form h for the whole grid; ``t=None`` -> steady state."""
        if t is None:
            # h = -(1/a) ln(D/E0); only 0 < D < E0 gives a positive thickness.
            out = np.full(D.shape, np.nan, dtype=np.float64)
            pos = D > 0
            out[pos] = -np.log(D[pos] / E0) / a_full[pos]
            # D <= 0 (convergent): no steady state -> accumulation, cap it.
            out[~pos] = h_max
            return out
        u0 = np.exp(a_full * h_init)
        u = np.empty(D.shape, dtype=np.float64)
        zero = D == 0.0
        nz = ~zero
        expo = np.clip(-a_full[nz] * D[nz] * t / rho_g, -700.0, 700.0)
        u[nz] = (E0 - (E0 - D[nz] * u0[nz]) * np.exp(expo)) / D[nz]
        # D == 0: pure production, no export
        u[zero] = u0[zero] + a_full[zero] * E0 * t / rho_g
        u = np.where(np.isfinite(u), u, np.inf)
        return np.where(u > 0.0, np.log(np.maximum(u, 1e-300)) / a_full, 0.0)

    steady = endtime is None or endtime <= 0
    composite = steady and hollow_endtime is not None and hollow_endtime > 0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        h = _solve(None if steady else float(endtime))
        if composite:
            # Divergent slopes have had >> 1e4 yr to equilibrate; hollows are
            # refilling since the last shallow landslide, so give them only
            # `hollow_endtime` years of accumulation instead of h_max.
            h = np.where(D > 0, h, _solve(float(hollow_endtime)))

    # Cells at/above the critical gradient are transport-unlimited (landsliding
    # in the Roering framework): soil_depth.m snaps them to bare bedrock.
    h = np.where(oversteep, h_min, h)
    h = np.clip(h, h_min, h_max)
    h[~valid] = np.nan

    if verbose:
        n_valid = max(int(valid.sum()), 1)
        n_conv = int((D[valid] <= 0).sum())
        n_bare = int((D[valid] >= E0).sum())
        n_over = int(oversteep[valid].sum())
        n_cap = int((h[valid] >= h_max).sum())
        mode = (f'steady + hollows {hollow_endtime:.0f} yr' if composite
                else 'steady state' if steady else f'transient {endtime:.0f} yr')
        print(f"  soil_depth (mass balance, {mode}, {params.transport} creep): "
              f"mean={np.nanmean(h):.3f} m, median={np.nanmedian(h):.3f} m, "
              f"max={np.nanmax(h):.3f} m", flush=True)
        note = (f'given {hollow_endtime:.0f} yr of refill' if composite
                else 'all held at h_max' if steady
                else 'still accumulating at the end of the run')
        print(f"    convergent cells (no steady state, {note}): "
              f"{n_conv:,}/{n_valid:,} ({100.0*n_conv/n_valid:.1f}%);  "
              f"reached h_max={h_max:.2f} m: {n_cap:,} "
              f"({100.0*n_cap/n_valid:.1f}%)", flush=True)
        print(f"    export >= E0 (production cannot keep up, thin/bare soil): "
              f"{n_bare:,} ({100.0*n_bare/n_valid:.1f}%);  "
              f"gradient >= Sc (forced to bare bedrock): {n_over:,} "
              f"({100.0*n_over/n_valid:.1f}%)", flush=True)

    return h
