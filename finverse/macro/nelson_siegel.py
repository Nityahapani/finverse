"""
finverse.macro.nelson_siegel — Nelson-Siegel (1987) yield curve model.

Fits a smooth parametric yield curve to observed Treasury yields.
Decomposes the curve into:
  - Level (β0): long-term rate
  - Slope (β1): short-term spread (= - yield curve slope)
  - Curvature (β2): hump/trough in medium-term rates

Extensions: Svensson (1994) adds a second hump (β3, β4).

Applications:
  - Interpolate yields at any maturity
  - Decompose curve into economic components
  - Forward rate extraction
  - Factor-based macro modeling
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class NelsonSiegelResult:
    beta0: float        # level
    beta1: float        # slope
    beta2: float        # curvature
    lambda_: float      # decay factor
    beta3: float | None = None   # Svensson second curvature
    lambda2: float | None = None
    model: str = "Nelson-Siegel"
    fit_error: float = 0.0
    maturities: list[float] = None
    observed_yields: list[float] = None

    @property
    def level(self) -> float:
        return self.beta0

    @property
    def slope(self) -> float:
        return -self.beta1  # negative because beta1 drives short-term down

    @property
    def curvature(self) -> float:
        return self.beta2

    def yield_at(self, maturity: float) -> float:
        """Get fitted yield at a given maturity (in years)."""
        return float(_ns_yield(maturity, self.beta0, self.beta1,
                               self.beta2, self.lambda_))

    def forward_rate(self, maturity: float) -> float:
        """Compute instantaneous forward rate at given maturity."""
        t = max(maturity, 0.001)
        lam = self.lambda_
        b0, b1, b2 = self.beta0, self.beta1, self.beta2
        exp_t = np.exp(-lam * t)
        fwd = (b0
               + b1 * exp_t
               + b2 * (lam * t) * exp_t)
        return float(fwd)

    def curve(self, maturities: list[float] | None = None) -> pd.Series:
        """Return the full fitted curve at given maturities."""
        if maturities is None:
            maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        yields = [self.yield_at(m) for m in maturities]
        return pd.Series(yields, index=maturities, name="yield")

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]{self.model} Yield Curve[/bold blue]\n")

        slope_val = self.yield_at(10) - self.yield_at(2)
        slope_label = "normal" if slope_val > 0.2 else ("flat" if slope_val > -0.1 else "inverted")

        console.print(f"  β₀ (level):     {self.beta0:.4f}  = long-run rate")
        console.print(f"  β₁ (slope):     {self.beta1:.4f}  = short-term loading")
        console.print(f"  β₂ (curvature): {self.beta2:.4f}  = medium-term hump")
        console.print(f"  λ  (decay):     {self.lambda_:.4f}")
        if self.beta3 is not None:
            console.print(f"  β₃ (2nd hump):  {self.beta3:.4f}")
        console.print(f"  Fit RMSE:       {self.fit_error:.4f}%")

        curve_tbl = Table(title="Fitted yield curve", box=box.SIMPLE_HEAD, header_style="bold blue")
        curve_tbl.add_column("Maturity")
        curve_tbl.add_column("Fitted yield", justify="right")

        for m, y in self.curve().items():
            label = {0.25: "3M", 0.5: "6M", 1: "1Y", 2: "2Y", 3: "3Y",
                     5: "5Y", 7: "7Y", 10: "10Y", 20: "20Y", 30: "30Y"}.get(m, f"{m}Y")
            curve_tbl.add_row(label, f"{y:.3%}")

        console.print(curve_tbl)
        console.print(f"\n  10Y-2Y spread: {slope_val:+.3%} ({slope_label})")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.curve().to_frame()


def _ns_yield(t, beta0, beta1, beta2, lam):
    """Nelson-Siegel yield formula."""
    t = max(t, 0.001)
    lt = lam * t
    exp_lt = np.exp(-lt)
    factor1 = (1 - exp_lt) / lt
    factor2 = factor1 - exp_lt
    return beta0 + beta1 * factor1 + beta2 * factor2


def _svensson_yield(t, beta0, beta1, beta2, beta3, lam1, lam2):
    """Svensson (1994) extended yield formula."""
    t = max(t, 0.001)
    lt1 = lam1 * t
    lt2 = lam2 * t
    exp1 = np.exp(-lt1)
    exp2 = np.exp(-lt2)
    f1 = (1 - exp1) / lt1
    f2 = (1 - exp2) / lt2
    return beta0 + beta1 * f1 + beta2 * (f1 - exp1) + beta3 * (f2 - exp2)


def fit(
    maturities: list[float],
    yields: list[float],
    model: str = "Nelson-Siegel",
    lambda_init: float = 0.7,
) -> NelsonSiegelResult:
    """
    Fit Nelson-Siegel (or Svensson) model to observed yields.

    Parameters
    ----------
    maturities   : list of maturities in years (e.g. [0.25, 1, 2, 5, 10, 30])
    yields       : list of observed yields (as decimals, e.g. 0.045 = 4.5%)
    model        : "Nelson-Siegel" or "Svensson" (default "Nelson-Siegel")
    lambda_init  : initial decay parameter (default 0.7)

    Returns
    -------
    NelsonSiegelResult

    Example
    -------
    >>> from finverse.macro import nelson_siegel
    >>> # US Treasury yields (typical)
    >>> maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    >>> yields = [0.053, 0.054, 0.052, 0.048, 0.046, 0.044, 0.045, 0.044, 0.047, 0.045]
    >>> result = nelson_siegel.fit(maturities, yields)
    >>> result.summary()
    >>> print(f"5Y yield: {result.yield_at(5):.3%}")
    >>> print(f"Curve level: {result.level:.3%}")

    From FRED data:
    >>> macro = pull.fred("DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30")
    >>> latest = macro.iloc[-1] / 100  # FRED gives percent
    >>> mats = [0.25, 1, 2, 5, 10, 30]
    >>> yields = latest.values.tolist()
    >>> result = nelson_siegel.fit(mats, yields)
    """
    from finverse.utils.display import console
    console.print(f"[dim]Fitting {model} yield curve ({len(maturities)} points)...[/dim]")

    maturities = np.array(maturities, dtype=float)
    yields = np.array(yields, dtype=float)

    if model == "Svensson":
        def objective(params):
            b0, b1, b2, b3, lam1, lam2 = params
            if lam1 <= 0 or lam2 <= 0:
                return 1e10
            fitted = np.array([_svensson_yield(t, b0, b1, b2, b3, lam1, lam2)
                               for t in maturities])
            return np.sum((fitted - yields)**2)

        x0 = [yields[-1], yields[0] - yields[-1], 0.0, 0.0, lambda_init, lambda_init * 0.5]
        bounds = [
            (0, 0.20), (-0.20, 0.20), (-0.20, 0.20), (-0.20, 0.20),
            (0.01, 5.0), (0.01, 5.0),
        ]
    else:
        def objective(params):
            b0, b1, b2, lam = params
            if lam <= 0:
                return 1e10
            fitted = np.array([_ns_yield(t, b0, b1, b2, lam) for t in maturities])
            return np.sum((fitted - yields)**2)

        x0 = [yields[-1], yields[0] - yields[-1], 0.0, lambda_init]
        bounds = [(0, 0.20), (-0.20, 0.20), (-0.20, 0.20), (0.01, 5.0)]

    best_res = None
    best_val = np.inf

    for lam_try in [0.3, 0.5, 0.7, 1.0, 1.5]:
        x0_try = x0.copy()
        x0_try[-1] = lam_try
        if model == "Svensson":
            x0_try[-2] = lam_try
            x0_try[-1] = lam_try * 0.5

        res = minimize(objective, x0_try, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    params = best_res.x
    rmse = float(np.sqrt(best_val / len(maturities))) * 100

    if model == "Svensson":
        b0, b1, b2, b3, lam1, lam2 = params
        result = NelsonSiegelResult(
            beta0=round(b0, 6), beta1=round(b1, 6),
            beta2=round(b2, 6), lambda_=round(lam1, 6),
            beta3=round(b3, 6), lambda2=round(lam2, 6),
            model="Svensson", fit_error=round(rmse, 6),
            maturities=list(maturities), observed_yields=list(yields),
        )
    else:
        b0, b1, b2, lam = params
        result = NelsonSiegelResult(
            beta0=round(b0, 6), beta1=round(b1, 6),
            beta2=round(b2, 6), lambda_=round(lam, 6),
            model="Nelson-Siegel", fit_error=round(rmse, 6),
            maturities=list(maturities), observed_yields=list(yields),
        )

    console.print(
        f"[green]✓[/green] {model} fitted — "
        f"RMSE: {rmse:.4f}%  |  "
        f"Level: {result.beta0:.3%}  |  "
        f"Slope: {result.beta1:.3%}  |  "
        f"10Y-2Y: {result.yield_at(10)-result.yield_at(2):+.3%}"
    )

    return result


def us_curve(macro_df: pd.DataFrame | None = None) -> NelsonSiegelResult:
    """
    Fit Nelson-Siegel to current US Treasury curve.

    Uses FRED data if provided (DGS3MO, DGS1, DGS2, DGS5, DGS10, DGS30),
    otherwise uses recent typical values as a demonstration.

    Example
    -------
    >>> macro = pull.fred("DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30")
    >>> result = nelson_siegel.us_curve(macro)
    >>> result.summary()
    """
    maturities = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]

    if macro_df is not None and not macro_df.empty:
        fred_map = {
            "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
            "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0,
            "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
        }
        latest = macro_df.iloc[-1]
        obs_mats = []
        obs_yields = []
        for series_id, mat in fred_map.items():
            if series_id in latest.index:
                val = latest[series_id]
                if not np.isnan(val) and val > 0:
                    obs_mats.append(mat)
                    obs_yields.append(val / 100)

        if len(obs_mats) >= 4:
            return fit(obs_mats, obs_yields)

    typical_yields = [0.0530, 0.0520, 0.0480, 0.0440, 0.0440, 0.0450]
    return fit(maturities, typical_yields)
