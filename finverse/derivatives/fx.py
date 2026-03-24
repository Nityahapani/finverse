"""
finverse.derivatives.fx
=======================
FX derivatives: forwards (CIP), cross-currency basis swaps, FX options
(Garman-Kohlhagen), and currency-adjusted WACC for multinational DCF.

Usage
-----
from finverse.derivatives import fx

fwd = fx.forward(spot=1.085, r_domestic=0.053, r_foreign=0.038, tenor=1.0, pair='EURUSD')
fwd.summary()

ccs = fx.cross_currency_swap(notional_usd=10_000_000, pair='EURUSD', spot=1.085,
                              tenor=3, basis_spread=-0.0010)
ccs.summary()

opt = fx.option(spot=1.085, strike=1.10, tenor=0.5, r_domestic=0.053,
                r_foreign=0.038, sigma=0.085, type='call', pair='EURUSD')
opt.summary()
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm  # type: ignore

OptionType = Literal["call", "put"]


# ── Garman-Kohlhagen (GK) helpers ────────────────────────────────────────────

def _gk_d1(S, K, T, r_d, r_f, sigma):
    return (math.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _gk_d2(S, K, T, r_d, r_f, sigma):
    return _gk_d1(S, K, T, r_d, r_f, sigma) - sigma * math.sqrt(T)


# ── FX Forward ───────────────────────────────────────────────────────────────

@dataclass
class FXForwardResult:
    pair: str
    spot: float
    r_domestic: float
    r_foreign: float
    tenor: float

    forward_rate: float = 0.0
    forward_points: float = 0.0   # in pips = (fwd - spot) * 10000
    cip_implied_rate: float = 0.0

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"FX Forward — {self.pair}  tenor={self.tenor}y")
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", justify="right")
            t.add_row("Spot", f"{self.spot:.5f}")
            t.add_row("Forward Rate (CIP)", f"{self.forward_rate:.5f}")
            t.add_row("Forward Points", f"{self.forward_points:+.2f} pips")
            t.add_row("Domestic Rate", f"{self.r_domestic:.3%}")
            t.add_row("Foreign Rate", f"{self.r_foreign:.3%}")
            t.add_row("CIP Implied Foreign Rate", f"{self.cip_implied_rate:.3%}")
            console.print(t)
        except ImportError:
            print(f"FX Fwd [{self.pair}]  fwd={self.forward_rate:.5f}  pts={self.forward_points:+.2f}")


def forward(
    spot: float,
    r_domestic: float,
    r_foreign: float,
    tenor: float,
    pair: str = "",
) -> FXForwardResult:
    """
    FX forward rate via Covered Interest Parity (CIP).

    F = S * (1 + r_d)^T / (1 + r_f)^T
    """
    fwd = spot * ((1 + r_domestic) ** tenor) / ((1 + r_foreign) ** tenor)
    fwd_points = (fwd - spot) * 10000
    # Implied foreign rate from observed forward
    cip_implied = (spot / fwd) ** (1 / tenor) * (1 + r_domestic) - 1

    return FXForwardResult(
        pair=pair,
        spot=spot,
        r_domestic=r_domestic,
        r_foreign=r_foreign,
        tenor=tenor,
        forward_rate=fwd,
        forward_points=fwd_points,
        cip_implied_rate=cip_implied,
    )


# ── Cross-Currency Swap ───────────────────────────────────────────────────────

@dataclass
class CrossCurrencySwapResult:
    pair: str
    notional_usd: float
    spot: float
    tenor: float
    basis_spread: float

    npv: float = 0.0
    notional_foreign: float = 0.0
    basis_spread_bps: float = 0.0
    fx_delta: float = 0.0  # approximate

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"Cross-Currency Swap — {self.pair}  tenor={self.tenor}y")
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", justify="right")
            t.add_row("Notional (USD)", f"${self.notional_usd:,.0f}")
            t.add_row("Notional (Foreign)", f"{self.notional_foreign:,.0f}")
            t.add_row("Basis Spread", f"{self.basis_spread_bps:+.1f} bps")
            t.add_row("NPV", f"${self.npv:,.2f}")
            t.add_row("FX Delta (approx.)", f"${self.fx_delta:,.2f}")
            console.print(t)
        except ImportError:
            print(f"CCS [{self.pair}]  NPV={self.npv:,.2f}  basis={self.basis_spread_bps:.1f}bps")


def cross_currency_swap(
    notional_usd: float,
    pair: str,
    spot: float,
    tenor: float,
    basis_spread: float = 0.0,
    r_usd: float = 0.053,
    r_foreign: float = 0.038,
) -> CrossCurrencySwapResult:
    """
    Price a cross-currency basis swap.

    Simplified: NPV = effect of basis spread on foreign leg payments.
    basis_spread : float — e.g. -0.0010 means -10 bps (pay basis)
    """
    notional_foreign = notional_usd / spot
    basis_spread_bps = basis_spread * 10000

    # Present value impact of the basis spread over the tenor
    # Approximate: NPV = notional * basis_spread * sum(P(0, t_i) * dt)
    dt = 0.5  # semi-annual
    n_periods = int(tenor / dt)
    annuity = sum(math.exp(-r_usd * i * dt) * dt for i in range(1, n_periods + 1))
    npv = notional_usd * basis_spread * annuity

    # FX delta: change in NPV for 1% move in spot
    fx_delta = notional_foreign * 0.01 * spot  # first-order approximation

    return CrossCurrencySwapResult(
        pair=pair,
        notional_usd=notional_usd,
        spot=spot,
        tenor=tenor,
        basis_spread=basis_spread,
        npv=npv,
        notional_foreign=notional_foreign,
        basis_spread_bps=basis_spread_bps,
        fx_delta=fx_delta,
    )


# ── FX Option (Garman-Kohlhagen) ─────────────────────────────────────────────

@dataclass
class FXOptionResult:
    pair: str
    option_type: str
    spot: float
    strike: float
    tenor: float
    r_domestic: float
    r_foreign: float
    sigma: float

    price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    breakeven: float = 0.0

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"FX Option ({self.option_type.upper()}) — {self.pair}")
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", justify="right")
            t.add_row("Spot", f"{self.spot:.5f}")
            t.add_row("Strike", f"{self.strike:.5f}")
            t.add_row("Tenor", f"{self.tenor:.2f}y")
            t.add_row("σ (vol)", f"{self.sigma:.2%}")
            t.add_row("Price (domestic)", f"{self.price:.6f}")
            t.add_row("Delta", f"{self.delta:+.4f}")
            t.add_row("Gamma", f"{self.gamma:.6f}")
            t.add_row("Vega (per 1% σ)", f"{self.vega:.6f}")
            t.add_row("Breakeven", f"{self.breakeven:.5f}")
            console.print(t)
        except ImportError:
            print(f"FX Opt [{self.pair} {self.option_type}]  price={self.price:.6f}  delta={self.delta:+.4f}")


def option(
    spot: float,
    strike: float,
    tenor: float,
    r_domestic: float,
    r_foreign: float,
    sigma: float,
    type: OptionType = "call",
    pair: str = "",
) -> FXOptionResult:
    """
    Price a European FX option using the Garman-Kohlhagen model.
    (Black-Scholes treating r_foreign as continuous dividend yield.)
    """
    if tenor <= 0 or sigma <= 0:
        raise ValueError("tenor and sigma must be positive")

    d1 = _gk_d1(spot, strike, tenor, r_domestic, r_foreign, sigma)
    d2 = _gk_d2(spot, strike, tenor, r_domestic, r_foreign, sigma)
    sqrt_T = math.sqrt(tenor)

    exp_rf_T = math.exp(-r_foreign * tenor)
    exp_rd_T = math.exp(-r_domestic * tenor)
    npdf_d1 = norm.pdf(d1)

    if type == "call":
        price_val = spot * exp_rf_T * norm.cdf(d1) - strike * exp_rd_T * norm.cdf(d2)
        delta = exp_rf_T * norm.cdf(d1)
        breakeven = strike + price_val
    else:
        price_val = strike * exp_rd_T * norm.cdf(-d2) - spot * exp_rf_T * norm.cdf(-d1)
        delta = -exp_rf_T * norm.cdf(-d1)
        breakeven = strike - price_val

    gamma = exp_rf_T * npdf_d1 / (spot * sigma * sqrt_T)
    vega = spot * exp_rf_T * npdf_d1 * sqrt_T / 100

    return FXOptionResult(
        pair=pair,
        option_type=type,
        spot=spot,
        strike=strike,
        tenor=tenor,
        r_domestic=r_domestic,
        r_foreign=r_foreign,
        sigma=sigma,
        price=max(price_val, 0.0),
        delta=delta,
        gamma=gamma,
        vega=vega,
        breakeven=breakeven,
    )


# ── Currency-Adjusted WACC ───────────────────────────────────────────────────

def currency_adjusted_wacc(
    base_wacc: float,
    revenue_fx_exposure: dict[str, float],
    tenor: float = 5.0,
    hedging_cost_spread: float = 0.003,
) -> float:
    """
    Compute a currency-adjusted WACC for multinational DCF models.

    For each foreign currency bucket, the hedging cost (forward basis) is
    added proportionally to the base WACC.

    Parameters
    ----------
    base_wacc : float — USD or domestic WACC (e.g. 0.095)
    revenue_fx_exposure : dict — {currency: share}, e.g. {'EUR': 0.35, 'GBP': 0.20}
    tenor : float — average hedging horizon in years
    hedging_cost_spread : float — assumed hedging cost spread per year (default 30bps)

    Returns
    -------
    float — adjusted WACC
    """
    total_foreign = sum(revenue_fx_exposure.values())
    if total_foreign > 1.0:
        revenue_fx_exposure = {k: v / total_foreign for k, v in revenue_fx_exposure.items()}

    fx_cost_adjustment = sum(
        share * hedging_cost_spread
        for share in revenue_fx_exposure.values()
    )

    adjusted_wacc = base_wacc + fx_cost_adjustment
    return round(adjusted_wacc, 6)
