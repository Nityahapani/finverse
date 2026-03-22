"""
finverse.credit.merton — Merton (1974) structural credit model.

Treats equity as a call option on the firm's assets. Derives:
- Asset value and asset volatility (from equity + debt)
- Distance to Default (DD)
- Probability of Default (PD)
- Implied credit spread

Pure math — no API keys needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import fsolve


@dataclass
class MertonResult:
    ticker: str
    asset_value: float              # implied firm asset value ($B)
    asset_vol: float                # implied asset volatility (annual)
    equity_value: float             # market cap ($B)
    equity_vol: float               # observed equity vol (annual)
    debt_face: float                # face value of debt ($B)
    distance_to_default: float      # DD in standard deviations
    prob_default_1y: float          # 1-year PD
    prob_default_5y: float          # 5-year PD
    implied_spread: float           # credit spread in bps
    risk_free: float
    rating_equivalent: str          # approximate rating

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        dd_color = (
            "green" if self.distance_to_default > 4
            else "yellow" if self.distance_to_default > 2
            else "red"
        )
        pd_color = (
            "green" if self.prob_default_1y < 0.01
            else "yellow" if self.prob_default_1y < 0.05
            else "red"
        )

        console.print(f"\n[bold blue]Merton Credit Model — {self.ticker}[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row("Asset value",          f"${self.asset_value:.1f}B")
        table.add_row("Asset volatility",     f"{self.asset_vol:.2%}")
        table.add_row("Equity value",         f"${self.equity_value:.1f}B")
        table.add_row("Equity volatility",    f"{self.equity_vol:.2%}")
        table.add_row("Debt face value",      f"${self.debt_face:.1f}B")
        table.add_row(
            "Distance to Default",
            f"[{dd_color}][bold]{self.distance_to_default:.2f}σ[/bold][/{dd_color}]"
        )
        table.add_row(
            "P(Default) 1-year",
            f"[{pd_color}][bold]{self.prob_default_1y:.2%}[/bold][/{pd_color}]"
        )
        table.add_row("P(Default) 5-year",   f"{self.prob_default_5y:.2%}")
        table.add_row("Implied credit spread",f"{self.implied_spread:.0f} bps")
        table.add_row("Rating equivalent",   f"[bold]{self.rating_equivalent}[/bold]")
        console.print(table)
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "asset_value": self.asset_value,
            "asset_vol": self.asset_vol,
            "distance_to_default": self.distance_to_default,
            "prob_default_1y": self.prob_default_1y,
            "implied_spread_bps": self.implied_spread,
            "rating": self.rating_equivalent,
        }])


def _d1(V, D, r, sigma, T):
    return (np.log(V / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(V, D, r, sigma, T):
    return _d1(V, D, r, sigma, T) - sigma * np.sqrt(T)


def _merton_equity(V, E, D, r, sigma_V, T):
    """System of equations to solve for asset value and vol."""
    d1 = _d1(V, D, r, sigma_V, T)
    d2 = _d2(V, D, r, sigma_V, T)
    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    return eq1


def _rating_from_pd(pd_1y: float) -> str:
    """Approximate rating from 1-year PD."""
    thresholds = [
        (0.001, "AAA"),
        (0.002, "AA+"),
        (0.004, "AA"),
        (0.006, "AA-"),
        (0.010, "A+"),
        (0.015, "A"),
        (0.020, "A-"),
        (0.030, "BBB+"),
        (0.050, "BBB"),
        (0.080, "BBB-"),
        (0.120, "BB+"),
        (0.180, "BB"),
        (0.250, "BB-"),
        (0.350, "B+"),
        (0.500, "B"),
        (0.700, "B-"),
        (1.000, "CCC/C"),
    ]
    for threshold, rating in thresholds:
        if pd_1y <= threshold:
            return rating
    return "D"


def analyze(
    data,
    risk_free: float = 0.045,
    debt_maturity: float = 1.0,
    garch_vol: float | None = None,
) -> MertonResult:
    """
    Compute Merton distance-to-default and probability of default.

    Parameters
    ----------
    data          : TickerData — needs price_history, balance_sheet, info
    risk_free     : float — risk-free rate (default 4.5%)
    debt_maturity : float — average debt maturity in years (default 1.0)
    garch_vol     : float — use GARCH conditional vol instead of historical
                    (pass result.current_vol from garch.fit())

    Returns
    -------
    MertonResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.credit import merton
    >>> data = pull.ticker("AAPL")
    >>> result = merton.analyze(data)
    >>> result.summary()

    With GARCH vol:
    >>> from finverse.ml import garch
    >>> garch_result = garch.fit(data)
    >>> result = merton.analyze(data, garch_vol=garch_result.current_vol)
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Computing Merton model for {ticker}...[/dim]")

    mkt_cap = data.market_cap / 1e9 if data.market_cap else None
    if mkt_cap is None and not data.price_history.empty:
        price = float(data.price_history["Close"].iloc[-1])
        shares = data.shares_outstanding or 15e9
        mkt_cap = price * shares / 1e9

    mkt_cap = mkt_cap or 100.0

    debt = data.total_debt or 0.0
    if debt == 0 and hasattr(data, "info"):
        debt = (data.info.get("totalDebt", 0) or 0) / 1e9

    if garch_vol is not None:
        equity_vol = garch_vol
    elif not data.price_history.empty:
        returns = data.price_history["Close"].pct_change().dropna()
        equity_vol = float(returns.std() * np.sqrt(252))
    else:
        equity_vol = 0.25

    E = mkt_cap
    D = max(debt, 0.01)
    sigma_E = equity_vol
    r = risk_free
    T = debt_maturity

    sigma_V_init = sigma_E * E / (E + D)
    V_init = E + D

    def equations(x):
        V, sigma_V = x
        if V <= 0 or sigma_V <= 0:
            return [1e6, 1e6]
        d1 = _d1(V, D, r, sigma_V, T)
        d2 = _d2(V, D, r, sigma_V, T)
        eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
        return [eq1, eq2]

    try:
        solution = fsolve(equations, [V_init, sigma_V_init], full_output=True)
        V_sol, sigma_V_sol = solution[0]
        if V_sol <= 0 or sigma_V_sol <= 0:
            raise ValueError("Invalid solution")
    except Exception:
        V_sol = E + D
        sigma_V_sol = sigma_E * E / max(E + D, 0.01)

    V_sol = max(V_sol, 0.01)
    sigma_V_sol = max(sigma_V_sol, 0.001)

    mu_V = r
    dd = (np.log(V_sol / D) + (mu_V - 0.5 * sigma_V_sol**2) * T) / (sigma_V_sol * np.sqrt(T))
    pd_1y = float(norm.cdf(-dd))
    pd_5y = float(norm.cdf(-(np.log(V_sol / D) + (mu_V - 0.5 * sigma_V_sol**2) * 5) / (sigma_V_sol * np.sqrt(5))))

    if pd_1y > 1e-6:
        spread_bps = -np.log(1 - pd_1y) * 10000 / T
    else:
        spread_bps = 1.0

    rating = _rating_from_pd(pd_1y)

    console.print(
        f"[green]✓[/green] Merton model — "
        f"DD={dd:.2f}σ, PD(1y)={pd_1y:.2%}, "
        f"spread={spread_bps:.0f}bps, rating≈{rating}"
    )

    return MertonResult(
        ticker=ticker,
        asset_value=round(V_sol, 2),
        asset_vol=round(sigma_V_sol, 4),
        equity_value=round(E, 2),
        equity_vol=round(sigma_E, 4),
        debt_face=round(D, 2),
        distance_to_default=round(dd, 4),
        prob_default_1y=round(pd_1y, 6),
        prob_default_5y=round(min(pd_5y, 1.0), 6),
        implied_spread=round(spread_bps, 1),
        risk_free=risk_free,
        rating_equivalent=rating,
    )
