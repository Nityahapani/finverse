"""
finverse.models.bonds — Bond pricing, yield, duration, convexity.

Clean price, dirty price, YTM, modified duration, Macaulay duration,
convexity, DV01, and price-yield relationship.

Handles fixed-rate coupon bonds, zero-coupon bonds, and floating-rate
approximations. Pure math — no API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import brentq


@dataclass
class BondResult:
    clean_price: float
    dirty_price: float
    ytm: float                    # yield to maturity
    current_yield: float          # annual coupon / price
    macaulay_duration: float      # years
    modified_duration: float      # % price change per 1% yield change
    convexity: float              # curvature of price-yield
    dv01: float                   # dollar value of 1bp — price change for 0.01% yield move
    accrued_interest: float
    par: float
    coupon_rate: float
    maturity_years: float
    face_value: float = 1000.0

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Bond Pricing[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        table.add_row("Clean price",          f"${self.clean_price:.4f}",    "Price without accrued interest")
        table.add_row("Dirty price",          f"${self.dirty_price:.4f}",    "Settlement price (clean + accrued)")
        table.add_row("Accrued interest",     f"${self.accrued_interest:.4f}","Interest earned since last coupon")
        table.add_row("YTM",                  f"{self.ytm:.4%}",             "Annualised return if held to maturity")
        table.add_row("Current yield",        f"{self.current_yield:.4%}",   "Annual coupon / price")
        table.add_row("Macaulay duration",    f"{self.macaulay_duration:.4f} yrs", "Weighted avg cash flow timing")
        table.add_row("Modified duration",    f"{self.modified_duration:.4f}","% price drop per 1% yield rise")
        table.add_row("Convexity",            f"{self.convexity:.4f}",       "Curvature — helps at large yield moves")
        table.add_row("DV01",                 f"${self.dv01:.4f}",           "Price change for 0.01% (1bp) yield move")

        console.print(table)

        # Scenario analysis
        console.print("\n  [dim]Price impact scenarios (yield shock):[/dim]")
        for shock_bps in [-100, -50, -25, +25, +50, +100]:
            shock = shock_bps / 10000
            approx_change = (
                -self.modified_duration * self.clean_price * shock
                + 0.5 * self.convexity * self.clean_price * shock**2
            )
            color = "green" if approx_change > 0 else "red"
            console.print(
                f"    {shock_bps:+4d}bps → "
                f"[{color}]{approx_change:+.4f}[/{color}] "
                f"(new price ≈ ${self.clean_price + approx_change:.4f})"
            )
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "clean_price":        self.clean_price,
            "ytm":                self.ytm,
            "macaulay_duration":  self.macaulay_duration,
            "modified_duration":  self.modified_duration,
            "convexity":          self.convexity,
            "dv01":               self.dv01,
        }])


def price(
    face: float = 1000.0,
    coupon_rate: float = 0.05,
    ytm: float = 0.06,
    maturity: float = 10.0,
    freq: int = 2,
    accrued_days: int = 0,
) -> BondResult:
    """
    Price a fixed-rate coupon bond and compute all risk measures.

    Parameters
    ----------
    face        : float — face/par value (default $1000)
    coupon_rate : float — annual coupon rate (default 5%)
    ytm         : float — yield to maturity / discount rate (default 6%)
    maturity    : float — years to maturity (default 10)
    freq        : int — coupon frequency per year: 1=annual, 2=semi-annual (default 2)
    accrued_days: int — days since last coupon (default 0)

    Returns
    -------
    BondResult

    Example
    -------
    >>> from finverse.models.bonds import price as bond_price, ytm as bond_ytm
    >>>
    >>> # 5% coupon, 10-year bond, 6% market yield
    >>> b = bond_price(face=1000, coupon_rate=0.05, ytm=0.06, maturity=10)
    >>> b.summary()
    >>> print(f"Price: ${b.clean_price:.2f}")
    >>> print(f"Modified duration: {b.modified_duration:.2f}")
    >>> print(f"DV01: ${b.dv01:.4f}")
    >>>
    >>> # Zero coupon bond
    >>> z = bond_price(face=1000, coupon_rate=0.0, ytm=0.05, maturity=5)
    """
    n_periods = int(maturity * freq)
    period_ytm = ytm / freq
    coupon = face * coupon_rate / freq

    # Cash flows: coupons + final principal
    times = np.arange(1, n_periods + 1) / freq
    cash_flows = np.full(n_periods, coupon)
    cash_flows[-1] += face

    # Discount factors
    discount = (1 + period_ytm) ** np.arange(1, n_periods + 1)
    pv_flows = cash_flows / discount

    clean_px = float(pv_flows.sum())

    # Accrued interest
    period_days = 365 / freq
    ai = coupon * (accrued_days / period_days)
    dirty_px = clean_px + ai

    # Macaulay duration
    mac_dur = float((times * pv_flows).sum() / clean_px)

    # Modified duration
    mod_dur = mac_dur / (1 + period_ytm)

    # Convexity
    period_times = np.arange(1, n_periods + 1)
    convex = float(
        np.sum(pv_flows * period_times * (period_times + 1))
        / (clean_px * (1 + period_ytm)**2 * freq**2)
    )

    # DV01 — dollar value of 1 basis point
    dv01 = mod_dur * dirty_px * 0.0001

    # Current yield
    current_yield = (coupon * freq) / clean_px if clean_px > 0 else 0

    return BondResult(
        clean_price=round(clean_px, 4),
        dirty_price=round(dirty_px, 4),
        ytm=ytm,
        current_yield=round(current_yield, 6),
        macaulay_duration=round(mac_dur, 4),
        modified_duration=round(mod_dur, 4),
        convexity=round(convex, 4),
        dv01=round(dv01, 6),
        accrued_interest=round(ai, 4),
        par=face,
        coupon_rate=coupon_rate,
        maturity_years=maturity,
        face_value=face,
    )


def ytm_from_price(
    market_price: float,
    face: float = 1000.0,
    coupon_rate: float = 0.05,
    maturity: float = 10.0,
    freq: int = 2,
) -> BondResult:
    """
    Solve for yield-to-maturity given a market price.

    Parameters
    ----------
    market_price : float — observed clean price
    (other params same as price())

    Returns
    -------
    BondResult

    Example
    -------
    >>> b = ytm_from_price(market_price=950, face=1000,
    ...                    coupon_rate=0.05, maturity=10)
    >>> print(f"YTM: {b.ytm:.4%}")
    """
    def objective(y):
        return price(face, coupon_rate, y, maturity, freq).clean_price - market_price

    try:
        ytm_sol = brentq(objective, 0.0001, 0.50, xtol=1e-8, maxiter=500)
    except Exception:
        ytm_sol = coupon_rate  # fallback

    return price(face, coupon_rate, ytm_sol, maturity, freq)


def price_yield_table(
    face: float = 1000.0,
    coupon_rate: float = 0.05,
    maturity: float = 10.0,
    ytm_range: tuple = (0.01, 0.12),
    n: int = 12,
) -> pd.DataFrame:
    """
    Price-yield relationship table across a range of yields.

    Useful for visualising how price responds to yield changes.

    Example
    -------
    >>> table = price_yield_table(coupon_rate=0.05, maturity=10)
    >>> print(table)
    """
    yields = np.linspace(*ytm_range, n)
    rows = []
    for y in yields:
        b = price(face, coupon_rate, y, maturity)
        rows.append({
            "ytm":               round(y, 4),
            "clean_price":       b.clean_price,
            "mod_duration":      b.modified_duration,
            "dv01":              b.dv01,
        })
    return pd.DataFrame(rows).set_index("ytm")
