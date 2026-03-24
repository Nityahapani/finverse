"""
finverse.options.black_scholes
Black-Scholes closed-form option pricing and Greeks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from scipy.stats import norm  # type: ignore


OptionType = Literal["call", "put"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class OptionResult:
    """Full pricing output for a single European option."""
    option_type: str
    S: float
    K: float
    T: float
    r: float
    sigma: float

    price: float = 0.0
    intrinsic_value: float = 0.0
    time_value: float = 0.0

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # $ per day
    vega: float = 0.0    # $ per 1% vol change
    rho: float = 0.0     # $ per 1% rate change

    implied_vol: float | None = None
    breakeven: float = 0.0

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"Option Pricing — {self.option_type.upper()}  S={self.S}  K={self.K}  T={self.T:.2f}y")
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", justify="right")
            rows = [
                ("Price", f"${self.price:.4f}"),
                ("Intrinsic Value", f"${self.intrinsic_value:.4f}"),
                ("Time Value", f"${self.time_value:.4f}"),
                ("Breakeven", f"${self.breakeven:.2f}"),
                ("─── Greeks ───", ""),
                ("Delta", f"{self.delta:+.4f}"),
                ("Gamma", f"{self.gamma:.6f}"),
                ("Theta ($/day)", f"${self.theta:.4f}"),
                ("Vega (per 1% σ)", f"${self.vega:.4f}"),
                ("Rho (per 1% r)", f"${self.rho:.4f}"),
            ]
            if self.implied_vol is not None:
                rows.append(("Implied Vol", f"{self.implied_vol:.2%}"))
            for k, v in rows:
                t.add_row(k, v)
            console.print(t)
        except ImportError:
            print(f"Option [{self.option_type}]  price={self.price:.4f}  delta={self.delta:+.4f}  "
                  f"gamma={self.gamma:.6f}  theta={self.theta:.4f}  vega={self.vega:.4f}")


# ── pricing engine ────────────────────────────────────────────────────────────

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    type: OptionType = "call",
    market_price: float | None = None,
) -> OptionResult:
    """
    Price a European option using Black-Scholes.

    Parameters
    ----------
    S : float  — spot price
    K : float  — strike price
    T : float  — time to expiry in years
    r : float  — risk-free rate (e.g. 0.053)
    sigma : float — volatility (e.g. 0.28)
    type : 'call' or 'put'
    market_price : optional — if supplied, also compute implied vol
    """
    if T <= 0:
        raise ValueError("T must be positive (time to expiry in years)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_neg_d1 = norm.cdf(-d1)
    n_neg_d2 = norm.cdf(-d2)
    npdf_d1 = norm.pdf(d1)

    if type == "call":
        opt_price = S * nd1 - K * math.exp(-r * T) * nd2
        intrinsic = max(S - K, 0.0)
        delta = nd1
        rho_val = K * T * math.exp(-r * T) * nd2 / 100
        breakeven = K + opt_price if type == "call" else K - opt_price
    else:
        opt_price = K * math.exp(-r * T) * n_neg_d2 - S * n_neg_d1
        intrinsic = max(K - S, 0.0)
        delta = nd1 - 1
        rho_val = -K * T * math.exp(-r * T) * n_neg_d2 / 100
        breakeven = K - opt_price

    gamma = npdf_d1 / (S * sigma * sqrt_T)
    # theta: daily decay
    theta = (-(S * npdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * (nd2 if type == "call" else -n_neg_d2)) / 365
    vega = S * npdf_d1 * sqrt_T / 100

    iv = None
    if market_price is not None:
        from finverse.options.implied_vol import solve_iv
        iv = solve_iv(market_price, S, K, T, r, type)

    return OptionResult(
        option_type=type,
        S=S, K=K, T=T, r=r, sigma=sigma,
        price=opt_price,
        intrinsic_value=intrinsic,
        time_value=opt_price - intrinsic,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho_val,
        implied_vol=iv,
        breakeven=breakeven,
    )
