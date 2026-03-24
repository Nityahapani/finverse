"""
finverse.options.implied_vol
Implied volatility solver using Brent's method via scipy.
"""
from __future__ import annotations

import math
from typing import Literal

from scipy.optimize import brentq  # type: ignore

OptionType = Literal["call", "put"]


def solve_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float | None:
    """
    Solve for implied volatility given a market price.
    Uses Brent's method on the Black-Scholes price function.

    Returns None if solution cannot be found (e.g. deep ITM/OTM price violations).
    """
    from finverse.options.black_scholes import price as bs_price

    def objective(sigma: float) -> float:
        return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, type=type).price - market_price

    # Intrinsic value bounds check
    intrinsic = max(S - K, 0.0) if type == "call" else max(K - S, 0.0)
    upper_bound = S if type == "call" else K * math.exp(-r * T)
    if market_price < intrinsic or market_price > upper_bound * 1.01:
        return None

    try:
        iv = brentq(objective, 1e-6, 20.0, xtol=tol, maxiter=max_iter)
        return float(iv)
    except ValueError:
        return None


def iv_from_params(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    type: OptionType = "call",
) -> float | None:
    """Convenience alias for solve_iv."""
    return solve_iv(market_price, S, K, T, r, type)
