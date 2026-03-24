"""
finverse.options.binomial
Cox-Ross-Rubinstein (CRR) binomial tree for American option pricing.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from finverse.options.black_scholes import OptionResult

OptionType = Literal["call", "put"]


def price_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    type: OptionType = "put",
    steps: int = 500,
) -> OptionResult:
    """
    Price an American option using the CRR binomial tree.

    Parameters
    ----------
    S : float  — spot price
    K : float  — strike price
    T : float  — time to expiry in years
    r : float  — risk-free rate
    sigma : float — volatility
    type : 'call' or 'put'
    steps : int — number of binomial steps (default 500 for accuracy)
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    discount = math.exp(-r * dt)

    # Build terminal stock prices
    S_T = np.array([S * (u ** (steps - 2 * j)) for j in range(steps + 1)])

    # Terminal payoffs
    if type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction with early exercise
    for i in range(steps - 1, -1, -1):
        S_i = np.array([S * (u ** (i - 2 * j)) for j in range(i + 1)])
        V = discount * (p * V[:-1] + (1 - p) * V[1:])
        if type == "call":
            early_ex = np.maximum(S_i - K, 0.0)
        else:
            early_ex = np.maximum(K - S_i, 0.0)
        V = np.maximum(V, early_ex)

    opt_price = float(V[0])
    intrinsic = max(S - K, 0.0) if type == "call" else max(K - S, 0.0)

    # Approximate delta from tree (two-step finite difference)
    S_up = S * u
    S_dn = S * d

    def _price_binomial(s: float) -> float:
        S_T_ = np.array([s * (u ** (steps - 2 * j)) for j in range(steps + 1)])
        V_ = np.maximum(S_T_ - K, 0.0) if type == "call" else np.maximum(K - S_T_, 0.0)
        for i in range(steps - 1, -1, -1):
            S_i_ = np.array([s * (u ** (i - 2 * j)) for j in range(i + 1)])
            V_ = discount * (p * V_[:-1] + (1 - p) * V_[1:])
            ex_ = np.maximum(S_i_ - K, 0.0) if type == "call" else np.maximum(K - S_i_, 0.0)
            V_ = np.maximum(V_, ex_)
        return float(V_[0])

    delta = (_price_binomial(S_up) - _price_binomial(S_dn)) / (S_up - S_dn)
    breakeven = K + opt_price if type == "call" else K - opt_price

    return OptionResult(
        option_type=f"american_{type}",
        S=S, K=K, T=T, r=r, sigma=sigma,
        price=opt_price,
        intrinsic_value=intrinsic,
        time_value=opt_price - intrinsic,
        delta=delta,
        gamma=0.0,   # not computed for American binomial to keep it fast
        theta=0.0,
        vega=0.0,
        rho=0.0,
        implied_vol=None,
        breakeven=breakeven,
    )
