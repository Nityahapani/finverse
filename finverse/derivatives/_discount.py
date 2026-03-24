"""
finverse.derivatives._discount
Bootstrap discount factors and forward rates from a Nelson-Siegel curve
or a flat rate assumption (fallback).
"""
from __future__ import annotations

import math
from typing import Any


def discount_factor(t: float, curve: Any | None = None, flat_rate: float = 0.05) -> float:
    """
    Return P(0, t) — the discount factor for maturity t years.

    Uses Nelson-Siegel curve if provided, otherwise flat rate.
    """
    if curve is None:
        return math.exp(-flat_rate * t)
    try:
        r = curve.yield_at(t)
        return math.exp(-r * t)
    except Exception:
        return math.exp(-flat_rate * t)


def forward_rate(t1: float, t2: float, curve: Any | None = None, flat_rate: float = 0.05) -> float:
    """
    Return the forward rate F(t1, t2) implied by discount factors.
    """
    P1 = discount_factor(t1, curve, flat_rate)
    P2 = discount_factor(t2, curve, flat_rate)
    dt = t2 - t1
    if dt <= 0 or P1 <= 0 or P2 <= 0:
        return flat_rate
    return (P1 / P2 - 1) / dt


def par_swap_rate(
    tenor: float,
    payment_freq: str = "semi-annual",
    curve: Any | None = None,
    flat_rate: float = 0.05,
) -> float:
    """
    Compute the par swap rate for a given tenor.

    par_rate = (1 - P(0, T)) / sum(P(0, t_i) * dt)
    """
    freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12}
    n_per_year = freq_map.get(payment_freq, 2)
    dt = 1 / n_per_year
    n_periods = int(tenor * n_per_year)

    annuity = sum(
        discount_factor(i * dt, curve, flat_rate) * dt
        for i in range(1, n_periods + 1)
    )
    P_T = discount_factor(tenor, curve, flat_rate)

    if annuity <= 0:
        return flat_rate

    return (1 - P_T) / annuity


def annuity_pv(
    tenor: float,
    payment_freq: str = "semi-annual",
    curve: Any | None = None,
    flat_rate: float = 0.05,
) -> float:
    """Return PV of a unit annuity over tenor."""
    freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12}
    n_per_year = freq_map.get(payment_freq, 2)
    dt = 1 / n_per_year
    n_periods = int(tenor * n_per_year)
    return sum(
        discount_factor(i * dt, curve, flat_rate) * dt
        for i in range(1, n_periods + 1)
    )
