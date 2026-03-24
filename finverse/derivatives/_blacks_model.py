"""
finverse.derivatives._blacks_model
Black's model for swaptions, caps, and floors.
"""
from __future__ import annotations

import math
from scipy.stats import norm  # type: ignore


def blacks_swaption(
    notional: float,
    strike_rate: float,
    swap_rate: float,
    annuity: float,
    vol: float,
    option_expiry: float,
    type: str = "payer",
) -> dict:
    """
    Price a European swaption using Black's model.

    Parameters
    ----------
    notional : float
    strike_rate : float  — exercise rate
    swap_rate : float    — current par swap rate
    annuity : float      — present value of annuity (PV01 * notional)
    vol : float          — Black's volatility (normal or lognormal)
    option_expiry : float — years to option expiry
    type : 'payer' (right to pay fixed) or 'receiver'

    Returns dict with price, delta, vega, breakeven_vol
    """
    if option_expiry <= 0 or vol <= 0:
        return {"price": 0.0, "delta": 0.0, "vega": 0.0, "breakeven_vol": vol}

    d1 = (math.log(swap_rate / strike_rate) + 0.5 * vol**2 * option_expiry) / (vol * math.sqrt(option_expiry))
    d2 = d1 - vol * math.sqrt(option_expiry)

    if type == "payer":
        price = notional * annuity * (swap_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2))
        delta = notional * annuity * norm.cdf(d1)
    else:
        price = notional * annuity * (strike_rate * norm.cdf(-d2) - swap_rate * norm.cdf(-d1))
        delta = -notional * annuity * norm.cdf(-d1)

    vega = notional * annuity * swap_rate * norm.pdf(d1) * math.sqrt(option_expiry) / 100

    # Approximate breakeven vol via Newton
    breakeven_vol = vol  # placeholder (market vol would come from chain)

    return {
        "price": max(price, 0.0),
        "delta": delta,
        "vega": vega,
        "breakeven_vol": breakeven_vol,
    }
