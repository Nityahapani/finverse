"""
finverse.risk._stress_engine
Core portfolio and model impact computation for stress scenarios.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finverse.risk._scenarios import ScenarioShocks


# ── sector beta map (approximate) ────────────────────────────────────────────
SECTOR_BETAS: dict[str, float] = {
    "tech": 1.35,
    "technology": 1.35,
    "finance": 1.20,
    "financial": 1.20,
    "energy": 0.95,
    "healthcare": 0.75,
    "consumer": 0.90,
    "utilities": 0.55,
    "materials": 1.05,
    "industrials": 1.10,
    "real estate": 1.15,
    "communication": 1.15,
}

DEFAULT_BETA = 1.0


def _get_beta(data: Any) -> float:
    """Extract or estimate a beta for a TickerData object."""
    # Try to read beta from yfinance info if cached on the data object
    if hasattr(data, "info") and isinstance(data.info, dict):
        b = data.info.get("beta")
        if b and 0.1 < b < 4.0:
            return float(b)
    if hasattr(data, "sector"):
        sector = (data.sector or "").lower()
        for k, v in SECTOR_BETAS.items():
            if k in sector:
                return v
    return DEFAULT_BETA


def _get_ticker(data: Any) -> str:
    return getattr(data, "ticker", str(data))


def _sector_of(data: Any) -> str:
    if hasattr(data, "sector"):
        return (data.sector or "").lower()
    if hasattr(data, "info") and isinstance(data.info, dict):
        return data.info.get("sector", "").lower()
    return ""


def compute_portfolio_impact(
    holdings: list[Any],
    shocks: ScenarioShocks,
    weights: list[float] | None = None,
) -> dict:
    """
    Estimate portfolio return under a stress scenario.

    Uses beta-adjusted equity shock + sector multipliers.
    Returns dict with per-holding returns, portfolio return, worst/best holding.
    """
    n = len(holdings)
    if weights is None:
        weights = [1.0 / n] * n

    holding_returns = {}
    for data, w in zip(holdings, weights):
        ticker = _get_ticker(data)
        beta = _get_beta(data)
        sector = _sector_of(data)

        # Base equity shock adjusted by beta
        base_return = shocks.equity_return * beta

        # Tech multiplier
        is_tech = any(k in sector for k in ["tech", "software", "semiconductor"])
        if is_tech:
            base_return *= shocks.tech_multiplier

        # EM multiplier (rough: if ticker has known EM exposure)
        # We don't flag EM specifically; leave as-is for domestic tickers

        holding_returns[ticker] = float(base_return)

    portfolio_return = sum(
        holding_returns[_get_ticker(d)] * w
        for d, w in zip(holdings, weights)
    )

    worst = min(holding_returns, key=holding_returns.get)
    best = max(holding_returns, key=holding_returns.get)

    return {
        "portfolio_return": portfolio_return,
        "holding_returns": holding_returns,
        "worst_holding": worst,
        "best_holding": best,
    }


def compute_dcf_impact(
    dcf_model: Any,
    shocks: ScenarioShocks,
) -> dict:
    """
    Stress a DCF model by adjusting WACC and growth assumptions.

    WACC stressed by rate_shift and credit_spread widening.
    Terminal growth stressed by equity_return magnitude.
    """
    base_wacc = getattr(dcf_model, "wacc", 0.10)
    base_growth = getattr(dcf_model, "terminal_growth", 0.025)
    base_price = getattr(dcf_model, "implied_price", None)

    if base_price is None:
        return {"dcf_price_impact": None, "wacc_stressed": base_wacc}

    # WACC stress: rate shift + credit spread (in bps → %)
    wacc_delta = (shocks.rate_shift_bps + shocks.credit_spread_bps * 0.3) / 10000
    wacc_stressed = base_wacc + wacc_delta

    # Growth stress: severe scenarios cut terminal growth
    growth_delta = max(shocks.equity_return * 0.02, -0.015)
    growth_stressed = max(base_growth + growth_delta, 0.005)

    # Re-run DCF with stressed assumptions (simplified Gordon Growth approximation)
    # Implied price ~ FCF / (WACC - g); use ratio for % impact
    try:
        ratio = (base_wacc - base_growth) / max(wacc_stressed - growth_stressed, 0.001)
        stressed_price = base_price * ratio
        price_impact = (stressed_price - base_price) / base_price
    except Exception:
        price_impact = shocks.equity_return * 0.6   # fallback
        stressed_price = base_price * (1 + price_impact)

    return {
        "dcf_price_impact": price_impact,
        "wacc_stressed": wacc_stressed,
        "growth_stressed": growth_stressed,
        "stressed_price": stressed_price,
        "base_price": base_price,
    }


def identify_key_risk_drivers(shocks: ScenarioShocks) -> list[str]:
    """Return the top 3 risk drivers for a scenario based on shock magnitudes."""
    drivers = {
        "Equity market decline": abs(shocks.equity_return),
        "Interest rate shift": abs(shocks.rate_shift_bps) / 100,
        "Credit spread widening": abs(shocks.credit_spread_bps) / 100,
        "Volatility spike (VIX)": shocks.vix_level / 40,
        "Oil price shock": abs(shocks.oil_return),
        "USD movement": abs(shocks.usd_return) * 3,
    }
    top_3 = sorted(drivers, key=drivers.get, reverse=True)[:3]
    return top_3


def build_commentary(shocks: ScenarioShocks, portfolio_return: float) -> str:
    direction = "loss" if portfolio_return < 0 else "gain"
    severity = "severe" if abs(portfolio_return) > 0.30 else "moderate" if abs(portfolio_return) > 0.15 else "mild"
    return (
        f"{shocks.name}: {severity.capitalize()} portfolio {direction} of {portfolio_return:.1%}. "
        f"Scenario featured equity markets {shocks.equity_return:.0%}, "
        f"rates {shocks.rate_shift_bps:+.0f}bps, "
        f"credit spreads +{shocks.credit_spread_bps}bps, VIX peak {shocks.vix_level:.0f}. "
        f"Duration: ~{shocks.duration_years:.1f}y."
    )
