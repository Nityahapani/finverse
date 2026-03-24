"""
finverse.options
================
Full options pricing library: Black-Scholes, American binomial,
implied vol, vol surface, and put-call parity arbitrage scanner.

Usage
-----
from finverse import options

# Price a European option
opt = options.price(S=185.0, K=190.0, T=0.25, r=0.053, sigma=0.28, type='call')
opt.summary()

# American option via binomial tree
opt = options.price_american(S=185, K=190, T=0.25, r=0.053, sigma=0.28, type='put', steps=500)
opt.summary()

# Implied volatility
iv = options.implied_vol(market_price=7.50, S=185, K=190, T=0.25, r=0.053, type='call')

# Live options chain (requires yfinance)
chain = options.chain(data)
chain.summary()
chain.vol_surface().plot()

# Arbitrage scanner
arb = options.scan_arbitrage(chain)
arb.summary()

# Tail hedge suggestion (integrates with risk.evt)
hedge = options.tail_hedge_suggestion(data, evt_result=tail)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from finverse.options.black_scholes import price, OptionResult
from finverse.options.binomial import price_american
from finverse.options.implied_vol import solve_iv
from finverse.options.chain import fetch_chain as chain, scan_arbitrage
from finverse.options.vol_surface import VolSurface

OptionType = Literal["call", "put"]

__all__ = [
    "price",
    "price_american",
    "implied_vol",
    "chain",
    "scan_arbitrage",
    "tail_hedge_suggestion",
]


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    type: OptionType = "call",
) -> float | None:
    """Compute implied volatility from a market option price."""
    return solve_iv(market_price=market_price, S=S, K=K, T=T, r=r, type=type)


def tail_hedge_suggestion(
    data: Any,
    evt_result: Any | None = None,
    r: float = 0.053,
    hedge_horizon: float = 0.25,
) -> "TailHedgeResult":
    """
    Suggest a tail hedge (put option) based on EVT tail loss estimates.

    Parameters
    ----------
    data : TickerData
    evt_result : EVTResult from finverse.risk.evt (optional)
    r : float — risk-free rate
    hedge_horizon : float — option expiry horizon in years
    """
    spot = 100.0
    var_99 = 0.20   # fallback

    if hasattr(data, "price_history") and data.price_history is not None:
        closes = data.price_history["Close"]
        if not closes.empty:
            spot = float(closes.iloc[-1])

    if evt_result is not None and hasattr(evt_result, "var_999"):
        var_99 = float(evt_result.var_999) if evt_result.var_999 else var_99
    elif evt_result is not None and hasattr(evt_result, "var_99"):
        var_99 = float(evt_result.var_99)

    # Strike = spot * (1 - tail loss), rounded to nearest 5
    strike = round(spot * (1 - var_99) / 5) * 5
    sigma = 0.30   # assume elevated vol for tail hedge

    put = price(S=spot, K=strike, T=hedge_horizon, r=r, sigma=sigma, type="put")
    cost_pct = put.price / spot

    return TailHedgeResult(
        spot=spot,
        suggested_strike=strike,
        expiry_years=hedge_horizon,
        put_price=put.price,
        cost_pct_of_spot=cost_pct,
        var_99_estimate=var_99,
        delta=put.delta,
    )


from dataclasses import dataclass


@dataclass
class TailHedgeResult:
    """Output of tail_hedge_suggestion()."""
    spot: float
    suggested_strike: float
    expiry_years: float
    put_price: float
    cost_pct_of_spot: float
    var_99_estimate: float
    delta: float

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title="Tail Hedge Suggestion")
            t.add_column("Parameter", style="bold cyan")
            t.add_column("Value", justify="right")
            t.add_row("Spot", f"${self.spot:.2f}")
            t.add_row("Suggested Put Strike", f"${self.suggested_strike:.2f}")
            t.add_row("Expiry", f"{self.expiry_years:.2f}y ({int(self.expiry_years*365)}d)")
            t.add_row("Put Price", f"${self.put_price:.4f}")
            t.add_row("Cost as % of Spot", f"{self.cost_pct_of_spot:.2%}")
            t.add_row("EVT VaR Estimate", f"{self.var_99_estimate:.2%}")
            t.add_row("Delta", f"{self.delta:+.4f}")
            console.print(t)
        except ImportError:
            print(f"Tail Hedge: strike={self.suggested_strike}, cost={self.cost_pct_of_spot:.2%} of spot")
