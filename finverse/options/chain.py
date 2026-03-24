"""
finverse.options.chain
yfinance options chain wrapper + put-call parity arbitrage scanner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finverse.options.implied_vol import solve_iv
from finverse.options.vol_surface import VolSurface, build_surface


@dataclass
class OptionsChain:
    """Parsed options chain for a ticker."""
    ticker: str
    spot: float
    expirations: list[str]
    calls: pd.DataFrame
    puts: pd.DataFrame
    r: float = 0.053   # risk-free rate used for IV computation

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"Options Chain — {self.ticker}  (spot={self.spot:.2f})")
            t.add_column("Expiry", style="bold")
            t.add_column("# Calls", justify="right")
            t.add_column("# Puts", justify="right")
            for exp in self.expirations[:10]:
                nc = len(self.calls[self.calls["expiry"] == exp]) if "expiry" in self.calls.columns else "—"
                np_ = len(self.puts[self.puts["expiry"] == exp]) if "expiry" in self.puts.columns else "—"
                t.add_row(exp, str(nc), str(np_))
            console.print(t)
        except ImportError:
            print(f"Options Chain [{self.ticker}] — {len(self.expirations)} expirations, spot={self.spot:.2f}")

    def vol_surface(self) -> VolSurface:
        """Build implied vol surface from the chain."""
        records = []
        for df, otype in [(self.calls, "call"), (self.puts, "put")]:
            if df.empty or "expiry" not in df.columns:
                continue
            for _, row in df.iterrows():
                if pd.isna(row.get("lastPrice", np.nan)) or row.get("lastPrice", 0) <= 0:
                    continue
                T = row.get("T", 0.25)
                if T <= 0:
                    continue
                iv = solve_iv(
                    market_price=float(row["lastPrice"]),
                    S=self.spot,
                    K=float(row["strike"]),
                    T=T,
                    r=self.r,
                    type=otype,
                )
                if iv is not None and 0.01 < iv < 5.0:
                    records.append({
                        "expiry": row["expiry"],
                        "moneyness": float(row["strike"]) / self.spot,
                        "iv": iv,
                    })
        return build_surface(records, ticker=self.ticker)


@dataclass
class ArbitrageResult:
    """Put-call parity arbitrage scan results."""
    ticker: str
    mispricings: pd.DataFrame   # rows where |C - P - S + Ke^{-rT}| > threshold

    def summary(self) -> None:
        try:
            from rich.console import Console
            console = Console()
            if self.mispricings.empty:
                console.print(f"[green]No put-call parity violations found for {self.ticker}.[/green]")
            else:
                from rich.table import Table
                t = Table(title=f"Put-Call Parity Violations — {self.ticker}")
                for col in self.mispricings.columns:
                    t.add_column(str(col))
                for _, row in self.mispricings.iterrows():
                    t.add_row(*[str(round(v, 4)) if isinstance(v, float) else str(v) for v in row])
                console.print(t)
        except ImportError:
            print(f"Arbitrage scan [{self.ticker}]: {len(self.mispricings)} violations")


def fetch_chain(data: Any, r: float = 0.053) -> OptionsChain:
    """
    Fetch live options chain from yfinance via TickerData object or ticker string.

    Parameters
    ----------
    data : TickerData or str ticker symbol
    r : float — risk-free rate for IV computation
    """
    import yfinance as yf  # type: ignore

    ticker_sym = data.ticker if hasattr(data, "ticker") else str(data)
    spot = (data.price_history["Close"].iloc[-1]
            if hasattr(data, "price_history") and data.price_history is not None and not data.price_history.empty
            else yf.Ticker(ticker_sym).fast_info.get("last_price", 100.0))

    yf_ticker = yf.Ticker(ticker_sym)
    expirations = list(yf_ticker.options)

    all_calls, all_puts = [], []
    import datetime
    today = datetime.date.today()

    for exp in expirations[:6]:   # limit to 6 nearest expiries
        try:
            chain = yf_ticker.option_chain(exp)
            exp_date = datetime.date.fromisoformat(exp)
            T = max((exp_date - today).days / 365.0, 1 / 365)

            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()
            calls_df["expiry"] = exp
            puts_df["expiry"] = exp
            calls_df["T"] = T
            puts_df["T"] = T
            all_calls.append(calls_df)
            all_puts.append(puts_df)
        except Exception:
            continue

    calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    return OptionsChain(
        ticker=ticker_sym,
        spot=float(spot),
        expirations=expirations,
        calls=calls,
        puts=puts,
        r=r,
    )


def scan_arbitrage(chain: OptionsChain, threshold: float = 0.10) -> ArbitrageResult:
    """
    Scan for put-call parity violations.

    C - P = S - K * e^{-rT}
    Flag pairs where |deviation| > threshold (in $).
    """
    if chain.calls.empty or chain.puts.empty:
        return ArbitrageResult(ticker=chain.ticker, mispricings=pd.DataFrame())

    import math
    violations = []
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    if "expiry" not in calls.columns or "strike" not in calls.columns:
        return ArbitrageResult(ticker=chain.ticker, mispricings=pd.DataFrame())

    merged = pd.merge(
        calls[["expiry", "strike", "lastPrice", "T"]].rename(columns={"lastPrice": "call_price"}),
        puts[["expiry", "strike", "lastPrice"]].rename(columns={"lastPrice": "put_price"}),
        on=["expiry", "strike"],
    )
    for _, row in merged.iterrows():
        T = row.get("T", 0.25)
        parity_rhs = chain.spot - row["strike"] * math.exp(-chain.r * T)
        lhs = row["call_price"] - row["put_price"]
        deviation = abs(lhs - parity_rhs)
        if deviation > threshold:
            violations.append({
                "expiry": row["expiry"],
                "strike": row["strike"],
                "call_price": row["call_price"],
                "put_price": row["put_price"],
                "parity_rhs": round(parity_rhs, 4),
                "deviation_$": round(deviation, 4),
            })

    return ArbitrageResult(
        ticker=chain.ticker,
        mispricings=pd.DataFrame(violations),
    )
