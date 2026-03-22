"""
finverse.models.ddm — Dividend Discount Model family.

Three variants:
  - Gordon Growth Model (single-stage constant growth)
  - H-Model (two-stage with linear decline from high to stable growth)
  - Multistage DDM (explicit high-growth phase + terminal value)

All pure math — no API keys needed.
Works best for dividend-paying stocks (utilities, REITs, consumer staples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DDMResult:
    ticker: str
    model: str
    implied_price: float
    current_price: float | None
    upside: float | None
    dividend_yield_implied: float
    cost_of_equity: float
    dividends_used: list[float]          # projected dividends
    terminal_value: float
    pv_dividends: float
    assumptions: dict

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]DDM — {self.ticker} ({self.model})[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Component")
        table.add_column("Value", justify="right")

        table.add_row("PV of dividends",      f"${self.pv_dividends:.2f}")
        table.add_row("Terminal value (PV)",   f"${self.terminal_value:.2f}")
        table.add_row("Implied price",         f"[bold green]${self.implied_price:.2f}[/bold green]")
        if self.current_price:
            table.add_row("Current price",     f"${self.current_price:.2f}")
            color = "green" if (self.upside or 0) > 0 else "red"
            table.add_row("Upside / downside", f"[{color}]{self.upside:.1%}[/{color}]" if self.upside else "—")
        table.add_row("Implied div yield",     f"{self.dividend_yield_implied:.2%}")
        table.add_row("Cost of equity (ke)",   f"{self.cost_of_equity:.2%}")
        console.print(table)

        console.print("\n  [dim]Projected dividends:[/dim]")
        for i, d in enumerate(self.dividends_used[:8], 1):
            console.print(f"    Year {i}: ${d:.4f}")
        if len(self.dividends_used) > 8:
            console.print(f"    ... ({len(self.dividends_used)} total)")

        console.print("\n  [dim]Assumptions:[/dim]")
        for k, v in self.assumptions.items():
            fmt = f"{v:.2%}" if isinstance(v, float) and v < 2 else str(v)
            console.print(f"    {k:<28} {fmt}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "model": self.model,
            "implied_price": self.implied_price,
            "current_price": self.current_price,
            "upside": self.upside,
            "cost_of_equity": self.cost_of_equity,
        }])


def _gordon(d1: float, ke: float, g: float) -> float:
    """Gordon Growth Model: P = D1 / (ke - g)"""
    if ke <= g:
        raise ValueError(f"ke ({ke:.2%}) must exceed g ({g:.2%})")
    return d1 / (ke - g)


def gordon(
    data=None,
    dividend: float | None = None,
    growth_rate: float = 0.04,
    cost_of_equity: float = 0.09,
    current_price: float | None = None,
) -> DDMResult:
    """
    Gordon Growth Model: P = D₁ / (ke - g)

    Best for mature, stable dividend-paying companies.

    Parameters
    ----------
    data          : TickerData (auto-extracts dividend, price if provided)
    dividend      : float — current annual dividend per share (override)
    growth_rate   : float — constant dividend growth rate (default 4%)
    cost_of_equity: float — required return on equity (default 9%)
    current_price : float — override current price for upside calculation

    Returns
    -------
    DDMResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.models.ddm import gordon
    >>> ko = pull.ticker("KO")  # Coca-Cola — classic dividend stock
    >>> result = gordon(ko, growth_rate=0.04, cost_of_equity=0.085)
    >>> result.summary()

    Manual:
    >>> result = gordon(dividend=1.84, growth_rate=0.04, cost_of_equity=0.085)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = "Company"
    d0 = dividend

    if data is not None:
        ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
        if d0 is None:
            info = data.info if hasattr(data, "info") and data.info else {}
            d0 = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
            if d0 is None:
                d0 = 1.0
                console.print(f"[yellow]No dividend data — using placeholder $1.00[/yellow]")
        if current_price is None:
            current_price = data.current_price

    if d0 is None or d0 <= 0:
        d0 = 1.0

    console.print(f"[dim]Gordon Growth Model for {ticker} (D₀=${d0:.2f}, g={growth_rate:.1%}, ke={cost_of_equity:.1%})...[/dim]")

    d1 = d0 * (1 + growth_rate)
    implied = _gordon(d1, cost_of_equity, growth_rate)
    terminal_pv = implied
    pv_divs = 0.0

    upside = (implied - current_price) / current_price if current_price else None
    div_yield = d1 / implied if implied > 0 else 0

    console.print(f"[green]✓[/green] Gordon DDM: ${implied:.2f}" + (f" ({upside:+.1%} vs current)" if upside else ""))

    return DDMResult(
        ticker=ticker, model="Gordon Growth Model",
        implied_price=round(implied, 2),
        current_price=current_price,
        upside=round(upside, 4) if upside else None,
        dividend_yield_implied=round(div_yield, 4),
        cost_of_equity=cost_of_equity,
        dividends_used=[round(d1, 4)],
        terminal_value=round(terminal_pv, 2),
        pv_dividends=round(pv_divs, 2),
        assumptions={"D₀ (current dividend)": d0, "g (growth)": growth_rate,
                     "ke (cost of equity)": cost_of_equity, "D₁": d1},
    )


def h_model(
    data=None,
    dividend: float | None = None,
    high_growth: float = 0.12,
    stable_growth: float = 0.04,
    half_life: float = 5.0,
    cost_of_equity: float = 0.09,
    current_price: float | None = None,
) -> DDMResult:
    """
    H-Model: two-stage DDM with linear growth decline.

    P = D₀ * [(1 + gn) + H * (ga - gn)] / (ke - gn)
    where H = half_life (years until high growth declines to stable).

    Better than Gordon for growth companies transitioning to maturity.

    Parameters
    ----------
    data         : TickerData
    dividend     : float — current annual dividend
    high_growth  : float — current high growth rate (default 12%)
    stable_growth: float — long-run stable growth rate (default 4%)
    half_life    : float — years until halfway through transition (default 5)
    cost_of_equity: float

    Example
    -------
    >>> result = h_model(data, high_growth=0.15, stable_growth=0.04, half_life=7)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = "Company"
    d0 = dividend

    if data is not None:
        ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
        if d0 is None:
            info = data.info if hasattr(data, "info") and data.info else {}
            d0 = info.get("dividendRate") or 1.0
        if current_price is None:
            current_price = data.current_price

    if d0 is None or d0 <= 0:
        d0 = 1.0

    console.print(f"[dim]H-Model for {ticker} (ga={high_growth:.1%}→gn={stable_growth:.1%}, H={half_life})...[/dim]")

    if cost_of_equity <= stable_growth:
        raise ValueError(f"ke must exceed stable growth: {cost_of_equity:.2%} vs {stable_growth:.2%}")

    implied = d0 * ((1 + stable_growth) + half_life * (high_growth - stable_growth)) / (cost_of_equity - stable_growth)

    n = int(half_life * 2)
    dividends = []
    for t in range(1, n + 1):
        g_t = high_growth - (high_growth - stable_growth) * t / n
        d_t = d0 * (1 + g_t) ** t
        dividends.append(round(d_t, 4))

    upside = (implied - current_price) / current_price if current_price else None
    div_yield = d0 * (1 + stable_growth) / implied if implied > 0 else 0

    console.print(f"[green]✓[/green] H-Model: ${implied:.2f}")

    return DDMResult(
        ticker=ticker, model="H-Model",
        implied_price=round(implied, 2),
        current_price=current_price,
        upside=round(upside, 4) if upside else None,
        dividend_yield_implied=round(div_yield, 4),
        cost_of_equity=cost_of_equity,
        dividends_used=dividends,
        terminal_value=round(implied, 2),
        pv_dividends=0.0,
        assumptions={"D₀": d0, "high growth": high_growth,
                     "stable growth": stable_growth, "half life (H)": half_life,
                     "ke": cost_of_equity},
    )


def multistage(
    data=None,
    dividend: float | None = None,
    stage1_growth: float = 0.15,
    stage1_years: int = 5,
    stage2_growth: float = 0.08,
    stage2_years: int = 5,
    terminal_growth: float = 0.04,
    cost_of_equity: float = 0.10,
    current_price: float | None = None,
) -> DDMResult:
    """
    Multistage DDM: explicit high-growth + transition + terminal phase.

    Most flexible DDM variant. Suitable for companies with distinct
    high-growth phase followed by normalization.

    Parameters
    ----------
    data           : TickerData
    dividend       : float — current annual dividend
    stage1_growth  : float — growth in years 1–n (default 15%)
    stage1_years   : int — duration of stage 1 (default 5)
    stage2_growth  : float — growth in years n+1 to n+m (default 8%)
    stage2_years   : int — duration of stage 2 (default 5)
    terminal_growth: float — terminal stable growth (default 4%)
    cost_of_equity : float

    Example
    -------
    >>> result = multistage(
    ...     data=data,
    ...     stage1_growth=0.20, stage1_years=5,
    ...     stage2_growth=0.10, stage2_years=5,
    ...     terminal_growth=0.04,
    ...     cost_of_equity=0.11,
    ... )
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = "Company"
    d0 = dividend

    if data is not None:
        ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
        if d0 is None:
            info = data.info if hasattr(data, "info") and data.info else {}
            d0 = info.get("dividendRate") or 1.0
        if current_price is None:
            current_price = data.current_price

    if d0 is None or d0 <= 0:
        d0 = 1.0

    console.print(
        f"[dim]Multistage DDM for {ticker} "
        f"({stage1_growth:.0%}/{stage1_years}y → {stage2_growth:.0%}/{stage2_years}y → {terminal_growth:.0%})...[/dim]"
    )

    if cost_of_equity <= terminal_growth:
        raise ValueError(f"ke must exceed terminal growth")

    pv_divs = 0.0
    dividends = []
    current_div = d0

    for t in range(1, stage1_years + 1):
        current_div *= (1 + stage1_growth)
        pv = current_div / (1 + cost_of_equity) ** t
        pv_divs += pv
        dividends.append(round(current_div, 4))

    for t in range(1, stage2_years + 1):
        current_div *= (1 + stage2_growth)
        yr = stage1_years + t
        pv = current_div / (1 + cost_of_equity) ** yr
        pv_divs += pv
        dividends.append(round(current_div, 4))

    total_years = stage1_years + stage2_years
    d_terminal = current_div * (1 + terminal_growth)
    terminal_price = d_terminal / (cost_of_equity - terminal_growth)
    terminal_pv = terminal_price / (1 + cost_of_equity) ** total_years

    implied = pv_divs + terminal_pv
    upside = (implied - current_price) / current_price if current_price else None
    div_yield = d0 * (1 + stage1_growth) / implied if implied > 0 else 0

    console.print(f"[green]✓[/green] Multistage DDM: ${implied:.2f}")

    return DDMResult(
        ticker=ticker, model="Multistage DDM",
        implied_price=round(implied, 2),
        current_price=current_price,
        upside=round(upside, 4) if upside else None,
        dividend_yield_implied=round(div_yield, 4),
        cost_of_equity=cost_of_equity,
        dividends_used=dividends,
        terminal_value=round(terminal_pv, 2),
        pv_dividends=round(pv_divs, 2),
        assumptions={
            "D₀": d0, "Stage 1 growth": stage1_growth, "Stage 1 years": stage1_years,
            "Stage 2 growth": stage2_growth, "Stage 2 years": stage2_years,
            "Terminal growth": terminal_growth, "ke": cost_of_equity,
        },
    )
