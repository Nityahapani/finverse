"""
finverse.risk.var — Value at Risk, CVaR, and stress testing for positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class VaRResult:
    ticker: str
    var_95: float          # 1-day 95% VaR (% loss)
    var_99: float          # 1-day 99% VaR
    cvar_95: float         # Conditional VaR (Expected Shortfall) at 95%
    cvar_99: float         # CVaR at 99%
    annualized_vol: float
    max_drawdown: float
    stress_scenarios: dict[str, float]
    method: str

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Risk Metrics — {self.ticker}[/bold blue]")
        console.print(f"[dim]Method: {self.method}[/dim]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        table.add_row("VaR (95%, 1-day)", f"{self.var_95:.2%}", "Max daily loss 95% of time")
        table.add_row("VaR (99%, 1-day)", f"{self.var_99:.2%}", "Max daily loss 99% of time")
        table.add_row("CVaR / ES (95%)", f"{self.cvar_95:.2%}", "Avg loss in worst 5% scenarios")
        table.add_row("CVaR / ES (99%)", f"{self.cvar_99:.2%}", "Avg loss in worst 1% scenarios")
        table.add_row("Annualised vol", f"{self.annualized_vol:.2%}", "Historical return volatility")
        table.add_row("Max drawdown", f"{self.max_drawdown:.2%}", "Worst peak-to-trough decline")
        console.print(table)

        if self.stress_scenarios:
            stress_table = Table(
                title="Stress scenarios (scaled to this stock's vol vs SPX)",
                box=box.SIMPLE_HEAD, header_style="bold blue"
            )
            stress_table.add_column("Scenario")
            stress_table.add_column("Estimated loss", justify="right")
            for scenario, loss in self.stress_scenarios.items():
                color = "red" if loss < -0.2 else "yellow"
                stress_table.add_row(scenario, f"[{color}]{loss:.1%}[/{color}]")
            console.print(stress_table)

        console.print()


def var(
    data,
    confidence: float = 0.95,
    window: int = 252,
    method: str = "historical",
) -> VaRResult:
    """
    Compute Value at Risk and risk metrics from price history.

    Parameters
    ----------
    data       : TickerData with price_history
    confidence : float — confidence level (default 0.95)
    window     : int — lookback days (default 252 = 1 year)
    method     : "historical" or "parametric" (default "historical")

    Returns
    -------
    VaRResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.risk import var as risk_var
    >>> data = pull.ticker("AAPL")
    >>> result = risk_var.var(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Computing VaR ({method}, {confidence:.0%} confidence, {window}d window)...[/dim]")

    if not hasattr(data, "price_history") or data.price_history.empty:
        raise ValueError("price_history required. Use pull.ticker() first.")

    prices = data.price_history["Close"].tail(window)
    returns = prices.pct_change().dropna()

    if len(returns) < 30:
        console.print("[yellow]Warning: fewer than 30 observations — VaR estimates unreliable[/yellow]")

    if method == "parametric":
        from scipy import stats
        mu = float(returns.mean())
        sigma = float(returns.std())
        var_95 = float(-stats.norm.ppf(0.05, mu, sigma))
        var_99 = float(-stats.norm.ppf(0.01, mu, sigma))
        threshold_95 = stats.norm.ppf(0.05, mu, sigma)
        threshold_99 = stats.norm.ppf(0.01, mu, sigma)
        cvar_95 = float(-stats.norm.expect(lambda x: x, loc=mu, scale=sigma,
                                            ub=threshold_95) / 0.05)
        cvar_99 = float(-stats.norm.expect(lambda x: x, loc=mu, scale=sigma,
                                            ub=threshold_99) / 0.01)
    else:
        sorted_returns = np.sort(returns.values)
        var_95 = float(-np.percentile(sorted_returns, 5))
        var_99 = float(-np.percentile(sorted_returns, 1))
        threshold_95 = np.percentile(sorted_returns, 5)
        threshold_99 = np.percentile(sorted_returns, 1)
        cvar_95 = float(-sorted_returns[sorted_returns <= threshold_95].mean())
        cvar_99 = float(-sorted_returns[sorted_returns <= threshold_99].mean())

    ann_vol = float(returns.std() * np.sqrt(252))
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Scale historical market shocks to this stock's own volatility
    # Using SPX annual vol ~16% as the reference for the historical events
    _spx_vol = 0.16
    _scale   = ann_vol / _spx_vol   # e.g. a vol-30% stock → scale 1.875x
    stress_scenarios = {
        "COVID crash (Feb-Mar 2020)":  round(-0.34 * _scale, 4),
        "GFC (Sep 2008)":              round(-0.47 * _scale, 4),
        "Dot-com bust (2000-2002)":    round(-0.49 * _scale, 4),
        "Black Monday (Oct 1987)":     round(-0.22 * _scale, 4),
        "2022 rate shock":             round(-0.19 * _scale, 4),
        "1-sigma shock (1 month)":     round(-ann_vol / np.sqrt(12), 4),
        "2-sigma shock (1 month)":     round(-2 * ann_vol / np.sqrt(12), 4),
        "3-sigma shock (tail event)":  round(-3 * ann_vol / np.sqrt(12), 4),
    }

    console.print(
        f"[green]✓[/green] VaR(95%)={var_95:.2%}  CVaR(95%)={cvar_95:.2%}  "
        f"Ann.vol={ann_vol:.2%}  MaxDD={max_dd:.2%}"
    )

    return VaRResult(
        ticker=ticker,
        var_95=round(var_95, 4),
        var_99=round(var_99, 4),
        cvar_95=round(cvar_95, 4),
        cvar_99=round(cvar_99, 4),
        annualized_vol=round(ann_vol, 4),
        max_drawdown=round(max_dd, 4),
        stress_scenarios={k: round(v, 4) for k, v in stress_scenarios.items()},
        method=method,
    )
