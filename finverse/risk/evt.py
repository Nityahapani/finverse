"""
finverse.risk.evt — Extreme Value Theory (EVT) for tail risk modeling.

Uses the Peaks-Over-Threshold (POT) method with Generalized Pareto
Distribution (GPD) to model extreme losses beyond VaR.

Why EVT matters:
  Normal distribution massively underestimates tail risk.
  GFC losses were 25+ standard deviations under Gaussian — impossible.
  GPD captures fat tails empirically without assuming normality.

Provides:
  - GPD parameter estimation (MLE)
  - Tail VaR and Expected Shortfall at extreme confidence levels (99.9%)
  - Return period analysis (how often does a loss this large occur?)
  - Tail index (shape parameter ξ — how heavy the tail is)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import genpareto


@dataclass
class EVTResult:
    ticker: str
    threshold: float               # POT threshold (negative return)
    n_exceedances: int             # observations above threshold
    xi: float                      # shape parameter (tail index)
    sigma: float                   # scale parameter
    tail_index: float              # = 1/xi (higher = heavier tail)
    var_99: float                  # VaR at 99%
    var_999: float                 # VaR at 99.9%
    var_9999: float                # VaR at 99.99%
    es_99: float                   # Expected Shortfall at 99%
    es_999: float                  # Expected Shortfall at 99.9%
    return_periods: dict[float, float]  # loss → return period (days)
    method: str = "Peaks-Over-Threshold (GPD)"

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        tail_label = (
            "heavy tail (fat-tailed)" if self.xi > 0.3
            else "moderate tail" if self.xi > 0
            else "thin tail (bounded)"
        )
        tail_color = "red" if self.xi > 0.3 else ("yellow" if self.xi > 0 else "green")

        console.print(f"\n[bold blue]Extreme Value Theory — {self.ticker}[/bold blue]")
        console.print(f"[dim]{self.method}[/dim]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        table.add_row("Threshold (u)",      f"{self.threshold:.2%}",   f"{self.n_exceedances} exceedances")
        table.add_row("Shape (ξ)",          f"{self.xi:.4f}",          f"[{tail_color}]{tail_label}[/{tail_color}]")
        table.add_row("Scale (σ)",          f"{self.sigma:.4f}",       "GPD scale")
        table.add_row("Tail index (1/ξ)",   f"{self.tail_index:.2f}",  "Higher = heavier tail")
        table.add_row("VaR 99%",            f"{self.var_99:.2%}",      "1-in-100 daily loss")
        table.add_row("VaR 99.9%",          f"{self.var_999:.2%}",     "1-in-1000 daily loss")
        table.add_row("VaR 99.99%",         f"{self.var_9999:.2%}",    "1-in-10000 daily loss")
        table.add_row("ES 99%",             f"{self.es_99:.2%}",       "Avg loss in worst 1%")
        table.add_row("ES 99.9%",           f"{self.es_999:.2%}",      "Avg loss in worst 0.1%")
        console.print(table)

        if self.return_periods:
            rp_table = Table(title="Return periods", box=box.SIMPLE_HEAD, header_style="bold blue")
            rp_table.add_column("Loss level")
            rp_table.add_column("Return period", justify="right")
            for loss, period in self.return_periods.items():
                years = period / 252
                rp_table.add_row(
                    f"{loss:.1%}",
                    f"{period:.0f} days ({years:.1f} years)" if years < 100
                    else f"{period:.0f} days ({years:.0f} years)"
                )
            console.print(rp_table)
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "xi": self.xi,
            "sigma": self.sigma,
            "var_99": self.var_99,
            "var_999": self.var_999,
            "es_99": self.es_99,
            "es_999": self.es_999,
        }])


def _gpd_loglik(params: np.ndarray, exceedances: np.ndarray) -> float:
    """Negative log-likelihood of Generalized Pareto Distribution."""
    xi, sigma = params
    if sigma <= 0:
        return 1e10
    n = len(exceedances)
    y = exceedances / sigma
    if xi == 0:
        return n * np.log(sigma) + np.sum(y)
    if xi < 0 and np.any(y > -1/xi):
        return 1e10
    ll = n * np.log(sigma) + (1 + 1/xi) * np.sum(np.log1p(xi * y))
    return ll


def _select_threshold(losses: np.ndarray, method: str = "10pct") -> float:
    """Select POT threshold. 'losses' are positive loss values."""
    if method == "10pct":
        return float(np.percentile(losses, 90))
    elif method == "mean_excess":
        candidates = np.linspace(np.percentile(losses, 80), np.percentile(losses, 97), 20)
        mean_excesses = [np.mean(losses[losses > u] - u) for u in candidates]
        diffs = np.diff(mean_excesses)
        linear_start = np.argmax(np.abs(diffs) < np.std(diffs) * 0.5)
        return float(candidates[linear_start])
    return float(np.percentile(losses, 90))


def analyze(
    data,
    threshold_pct: float | None = None,
    confidence_levels: list[float] | None = None,
    return_period_losses: list[float] | None = None,
    window: int = 756,
) -> EVTResult:
    """
    Fit a Generalized Pareto Distribution to extreme losses.

    Parameters
    ----------
    data                 : TickerData with price_history, or pd.Series of returns
    threshold_pct        : float — threshold as % of data (default auto-select at 90th pct)
    confidence_levels    : list of confidence levels for VaR/ES (default [0.99, 0.999, 0.9999])
    return_period_losses : list of loss levels to compute return periods for
    window               : int — lookback days (default 756)

    Returns
    -------
    EVTResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.risk import evt
    >>> data = pull.ticker("AAPL")
    >>> result = evt.analyze(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    if isinstance(data, pd.Series):
        returns = data.dropna().values
        ticker = data.name or "series"
    elif hasattr(data, "price_history") and not data.price_history.empty:
        prices = data.price_history["Close"].tail(window)
        returns = prices.pct_change().dropna().values
        ticker = data.ticker
    else:
        raise ValueError("Provide TickerData with price_history or pd.Series of returns")

    losses = -returns[returns < 0]

    console.print(f"[dim]Fitting GPD to {ticker} ({len(losses)} negative returns)...[/dim]")

    if threshold_pct is not None:
        threshold = float(np.percentile(losses, threshold_pct * 100))
    else:
        threshold = _select_threshold(losses)

    exceedances = losses[losses > threshold] - threshold
    n_exc = len(exceedances)

    if n_exc < 15:
        console.print(f"[yellow]Warning: only {n_exc} exceedances — lowering threshold[/yellow]")
        threshold = float(np.percentile(losses, 85))
        exceedances = losses[losses > threshold] - threshold
        n_exc = len(exceedances)

    x0 = [0.1, float(np.mean(exceedances))]
    bounds = [(-0.5, 1.0), (1e-6, None)]

    res = minimize(
        _gpd_loglik, x0, args=(exceedances,),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    xi, sigma = res.x
    n_total = len(losses)
    p_threshold = n_exc / n_total

    def var_gpd(p: float) -> float:
        """VaR at probability p using fitted GPD."""
        if p <= 1 - p_threshold:
            return float(np.percentile(losses, p * 100))
        p_excess = (p - (1 - p_threshold)) / p_threshold
        if xi == 0:
            return float(threshold - sigma * np.log(1 - p_excess))
        return float(threshold + sigma * ((1 - p_excess) ** (-xi) - 1) / xi)

    def es_gpd(p: float) -> float:
        """Expected Shortfall (CVaR) at probability p."""
        v = var_gpd(p)
        if xi >= 1:
            return float(v * 3)
        return float((v + sigma - xi * threshold) / (1 - xi))

    cls = confidence_levels or [0.99, 0.999, 0.9999]
    var_99   = var_gpd(0.99)
    var_999  = var_gpd(0.999)
    var_9999 = var_gpd(0.9999)
    es_99    = es_gpd(0.99)
    es_999   = es_gpd(0.999)

    loss_levels = return_period_losses or [0.05, 0.10, 0.15, 0.20, 0.30]
    return_periods = {}
    for loss in loss_levels:
        if loss <= threshold:
            p_exceed = (losses >= loss).mean()
        else:
            y = (loss - threshold) / sigma
            if xi == 0:
                p_exceed_given_thresh = np.exp(-y)
            else:
                base = 1 + xi * y
                if base <= 0:
                    # bounded tail (xi < 0) — loss beyond maximum possible
                    p_exceed_given_thresh = 0.0
                else:
                    p_exceed_given_thresh = max(float(base ** (-1/xi)), 0)
            p_exceed = p_threshold * p_exceed_given_thresh
        rp = 1 / max(p_exceed, 1e-8)
        return_periods[loss] = round(rp, 1)

    tail_index = 1 / xi if xi > 0.01 else float("inf")

    console.print(
        f"[green]✓[/green] EVT fitted — "
        f"ξ={xi:.4f} ({('heavy' if xi>0.2 else 'moderate')} tail), "
        f"VaR(99%)={var_99:.2%}, "
        f"VaR(99.9%)={var_999:.2%}"
    )

    return EVTResult(
        ticker=ticker,
        threshold=round(float(threshold), 6),
        n_exceedances=n_exc,
        xi=round(float(xi), 6),
        sigma=round(float(sigma), 6),
        tail_index=round(float(tail_index), 4) if tail_index != float("inf") else 999.0,
        var_99=round(var_99, 6),
        var_999=round(var_999, 6),
        var_9999=round(var_9999, 6),
        es_99=round(es_99, 6),
        es_999=round(es_999, 6),
        return_periods=return_periods,
    )


def compare_tails(data_list: list, window: int = 756) -> pd.DataFrame:
    """
    Compare tail risk across multiple stocks.

    Parameters
    ----------
    data_list : list of TickerData
    window    : int — lookback days

    Returns
    -------
    pd.DataFrame comparing ξ, VaR(99%), VaR(99.9%), ES(99%) across stocks

    Example
    -------
    >>> results = evt.compare_tails([apple, msft, googl])
    >>> print(results)
    """
    from finverse.utils.display import console
    rows = []
    for d in data_list:
        try:
            r = analyze(d, window=window)
            rows.append({
                "ticker": r.ticker,
                "xi (tail index)": r.xi,
                "VaR 99%": r.var_99,
                "VaR 99.9%": r.var_999,
                "ES 99%": r.es_99,
                "threshold": r.threshold,
                "n_exceedances": r.n_exceedances,
            })
        except Exception as e:
            console.print(f"[yellow]Skipping {getattr(d,'ticker','?')}: {e}[/yellow]")

    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values("xi (tail index)", ascending=False).round(4)
