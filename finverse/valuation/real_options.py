"""
finverse.valuation.real_options — Real Options valuation using Black-Scholes.

Treats strategic corporate investments as options:
- Option to expand (call)
- Option to abandon (put)
- Option to defer/wait (call on underlying project)
- Option to switch (exchange option)

Pure Black-Scholes math, no external APIs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class RealOptionResult:
    option_type: str
    option_value: float            # $ value of the option
    dcf_npv: float                 # static NPV (option-free)
    expanded_npv: float            # NPV including option value
    delta: float                   # sensitivity to underlying
    intrinsic_value: float         # max(S-X, 0) or max(X-S, 0)
    time_value: float              # option_value - intrinsic_value
    inputs: dict

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Real Option — {self.option_type}[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Component")
        table.add_column("Value ($M)", justify="right")

        table.add_row("Static NPV (DCF)",      f"{self.dcf_npv:.1f}")
        table.add_row("Option value",           f"[bold green]{self.option_value:.1f}[/bold green]")
        table.add_row("  Intrinsic value",      f"  {self.intrinsic_value:.1f}")
        table.add_row("  Time value",           f"  {self.time_value:.1f}")
        table.add_row("Expanded NPV (total)",   f"[bold]{self.expanded_npv:.1f}[/bold]")
        table.add_row("Delta (Δ)",              f"{self.delta:.4f}")
        console.print(table)

        console.print("\n  [dim]Inputs:[/dim]")
        for k, v in self.inputs.items():
            if isinstance(v, float):
                console.print(f"    {k:<25} {v:.4f}" if v < 100 else f"    {k:<25} {v:.1f}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "option_type": self.option_type,
            "option_value": self.option_value,
            "dcf_npv": self.dcf_npv,
            "expanded_npv": self.expanded_npv,
            "delta": self.delta,
        }])


def _bs_call(S, X, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(S - X, 0), 0
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return float(price), float(delta)


def _bs_put(S, X, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(X - S, 0), -1
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    return float(price), float(delta)


def expand(
    project_value: float,
    expansion_cost: float,
    sigma: float,
    time_to_expiry: float,
    risk_free: float = 0.045,
    dcf_npv: float | None = None,
) -> RealOptionResult:
    """
    Option to expand — value of being able to scale up a project.

    S = current project value, X = cost of expansion.

    Parameters
    ----------
    project_value   : float — current PV of project cash flows ($M)
    expansion_cost  : float — investment required to expand ($M)
    sigma           : float — volatility of project value (annual)
    time_to_expiry  : float — years until option expires
    risk_free       : float — risk-free rate (default 4.5%)
    dcf_npv         : float — static DCF NPV (if known)

    Returns
    -------
    RealOptionResult

    Example
    -------
    >>> from finverse.valuation import real_options
    >>> result = real_options.expand(
    ...     project_value=500,   # $500M project
    ...     expansion_cost=200,  # $200M to expand
    ...     sigma=0.30,          # 30% project volatility
    ...     time_to_expiry=3.0,  # 3-year window
    ... )
    >>> result.summary()
    """
    from finverse.utils.display import console
    console.print("[dim]Valuing expansion option...[/dim]")

    option_value, delta = _bs_call(project_value, expansion_cost, risk_free, sigma, time_to_expiry)
    intrinsic = max(project_value - expansion_cost, 0)
    time_value = option_value - intrinsic
    base_npv = dcf_npv if dcf_npv is not None else project_value - expansion_cost
    expanded_npv = base_npv + option_value

    console.print(f"[green]✓[/green] Expansion option value: ${option_value:.1f}M (expanded NPV: ${expanded_npv:.1f}M)")

    return RealOptionResult(
        option_type="Option to expand",
        option_value=round(option_value, 2),
        dcf_npv=round(base_npv, 2),
        expanded_npv=round(expanded_npv, 2),
        delta=round(delta, 4),
        intrinsic_value=round(intrinsic, 2),
        time_value=round(time_value, 2),
        inputs={
            "Project value (S)": project_value,
            "Expansion cost (X)": expansion_cost,
            "Volatility (σ)": sigma,
            "Time to expiry (T)": time_to_expiry,
            "Risk-free rate (r)": risk_free,
        },
    )


def abandon(
    project_value: float,
    salvage_value: float,
    sigma: float,
    time_to_expiry: float,
    risk_free: float = 0.045,
    dcf_npv: float | None = None,
) -> RealOptionResult:
    """
    Option to abandon — value of being able to exit and recover salvage.

    Modeled as a put option: right to sell project at salvage price.

    Parameters
    ----------
    project_value : float — current PV of project ($M)
    salvage_value : float — value if project abandoned ($M)
    sigma         : float — project volatility
    time_to_expiry: float — years until option expires
    risk_free     : float — risk-free rate

    Example
    -------
    >>> result = real_options.abandon(
    ...     project_value=300,
    ...     salvage_value=150,
    ...     sigma=0.35,
    ...     time_to_expiry=2.0,
    ... )
    >>> result.summary()
    """
    from finverse.utils.display import console
    console.print("[dim]Valuing abandonment option...[/dim]")

    option_value, delta = _bs_put(project_value, salvage_value, risk_free, sigma, time_to_expiry)
    intrinsic = max(salvage_value - project_value, 0)
    time_value = option_value - intrinsic
    base_npv = dcf_npv if dcf_npv is not None else project_value - salvage_value
    expanded_npv = base_npv + option_value

    console.print(f"[green]✓[/green] Abandonment option value: ${option_value:.1f}M")

    return RealOptionResult(
        option_type="Option to abandon",
        option_value=round(option_value, 2),
        dcf_npv=round(base_npv, 2),
        expanded_npv=round(expanded_npv, 2),
        delta=round(delta, 4),
        intrinsic_value=round(intrinsic, 2),
        time_value=round(time_value, 2),
        inputs={
            "Project value (S)": project_value,
            "Salvage value (X)": salvage_value,
            "Volatility (σ)": sigma,
            "Time to expiry (T)": time_to_expiry,
            "Risk-free rate (r)": risk_free,
        },
    )


def defer(
    project_value: float,
    investment_cost: float,
    sigma: float,
    time_to_expiry: float,
    risk_free: float = 0.045,
    dividend_yield: float = 0.0,
) -> RealOptionResult:
    """
    Option to defer — value of waiting before committing to an investment.

    Merton (1973) call with continuous dividend yield (opportunity cost of waiting).

    Parameters
    ----------
    project_value   : float — current PV of project ($M)
    investment_cost : float — investment required ($M)
    sigma           : float — project value volatility
    time_to_expiry  : float — how long the option to wait lasts
    risk_free       : float — risk-free rate
    dividend_yield  : float — cash flows lost by waiting (opportunity cost)

    Example
    -------
    >>> result = real_options.defer(
    ...     project_value=400,
    ...     investment_cost=350,
    ...     sigma=0.25,
    ...     time_to_expiry=2.0,
    ...     dividend_yield=0.04,  # 4% annual cash flows foregone
    ... )
    >>> result.summary()
    """
    from finverse.utils.display import console
    console.print("[dim]Valuing deferral option...[/dim]")

    S_adj = project_value * np.exp(-dividend_yield * time_to_expiry)
    option_value, delta = _bs_call(S_adj, investment_cost, risk_free, sigma, time_to_expiry)
    intrinsic = max(project_value - investment_cost, 0)
    time_value = option_value - intrinsic
    base_npv = project_value - investment_cost
    expanded_npv = option_value  # option value IS the value of the deferred investment

    console.print(f"[green]✓[/green] Deferral option value: ${option_value:.1f}M (vs invest now NPV: ${base_npv:.1f}M)")

    return RealOptionResult(
        option_type="Option to defer",
        option_value=round(option_value, 2),
        dcf_npv=round(base_npv, 2),
        expanded_npv=round(expanded_npv, 2),
        delta=round(delta, 4),
        intrinsic_value=round(intrinsic, 2),
        time_value=round(time_value, 2),
        inputs={
            "Project value (S)": project_value,
            "Investment cost (X)": investment_cost,
            "Volatility (σ)": sigma,
            "Time to expiry (T)": time_to_expiry,
            "Risk-free rate (r)": risk_free,
            "Dividend yield (δ)": dividend_yield,
        },
    )


def sensitivity_grid(
    project_value: float,
    cost: float,
    option_fn,
    sigma_range: tuple = (0.15, 0.50),
    time_range: tuple = (0.5, 5.0),
    n: int = 5,
) -> pd.DataFrame:
    """
    Build a sensitivity grid of option values over sigma and time.

    Parameters
    ----------
    project_value : float
    cost          : float — exercise cost / strike
    option_fn     : callable — one of expand, abandon, defer
    sigma_range   : (min_sigma, max_sigma)
    time_range    : (min_T, max_T)
    n             : grid size (default 5)

    Returns
    -------
    pd.DataFrame — sigma on rows, time on columns

    Example
    -------
    >>> grid = real_options.sensitivity_grid(500, 200, real_options.expand)
    >>> print(grid)
    """
    sigmas = np.linspace(*sigma_range, n)
    times = np.linspace(*time_range, n)

    rows = {}
    for sigma in sigmas:
        row = {}
        for T in times:
            try:
                r = option_fn(project_value, cost, sigma, T)
                row[f"T={T:.1f}y"] = round(r.option_value, 1)
            except Exception:
                row[f"T={T:.1f}y"] = np.nan
        rows[f"σ={sigma:.0%}"] = row

    return pd.DataFrame(rows).T
