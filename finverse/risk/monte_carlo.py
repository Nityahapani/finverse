"""
finverse.risk.monte_carlo — Monte Carlo simulation over DCF assumptions.

Runs thousands of scenarios by sampling from distributions over key
assumptions, producing a full probability distribution of outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class MonteCarloResult:
    ticker: str
    n_simulations: int
    implied_prices: np.ndarray
    mean_price: float
    median_price: float
    std_price: float
    percentiles: dict[int, float]        # 5, 10, 25, 50, 75, 90, 95
    prob_upside: float | None            # probability price > current
    current_price: float | None
    assumption_distributions: dict       # what was sampled

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Monte Carlo Simulation — {self.ticker}[/bold blue]")
        console.print(f"[dim]{self.n_simulations:,} simulations[/dim]\n")

        table = Table(title="Price distribution", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Percentile")
        table.add_column("Implied price", justify="right")
        table.add_column("")

        pct_labels = {5: "5th (bear)", 25: "25th", 50: "50th (median)",
                      75: "75th", 95: "95th (bull)"}
        for p, label in pct_labels.items():
            price = self.percentiles[p]
            bar_len = int((price / (self.percentiles[95] + 1)) * 20)
            bar = "█" * bar_len
            table.add_row(label, f"${price:.2f}", f"[dim]{bar}[/dim]")

        console.print(table)
        console.print(f"\n  Mean:   ${self.mean_price:.2f}")
        console.print(f"  Median: ${self.median_price:.2f}")
        console.print(f"  Std dev: ${self.std_price:.2f}")

        if self.prob_upside is not None and self.current_price:
            color = "green" if self.prob_upside > 0.5 else "red"
            console.print(
                f"\n  Current price: ${self.current_price:.2f}  |  "
                f"P(upside) = [{color}][bold]{self.prob_upside:.0%}[/bold][/{color}]"
            )

        if self.assumption_distributions:
            console.print("\n  [dim]Assumptions sampled (Normal distributions):[/dim]")
            for param, dist in self.assumption_distributions.items():
                mean_v = dist["mean"]
                std_v  = dist["std"]
                lo = mean_v - 2 * std_v
                hi = mean_v + 2 * std_v
                fmt = lambda x: f"{x:.1%}" if x < 2 else f"${x:.1f}B"
                console.print(
                    f"    {param:<22} μ={fmt(mean_v)}  σ={fmt(std_v)}"
                    f"  [dim](±2σ: {fmt(lo)} – {fmt(hi)})[/dim]"
                )
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "implied_price": self.implied_prices,
        })

    def plot(self):
        """Plot histogram of simulated prices."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(self.implied_prices, bins=80, ax=ax, color="#185FA5", alpha=0.7)

            for p, color, label in [
                (5,  "#E24B4A", "5th pct"),
                (50, "#1D9E75", "Median"),
                (95, "#EF9F27", "95th pct"),
            ]:
                ax.axvline(self.percentiles[p], color=color, linestyle="--", linewidth=1.5, label=f"{label}: ${self.percentiles[p]:.2f}")

            if self.current_price:
                ax.axvline(self.current_price, color="black", linestyle="-", linewidth=2, label=f"Current: ${self.current_price:.2f}")

            ax.set_xlabel("Implied Share Price ($)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Monte Carlo — {self.ticker} ({self.n_simulations:,} simulations)", fontsize=13)
            ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            from finverse.utils.display import console
            console.print("[yellow]matplotlib/seaborn required for plot: pip install matplotlib seaborn[/yellow]")


def simulate(
    model,
    n_simulations: int = 10_000,
    wacc_std: float = 0.015,
    growth_std: float = 0.02,
    margin_std: float = 0.03,
    terminal_growth_std: float = 0.005,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation over DCF assumptions.

    Samples from normal distributions around each assumption and
    runs a full DCF for each simulation.

    Parameters
    ----------
    model             : DCF model instance (configured but not necessarily run)
    n_simulations     : int — number of scenarios (default 10,000)
    wacc_std          : float — std dev for WACC sampling (default 1.5%)
    growth_std        : float — std dev for revenue growth (default 2%)
    margin_std        : float — std dev for EBITDA margin (default 3%)
    terminal_growth_std : float — std dev for terminal growth (default 0.5%)
    seed              : int — random seed for reproducibility

    Returns
    -------
    MonteCarloResult

    Example
    -------
    >>> from finverse import pull, DCF
    >>> from finverse.risk import monte_carlo
    >>> data = pull.ticker("AAPL")
    >>> model = DCF(data)
    >>> result = monte_carlo.simulate(model, n_simulations=10000)
    >>> result.summary()
    >>> result.plot()
    """
    from finverse.utils.display import console

    ticker = "Unknown"
    if model._data is not None and hasattr(model._data, "ticker"):
        ticker = model._data.ticker

    console.print(f"[dim]Running Monte Carlo: {n_simulations:,} simulations for {ticker}...[/dim]")

    np.random.seed(seed)
    a = model._assumptions

    wacc_samples          = np.random.normal(a.wacc, wacc_std, n_simulations).clip(0.04, 0.25)
    tg_samples            = np.random.normal(a.terminal_growth, terminal_growth_std, n_simulations).clip(0.005, 0.05)
    margin_samples        = np.random.normal(a.ebitda_margin, margin_std, n_simulations).clip(0.01, 0.80)
    growth_samples        = np.random.normal(
        a.revenue_growth[0] if isinstance(a.revenue_growth, (list, tuple)) and a.revenue_growth
        else (float(a.revenue_growth) if isinstance(a.revenue_growth, (int, float)) else 0.08),
        growth_std, n_simulations
    ).clip(-0.15, 0.40)

    implied_prices = np.zeros(n_simulations)

    base_revenue = model._base_revenue or 100.0
    shares = model._shares or 1.0
    net_debt = model._net_debt or 0.0
    n_years = a.projection_years
    tax = a.tax_rate
    capex_pct = a.capex_pct_revenue
    nwc_pct = a.nwc_pct_revenue

    for i in range(n_simulations):
        wacc   = wacc_samples[i]
        tg     = tg_samples[i]
        margin = margin_samples[i]
        growth = growth_samples[i]

        revenue = base_revenue
        pv_sum = 0.0

        for yr in range(1, n_years + 1):
            revenue *= (1 + growth)
            ebitda   = revenue * margin
            capex    = revenue * capex_pct
            nwc_ch   = revenue * nwc_pct * growth
            fcf      = ebitda * (1 - tax) - capex - nwc_ch
            pv_sum  += fcf / (1 + wacc) ** yr

        terminal_fcf = revenue * margin * (1 - tax) * (1 + tg)
        if wacc > tg:
            terminal_pv = (terminal_fcf / (wacc - tg)) / (1 + wacc) ** n_years
        else:
            terminal_pv = pv_sum * 2

        ev = pv_sum + terminal_pv
        equity = ev - net_debt
        price = (equity * 1e9) / (shares * 1e9) if shares > 0 else 0
        implied_prices[i] = max(price, 0)

    implied_prices = implied_prices[implied_prices > 0]

    percentiles = {p: float(np.percentile(implied_prices, p))
                   for p in [5, 10, 25, 50, 75, 90, 95]}

    current = model._current_price
    prob_up = float(np.mean(implied_prices > current)) if current else None

    console.print(
        f"[green]✓[/green] Monte Carlo complete — "
        f"median ${np.median(implied_prices):.2f}  |  "
        f"5th–95th: ${percentiles[5]:.2f}–${percentiles[95]:.2f}"
        + (f"  |  P(upside)={prob_up:.0%}" if prob_up else "")
    )

    return MonteCarloResult(
        ticker=ticker,
        n_simulations=len(implied_prices),
        implied_prices=implied_prices,
        mean_price=round(float(np.mean(implied_prices)), 2),
        median_price=round(float(np.median(implied_prices)), 2),
        std_price=round(float(np.std(implied_prices)), 2),
        percentiles={k: round(v, 2) for k, v in percentiles.items()},
        prob_upside=round(prob_up, 4) if prob_up else None,
        current_price=current,
        assumption_distributions={
            "wacc": {"mean": a.wacc, "std": wacc_std},
            "terminal_growth": {"mean": a.terminal_growth, "std": terminal_growth_std},
            "ebitda_margin": {"mean": a.ebitda_margin, "std": margin_std},
            "revenue_growth": {
                "mean": a.revenue_growth[0] if isinstance(a.revenue_growth, (list, tuple)) and a.revenue_growth
                        else (float(a.revenue_growth) if isinstance(a.revenue_growth, (int, float)) else 0.08),
                "std": growth_std,
            },
        },
    )
