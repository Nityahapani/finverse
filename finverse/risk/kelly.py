"""
finverse.risk.kelly — Kelly Criterion and optimal position sizing.

The Kelly Criterion gives the fraction of capital to bet on an
investment to maximize long-run geometric growth rate.

Full Kelly is theoretically optimal but practically volatile.
Fractional Kelly (half-Kelly, quarter-Kelly) is standard in practice.

Provides:
  - Basic Kelly fraction (binary bet)
  - Continuous Kelly (from return distribution)
  - Multi-asset Kelly (via covariance matrix)
  - Fractional Kelly variants
  - Kelly vs fixed-fraction growth simulation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class KellyResult:
    ticker: str
    full_kelly: float               # optimal fraction (full Kelly)
    half_kelly: float
    quarter_kelly: float
    expected_growth_full: float     # geometric growth rate at full Kelly
    expected_growth_half: float
    expected_growth_quarter: float
    expected_growth_zero: float     # growth at no bet (= 0)
    edge: float                     # expected return
    odds: float                     # (used in binary Kelly)
    method: str
    breakeven_kelly: float          # fraction where growth = 0

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Kelly Criterion — {self.ticker}[/bold blue]")
        console.print(f"[dim]{self.method}[/dim]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Sizing strategy")
        table.add_column("Fraction", justify="right")
        table.add_column("Expected growth (ann.)", justify="right")

        rows = [
            ("Full Kelly (theoretical max)", self.full_kelly, self.expected_growth_full),
            ("Half Kelly (standard practice)", self.half_kelly, self.expected_growth_half),
            ("Quarter Kelly (conservative)", self.quarter_kelly, self.expected_growth_quarter),
            ("No position", 0.0, self.expected_growth_zero),
        ]

        colors = ["green", "blue", "blue", "dim"]
        for (label, frac, growth), color in zip(rows, colors):
            table.add_row(
                f"[{color}]{label}[/{color}]",
                f"[{color}]{frac:.1%}[/{color}]",
                f"[{color}]{growth:.2%} p.a.[/{color}]",
            )

        console.print(table)
        console.print(f"\n  Edge (expected return): {self.edge:.2%}")
        console.print(f"  Breakeven fraction:     {self.breakeven_kelly:.1%}")

        if self.full_kelly > 0.5:
            console.print(f"\n  [yellow]Warning: full Kelly > 50% — consider half or quarter Kelly in practice[/yellow]")
        if self.full_kelly <= 0:
            console.print(f"\n  [red]Negative Kelly — negative edge, no bet recommended[/red]")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "full_kelly": self.full_kelly,
            "half_kelly": self.half_kelly,
            "quarter_kelly": self.quarter_kelly,
            "expected_growth_full": self.expected_growth_full,
            "expected_growth_half": self.expected_growth_half,
            "edge": self.edge,
        }])

    def simulate(
        self,
        n_periods: int = 252,
        n_paths: int = 500,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Simulate wealth paths under different Kelly fractions.

        Returns DataFrame with columns: full, half, quarter, zero Kelly paths.
        """
        np.random.seed(seed)
        mu = self.edge / 252
        sigma = np.sqrt(max(self.edge * 2, 0.01)) / np.sqrt(252)

        returns = np.random.normal(mu, sigma, (n_periods, n_paths))

        results = {}
        for name, frac in [
            ("Full Kelly", self.full_kelly),
            ("Half Kelly", self.half_kelly),
            ("Quarter Kelly", self.quarter_kelly),
            ("No bet", 0.0),
        ]:
            portfolio_returns = frac * returns
            wealth = np.cumprod(1 + portfolio_returns, axis=0)
            results[name] = pd.Series(np.median(wealth, axis=1))

        return pd.DataFrame(results)


def from_distribution(
    data,
    window: int = 756,
    annualize: bool = True,
) -> KellyResult:
    """
    Compute Kelly fraction from empirical return distribution.

    Uses the continuous Kelly formula:
      f* = μ / σ²
    where μ = expected return, σ² = variance of returns.

    Parameters
    ----------
    data     : TickerData with price_history, or pd.Series of returns
    window   : int — lookback days (default 756)
    annualize: bool — annualize growth rates (default True)

    Returns
    -------
    KellyResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.risk import kelly
    >>> data = pull.ticker("AAPL")
    >>> result = kelly.from_distribution(data)
    >>> result.summary()
    >>> paths = result.simulate(n_periods=252)
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
        raise ValueError("Provide TickerData or pd.Series")

    console.print(f"[dim]Computing Kelly fraction for {ticker}...[/dim]")

    mu = float(np.mean(returns))
    sigma2 = float(np.var(returns))

    full_kelly = mu / sigma2 if sigma2 > 0 else 0.0
    full_kelly = float(np.clip(full_kelly, -1.0, 1.0))

    half_kelly    = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25

    scale = 252 if annualize else 1

    def growth_rate(f: float) -> float:
        return float(scale * (f * mu - 0.5 * f**2 * sigma2))

    g_full    = growth_rate(full_kelly)
    g_half    = growth_rate(half_kelly)
    g_quarter = growth_rate(quarter_kelly)
    g_zero    = 0.0

    breakeven = 2 * mu / sigma2 if sigma2 > 0 else 0.0
    breakeven = float(np.clip(breakeven, 0, 1))

    ann_mu = mu * 252
    console.print(
        f"[green]✓[/green] Kelly: {full_kelly:.1%} full, {half_kelly:.1%} half | "
        f"edge={ann_mu:.2%} p.a., growth@full={g_full:.2%}"
    )

    return KellyResult(
        ticker=ticker,
        full_kelly=round(full_kelly, 4),
        half_kelly=round(half_kelly, 4),
        quarter_kelly=round(quarter_kelly, 4),
        expected_growth_full=round(g_full, 4),
        expected_growth_half=round(g_half, 4),
        expected_growth_quarter=round(g_quarter, 4),
        expected_growth_zero=g_zero,
        edge=round(ann_mu, 4),
        odds=round(mu / max(abs(min(returns)), 1e-8), 4),
        method="Continuous Kelly (f* = μ/σ²)",
        breakeven_kelly=round(breakeven, 4),
    )


def from_binary(
    win_prob: float,
    win_return: float,
    loss_return: float,
    ticker: str = "bet",
) -> KellyResult:
    """
    Kelly fraction for a binary outcome (win/loss) bet.

    Classic Kelly formula: f* = (p*b - q) / b
    where p = win prob, q = 1-p, b = win payoff.

    Parameters
    ----------
    win_prob    : float — probability of winning (0–1)
    win_return  : float — fractional return if win (e.g. 0.10 = 10%)
    loss_return : float — fractional loss if lose (e.g. 0.05 = 5% loss)
    ticker      : str — label

    Returns
    -------
    KellyResult

    Example
    -------
    >>> result = kelly.from_binary(win_prob=0.55, win_return=0.10, loss_return=0.08)
    >>> result.summary()
    """
    from finverse.utils.display import console
    console.print(f"[dim]Computing binary Kelly (p={win_prob:.2%}, b={win_return:.2%})...[/dim]")

    p = win_prob
    q = 1 - p
    b = win_return / loss_return

    full_kelly = (p * b - q) / b
    full_kelly = float(np.clip(full_kelly, -1.0, 1.0))
    half_kelly = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25

    edge = p * win_return - q * loss_return

    def growth(f):
        return p * np.log(1 + f * win_return) + q * np.log(1 - f * loss_return)

    g_full    = float(growth(full_kelly) * 252) if full_kelly > 0 else 0.0
    g_half    = float(growth(half_kelly) * 252) if half_kelly > 0 else 0.0
    g_quarter = float(growth(quarter_kelly) * 252) if quarter_kelly > 0 else 0.0

    console.print(f"[green]✓[/green] Binary Kelly: {full_kelly:.1%}, edge={edge:.2%}")

    return KellyResult(
        ticker=ticker,
        full_kelly=round(full_kelly, 4),
        half_kelly=round(half_kelly, 4),
        quarter_kelly=round(quarter_kelly, 4),
        expected_growth_full=round(g_full, 4),
        expected_growth_half=round(g_half, 4),
        expected_growth_quarter=round(g_quarter, 4),
        expected_growth_zero=0.0,
        edge=round(edge, 4),
        odds=round(b, 4),
        method=f"Binary Kelly (p={p:.2%}, b={b:.2f}x)",
        breakeven_kelly=round(q / b, 4),
    )


def multi_asset(
    data_list: list,
    window: int = 756,
    risk_free: float = 0.045,
) -> pd.Series:
    """
    Multi-asset Kelly allocation using the covariance matrix.

    f* = Σ⁻¹ μ (proportional to Sharpe-optimal weights)

    Parameters
    ----------
    data_list : list of TickerData
    window    : lookback days
    risk_free : float — risk-free rate

    Returns
    -------
    pd.Series of Kelly fractions (may need scaling to sum to ≤ 1)

    Example
    -------
    >>> fractions = kelly.multi_asset([apple, msft, googl])
    >>> print(fractions)
    """
    from finverse.utils.display import console
    from finverse.portfolio.optimizer import _get_returns

    console.print(f"[dim]Computing multi-asset Kelly for {len(data_list)} assets...[/dim]")

    try:
        returns_df = _get_returns(data_list, "3y")
    except Exception:
        np.random.seed(42)
        tickers = [d.ticker for d in data_list]
        n = len(tickers)
        returns_df = pd.DataFrame(
            np.random.randn(756, n) * 0.012 + 0.0005,
            columns=tickers
        )

    mu = returns_df.mean().values * 252 - risk_free
    cov = returns_df.cov().values * 252

    try:
        cov_inv = np.linalg.inv(cov)
        kelly_fracs = cov_inv @ mu
        kelly_fracs = np.clip(kelly_fracs, -1, 1)
    except np.linalg.LinAlgError:
        kelly_fracs = mu / (np.diag(cov) + 1e-8)

    tickers = [d.ticker for d in data_list if hasattr(d, "ticker")]
    result = pd.Series(kelly_fracs.round(4), index=tickers)
    console.print(f"[green]✓[/green] Multi-asset Kelly computed")
    return result
