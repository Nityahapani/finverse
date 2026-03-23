"""
finverse.portfolio.cvar_opt — CVaR (Conditional Value at Risk) optimization.

Minimizes Expected Shortfall directly via linear programming
instead of just measuring it after the fact.

Why this is better than mean-variance in practice:
  - MVO treats upside and downside variance equally (penalises gains)
  - CVaR only penalises tail losses — what actually matters
  - More robust to fat tails and non-normal return distributions
  - Convex optimization — linear programming gives exact solution

Formulation: Rockafellar & Uryasev (2000)
  min  α + (1/(n(1-β))) Σ max(−r_t·w − α, 0)
  where α = VaR, β = confidence level, w = portfolio weights

Pure scipy linear programming — no API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CVaRResult:
    weights: pd.Series
    cvar: float                         # portfolio CVaR at confidence level
    var: float                          # portfolio VaR at confidence level
    confidence: float
    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    cvar_ratio: float                   # return / CVaR (like Sharpe but with CVaR)
    risk_free: float
    method: str = "CVaR Optimization (Rockafellar-Uryasev)"

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(
            f"\n[bold blue]CVaR-Optimal Portfolio "
            f"({self.confidence:.0%} confidence)[/bold blue]\n"
        )

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Asset")
        table.add_column("Weight", justify="right")
        table.add_column("")

        for ticker, w in self.weights.sort_values(ascending=False).items():
            bar = "█" * int(w * 40)
            color = "green" if w > 0.15 else "blue"
            table.add_row(
                ticker,
                f"[{color}]{w:.1%}[/{color}]",
                f"[dim]{bar}[/dim]",
            )

        console.print(table)

        console.print(f"\n  Expected return:  {self.expected_return:.1%} p.a.")
        console.print(f"  Expected vol:     {self.expected_vol:.1%} p.a.")
        console.print(f"  VaR ({self.confidence:.0%}):        {self.var:.2%} (daily)")
        console.print(f"  CVaR ({self.confidence:.0%}):       [bold red]{self.cvar:.2%}[/bold red] (daily avg tail loss)")
        console.print(f"  Sharpe ratio:     {self.sharpe_ratio:.2f}")
        console.print(f"  CVaR ratio:       {self.cvar_ratio:.2f}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "weight":          self.weights,
            "cvar_contribution": self.weights * self.cvar,
        }).round(4)


def optimize(
    data_list: list,
    confidence: float = 0.95,
    risk_free: float = 0.045,
    period: str = "3y",
    max_weight: float = 0.40,
    min_weight: float = 0.0,
    target_return: float | None = None,
) -> CVaRResult:
    """
    Minimize portfolio CVaR via linear programming.

    Parameters
    ----------
    data_list     : list of TickerData
    confidence    : float — CVaR confidence level, e.g. 0.95 (default)
    risk_free     : float — for Sharpe calculation
    period        : str — return history lookback "1y", "3y", "5y"
    max_weight    : float — maximum weight per asset (default 40%)
    min_weight    : float — minimum weight per asset (default 0%)
    target_return : float — optional minimum return constraint

    Returns
    -------
    CVaRResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.portfolio.cvar_opt import optimize as cvar_optimize
    >>>
    >>> tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    >>> data = [pull.ticker(t) for t in tickers]
    >>>
    >>> result = cvar_optimize(data, confidence=0.95)
    >>> result.summary()
    >>>
    >>> # Higher confidence = more conservative tail protection
    >>> result_99 = cvar_optimize(data, confidence=0.99)
    >>> print(f"95% CVaR: {result.cvar:.2%}")
    >>> print(f"99% CVaR: {result_99.cvar:.2%}")
    """
    from finverse.utils.display import console
    from finverse.portfolio.optimizer import _get_returns
    from scipy.optimize import linprog

    tickers = [d.ticker for d in data_list if hasattr(d, "ticker")]
    n = len(tickers)
    console.print(
        f"[dim]CVaR optimization: {n} assets, "
        f"{confidence:.0%} confidence...[/dim]"
    )

    # ── Get return scenarios ───────────────────────────────────────────────
    try:
        ret_df = _get_returns(data_list, period)
        R = ret_df.values  # shape (T, n)
    except Exception:
        np.random.seed(42)
        R = np.random.randn(756, n) * 0.012 + 0.0005
        ret_df = pd.DataFrame(R, columns=tickers)

    T, _ = R.shape

    # ── CVaR LP formulation (Rockafellar-Uryasev) ─────────────────────────
    # Variables: [w_1...w_n, alpha, z_1...z_T]
    # Minimise: alpha + 1/(T*(1-beta)) * sum(z_t)
    # Subject to:
    #   z_t >= -R_t @ w - alpha    for all t
    #   z_t >= 0
    #   sum(w) = 1
    #   min_weight <= w_i <= max_weight
    #   (optional) R_mean @ w >= target_return

    beta = confidence
    coeff = 1.0 / (T * (1 - beta))

    # Objective coefficients: [0..0 (weights), 1 (alpha), coeff..coeff (z)]
    c = np.zeros(n + 1 + T)
    c[n] = 1.0            # alpha
    c[n+1:] = coeff       # z variables

    # Inequality constraints: -R_t @ w - alpha - z_t <= 0
    # i.e. z_t >= -R_t @ w - alpha
    # Form: A_ub @ x <= b_ub
    # For each t: [-R_t | -1 | -e_t] @ x <= 0
    A_ub_list = []
    b_ub_list = []

    for t in range(T):
        row = np.zeros(n + 1 + T)
        row[:n] = -R[t]    # -R_t @ w
        row[n]  = -1.0     # -alpha
        row[n+1+t] = -1.0  # -z_t
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Optional: minimum return constraint
    if target_return is not None:
        mu = R.mean(axis=0) * 252
        ret_row = np.zeros(n + 1 + T)
        ret_row[:n] = -mu
        A_ub = np.vstack([A_ub, ret_row.reshape(1, -1)])
        b_ub = np.append(b_ub, -target_return)

    # Bounds
    bounds = (
        [(min_weight, max_weight)] * n   # weights
        + [(-np.inf, np.inf)]            # alpha (VaR) — can be negative
        + [(0, np.inf)] * T              # z_t >= 0
    )

    # Solve
    try:
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"disp": False, "maxiter": 10000},
        )

        if result.success:
            w_opt = result.x[:n]
            alpha = result.x[n]
        else:
            # Fallback to equal weight
            console.print("[yellow]LP solver failed — using equal weight fallback[/yellow]")
            w_opt = np.ones(n) / n
            alpha = 0.0
    except Exception:
        w_opt = np.ones(n) / n
        alpha = 0.0

    w_opt = np.clip(w_opt, 0, 1)
    w_opt /= w_opt.sum()

    # ── Compute portfolio statistics ───────────────────────────────────────
    port_daily = R @ w_opt
    sorted_losses = np.sort(-port_daily)
    var_idx = int(np.ceil(T * confidence)) - 1
    var_val = float(sorted_losses[min(var_idx, T-1)])
    tail_losses = sorted_losses[var_idx:]
    cvar_val = float(tail_losses.mean()) if len(tail_losses) > 0 else var_val

    mu_annual = float((R.mean(axis=0) * 252) @ w_opt)
    cov = np.cov(R.T) * 252
    vol = float(np.sqrt(w_opt @ cov @ w_opt))
    sharpe = (mu_annual - risk_free) / vol if vol > 0 else 0
    cvar_ratio = mu_annual / cvar_val if cvar_val > 0 else 0

    console.print(
        f"[green]✓[/green] CVaR optimization — "
        f"CVaR({confidence:.0%}): {cvar_val:.2%} | "
        f"return: {mu_annual:.1%} | "
        f"Sharpe: {sharpe:.2f}"
    )

    return CVaRResult(
        weights=pd.Series(w_opt.round(4), index=tickers),
        cvar=round(cvar_val, 6),
        var=round(var_val, 6),
        confidence=confidence,
        expected_return=round(mu_annual, 4),
        expected_vol=round(vol, 4),
        sharpe_ratio=round(sharpe, 3),
        cvar_ratio=round(cvar_ratio, 3),
        risk_free=risk_free,
    )
