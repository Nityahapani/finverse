"""
finverse.ml.garch — GARCH family volatility modeling.

GARCH(1,1), EGARCH, and GJR-GARCH implemented from scratch using
maximum likelihood estimation via scipy. No arch package required.

Provides:
- Time-varying volatility estimates
- Multi-step volatility forecasts with confidence bands
- Volatility regime classification
- Annualised conditional volatility series
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GARCHResult:
    model: str                        # "GARCH(1,1)", "EGARCH", "GJR-GARCH"
    ticker: str
    omega: float                      # constant term
    alpha: float                      # ARCH coefficient
    beta: float                       # GARCH coefficient
    gamma: float | None               # asymmetry (GJR/EGARCH only)
    persistence: float                # alpha + beta (or equivalent)
    long_run_vol: float               # annualised unconditional vol
    current_vol: float                # annualised conditional vol (latest)
    conditional_vol: pd.Series        # full time series
    forecast: pd.Series               # n-step ahead vol forecast
    log_likelihood: float
    aic: float
    bic: float

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]{self.model} — {self.ticker}[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Parameter")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        table.add_row("ω (omega)",      f"{self.omega:.6f}", "Baseline variance")
        table.add_row("α (alpha)",      f"{self.alpha:.4f}", "ARCH effect (shock impact)")
        table.add_row("β (beta)",       f"{self.beta:.4f}",  "GARCH effect (vol persistence)")
        if self.gamma is not None:
            table.add_row("γ (gamma)",  f"{self.gamma:.4f}", "Asymmetry (leverage effect)")
        table.add_row("Persistence",    f"{self.persistence:.4f}", "α+β (>0.95 = long memory)")
        table.add_row("Long-run vol",   f"{self.long_run_vol:.2%}",  "Unconditional annual vol")
        table.add_row("Current vol",    f"{self.current_vol:.2%}",   "Latest conditional annual vol")
        table.add_row("Log-likelihood", f"{self.log_likelihood:.2f}", "")
        table.add_row("AIC",            f"{self.aic:.2f}", "Lower = better fit")
        console.print(table)

        vol_regime = (
            "low" if self.current_vol < 0.15
            else "elevated" if self.current_vol < 0.30
            else "high"
        )
        color = {"low": "green", "elevated": "yellow", "high": "red"}[vol_regime]
        console.print(f"\n  Volatility regime: [{color}][bold]{vol_regime}[/bold][/{color}]")

        if not self.forecast.empty:
            console.print(f"\n  [dim]Volatility forecast:[/dim]")
            for h, v in self.forecast.items():
                console.print(f"    {h}: {v:.2%}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "conditional_vol": self.conditional_vol,
        })


def _garch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / h)
    return -ll


def _egarch_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1:
        return 1e10
    n = len(returns)
    log_h = np.zeros(n)
    log_h[0] = np.log(np.var(returns) + 1e-8)
    for t in range(1, n):
        std_resid = returns[t-1] / np.exp(0.5 * log_h[t-1])
        log_h[t] = (omega
                    + alpha * (np.abs(std_resid) - np.sqrt(2/np.pi))
                    + gamma * std_resid
                    + beta * log_h[t-1])
    h = np.exp(log_h)
    ll = -0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / h)
    return -ll


def _gjr_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
        return 1e10
    if alpha + 0.5 * gamma + beta >= 1:
        return 1e10
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    for t in range(1, n):
        indicator = 1.0 if returns[t-1] < 0 else 0.0
        h[t] = omega + (alpha + gamma * indicator) * returns[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / h)
    return -ll


def _compute_conditional_vol(params, returns, model_type):
    n = len(returns)
    if model_type == "GARCH(1,1)":
        omega, alpha, beta = params
        h = np.zeros(n)
        h[0] = np.var(returns)
        for t in range(1, n):
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        return h, alpha, beta, None
    elif model_type == "EGARCH":
        omega, alpha, gamma, beta = params
        log_h = np.zeros(n)
        log_h[0] = np.log(np.var(returns) + 1e-8)
        for t in range(1, n):
            std_resid = returns[t-1] / (np.exp(0.5 * log_h[t-1]) + 1e-8)
            log_h[t] = (omega + alpha * (np.abs(std_resid) - np.sqrt(2/np.pi))
                        + gamma * std_resid + beta * log_h[t-1])
        return np.exp(log_h), alpha, beta, gamma
    else:  # GJR
        omega, alpha, gamma, beta = params
        h = np.zeros(n)
        h[0] = np.var(returns)
        for t in range(1, n):
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * indicator) * returns[t-1]**2 + beta * h[t-1]
        return h, alpha, beta, gamma


def fit(
    data,
    model_type: str = "GJR-GARCH",
    forecast_horizon: int = 22,
    window: int = 756,
) -> GARCHResult:
    """
    Fit a GARCH family model to price return data.

    Parameters
    ----------
    data            : TickerData with price_history, or pd.Series of returns
    model_type      : "GARCH(1,1)", "EGARCH", "GJR-GARCH" (default "GJR-GARCH")
    forecast_horizon: days ahead to forecast vol (default 22 = 1 month)
    window          : lookback days (default 756 = 3 years)

    Returns
    -------
    GARCHResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import garch
    >>> data = pull.ticker("AAPL")
    >>> result = garch.fit(data, model_type="GJR-GARCH")
    >>> result.summary()
    """
    from finverse.utils.display import console

    if isinstance(data, pd.Series):
        returns = data.dropna().values
        ticker = data.name or "series"
        index = data.dropna().index
    elif hasattr(data, "price_history") and not data.price_history.empty:
        prices = data.price_history["Close"].tail(window)
        returns_s = prices.pct_change().dropna()
        returns = returns_s.values
        ticker = data.ticker
        index = returns_s.index
    else:
        raise ValueError("Provide TickerData with price_history or a pd.Series of returns")

    console.print(f"[dim]Fitting {model_type} to {ticker} ({len(returns)} obs)...[/dim]")

    var0 = np.var(returns)

    if model_type == "GARCH(1,1)":
        x0 = [var0 * 0.05, 0.08, 0.88]
        bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
        res = minimize(_garch11_loglik, x0, args=(returns,),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
        params = res.x
        h, alpha, beta, gamma = _compute_conditional_vol(params, returns, model_type)
        omega = params[0]
        persistence = alpha + beta
        long_run_var = omega / max(1 - persistence, 1e-8)
        n_params = 3

    elif model_type == "EGARCH":
        x0 = [-0.1, 0.1, -0.05, 0.95]
        bounds = [(-1, 1), (1e-6, 0.5), (-0.5, 0.5), (-0.999, 0.999)]
        res = minimize(_egarch_loglik, x0, args=(returns,),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
        params = res.x
        h, alpha, beta, gamma = _compute_conditional_vol(params, returns, model_type)
        omega = params[0]
        persistence = abs(beta)
        long_run_var = var0
        n_params = 4

    else:  # GJR-GARCH
        x0 = [var0 * 0.03, 0.04, 0.08, 0.88]
        bounds = [(1e-8, None), (1e-6, 0.4), (1e-6, 0.4), (1e-6, 0.999)]
        res = minimize(_gjr_loglik, x0, args=(returns,),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
        params = res.x
        h, alpha, beta, gamma = _compute_conditional_vol(params, returns, model_type)
        omega = params[0]
        persistence = alpha + 0.5 * gamma + beta
        long_run_var = omega / max(1 - persistence, 1e-8)
        n_params = 4

    n = len(returns)
    ll = -res.fun
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n) - 2 * ll

    h_forecast = np.zeros(forecast_horizon)
    h_last = h[-1]
    eps_last = returns[-1]

    for t in range(forecast_horizon):
        if model_type == "GARCH(1,1)":
            if t == 0:
                h_forecast[t] = omega + alpha * eps_last**2 + beta * h_last
            else:
                h_forecast[t] = omega + (alpha + beta) * h_forecast[t-1]
        elif model_type == "GJR-GARCH":
            if t == 0:
                indicator = 1.0 if eps_last < 0 else 0.0
                h_forecast[t] = omega + (alpha + gamma * indicator) * eps_last**2 + beta * h_last
            else:
                h_forecast[t] = omega + (alpha + 0.5 * gamma + beta) * h_forecast[t-1]
        else:
            h_forecast[t] = long_run_var + (persistence ** t) * (h_last - long_run_var)

    forecast_vol = np.sqrt(h_forecast * 252)
    forecast_labels = [f"Day +{i+1}" for i in range(min(5, forecast_horizon))]
    forecast_series = pd.Series(
        forecast_vol[:len(forecast_labels)],
        index=forecast_labels,
    ).round(4)

    cond_vol = pd.Series(
        np.sqrt(h * 252),
        index=index[:len(h)],
    )

    console.print(
        f"[green]✓[/green] {model_type} fitted — "
        f"persistence={persistence:.3f}, "
        f"current vol={float(cond_vol.iloc[-1]):.2%}, "
        f"long-run vol={np.sqrt(long_run_var * 252):.2%}"
    )

    return GARCHResult(
        model=model_type,
        ticker=ticker,
        omega=round(float(omega), 8),
        alpha=round(float(alpha), 6),
        beta=round(float(beta), 6),
        gamma=round(float(gamma), 6) if gamma is not None else None,
        persistence=round(float(persistence), 6),
        long_run_vol=round(float(np.sqrt(long_run_var * 252)), 4),
        current_vol=round(float(cond_vol.iloc[-1]), 4),
        conditional_vol=cond_vol,
        forecast=forecast_series,
        log_likelihood=round(float(ll), 4),
        aic=round(float(aic), 4),
        bic=round(float(bic), 4),
    )


def compare(data, horizon: int = 22) -> pd.DataFrame:
    """
    Fit all three GARCH models and compare by AIC/BIC.

    Returns
    -------
    pd.DataFrame with model comparison

    Example
    -------
    >>> table = garch.compare(data)
    >>> print(table)
    """
    from finverse.utils.display import console

    results = []
    for m in ["GARCH(1,1)", "EGARCH", "GJR-GARCH"]:
        try:
            r = fit(data, model_type=m, forecast_horizon=horizon)
            results.append({
                "model": m,
                "persistence": r.persistence,
                "current_vol": r.current_vol,
                "long_run_vol": r.long_run_vol,
                "log_likelihood": r.log_likelihood,
                "aic": r.aic,
                "bic": r.bic,
            })
        except Exception as e:
            console.print(f"[yellow]Warning: {m} failed: {e}[/yellow]")

    df = pd.DataFrame(results).set_index("model")
    best_aic = df["aic"].idxmin()
    console.print(f"\n[green]Best model by AIC:[/green] [bold]{best_aic}[/bold]")
    return df.round(4)
