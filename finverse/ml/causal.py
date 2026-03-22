"""
finverse.ml.causal — find which macro variables actually drive a company's
earnings using Granger causality tests and structural analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CausalResult:
    ticker: str
    drivers: list[dict]             # ranked list of causal drivers
    top_driver: str
    explained_variance: float
    model_summary: pd.DataFrame

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Causal Drivers — {self.ticker}[/bold blue]")
        console.print(f"[dim]Top driver: {self.top_driver}  |  Explained variance: {self.explained_variance:.0%}[/dim]\n")

        table = Table(
            title="Macro → Earnings causal links",
            box=box.SIMPLE_HEAD,
            header_style="bold blue",
        )
        table.add_column("Macro variable")
        table.add_column("Causal strength", justify="right")
        table.add_column("Direction")
        table.add_column("Lag (quarters)")

        for d in self.drivers[:8]:
            strength = d.get("strength", 0)
            bar = "█" * int(strength * 15)
            direction = "[green]+[/green]" if d.get("direction", 1) > 0 else "[red]−[/red]"
            table.add_row(
                d.get("variable", ""),
                f"{bar} {strength:.2f}",
                direction,
                str(d.get("lag", 1)),
            )
        console.print(table)
        console.print()


def _granger_test(y: np.ndarray, x: np.ndarray, max_lag: int = 4) -> dict:
    """
    Simplified Granger causality: test if lagged x improves forecast of y.
    Returns F-statistic proxy and best lag.
    """
    from sklearn.linear_model import LinearRegression

    n = min(len(y), len(x))
    y = y[-n:]
    x = x[-n:]

    best_score = 0.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        if n - lag < 4:
            continue

        y_target = y[lag:]

        X_restricted = y[:-lag].reshape(-1, 1)
        X_unrestricted = np.column_stack([y[:-lag], x[:-lag]])

        if len(y_target) < 4:
            continue

        reg_r = LinearRegression().fit(X_restricted, y_target)
        reg_u = LinearRegression().fit(X_unrestricted, y_target)

        ss_r = np.sum((y_target - reg_r.predict(X_restricted)) ** 2)
        ss_u = np.sum((y_target - reg_u.predict(X_unrestricted)) ** 2)

        if ss_r > 0:
            f_stat = (ss_r - ss_u) / ss_u * (len(y_target) - 3)
            score = min(float(f_stat) / 10.0, 1.0)
            if score > best_score:
                best_score = score
                best_lag = lag

    direction_corr = np.corrcoef(x[:-1], y[1:])[0, 1] if len(x) > 1 else 0
    return {
        "strength": round(max(best_score, 0), 3),
        "lag": best_lag,
        "direction": 1 if direction_corr >= 0 else -1,
    }


def analyze(
    data,
    macro_df: pd.DataFrame | None = None,
    target: str = "revenue",
    max_lag: int = 4,
) -> CausalResult:
    """
    Find which macro variables causally drive a company's financials.

    Uses Granger causality tests on quarterly data. Identifies:
    - Which macro variable has the strongest causal link
    - The direction (positive/negative)
    - The lag (how many quarters it takes to show up)

    Parameters
    ----------
    data     : TickerData
    macro_df : pd.DataFrame from pull.fred() — optional
               If None, uses synthetic macro data for demonstration
    target   : "revenue" or "earnings" (default "revenue")
    max_lag  : max quarters to test (default 4)

    Returns
    -------
    CausalResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import causal
    >>> import os
    >>> os.environ["FRED_API_KEY"] = "your_key"
    >>> data = pull.ticker("AAPL")
    >>> macro = pull.fred("GDP", "FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE")
    >>> result = causal.analyze(data, macro_df=macro)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Running causal analysis for {ticker} (target: {target})...[/dim]")

    if target == "revenue":
        target_series = data.revenue_history if hasattr(data, "revenue_history") else pd.Series()
    else:
        target_series = data.net_income_history if hasattr(data, "net_income_history") else pd.Series()

    if target_series.empty or len(target_series) < 4:
        console.print("[yellow]Warning: insufficient historical data, using synthetic data[/yellow]")
        np.random.seed(42)
        target_series = pd.Series(
            np.cumsum(np.random.normal(0.02, 0.05, 20)) + 100,
            index=pd.date_range("2019", periods=20, freq="QE"),
        )

    MACRO_DESCRIPTIONS = {
        "GDP": "US GDP growth",
        "FEDFUNDS": "Federal funds rate",
        "DGS10": "10Y treasury yield",
        "CPIAUCSL": "Consumer price inflation",
        "UNRATE": "Unemployment rate",
        "T10Y2Y": "Yield curve slope",
        "VIXCLS": "Market volatility (VIX)",
        "BAMLH0A0HYM2": "High yield credit spread",
        "DCOILWTICO": "WTI crude oil price",
        "DEXUSEU": "USD/EUR exchange rate",
    }

    if macro_df is None or macro_df.empty:
        np.random.seed(42)
        n = len(target_series) + 8
        macro_df = pd.DataFrame({
            "GDP":       np.cumsum(np.random.normal(0.005, 0.01, n)) + 2.0,
            "FEDFUNDS":  np.random.normal(3.5, 1.5, n).clip(0, 8),
            "CPIAUCSL":  np.random.normal(2.5, 1.0, n).clip(0, 10),
            "UNRATE":    np.random.normal(4.5, 1.0, n).clip(2, 10),
            "VIXCLS":    np.random.normal(18, 8, n).clip(9, 80),
            "T10Y2Y":    np.random.normal(0.5, 0.8, n),
        }, index=pd.date_range("2017", periods=n, freq="QE"))

    y = target_series.pct_change().dropna().values

    results = []
    for col in macro_df.columns:
        macro_series = macro_df[col].resample("QE").last().dropna()
        macro_pct = macro_series.pct_change().dropna()

        aligned = macro_pct.reindex(
            pd.date_range(target_series.index[0], target_series.index[-1], freq="QE"),
            method="nearest",
        ).fillna(0).values

        n_common = min(len(y), len(aligned))
        if n_common < 4:
            continue

        result = _granger_test(y[:n_common], aligned[:n_common], max_lag)
        result["variable"] = MACRO_DESCRIPTIONS.get(col, col)
        result["series_id"] = col
        results.append(result)

    results.sort(key=lambda x: -x["strength"])

    top_driver = results[0]["variable"] if results else "Unknown"
    explained_var = min(sum(r["strength"] for r in results[:3]) / 3, 0.85)

    summary_df = pd.DataFrame(results)[["variable", "strength", "direction", "lag"]] if results else pd.DataFrame()

    console.print(
        f"[green]✓[/green] Top causal driver: [bold]{top_driver}[/bold]  |  "
        f"Explained variance: {explained_var:.0%}"
    )

    return CausalResult(
        ticker=ticker,
        drivers=results,
        top_driver=top_driver,
        explained_variance=round(explained_var, 3),
        model_summary=summary_df,
    )
