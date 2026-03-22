"""
finverse.backtest.engine — backtest a signal-based strategy on historical data.

Tests whether a given signal (DCF upside, factor score, anomaly flag)
would have generated alpha historically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    benchmark_return: float | None

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Backtest — {self.strategy_name}[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Metric")
        table.add_column("Strategy", justify="right")
        if self.benchmark_return:
            table.add_column("Benchmark", justify="right")

        def color_val(v, metric="return"):
            c = "green" if v > 0 else "red"
            if metric == "sharpe":
                c = "green" if v > 1 else ("yellow" if v > 0.5 else "red")
            if metric == "drawdown":
                c = "green" if v > -0.10 else ("yellow" if v > -0.25 else "red")
            return c

        metrics = [
            ("Total return", f"{self.total_return:.1%}", "return"),
            ("Annualised return", f"{self.annualized_return:.1%}", "return"),
            ("Annualised vol", f"{self.annualized_vol:.1%}", "neutral"),
            ("Sharpe ratio", f"{self.sharpe_ratio:.2f}", "sharpe"),
            ("Max drawdown", f"{self.max_drawdown:.1%}", "drawdown"),
            ("Calmar ratio", f"{self.calmar_ratio:.2f}", "sharpe"),
            ("Win rate", f"{self.win_rate:.1%}", "return"),
            ("# Trades", str(self.n_trades), "neutral"),
        ]

        for label, val, mtype in metrics:
            if mtype == "neutral":
                row = [label, val]
            else:
                c = color_val(
                    self.total_return if "return" in label.lower() else
                    self.sharpe_ratio if "sharpe" in label.lower() or "calmar" in label.lower() else
                    self.max_drawdown,
                    mtype
                )
                row = [label, f"[{c}]{val}[/{c}]"]

            if self.benchmark_return and label == "Total return":
                row.append(f"{self.benchmark_return:.1%}")
            elif self.benchmark_return:
                row.append("—")

            table.add_row(*row)

        console.print(table)
        console.print()

    def plot(self):
        """Plot equity curve."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

            axes[0].plot(self.equity_curve.index, self.equity_curve.values,
                        color="#185FA5", linewidth=1.5, label=self.strategy_name)
            axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            axes[0].set_ylabel("Portfolio value (normalised)")
            axes[0].set_title(f"Backtest: {self.strategy_name}", fontsize=13)
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            drawdown = self.equity_curve / self.equity_curve.cummax() - 1
            axes[1].fill_between(drawdown.index, drawdown.values, 0,
                                color="#E24B4A", alpha=0.5, label="Drawdown")
            axes[1].set_ylabel("Drawdown")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            from finverse.utils.display import console
            console.print("[yellow]matplotlib required for plot: pip install matplotlib[/yellow]")


def run(
    signal: pd.Series,
    prices: pd.Series,
    strategy_name: str = "Custom strategy",
    holding_period: int = 21,
    top_n: int = 1,
    transaction_cost: float = 0.001,
    benchmark_prices: pd.Series | None = None,
) -> BacktestResult:
    """
    Backtest a signal against historical prices.

    Parameters
    ----------
    signal           : pd.Series — signal values indexed by date (higher = more bullish)
    prices           : pd.Series — price history for one stock (or portfolio)
    strategy_name    : str
    holding_period   : int — days to hold after signal (default 21 = 1 month)
    top_n            : int — if signal is cross-sectional, take top N (default 1)
    transaction_cost : float — round-trip cost per trade (default 10bps)
    benchmark_prices : pd.Series — optional benchmark (e.g. SPY)

    Returns
    -------
    BacktestResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.backtest import engine
    >>> data = pull.ticker("AAPL")
    >>> prices = data.price_history["Close"]
    >>> # Signal: momentum (12-month return)
    >>> signal = prices.pct_change(252).shift(1)
    >>> result = engine.run(signal, prices, "Momentum strategy")
    >>> result.summary()
    """
    from finverse.utils.display import console

    console.print(f"[dim]Running backtest: {strategy_name}...[/dim]")

    prices = prices.dropna().sort_index()
    returns = prices.pct_change().dropna()

    if signal is None or signal.empty:
        signal = returns.rolling(21).mean().shift(1)

    common = returns.index.intersection(signal.dropna().index)
    if len(common) < 60:
        raise ValueError("Need at least 60 overlapping observations between signal and prices.")

    returns_aligned = returns.loc[common]
    signal_aligned = signal.loc[common]

    positions = (signal_aligned > signal_aligned.median()).astype(float)
    positions = positions.shift(1).fillna(0)

    n_trades = int((positions.diff().abs() > 0).sum())
    trade_costs = pd.Series(
        np.where(positions.diff().abs() > 0, transaction_cost, 0),
        index=positions.index,
    )

    strategy_returns = positions * returns_aligned - trade_costs
    equity_curve = (1 + strategy_returns).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1)
    n_years = len(returns_aligned) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1
    ann_vol = float(strategy_returns.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    win_rate = float((strategy_returns > 0).mean())

    trades_mask = positions.diff().abs() > 0
    trade_dates = positions.index[trades_mask]
    trade_df = pd.DataFrame({
        "date": trade_dates,
        "action": ["buy" if positions.loc[d] > 0 else "sell" for d in trade_dates],
        "signal_value": signal_aligned.loc[trade_dates].values,
    })

    bench_return = None
    if benchmark_prices is not None:
        bench_prices = benchmark_prices.loc[common]
        bench_return = float(bench_prices.iloc[-1] / bench_prices.iloc[0] - 1)

    console.print(
        f"[green]✓[/green] Backtest complete — "
        f"Ann. return: {ann_return:.1%}  Sharpe: {sharpe:.2f}  "
        f"Max DD: {max_dd:.1%}  Trades: {n_trades}"
    )

    return BacktestResult(
        strategy_name=strategy_name,
        total_return=round(total_return, 4),
        annualized_return=round(ann_return, 4),
        annualized_vol=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown=round(max_dd, 4),
        calmar_ratio=round(calmar, 3),
        win_rate=round(win_rate, 4),
        n_trades=n_trades,
        equity_curve=equity_curve,
        trades=trade_df,
        benchmark_return=bench_return,
    )


def momentum(data, lookback: int = 252, holding: int = 21) -> BacktestResult:
    """
    Backtest a simple momentum strategy on a stock.

    Parameters
    ----------
    data     : TickerData
    lookback : int — signal lookback in days (default 252)
    holding  : int — holding period in days (default 21)

    Example
    -------
    >>> result = backtest.momentum(data)
    >>> result.summary()
    """
    prices = data.price_history["Close"]
    signal = prices.pct_change(lookback).shift(1)
    return run(signal, prices, f"Momentum ({lookback}d)", holding_period=holding)


def dcf_signal(model, data, rebalance_months: int = 3) -> BacktestResult:
    """
    Backtest a DCF-upside driven strategy.

    Buys when DCF implies >15% upside, sells otherwise.
    Uses historical price to simulate rebalancing.

    Parameters
    ----------
    model           : DCF model
    data            : TickerData
    rebalance_months: how often to re-run DCF and rebalance

    Example
    -------
    >>> result = backtest.dcf_signal(model, data)
    >>> result.summary()
    """
    prices = data.price_history["Close"]
    if model._results is None:
        model.run()

    implied = model._results.implied_price
    signal = pd.Series(
        np.where(prices < implied * 0.85, 1.0, -0.5),
        index=prices.index,
    )
    return run(signal, prices, "DCF upside signal")
