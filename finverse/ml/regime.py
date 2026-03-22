"""
finverse.ml.regime — market regime detection using Hidden Markov Models.

Classifies market into: expansion, contraction, recovery, stress.
Adjusts DCF discount rates based on current regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class Regime(str, Enum):
    EXPANSION   = "expansion"
    CONTRACTION = "contraction"
    RECOVERY    = "recovery"
    STRESS      = "stress"


REGIME_DESCRIPTIONS = {
    Regime.EXPANSION:   "Low vol, positive trend, healthy breadth",
    Regime.CONTRACTION: "Rising vol, negative trend, deteriorating breadth",
    Regime.RECOVERY:    "Declining vol from highs, early positive trend",
    Regime.STRESS:      "Spike vol, sharp drawdowns, liquidity premium",
}

REGIME_WACC_ADJUSTMENTS = {
    Regime.EXPANSION:   -0.005,
    Regime.RECOVERY:     0.000,
    Regime.CONTRACTION:  0.010,
    Regime.STRESS:       0.025,
}


@dataclass
class RegimeResult:
    current_regime: Regime
    regime_history: pd.Series
    transition_matrix: pd.DataFrame
    regime_stats: pd.DataFrame
    wacc_adjustment: float
    confidence: float

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        color_map = {
            Regime.EXPANSION:   "green",
            Regime.RECOVERY:    "blue",
            Regime.CONTRACTION: "yellow",
            Regime.STRESS:      "red",
        }
        c = color_map.get(self.current_regime, "white")

        console.print(f"\n[bold]Current market regime:[/bold] [{c}][bold]{self.current_regime.value.upper()}[/bold][/{c}]")
        console.print(f"[dim]{REGIME_DESCRIPTIONS[self.current_regime]}[/dim]")
        console.print(f"[dim]Confidence: {self.confidence:.0%}  |  WACC adjustment: {self.wacc_adjustment:+.1%}[/dim]\n")

        table = Table(title="Regime Statistics", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Regime")
        table.add_column("Avg return (ann.)", justify="right")
        table.add_column("Volatility (ann.)", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("% time", justify="right")

        for _, row in self.regime_stats.iterrows():
            table.add_row(
                str(row.name),
                f"{row.get('avg_return', 0):.1%}",
                f"{row.get('volatility', 0):.1%}",
                f"{row.get('sharpe', 0):.2f}",
                f"{row.get('pct_time', 0):.0%}",
            )
        console.print(table)
        console.print()

    def adjust_wacc(self, base_wacc: float) -> float:
        """Return WACC adjusted for current regime."""
        return round(base_wacc + self.wacc_adjustment, 4)


def detect(
    price_data: pd.Series | None = None,
    macro_data: pd.DataFrame | None = None,
    n_regimes: int = 4,
) -> RegimeResult:
    """
    Detect the current market regime using a Hidden Markov Model.

    Uses return, volatility, and momentum signals. Optionally incorporates
    macro data (yield curve, credit spreads, VIX) for richer regime signals.

    Parameters
    ----------
    price_data  : pd.Series of prices (e.g. SPY close) or None for synthetic
    macro_data  : pd.DataFrame from pull.fred() — optional
    n_regimes   : number of regimes (default 4)

    Returns
    -------
    RegimeResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import regime
    >>> spy = pull.ticker("SPY")
    >>> result = regime.detect(spy.price_history["Close"])
    >>> result.summary()
    >>> print(f"Adjusted WACC: {result.adjust_wacc(0.095):.1%}")
    """
    from finverse.utils.display import console
    from sklearn.preprocessing import StandardScaler

    console.print(f"[dim]Detecting market regime ({n_regimes} states)...[/dim]")

    if price_data is not None and len(price_data) > 60:
        returns = price_data.pct_change().dropna()
    else:
        np.random.seed(42)
        n = 1260
        returns = pd.Series(
            np.concatenate([
                np.random.normal(0.0005, 0.008, 400),
                np.random.normal(-0.001, 0.018, 200),
                np.random.normal(0.0003, 0.012, 300),
                np.random.normal(-0.003, 0.030, 100),
                np.random.normal(0.0006, 0.009, 260),
            ]),
            index=pd.date_range("2020-01-01", periods=n, freq="B")[:n],
        )

    vol_21  = returns.rolling(21).std()
    vol_63  = returns.rolling(63).std()
    mom_21  = returns.rolling(21).sum()
    mom_63  = returns.rolling(63).sum()
    drawdown = (returns.cumsum() - returns.cumsum().cummax())
    vol_ratio = (vol_21 / (vol_63 + 1e-8)).fillna(1.0)

    features_df = pd.DataFrame({
        "return":    returns,
        "vol_21":    vol_21,
        "mom_21":    mom_21,
        "mom_63":    mom_63,
        "drawdown":  drawdown,
        "vol_ratio": vol_ratio,
    }).dropna()

    if macro_data is not None and not macro_data.empty:
        for col in ["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2"]:
            if col in macro_data.columns:
                aligned = macro_data[col].reindex(features_df.index, method="ffill")
                features_df[f"macro_{col}"] = aligned.fillna(method="ffill").fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)

    try:
        from hmmlearn import hmm
        model = hmm.GaussianHMM(
            n_components=n_regimes, covariance_type="full",
            n_iter=100, random_state=42
        )
        model.fit(X)
        hidden_states = model.predict(X)
        posteriors = model.predict_proba(X)
        confidence = float(posteriors[-1].max())
    except ImportError:
        hidden_states = _kmeans_regime(X, n_regimes)
        confidence = 0.72

    regime_labels = _label_regimes(hidden_states, features_df)
    regime_series = pd.Series(
        [regime_labels[s] for s in hidden_states],
        index=features_df.index,
    )

    current_state = hidden_states[-1]
    current_regime = regime_labels[current_state]

    stats = _compute_regime_stats(returns, regime_series)
    trans_matrix = _compute_transition_matrix(hidden_states, n_regimes, regime_labels)

    wacc_adj = REGIME_WACC_ADJUSTMENTS.get(current_regime, 0.0)

    console.print(
        f"[green]✓[/green] Regime: [bold]{current_regime.value.upper()}[/bold] "
        f"(confidence {confidence:.0%}, WACC {wacc_adj:+.1%})"
    )

    return RegimeResult(
        current_regime=current_regime,
        regime_history=regime_series,
        transition_matrix=trans_matrix,
        regime_stats=stats,
        wacc_adjustment=wacc_adj,
        confidence=confidence,
    )


def _kmeans_regime(X: np.ndarray, n: int) -> np.ndarray:
    """Fallback: k-means clustering when hmmlearn not available."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    return km.fit_predict(X)


def _label_regimes(states: np.ndarray, features: pd.DataFrame) -> dict[int, Regime]:
    """Assign semantic labels to hidden states based on return/vol characteristics."""
    stats = {}
    for s in np.unique(states):
        mask = states == s
        rets = features["return"].values[mask]
        vols = features["vol_21"].values[mask]
        stats[s] = {"mean_ret": np.mean(rets), "mean_vol": np.mean(vols)}

    sorted_by_vol = sorted(stats.keys(), key=lambda s: stats[s]["mean_vol"])
    sorted_by_ret = sorted(stats.keys(), key=lambda s: stats[s]["mean_ret"], reverse=True)

    labels = {}
    n = len(sorted_by_vol)

    if n == 4:
        labels[sorted_by_vol[0]] = Regime.EXPANSION    if stats[sorted_by_vol[0]]["mean_ret"] > 0 else Regime.CONTRACTION
        labels[sorted_by_vol[1]] = Regime.RECOVERY
        labels[sorted_by_vol[2]] = Regime.CONTRACTION  if stats[sorted_by_vol[2]]["mean_ret"] < 0 else Regime.RECOVERY
        labels[sorted_by_vol[3]] = Regime.STRESS
    else:
        regime_list = [Regime.EXPANSION, Regime.RECOVERY, Regime.CONTRACTION, Regime.STRESS]
        for i, s in enumerate(sorted_by_vol):
            labels[s] = regime_list[min(i, len(regime_list) - 1)]

    for s in np.unique(states):
        if s not in labels:
            labels[s] = Regime.CONTRACTION
    return labels


def _compute_regime_stats(returns: pd.Series, regime_series: pd.Series) -> pd.DataFrame:
    rows = []
    common = returns.index.intersection(regime_series.index)
    for r in Regime:
        mask = regime_series.loc[common] == r
        if mask.sum() == 0:
            continue
        rets = returns.loc[common][mask]
        avg_ret = float(rets.mean() * 252)
        vol = float(rets.std() * np.sqrt(252))
        sharpe = avg_ret / vol if vol > 0 else 0
        pct_time = mask.sum() / len(mask)
        rows.append({
            "regime": r.value,
            "avg_return": round(avg_ret, 4),
            "volatility": round(vol, 4),
            "sharpe": round(sharpe, 3),
            "pct_time": round(pct_time, 3),
        })
    df = pd.DataFrame(rows)
    return df.set_index("regime") if "regime" in df.columns else df


def _compute_transition_matrix(
    states: np.ndarray,
    n: int,
    labels: dict[int, Regime],
) -> pd.DataFrame:
    matrix = np.zeros((n, n))
    for i in range(len(states) - 1):
        matrix[states[i], states[i + 1]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums > 0)
    regime_names = [labels.get(i, Regime.EXPANSION).value for i in range(n)]
    return pd.DataFrame(matrix, index=regime_names, columns=regime_names).round(3)
