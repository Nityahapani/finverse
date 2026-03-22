"""
finverse.ml.cross_sectional — train forecasting models on a universe of
companies simultaneously rather than per-company.

Cross-sectional approach: for each company, use its own features PLUS
features from similar companies to improve generalization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CrossSectionalForecast:
    target: str                        # "revenue_growth", "ebitda_margin", etc.
    ticker: str
    forecast: float
    confidence_interval: tuple[float, float]
    percentile_rank: float             # where this company ranks in universe
    universe_mean: float
    universe_std: float
    feature_importance: dict[str, float]
    n_companies_trained: int
    model_name: str = "XGBoost (cross-sectional)"

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Cross-Sectional Forecast — {self.ticker}[/bold blue]")
        console.print(f"[dim]Target: {self.target} | Trained on {self.n_companies_trained} companies[/dim]\n")

        lo, hi = self.confidence_interval
        console.print(f"  Forecast:    [bold]{self.forecast:.2%}[/bold]")
        console.print(f"  80% CI:      {lo:.2%} – {hi:.2%}")
        console.print(f"  Universe avg: {self.universe_mean:.2%}")
        console.print(f"  Percentile rank: {self.percentile_rank:.0%} of universe")

        if self.feature_importance:
            console.print(f"\n  [dim]Top drivers:[/dim]")
            for feat, imp in list(self.feature_importance.items())[:5]:
                bar = "█" * int(imp * 20)
                console.print(f"    {feat:<28} {bar} {imp:.2%}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "target": self.target,
            "forecast": self.forecast,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "percentile_rank": self.percentile_rank,
        }])


def _build_universe_features(
    ticker_data_list: list,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix from a list of TickerData objects."""
    rows = []
    targets = []

    for d in ticker_data_list:
        try:
            row = {}
            ticker = d.ticker if hasattr(d, "ticker") else "unknown"

            rev = d.revenue_history if hasattr(d, "revenue_history") else pd.Series()
            ebitda = d.ebitda_history if hasattr(d, "ebitda_history") else pd.Series()
            fcf = d.fcf_history if hasattr(d, "fcf_history") else pd.Series()
            ni = d.net_income_history if hasattr(d, "net_income_history") else pd.Series()

            if not rev.empty and len(rev) >= 3:
                rev_growth = rev.pct_change().dropna()
                row["rev_growth_1y"] = float(rev_growth.iloc[-1])
                row["rev_growth_3y_avg"] = float(rev_growth.mean())
                row["rev_growth_std"] = float(rev_growth.std())
                row["rev_acceleration"] = float(rev_growth.diff().iloc[-1]) if len(rev_growth) > 1 else 0
                row["rev_level"] = float(np.log1p(rev.iloc[-1]))

            if not ebitda.empty and not rev.empty:
                common = ebitda.index.intersection(rev.index)
                if len(common) >= 2:
                    margin = ebitda.loc[common] / rev.loc[common]
                    row["ebitda_margin_latest"] = float(margin.iloc[-1])
                    row["ebitda_margin_avg"] = float(margin.mean())
                    row["margin_trend"] = float(margin.diff().mean())

            if not fcf.empty and not rev.empty:
                common = fcf.index.intersection(rev.index)
                if len(common) >= 2:
                    fcf_margin = fcf.loc[common] / rev.loc[common]
                    row["fcf_margin"] = float(fcf_margin.iloc[-1])

            if not ni.empty and not rev.empty:
                common = ni.index.intersection(rev.index)
                if len(common) >= 1:
                    net_margin = ni.loc[common] / rev.loc[common]
                    row["net_margin"] = float(net_margin.iloc[-1])

            if hasattr(d, "info") and d.info:
                info = d.info
                row["beta"] = info.get("beta", 1.0) or 1.0
                row["pe_ratio"] = min(info.get("trailingPE", 20) or 20, 100)
                row["ev_ebitda"] = min(info.get("enterpriseToEbitda", 15) or 15, 80)
                row["debt_to_eq"] = min(info.get("debtToEquity", 0.5) or 0.5, 10)

            if not row:
                continue

            if target == "revenue_growth" and "rev_growth_1y" in row:
                target_val = row.pop("rev_growth_1y")
            elif target == "ebitda_margin" and "ebitda_margin_latest" in row:
                target_val = row.pop("ebitda_margin_latest")
            else:
                target_val = row.get("rev_growth_3y_avg", 0.05)

            rows.append({"ticker": ticker, **row})
            targets.append(target_val)

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = pd.DataFrame(rows).set_index("ticker")
    target_series = pd.Series(targets, index=df.index, name=target)
    return df, target_series


def _synthetic_universe_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic cross-sectional data for demo/fallback."""
    np.random.seed(seed)
    sectors = ["tech", "finance", "healthcare", "energy", "consumer"]

    rows = []
    for i in range(n):
        sector = sectors[i % len(sectors)]
        base_growth = {"tech": 0.12, "finance": 0.05, "healthcare": 0.08,
                       "energy": 0.03, "consumer": 0.06}[sector]
        base_margin = {"tech": 0.28, "finance": 0.35, "healthcare": 0.22,
                       "energy": 0.18, "consumer": 0.12}[sector]

        rows.append({
            "ticker": f"CO{i:03d}",
            "rev_growth_3y_avg": np.random.normal(base_growth, 0.04),
            "rev_growth_std": np.random.uniform(0.01, 0.08),
            "rev_acceleration": np.random.normal(0, 0.02),
            "rev_level": np.random.uniform(3, 10),
            "ebitda_margin_avg": np.random.normal(base_margin, 0.05),
            "margin_trend": np.random.normal(0.005, 0.01),
            "fcf_margin": np.random.normal(base_margin * 0.8, 0.04),
            "net_margin": np.random.normal(base_margin * 0.6, 0.04),
            "beta": np.random.uniform(0.5, 1.8),
            "pe_ratio": np.random.uniform(10, 45),
            "ev_ebitda": np.random.uniform(6, 30),
            "debt_to_eq": np.random.uniform(0, 2),
            "sector_tech": 1 if sector == "tech" else 0,
            "sector_finance": 1 if sector == "finance" else 0,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    return df


def forecast(
    data,
    universe: list | None = None,
    target: str = "revenue_growth",
    n_estimators: int = 200,
) -> CrossSectionalForecast:
    """
    Forecast a target metric using cross-sectional ML.

    Trains on all companies in the universe simultaneously, then
    predicts for the target company. This leverages information from
    similar companies rather than only the company's own history.

    Parameters
    ----------
    data        : TickerData — the company to forecast for
    universe    : list of TickerData — training universe (uses synthetic if None)
    target      : "revenue_growth" or "ebitda_margin" (default "revenue_growth")
    n_estimators: XGBoost trees (default 200)

    Returns
    -------
    CrossSectionalForecast

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import cross_sectional
    >>> apple = pull.ticker("AAPL")
    >>> result = cross_sectional.forecast(apple, target="revenue_growth")
    >>> result.summary()
    """
    from finverse.utils.display import console
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Cross-sectional {target} forecast for {ticker}...[/dim]")

    if universe and len(universe) >= 10:
        X_train, y_train = _build_universe_features(universe, target)
    else:
        X_train = _synthetic_universe_data(n=80)
        np.random.seed(42)
        if target == "revenue_growth":
            y_train = pd.Series(
                X_train["rev_growth_3y_avg"] + np.random.normal(0, 0.02, len(X_train)),
                index=X_train.index,
            )
        else:
            y_train = pd.Series(
                X_train["ebitda_margin_avg"] + np.random.normal(0, 0.02, len(X_train)),
                index=X_train.index,
            )

    X_target, _ = _build_universe_features([data], target)

    if X_target.empty:
        rev = data.revenue_history if hasattr(data, "revenue_history") else pd.Series()
        fallback_val = float(rev.pct_change().mean()) if len(rev) > 1 else 0.08
        console.print(f"[yellow]Warning: insufficient data — using fallback {fallback_val:.1%}[/yellow]")
        return CrossSectionalForecast(
            target=target, ticker=ticker,
            forecast=fallback_val,
            confidence_interval=(fallback_val - 0.03, fallback_val + 0.03),
            percentile_rank=0.5, universe_mean=0.08, universe_std=0.04,
            feature_importance={}, n_companies_trained=len(X_train),
        )

    common_cols = X_train.columns.intersection(X_target.columns)
    if len(common_cols) < 2:
        common_cols = X_train.columns[:min(5, len(X_train.columns))]

    X_tr = X_train[common_cols].copy()
    X_te = X_target.reindex(columns=common_cols).copy()

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    np.random.seed(42)
    bootstrap_preds = []
    for seed in range(30):
        idx = np.random.choice(len(X_tr_scaled), size=len(X_tr_scaled), replace=True)
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=seed,
        )
        gb.fit(X_tr_scaled[idx], y_train.values[idx])
        bootstrap_preds.append(float(gb.predict(X_te_scaled)[0]))

    point_forecast = float(np.median(bootstrap_preds))
    ci_lower = float(np.percentile(bootstrap_preds, 10))
    ci_upper = float(np.percentile(bootstrap_preds, 90))

    final_model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    final_model.fit(X_tr_scaled, y_train.values)
    importances = dict(zip(common_cols, final_model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    universe_mean = float(y_train.mean())
    universe_std = float(y_train.std())
    percentile_rank = float((y_train < point_forecast).mean())

    n_trained = len(X_train)
    console.print(
        f"[green]✓[/green] Cross-sectional forecast: {point_forecast:.1%} "
        f"(CI: {ci_lower:.1%}–{ci_upper:.1%}) | "
        f"Universe avg: {universe_mean:.1%} | "
        f"Rank: {percentile_rank:.0%}"
    )

    return CrossSectionalForecast(
        target=target,
        ticker=ticker,
        forecast=round(point_forecast, 4),
        confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
        percentile_rank=round(percentile_rank, 3),
        universe_mean=round(universe_mean, 4),
        universe_std=round(universe_std, 4),
        feature_importance={k: round(v, 4) for k, v in list(importances.items())[:8]},
        n_companies_trained=n_trained,
    )


def rank_universe(
    universe: list,
    target: str = "revenue_growth",
) -> pd.DataFrame:
    """
    Rank all companies in a universe by cross-sectional ML forecast.

    Parameters
    ----------
    universe : list of TickerData
    target   : metric to forecast and rank by

    Returns
    -------
    pd.DataFrame ranked by forecast

    Example
    -------
    >>> tickers = ["AAPL", "MSFT", "GOOGL"]
    >>> data = [pull.ticker(t) for t in tickers]
    >>> ranking = cross_sectional.rank_universe(data)
    >>> print(ranking)
    """
    from finverse.utils.display import console

    console.print(f"[dim]Ranking {len(universe)} companies by {target}...[/dim]")
    rows = []
    for d in universe:
        try:
            r = forecast(d, universe=universe, target=target)
            rows.append({
                "ticker": r.ticker,
                "forecast": r.forecast,
                "ci_lower": r.confidence_interval[0],
                "ci_upper": r.confidence_interval[1],
                "percentile_rank": r.percentile_rank,
            })
        except Exception as e:
            console.print(f"[yellow]Skipping {getattr(d, 'ticker', '?')}: {e}[/yellow]")

    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values("forecast", ascending=False).round(4)
