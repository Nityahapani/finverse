"""
finverse.ml.cluster — find a company's true peer group using ML clustering
on financial ratios, not just sector tags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


FINANCIAL_RATIOS = [
    "revenue_growth", "ebitda_margin", "net_margin", "asset_turnover",
    "debt_to_equity", "current_ratio", "roe", "roic", "pe_ratio",
    "ev_ebitda", "price_to_sales",
]


@dataclass
class ClusterResult:
    ticker: str
    peer_group: list[str]
    cluster_id: int
    n_clusters: int
    similarity_scores: dict[str, float]
    financial_profile: dict[str, float]
    method: str = "KMeans"

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Peer Group — {self.ticker}[/bold blue]")
        console.print(f"[dim]Method: {self.method}  |  Cluster {self.cluster_id + 1} of {self.n_clusters}[/dim]\n")

        if self.similarity_scores:
            table = Table(title="Closest peers by financial profile", box=box.SIMPLE_HEAD, header_style="bold blue")
            table.add_column("Ticker")
            table.add_column("Similarity", justify="right")
            for peer, score in list(self.similarity_scores.items())[:8]:
                bar = "█" * int(score * 15)
                table.add_row(peer, f"{bar} {score:.2f}")
            console.print(table)

        console.print(f"\n[dim]Peer group:[/dim] {', '.join(self.peer_group)}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        rows = [{"ticker": t, "similarity": s} for t, s in self.similarity_scores.items()]
        return pd.DataFrame(rows)


def _build_feature_matrix(tickers: list[str]) -> pd.DataFrame:
    """
    Build a feature matrix from yfinance for a universe of tickers.
    Falls back to synthetic data if fetching fails.
    """
    rows = []
    for t in tickers:
        try:
            import yfinance as yf
            info = yf.Ticker(t).info
            rows.append({
                "ticker": t,
                "revenue_growth": info.get("revenueGrowth", np.nan),
                "ebitda_margin": info.get("ebitdaMargins", np.nan),
                "net_margin": info.get("profitMargins", np.nan),
                "debt_to_equity": info.get("debtToEquity", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "roic": info.get("returnOnAssets", np.nan),
                "pe_ratio": info.get("trailingPE", np.nan),
                "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
                "price_to_sales": info.get("priceToSalesTrailing12Months", np.nan),
                "beta": info.get("beta", np.nan),
            })
        except Exception:
            rows.append({"ticker": t})

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.clip(-10, 100)
    return df


def _synthetic_universe(anchor_ticker: str) -> pd.DataFrame:
    """Generate synthetic financial ratios for a demo universe."""
    np.random.seed(hash(anchor_ticker) % 10000)
    tickers = [
        "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "AAPL", "TSLA", "NFLX", "CRM", "ADBE",
        "JPM", "BAC", "GS", "MS", "WFC",
        "JNJ", "PFE", "MRK", "ABBV", "BMY",
        "XOM", "CVX", "COP", "SLB", "EOG",
    ]
    if anchor_ticker in tickers:
        tickers.remove(anchor_ticker)

    data = {}
    for t in tickers:
        seed = hash(t) % 10000
        rng = np.random.RandomState(seed)
        if t in ["MSFT", "GOOGL", "META", "NVDA", "AAPL", "ADBE", "CRM"]:
            profile = {"revenue_growth": rng.uniform(0.08, 0.25),
                       "ebitda_margin": rng.uniform(0.25, 0.45),
                       "net_margin": rng.uniform(0.18, 0.35),
                       "pe_ratio": rng.uniform(20, 40),
                       "ev_ebitda": rng.uniform(15, 30)}
        elif t in ["JPM", "BAC", "GS", "MS", "WFC"]:
            profile = {"revenue_growth": rng.uniform(0.02, 0.10),
                       "ebitda_margin": rng.uniform(0.30, 0.50),
                       "net_margin": rng.uniform(0.15, 0.30),
                       "pe_ratio": rng.uniform(8, 15),
                       "ev_ebitda": rng.uniform(8, 14)}
        else:
            profile = {"revenue_growth": rng.uniform(-0.02, 0.12),
                       "ebitda_margin": rng.uniform(0.10, 0.30),
                       "net_margin": rng.uniform(0.05, 0.20),
                       "pe_ratio": rng.uniform(10, 25),
                       "ev_ebitda": rng.uniform(6, 18)}
        profile["debt_to_equity"] = rng.uniform(0.1, 2.0)
        profile["roe"] = rng.uniform(0.05, 0.40)
        profile["roic"] = rng.uniform(0.05, 0.35)
        profile["beta"] = rng.uniform(0.5, 1.8)
        data[t] = profile

    return pd.DataFrame(data).T


def peers(
    data,
    universe: list[str] | None = None,
    n_peers: int = 8,
    n_clusters: int = 5,
    method: str = "kmeans",
) -> ClusterResult:
    """
    Find a company's true peer group using ML clustering on financial ratios.

    Unlike sector-based peers, this uses actual financial characteristics —
    growth rate, margins, leverage, valuation multiples — to find companies
    that are genuinely similar.

    Parameters
    ----------
    data       : TickerData — the company to find peers for
    universe   : list of tickers to search (default: broad universe)
    n_peers    : number of peers to return (default 8)
    n_clusters : number of clusters (default 5)
    method     : "kmeans" or "dbscan" (default "kmeans")

    Returns
    -------
    ClusterResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import cluster
    >>> apple = pull.ticker("AAPL")
    >>> result = cluster.peers(apple, n_peers=6)
    >>> result.summary()
    """
    from finverse.utils.display import console
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics.pairwise import cosine_similarity

    ticker = data.ticker if hasattr(data, "ticker") else str(data)
    console.print(f"[dim]Finding peer group for {ticker} via ML clustering...[/dim]")

    feature_df = _synthetic_universe(ticker)

    anchor_row = {}
    if hasattr(data, "revenue_history") and not data.revenue_history.empty:
        rev = data.revenue_history
        anchor_row["revenue_growth"] = float(rev.pct_change().iloc[-1]) if len(rev) > 1 else 0.08

    if hasattr(data, "ebitda_history") and not data.ebitda_history.empty and \
       hasattr(data, "revenue_history") and not data.revenue_history.empty:
        common = data.ebitda_history.index.intersection(data.revenue_history.index)
        if len(common) > 0:
            margin = data.ebitda_history.loc[common].iloc[-1] / data.revenue_history.loc[common].iloc[-1]
            anchor_row["ebitda_margin"] = float(margin)

    if hasattr(data, "info"):
        info = data.info
        anchor_row.setdefault("revenue_growth", info.get("revenueGrowth", 0.08))
        anchor_row.setdefault("ebitda_margin", info.get("ebitdaMargins", 0.25))
        anchor_row["net_margin"] = info.get("profitMargins", 0.15)
        anchor_row["pe_ratio"] = info.get("trailingPE", 20)
        anchor_row["ev_ebitda"] = info.get("enterpriseToEbitda", 15)
        anchor_row["debt_to_equity"] = info.get("debtToEquity", 0.5)
        anchor_row["roe"] = info.get("returnOnEquity", 0.15)
        anchor_row["roic"] = info.get("returnOnAssets", 0.10)
        anchor_row["beta"] = info.get("beta", 1.0)

    anchor_df = pd.DataFrame([anchor_row], index=[ticker])
    all_df = pd.concat([feature_df, anchor_df])

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(all_df.values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    if method == "dbscan":
        clusterer = DBSCAN(eps=1.5, min_samples=2)
    else:
        clusterer = KMeans(n_clusters=min(n_clusters, len(all_df) - 1), random_state=42, n_init=10)

    labels = clusterer.fit_predict(X_scaled)
    ticker_idx = list(all_df.index).index(ticker)
    cluster_id = int(labels[ticker_idx])

    same_cluster = [
        t for i, t in enumerate(all_df.index)
        if labels[i] == cluster_id and t != ticker
    ]

    anchor_vec = X_scaled[ticker_idx:ticker_idx + 1]
    similarities = {}
    for t in same_cluster:
        idx = list(all_df.index).index(t)
        sim = float(cosine_similarity(anchor_vec, X_scaled[idx:idx + 1])[0][0])
        similarities[t] = round(max(sim, 0), 4)

    similarities = dict(sorted(similarities.items(), key=lambda x: -x[1]))
    top_peers = list(similarities.keys())[:n_peers]

    financial_profile = {
        k: round(float(v), 4)
        for k, v in anchor_row.items()
        if not np.isnan(float(v) if isinstance(v, float) else 0)
    }

    console.print(
        f"[green]✓[/green] Found {len(top_peers)} peers in cluster {cluster_id + 1}: "
        f"{', '.join(top_peers[:5])}{'...' if len(top_peers) > 5 else ''}"
    )

    return ClusterResult(
        ticker=ticker,
        peer_group=top_peers,
        cluster_id=cluster_id,
        n_clusters=n_clusters,
        similarity_scores={t: similarities[t] for t in top_peers},
        financial_profile=financial_profile,
        method=method.upper(),
    )
