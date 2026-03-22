"""
finverse.models.comps — Comparable Company Analysis (Trading Comps).

Auto-pulls peer multiples, builds a comps table, and derives an
implied valuation range for the target company.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


MULTIPLES = ["ev_ebitda", "ev_revenue", "pe_ratio", "price_to_sales", "price_to_book"]

MULTIPLE_LABELS = {
    "ev_ebitda":      "EV/EBITDA",
    "ev_revenue":     "EV/Revenue",
    "pe_ratio":       "P/E",
    "price_to_sales": "P/Sales",
    "price_to_book":  "P/Book",
}


@dataclass
class CompsResult:
    target_ticker: str
    comps_table: pd.DataFrame
    implied_prices: pd.DataFrame       # implied price per multiple
    summary_stats: pd.DataFrame        # mean, median, 25th, 75th per multiple
    target_metrics: dict

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Comparable Company Analysis — {self.target_ticker}[/bold blue]\n")

        ct = Table(title="Trading multiples", box=box.SIMPLE_HEAD, header_style="bold blue")
        ct.add_column("Company")
        for col in self.comps_table.columns:
            ct.add_column(col, justify="right")
        for idx, row in self.comps_table.iterrows():
            ct.add_row(
                str(idx),
                *[f"{v:.1f}x" if not np.isnan(v) else "—" for v in row],
            )
        console.print(ct)

        if not self.implied_prices.empty:
            ip = Table(title=f"Implied share price for {self.target_ticker}", box=box.SIMPLE_HEAD, header_style="bold blue")
            ip.add_column("Multiple")
            ip.add_column("25th pct", justify="right")
            ip.add_column("Median", justify="right")
            ip.add_column("75th pct", justify="right")
            for idx, row in self.implied_prices.iterrows():
                ip.add_row(
                    str(idx),
                    f"${row.get('p25', 0):.2f}",
                    f"${row.get('median', 0):.2f}",
                    f"${row.get('p75', 0):.2f}",
                )
            console.print(ip)

        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.comps_table.copy()


def _get_multiples_synthetic(tickers: list[str]) -> pd.DataFrame:
    """Generate realistic multiples for demo when yfinance unavailable."""
    data = {}
    for t in tickers:
        seed = hash(t) % 10000
        rng = np.random.RandomState(seed)
        is_tech = t in ["MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "ORCL"]
        if is_tech:
            data[t] = {
                "ev_ebitda":      rng.uniform(18, 35),
                "ev_revenue":     rng.uniform(5, 15),
                "pe_ratio":       rng.uniform(22, 45),
                "price_to_sales": rng.uniform(5, 14),
            }
        else:
            data[t] = {
                "ev_ebitda":      rng.uniform(8, 18),
                "ev_revenue":     rng.uniform(1, 6),
                "pe_ratio":       rng.uniform(10, 25),
                "price_to_sales": rng.uniform(1, 5),
            }
    return pd.DataFrame(data).T


def _get_multiples_live(tickers: list[str]) -> pd.DataFrame:
    """Try to pull live multiples from yfinance."""
    rows = {}
    try:
        import yfinance as yf
        for t in tickers:
            info = yf.Ticker(t).info
            rows[t] = {
                "ev_ebitda":      info.get("enterpriseToEbitda"),
                "ev_revenue":     info.get("enterpriseToRevenue"),
                "pe_ratio":       info.get("trailingPE"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
            }
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")
    return df.clip(0, 200)


def analyze(
    data,
    peers: list[str] | None = None,
    multiples: list[str] | None = None,
    use_live: bool = True,
) -> CompsResult:
    """
    Build a comparable company analysis for a target.

    Pulls trading multiples for a peer set and derives an implied
    valuation range for the target company.

    Parameters
    ----------
    data     : TickerData — the target company
    peers    : list of peer tickers (auto-detected if None)
    multiples: list of multiples to use (default: ev_ebitda, pe_ratio, ev_revenue)
    use_live : bool — try yfinance for live data (default True)

    Returns
    -------
    CompsResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.models.comps import analyze
    >>> apple = pull.ticker("AAPL")
    >>> result = analyze(apple, peers=["MSFT", "GOOGL", "META", "NVDA"])
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Building comps for {ticker}...[/dim]")

    if peers is None:
        from finverse.ml.cluster import peers as find_peers
        cluster_result = find_peers(data, n_peers=6)
        peers = cluster_result.peer_group[:6]

    active_multiples = multiples or ["ev_ebitda", "pe_ratio", "ev_revenue"]

    comps_df = pd.DataFrame()
    if use_live:
        comps_df = _get_multiples_live(peers)

    if comps_df.empty:
        comps_df = _get_multiples_synthetic(peers)

    comps_df = comps_df[[m for m in active_multiples if m in comps_df.columns]]
    comps_df.columns = [MULTIPLE_LABELS.get(c, c) for c in comps_df.columns]

    stats = comps_df.agg(["mean", "median",
                          lambda x: x.quantile(0.25),
                          lambda x: x.quantile(0.75)]).round(2)
    stats.index = ["Mean", "Median", "25th pct", "75th pct"]

    target_metrics = {}
    ebitda_hist = data.ebitda_history
    rev_hist = data.revenue_history
    ni_hist = data.net_income_history
    shares = data.shares_outstanding
    price = data.current_price

    if not ebitda_hist.empty:
        target_metrics["ebitda"] = float(ebitda_hist.iloc[-1])
    if not rev_hist.empty:
        target_metrics["revenue"] = float(rev_hist.iloc[-1])
    if not ni_hist.empty:
        target_metrics["net_income"] = float(ni_hist.iloc[-1])
    if shares:
        target_metrics["shares"] = float(shares) / 1e9
    if price:
        target_metrics["current_price"] = price

    implied_rows = {}
    for orig_m, label in MULTIPLE_LABELS.items():
        if label not in comps_df.columns:
            continue
        col = comps_df[label].dropna()
        if col.empty:
            continue

        p25, median, p75 = col.quantile(0.25), col.median(), col.quantile(0.75)

        def _implied(multiple_val):
            if orig_m == "ev_ebitda" and "ebitda" in target_metrics:
                ev = target_metrics["ebitda"] * multiple_val
                net_debt = (data.total_debt or 0) - (data.cash or 0)
                equity = ev - net_debt
                sh = target_metrics.get("shares", 1)
                return (equity * 1e9) / (sh * 1e9) if sh > 0 else None
            elif orig_m == "pe_ratio" and "net_income" in target_metrics:
                sh = target_metrics.get("shares", 1)
                eps = target_metrics["net_income"] / sh if sh > 0 else 0
                return eps * multiple_val
            elif orig_m == "ev_revenue" and "revenue" in target_metrics:
                ev = target_metrics["revenue"] * multiple_val
                net_debt = (data.total_debt or 0) - (data.cash or 0)
                equity = ev - net_debt
                sh = target_metrics.get("shares", 1)
                return (equity * 1e9) / (sh * 1e9) if sh > 0 else None
            return None

        implied_rows[label] = {
            "p25":    _implied(p25),
            "median": _implied(median),
            "p75":    _implied(p75),
        }

    implied_df = pd.DataFrame(implied_rows).T.dropna()
    implied_df = implied_df.round(2)

    console.print(
        f"[green]✓[/green] Comps built — {len(peers)} peers, "
        f"{len(active_multiples)} multiples"
    )

    return CompsResult(
        target_ticker=ticker,
        comps_table=comps_df.round(2),
        implied_prices=implied_df,
        summary_stats=stats,
        target_metrics=target_metrics,
    )
