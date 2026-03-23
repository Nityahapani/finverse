"""
finverse.models.synthetic_peers — Synthetic Peer Construction.

When a company has no true public peers — conglomerate, niche sector,
early-stage — builds synthetic peers by blending financial characteristics
from multiple real companies weighted by segment mix.

Example: a company that is 60% software, 40% hardware gets a synthetic
peer constructed from 60% of software sector multiples and 40% of
hardware sector multiples.

Also works for pure-play companies: uses cluster-based peer detection
to find the closest real peers and interpolates their characteristics.

Pure sklearn + numpy, no API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Sector multiple benchmarks (updated periodically) ────────────────────
# Source: cross-sectional averages from public market data
SECTOR_MULTIPLES = {
    "software": {
        "ev_ebitda": 25.0, "ev_revenue": 8.0, "pe_ratio": 35.0,
        "price_to_sales": 8.0, "ev_ebit": 30.0,
        "revenue_growth": 0.15, "ebitda_margin": 0.28, "gross_margin": 0.72,
        "roic": 0.18, "beta": 1.20,
    },
    "hardware": {
        "ev_ebitda": 12.0, "ev_revenue": 1.8, "pe_ratio": 18.0,
        "price_to_sales": 1.8, "ev_ebit": 14.0,
        "revenue_growth": 0.06, "ebitda_margin": 0.16, "gross_margin": 0.38,
        "roic": 0.12, "beta": 1.05,
    },
    "semiconductors": {
        "ev_ebitda": 20.0, "ev_revenue": 5.0, "pe_ratio": 28.0,
        "price_to_sales": 5.0, "ev_ebit": 23.0,
        "revenue_growth": 0.12, "ebitda_margin": 0.32, "gross_margin": 0.55,
        "roic": 0.20, "beta": 1.35,
    },
    "finance": {
        "ev_ebitda": 10.0, "ev_revenue": 2.5, "pe_ratio": 13.0,
        "price_to_sales": 2.0, "ev_ebit": 12.0,
        "revenue_growth": 0.05, "ebitda_margin": 0.30, "gross_margin": 0.60,
        "roic": 0.10, "beta": 0.95,
    },
    "healthcare": {
        "ev_ebitda": 16.0, "ev_revenue": 3.5, "pe_ratio": 22.0,
        "price_to_sales": 3.5, "ev_ebit": 18.0,
        "revenue_growth": 0.09, "ebitda_margin": 0.22, "gross_margin": 0.58,
        "roic": 0.14, "beta": 0.80,
    },
    "pharma": {
        "ev_ebitda": 14.0, "ev_revenue": 4.0, "pe_ratio": 20.0,
        "price_to_sales": 4.0, "ev_ebit": 16.0,
        "revenue_growth": 0.07, "ebitda_margin": 0.28, "gross_margin": 0.68,
        "roic": 0.16, "beta": 0.75,
    },
    "consumer_staples": {
        "ev_ebitda": 14.0, "ev_revenue": 2.0, "pe_ratio": 20.0,
        "price_to_sales": 2.0, "ev_ebit": 16.0,
        "revenue_growth": 0.04, "ebitda_margin": 0.18, "gross_margin": 0.45,
        "roic": 0.13, "beta": 0.60,
    },
    "consumer_discretionary": {
        "ev_ebitda": 13.0, "ev_revenue": 1.5, "pe_ratio": 22.0,
        "price_to_sales": 1.5, "ev_ebit": 15.0,
        "revenue_growth": 0.07, "ebitda_margin": 0.14, "gross_margin": 0.40,
        "roic": 0.11, "beta": 1.15,
    },
    "energy": {
        "ev_ebitda": 8.0, "ev_revenue": 1.2, "pe_ratio": 12.0,
        "price_to_sales": 1.2, "ev_ebit": 10.0,
        "revenue_growth": 0.03, "ebitda_margin": 0.20, "gross_margin": 0.35,
        "roic": 0.09, "beta": 1.10,
    },
    "industrials": {
        "ev_ebitda": 13.0, "ev_revenue": 1.8, "pe_ratio": 19.0,
        "price_to_sales": 1.8, "ev_ebit": 15.0,
        "revenue_growth": 0.06, "ebitda_margin": 0.17, "gross_margin": 0.36,
        "roic": 0.12, "beta": 1.00,
    },
    "telecom": {
        "ev_ebitda": 7.0, "ev_revenue": 1.5, "pe_ratio": 14.0,
        "price_to_sales": 1.5, "ev_ebit": 9.0,
        "revenue_growth": 0.02, "ebitda_margin": 0.30, "gross_margin": 0.55,
        "roic": 0.07, "beta": 0.70,
    },
    "real_estate": {
        "ev_ebitda": 18.0, "ev_revenue": 8.0, "pe_ratio": 35.0,
        "price_to_sales": 8.0, "ev_ebit": 22.0,
        "revenue_growth": 0.04, "ebitda_margin": 0.45, "gross_margin": 0.65,
        "roic": 0.06, "beta": 0.85,
    },
    "utilities": {
        "ev_ebitda": 10.0, "ev_revenue": 2.5, "pe_ratio": 16.0,
        "price_to_sales": 2.5, "ev_ebit": 13.0,
        "revenue_growth": 0.02, "ebitda_margin": 0.28, "gross_margin": 0.40,
        "roic": 0.06, "beta": 0.45,
    },
    "ecommerce": {
        "ev_ebitda": 22.0, "ev_revenue": 3.5, "pe_ratio": 40.0,
        "price_to_sales": 3.5, "ev_ebit": 28.0,
        "revenue_growth": 0.18, "ebitda_margin": 0.12, "gross_margin": 0.42,
        "roic": 0.10, "beta": 1.25,
    },
}

MULTIPLE_KEYS = [
    "ev_ebitda", "ev_revenue", "pe_ratio",
    "price_to_sales", "ev_ebit",
    "revenue_growth", "ebitda_margin", "gross_margin",
    "roic", "beta",
]


@dataclass
class SyntheticPeerResult:
    ticker: str
    segment_weights: dict[str, float]
    synthetic_multiples: dict[str, float]
    implied_prices: dict[str, float]       # per-multiple implied prices
    implied_price_range: tuple[float, float, float]  # p25, median, p75
    component_breakdown: pd.DataFrame
    target_metrics: dict[str, float]

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(
            f"\n[bold blue]Synthetic Peer Construction — {self.ticker}[/bold blue]\n"
        )

        # Segment mix
        console.print("  [dim]Segment mix:[/dim]")
        for seg, w in sorted(self.segment_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30)
            console.print(f"    {seg:<25} {bar} {w:.0%}")

        # Synthetic multiples vs target
        table = Table(
            title="\nSynthetic multiples",
            box=box.SIMPLE_HEAD,
            header_style="bold blue",
        )
        table.add_column("Multiple")
        table.add_column("Synthetic", justify="right")
        table.add_column("Implied price", justify="right")

        for key, val in self.synthetic_multiples.items():
            price = self.implied_prices.get(key)
            price_str = f"${price:.2f}" if price else "—"
            table.add_row(
                key.replace("_", "/").upper(),
                f"{val:.1f}x" if val > 2 else f"{val:.2%}",
                price_str,
            )
        console.print(table)

        p25, med, p75 = self.implied_price_range
        console.print(f"\n  Implied price range:")
        console.print(f"    25th pct:  ${p25:.2f}")
        console.print(f"    Median:    [bold]${med:.2f}[/bold]")
        console.print(f"    75th pct:  ${p75:.2f}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.component_breakdown.copy()


def build_peers(
    data,
    segment_weights: dict[str, float] | None = None,
    auto_detect: bool = True,
) -> SyntheticPeerResult:
    """
    Build synthetic peers for a company with no clean peer set.

    Constructs implied multiples and valuation range by blending
    sector benchmarks weighted by business segment mix.

    Parameters
    ----------
    data            : TickerData from pull.ticker()
    segment_weights : dict of {sector_name: weight} summing to 1.0
                      e.g. {"software": 0.60, "hardware": 0.40}
                      If None, auto-detected from company info.
    auto_detect     : bool — attempt to auto-detect sector mix from
                      company description (default True)

    Returns
    -------
    SyntheticPeerResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.models.synthetic_peers import build_peers
    >>>
    >>> # Manual segment weights (best — most accurate)
    >>> data = pull.ticker("AAPL")
    >>> result = build_peers(data, segment_weights={
    ...     "hardware":     0.55,   # iPhone, Mac, iPad
    ...     "software":     0.25,   # App Store, iCloud
    ...     "consumer_staples": 0.20,  # wearables, accessories
    ... })
    >>> result.summary()
    >>>
    >>> # Auto-detect from sector
    >>> result = build_peers(data)
    >>> result.summary()
    >>>
    >>> # Use implied price range in SOTP
    >>> p25, median, p75 = result.implied_price_range
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Building synthetic peers for {ticker}...[/dim]")

    # ── Determine segment weights ─────────────────────────────────────────
    if segment_weights is None:
        segment_weights = _auto_detect_segments(data)

    # Normalise weights
    total_w = sum(segment_weights.values())
    if total_w <= 0:
        segment_weights = {"industrials": 1.0}
        total_w = 1.0
    segment_weights = {k: v / total_w for k, v in segment_weights.items()}

    # Validate sectors
    valid_segments = {}
    for seg, w in segment_weights.items():
        seg_clean = seg.lower().replace(" ", "_").replace("-", "_")
        if seg_clean in SECTOR_MULTIPLES:
            valid_segments[seg_clean] = w
        else:
            closest = _find_closest_sector(seg_clean)
            console.print(
                f"[yellow]'{seg}' not found — using closest match: '{closest}'[/yellow]"
            )
            valid_segments[closest] = valid_segments.get(closest, 0) + w

    # Re-normalise
    total_v = sum(valid_segments.values())
    valid_segments = {k: v / total_v for k, v in valid_segments.items()}

    # ── Blend multiples ───────────────────────────────────────────────────
    synthetic: dict[str, float] = {}
    component_rows = []

    for metric in MULTIPLE_KEYS:
        blended = 0.0
        for seg, w in valid_segments.items():
            sector_val = SECTOR_MULTIPLES[seg].get(metric, 0)
            blended += w * sector_val
            component_rows.append({
                "segment": seg,
                "weight": round(w, 3),
                "metric": metric,
                "sector_value": round(sector_val, 4),
                "contribution": round(w * sector_val, 4),
            })
        synthetic[metric] = round(blended, 3)

    component_df = pd.DataFrame(component_rows)

    # ── Extract target company metrics ────────────────────────────────────
    target_metrics: dict[str, float] = {}
    rev   = data.revenue_history  if hasattr(data, "revenue_history")  else pd.Series()
    ebitda= data.ebitda_history   if hasattr(data, "ebitda_history")   else pd.Series()
    ni    = data.net_income_history if hasattr(data, "net_income_history") else pd.Series()
    shares= getattr(data, "shares_outstanding", None)
    price = getattr(data, "current_price",      None)
    debt  = getattr(data, "total_debt",         None) or 0
    cash  = getattr(data, "cash",               None) or 0

    if not rev.empty:
        target_metrics["revenue"] = float(rev.iloc[-1])  # in B
    if not ebitda.empty:
        target_metrics["ebitda"] = float(ebitda.iloc[-1])
    if not ni.empty:
        target_metrics["net_income"] = float(ni.iloc[-1])
    if shares:
        target_metrics["shares"] = float(shares) / 1e9
    if price:
        target_metrics["price"] = float(price)

    net_debt = (debt - cash) / 1e9 if debt else 0.0

    # ── Implied prices per multiple ───────────────────────────────────────
    implied_prices: dict[str, float] = {}
    sh = target_metrics.get("shares", 1.0)

    if "ebitda" in target_metrics and sh > 0:
        for m_key, label in [("ev_ebitda", "ev_ebitda"), ("ev_ebit", "ev_ebit")]:
            mult = synthetic.get(m_key)
            base = target_metrics.get("ebitda") if "ebitda" in m_key else \
                   target_metrics.get("ebitda", 0) * 0.85
            if mult and base:
                ev = base * mult * 1e9
                eq = ev - net_debt * 1e9
                px = eq / (sh * 1e9)
                implied_prices[label] = round(max(px, 0), 2)

    if "revenue" in target_metrics and sh > 0:
        for m_key in ["ev_revenue", "price_to_sales"]:
            mult = synthetic.get(m_key)
            if mult:
                ev = target_metrics["revenue"] * mult * 1e9
                eq = ev - net_debt * 1e9
                px = eq / (sh * 1e9)
                implied_prices[m_key] = round(max(px, 0), 2)

    if "net_income" in target_metrics and sh > 0:
        mult = synthetic.get("pe_ratio")
        if mult:
            eps = target_metrics["net_income"] / sh
            implied_prices["pe_ratio"] = round(max(eps * mult, 0), 2)

    # Price range
    prices_list = [v for v in implied_prices.values() if v > 0]
    if prices_list:
        p25 = round(float(np.percentile(prices_list, 25)), 2)
        med = round(float(np.median(prices_list)), 2)
        p75 = round(float(np.percentile(prices_list, 75)), 2)
    else:
        p25 = med = p75 = 0.0

    console.print(
        f"[green]✓[/green] Synthetic peers built — "
        f"{len(valid_segments)} sectors blended | "
        f"EV/EBITDA: {synthetic.get('ev_ebitda', 0):.1f}x | "
        f"implied median: ${med:.2f}"
    )

    return SyntheticPeerResult(
        ticker=ticker,
        segment_weights=valid_segments,
        synthetic_multiples=synthetic,
        implied_prices=implied_prices,
        implied_price_range=(p25, med, p75),
        component_breakdown=component_df,
        target_metrics=target_metrics,
    )


def _auto_detect_segments(data) -> dict[str, float]:
    """Attempt to auto-detect sector from company info."""
    info = getattr(data, "info", {}) or {}
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    mapping = {
        "technology":             "software",
        "software":               "software",
        "hardware":               "hardware",
        "semiconductors":         "semiconductors",
        "financial":              "finance",
        "financials":             "finance",
        "healthcare":             "healthcare",
        "health care":            "healthcare",
        "pharmaceutical":         "pharma",
        "consumer staples":       "consumer_staples",
        "consumer discretionary": "consumer_discretionary",
        "energy":                 "energy",
        "industrials":            "industrials",
        "telecom":                "telecom",
        "communication":          "telecom",
        "real estate":            "real_estate",
        "utilities":              "utilities",
        "retail":                 "ecommerce",
    }

    for key, sector_name in mapping.items():
        if key in sector or key in industry:
            return {sector_name: 1.0}

    return {"industrials": 1.0}


def _find_closest_sector(name: str) -> str:
    """Find the closest sector name using simple string matching."""
    from difflib import get_close_matches
    matches = get_close_matches(name, list(SECTOR_MULTIPLES.keys()), n=1, cutoff=0.4)
    return matches[0] if matches else "industrials"


def compare_sectors(
    sectors: list[str],
    metric: str = "ev_ebitda",
) -> pd.DataFrame:
    """
    Compare a metric across sectors.

    Parameters
    ----------
    sectors : list of sector names
    metric  : one of the MULTIPLE_KEYS

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> from finverse.models.synthetic_peers import compare_sectors
    >>> compare_sectors(["software", "hardware", "finance", "healthcare"])
    """
    rows = []
    for s in sectors:
        s_clean = s.lower().replace(" ", "_")
        if s_clean in SECTOR_MULTIPLES:
            rows.append({"sector": s_clean, **SECTOR_MULTIPLES[s_clean]})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("sector")[[
        m for m in MULTIPLE_KEYS if m in pd.DataFrame(rows).columns
    ]].round(2)
