"""
finverse.models.sotp — Sum of the Parts (SOTP) valuation.

Values each business segment separately using appropriate multiples
or DCF, then aggregates to derive total enterprise value.

Particularly useful for:
  - Conglomerates (GE, Berkshire, Alphabet)
  - Companies with distinct business units at different stages
  - M&A target analysis (which segments are most valuable?)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Segment:
    name: str
    metric_value: float              # e.g. EBITDA $M, revenue $M
    metric_type: str                 # "ebitda", "revenue", "earnings", "dcf_value"
    multiple: float | None = None    # EV/EBITDA, EV/Revenue, P/E etc.
    dcf_value: float | None = None   # direct DCF value if no multiple
    growth_label: str = "stable"     # "high", "moderate", "stable", "declining"
    notes: str = ""


@dataclass
class SOTPResult:
    ticker: str
    segments: list[dict]
    segment_values: pd.DataFrame
    total_ev: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    implied_price: float
    current_price: float | None
    upside: float | None
    largest_segment: str
    concentration: float             # % from largest segment

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Sum of the Parts — {self.ticker}[/bold blue]\n")

        table = Table(title="Segment valuation", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Segment")
        table.add_column("Metric", justify="right")
        table.add_column("Multiple", justify="right")
        table.add_column("EV ($B)", justify="right")
        table.add_column("% of total", justify="right")

        for _, row in self.segment_values.iterrows():
            pct = row["pct_of_total"]
            color = "green" if pct > 30 else ("blue" if pct > 15 else "dim")
            table.add_row(
                f"[{color}]{row['segment']}[/{color}]",
                f"{row['metric_value']:.1f} ({row['metric_type']})",
                f"{row['multiple']:.1f}x" if row['multiple'] > 0 else "DCF",
                f"[{color}]{row['ev_value']:.2f}[/{color}]",
                f"[{color}]{pct:.1f}%[/{color}]",
            )

        console.print(table)

        summary_table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        summary_table.add_column("Item")
        summary_table.add_column("Value ($B)", justify="right")
        summary_table.add_row("Total enterprise value", f"[bold]{self.total_ev:.2f}[/bold]")
        summary_table.add_row("(-) Net debt", f"{self.net_debt:.2f}")
        summary_table.add_row("Equity value", f"[bold]{self.equity_value:.2f}[/bold]")
        summary_table.add_row("Implied share price", f"[bold green]${self.implied_price:.2f}[/bold green]")
        if self.current_price:
            color = "green" if (self.upside or 0) > 0 else "red"
            summary_table.add_row("Current price", f"${self.current_price:.2f}")
            summary_table.add_row("Upside", f"[{color}]{self.upside:.1%}[/{color}]" if self.upside else "—")
        console.print(summary_table)
        console.print(
            f"\n  Largest segment: [bold]{self.largest_segment}[/bold] "
            f"({self.concentration:.0f}% of EV)"
        )
        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.segment_values.copy()


def analyze(
    segments: list[Segment],
    ticker: str = "Company",
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
    current_price: float | None = None,
    conglomerate_discount: float = 0.0,
) -> SOTPResult:
    """
    Compute Sum of the Parts valuation.

    Parameters
    ----------
    segments             : list of Segment objects
    ticker               : str — company ticker/name
    net_debt             : float ($B) — net debt (debt - cash)
    shares_outstanding   : float (billions)
    current_price        : float — for upside calculation
    conglomerate_discount: float — % discount applied to total EV (default 0%)

    Returns
    -------
    SOTPResult

    Example
    -------
    >>> from finverse.models.sotp import Segment, analyze
    >>>
    >>> # Alphabet SOTP
    >>> segments = [
    ...     Segment("Google Search",  ebitda_value=80.0,  metric_type="ebitda", multiple=18.0),
    ...     Segment("YouTube",        ebitda_value=12.0,  metric_type="ebitda", multiple=20.0),
    ...     Segment("Google Cloud",   rev_value=35.0,     metric_type="revenue", multiple=8.0),
    ...     Segment("Other Bets",     dcf_value=20.0,     metric_type="dcf_value"),
    ... ]
    >>> # Simpler:
    >>> segments = [
    ...     Segment(name="Search",  metric_value=80, metric_type="ebitda", multiple=18),
    ...     Segment(name="Cloud",   metric_value=35, metric_type="revenue", multiple=8),
    ...     Segment(name="YouTube", metric_value=12, metric_type="ebitda", multiple=20),
    ... ]
    >>> result = analyze(segments, ticker="GOOGL", net_debt=-100, shares_outstanding=12.8,
    ...                  current_price=175.0)
    >>> result.summary()
    """
    from finverse.utils.display import console

    console.print(f"[dim]Computing SOTP for {ticker} ({len(segments)} segments)...[/dim]")

    rows = []
    total_ev = 0.0

    for seg in segments:
        if seg.metric_type == "dcf_value" and seg.dcf_value is not None:
            ev = seg.dcf_value / 1000  # assume $M input, convert to $B
            multiple_used = 0.0
        elif seg.multiple is not None and seg.metric_value > 0:
            ev = seg.metric_value * seg.multiple / 1000
            multiple_used = seg.multiple
        else:
            ev = 0.0
            multiple_used = 0.0

        rows.append({
            "segment": seg.name,
            "metric_value": seg.metric_value,
            "metric_type": seg.metric_type,
            "multiple": multiple_used,
            "ev_value": round(ev, 3),
            "growth_label": seg.growth_label,
            "notes": seg.notes,
            "pct_of_total": 0.0,
        })
        total_ev += ev

    if conglomerate_discount > 0:
        total_ev *= (1 - conglomerate_discount)
        console.print(f"[dim]Applying {conglomerate_discount:.0%} conglomerate discount[/dim]")

    for row in rows:
        row["pct_of_total"] = round(row["ev_value"] / total_ev * 100, 1) if total_ev > 0 else 0

    df = pd.DataFrame(rows)
    df = df.sort_values("ev_value", ascending=False).reset_index(drop=True)

    equity_value = total_ev - net_debt
    implied_price = (equity_value * 1e9) / (shares_outstanding * 1e9) if shares_outstanding > 0 else 0
    upside = (implied_price - current_price) / current_price if current_price and current_price > 0 else None
    largest = df.iloc[0]["segment"] if not df.empty else "Unknown"
    concentration = df.iloc[0]["pct_of_total"] if not df.empty else 0

    console.print(
        f"[green]✓[/green] SOTP: total EV=${total_ev:.2f}B, "
        f"equity=${equity_value:.2f}B, "
        f"implied=${implied_price:.2f}"
        + (f" ({upside:+.1%})" if upside else "")
    )

    return SOTPResult(
        ticker=ticker,
        segments=[{"name": s.name, "metric": s.metric_value, "multiple": s.multiple} for s in segments],
        segment_values=df,
        total_ev=round(total_ev, 3),
        net_debt=round(net_debt, 3),
        equity_value=round(equity_value, 3),
        shares_outstanding=shares_outstanding,
        implied_price=round(implied_price, 2),
        current_price=current_price,
        upside=round(upside, 4) if upside else None,
        largest_segment=largest,
        concentration=round(concentration, 1),
    )


def from_ticker(
    data,
    segment_definitions: list[dict] | None = None,
) -> SOTPResult:
    """
    Build SOTP from TickerData with auto-estimated segments.

    Uses yfinance segment data if available, otherwise builds
    a single-segment SOTP from consolidated financials.

    Parameters
    ----------
    data               : TickerData
    segment_definitions: optional list of dicts with segment overrides

    Example
    -------
    >>> sotp = sotp_module.from_ticker(data, segment_definitions=[
    ...     {"name": "Services",  "metric_value": 85,  "metric_type": "revenue", "multiple": 6},
    ...     {"name": "Products",  "metric_value": 300, "metric_type": "revenue", "multiple": 2},
    ... ])
    >>> sotp.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"

    if segment_definitions:
        segs = [
            Segment(
                name=s["name"],
                metric_value=s.get("metric_value", 0),
                metric_type=s.get("metric_type", "ebitda"),
                multiple=s.get("multiple"),
                dcf_value=s.get("dcf_value"),
                growth_label=s.get("growth_label", "stable"),
            )
            for s in segment_definitions
        ]
    else:
        rev = data.revenue_history if hasattr(data, "revenue_history") else pd.Series()
        ebitda = data.ebitda_history if hasattr(data, "ebitda_history") else pd.Series()
        rev_val = float(rev.iloc[-1]) * 1000 if not rev.empty else 100.0
        ebitda_val = float(ebitda.iloc[-1]) * 1000 if not ebitda.empty else rev_val * 0.25

        ev_ebitda = data.info.get("enterpriseToEbitda", 15) if hasattr(data, "info") and data.info else 15
        segs = [
            Segment(
                name=f"{ticker} (consolidated)",
                metric_value=ebitda_val,
                metric_type="ebitda",
                multiple=float(ev_ebitda),
            )
        ]
        console.print(f"[yellow]No segment data — using consolidated EBITDA × {ev_ebitda:.1f}x[/yellow]")

    net_debt = (data.total_debt or 0) - (data.cash or 0)
    shares = (data.shares_outstanding or 1e10) / 1e9
    price = data.current_price

    return analyze(segs, ticker=ticker, net_debt=net_debt,
                   shares_outstanding=shares, current_price=price)
