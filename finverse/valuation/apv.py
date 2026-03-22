"""
finverse.valuation.apv — Adjusted Present Value (APV) model.

Modigliani-Miller based: separate the unlevered firm value from
financing effects (tax shield, distress costs, issue costs).

APV = Unlevered NPV + PV(Tax shield) - PV(Financial distress costs)

More flexible than WACC-DCF for companies with changing capital structure
(LBOs, project finance, highly leveraged firms).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class APVResult:
    ticker: str
    unlevered_value: float          # PV of FCFs at unlevered cost of equity
    pv_tax_shield: float            # PV of debt tax shield
    pv_distress_costs: float        # PV of financial distress costs
    pv_issuance_costs: float        # PV of debt issuance costs
    apv: float                      # total APV
    wacc_dcf_value: float           # comparison: traditional WACC-DCF
    apv_vs_wacc: float              # difference ($B)
    fcf_projections: pd.DataFrame
    assumptions: dict

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Adjusted Present Value — {self.ticker}[/bold blue]\n")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Component")
        table.add_column("Value ($B)", justify="right")

        table.add_row("Unlevered firm value",        f"{self.unlevered_value:.2f}")
        table.add_row("(+) PV of tax shield",        f"[green]+{self.pv_tax_shield:.2f}[/green]")
        table.add_row("(-) PV of distress costs",    f"[red]-{self.pv_distress_costs:.2f}[/red]")
        table.add_row("(-) PV of issuance costs",    f"[red]-{self.pv_issuance_costs:.2f}[/red]")
        table.add_row("[bold]APV[/bold]",             f"[bold]{self.apv:.2f}[/bold]")
        table.add_row("────────────────────", "────────")
        table.add_row("WACC-DCF (comparison)",       f"{self.wacc_dcf_value:.2f}")

        diff_color = "green" if self.apv_vs_wacc > 0 else "red"
        table.add_row("APV vs WACC-DCF",
                      f"[{diff_color}]{self.apv_vs_wacc:+.2f}[/{diff_color}]")
        console.print(table)

        console.print("\n  [dim]Key inputs:[/dim]")
        for k, v in self.assumptions.items():
            if isinstance(v, float):
                fmt = f"{v:.1%}" if v < 2 else f"{v:.2f}"
                console.print(f"    {k:<30} {fmt}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.fcf_projections.copy()


def analyze(
    data=None,
    base_revenue: float | None = None,
    unlevered_cost_of_equity: float = 0.10,
    revenue_growth: float = 0.08,
    ebitda_margin: float = 0.30,
    capex_pct: float = 0.05,
    tax_rate: float = 0.21,
    debt: float | None = None,
    cost_of_debt: float = 0.05,
    distress_prob: float = 0.05,
    distress_cost_pct: float = 0.20,
    issuance_cost_pct: float = 0.02,
    projection_years: int = 5,
    terminal_growth: float = 0.025,
    wacc: float | None = None,
) -> APVResult:
    """
    Compute APV for a company or project.

    APV separates the value of the unlevered firm from financing side effects.
    More accurate than WACC-DCF when leverage changes over time.

    Parameters
    ----------
    data                    : TickerData (optional — fills base_revenue, debt etc.)
    base_revenue            : float ($B) — if not using data
    unlevered_cost_of_equity: float — cost of equity assuming 100% equity (default 10%)
    revenue_growth          : float — annual revenue growth
    ebitda_margin           : float — EBITDA margin
    capex_pct               : float — capex % of revenue
    tax_rate                : float — corporate tax rate
    debt                    : float ($B) — total debt
    cost_of_debt            : float — pre-tax cost of debt
    distress_prob           : float — probability of financial distress (default 5%)
    distress_cost_pct       : float — distress costs as % of firm value (default 20%)
    issuance_cost_pct       : float — debt issuance costs % of debt (default 2%)
    projection_years        : int
    terminal_growth         : float
    wacc                    : float — for WACC-DCF comparison (estimated if None)

    Returns
    -------
    APVResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.valuation import apv
    >>> data = pull.ticker("AAPL")
    >>> result = apv.analyze(data)
    >>> result.summary()

    Manual:
    >>> result = apv.analyze(
    ...     base_revenue=383.0,
    ...     debt=100.0,
    ...     unlevered_cost_of_equity=0.10,
    ...     distress_prob=0.03,
    ... )
    """
    from finverse.utils.display import console

    ticker = "Project"
    if data is not None:
        ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
        if base_revenue is None:
            rev = data.revenue_history if hasattr(data, "revenue_history") else pd.Series()
            base_revenue = float(rev.iloc[-1]) if not rev.empty else 100.0
        if debt is None:
            debt = data.total_debt or 0.0
        ebitda_h = data.ebitda_history if hasattr(data, "ebitda_history") else pd.Series()
        rev_h = data.revenue_history if hasattr(data, "revenue_history") else pd.Series()
        if not ebitda_h.empty and not rev_h.empty:
            common = ebitda_h.index.intersection(rev_h.index)
            if len(common) > 0:
                ebitda_margin = float(ebitda_h.loc[common].iloc[-1] / rev_h.loc[common].iloc[-1])

    base_revenue = base_revenue or 100.0
    debt = debt or 0.0

    console.print(f"[dim]Computing APV for {ticker}...[/dim]")

    ku = unlevered_cost_of_equity
    revenue = base_revenue
    rows = []
    pv_unlevered = 0.0

    for yr in range(1, projection_years + 1):
        revenue *= (1 + revenue_growth)
        ebitda = revenue * ebitda_margin
        tax = ebitda * tax_rate
        capex = revenue * capex_pct
        fcf = ebitda - tax - capex
        pv = fcf / (1 + ku) ** yr
        pv_unlevered += pv
        rows.append({"year": yr, "revenue": round(revenue, 2),
                     "fcf": round(fcf, 2), "pv_fcf": round(pv, 2)})

    terminal_fcf = rows[-1]["fcf"] * (1 + terminal_growth)
    terminal_pv = (terminal_fcf / (ku - terminal_growth)) / (1 + ku) ** projection_years
    unlevered_value = pv_unlevered + terminal_pv

    interest = debt * cost_of_debt
    annual_tax_shield = interest * tax_rate
    kd = cost_of_debt
    pv_tax_shield = annual_tax_shield / kd if kd > 0 else 0

    pv_distress = distress_prob * distress_cost_pct * unlevered_value

    pv_issuance = issuance_cost_pct * debt

    apv_value = unlevered_value + pv_tax_shield - pv_distress - pv_issuance

    if wacc is None:
        equity = max(apv_value - debt, 0)
        total = equity + debt
        w_e = equity / total if total > 0 else 0.8
        w_d = debt / total if total > 0 else 0.2
        wacc = w_e * ku + w_d * kd * (1 - tax_rate)

    wacc_dcf_pv = 0.0
    rev2 = base_revenue
    for yr in range(1, projection_years + 1):
        rev2 *= (1 + revenue_growth)
        fcf = rev2 * ebitda_margin * (1 - tax_rate) - rev2 * capex_pct
        wacc_dcf_pv += fcf / (1 + wacc) ** yr
    tv_wacc = (rows[-1]["fcf"] * (1 + terminal_growth)) / (wacc - terminal_growth) / (1 + wacc) ** projection_years
    wacc_dcf_value = wacc_dcf_pv + tv_wacc

    fcf_df = pd.DataFrame(rows).set_index("year")

    console.print(
        f"[green]✓[/green] APV: ${apv_value:.2f}B  |  "
        f"Tax shield: +${pv_tax_shield:.2f}B  |  "
        f"Distress: -${pv_distress:.2f}B  |  "
        f"vs WACC-DCF: {apv_value - wacc_dcf_value:+.2f}B"
    )

    return APVResult(
        ticker=ticker,
        unlevered_value=round(unlevered_value, 3),
        pv_tax_shield=round(pv_tax_shield, 3),
        pv_distress_costs=round(pv_distress, 3),
        pv_issuance_costs=round(pv_issuance, 3),
        apv=round(apv_value, 3),
        wacc_dcf_value=round(wacc_dcf_value, 3),
        apv_vs_wacc=round(apv_value - wacc_dcf_value, 3),
        fcf_projections=fcf_df,
        assumptions={
            "unlevered_cost_of_equity": unlevered_cost_of_equity,
            "revenue_growth": revenue_growth,
            "ebitda_margin": ebitda_margin,
            "tax_rate": tax_rate,
            "debt": debt,
            "cost_of_debt": cost_of_debt,
            "distress_probability": distress_prob,
            "terminal_growth": terminal_growth,
        },
    )
