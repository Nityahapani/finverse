"""
finverse.models.lbo — Leveraged Buyout model.

Models a private equity acquisition: entry, debt schedule,
operational improvements, and exit returns (IRR, MoM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LBOAssumptions:
    # Entry
    entry_ev_ebitda: float = 10.0
    entry_ebitda: float = 100.0        # $M
    equity_pct: float = 0.40           # equity as % of EV
    management_rollover: float = 0.05  # mgmt equity

    # Debt tranches
    senior_leverage: float = 4.0       # x EBITDA
    senior_rate: float = 0.075         # interest rate
    sub_leverage: float = 1.5          # x EBITDA (subordinated)
    sub_rate: float = 0.110

    # Operations
    revenue_growth: float = 0.08
    ebitda_margin_entry: float = 0.25
    margin_improvement: float = 0.02   # annual margin improvement
    capex_pct: float = 0.05
    tax_rate: float = 0.25

    # Exit
    hold_years: int = 5
    exit_ev_ebitda: float = 11.0
    exit_costs: float = 0.02           # as % of exit EV


@dataclass
class LBOResults:
    irr: float
    mom: float                          # money-on-money multiple
    entry_ev: float
    exit_ev: float
    equity_invested: float
    equity_at_exit: float
    debt_schedule: pd.DataFrame
    income_projections: pd.DataFrame
    assumptions: LBOAssumptions

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        a = self.assumptions
        console.print(f"\n[bold blue]LBO Model[/bold blue]\n")

        color = "green" if self.irr > 0.20 else ("yellow" if self.irr > 0.15 else "red")

        overview = Table(title="Returns summary", box=box.SIMPLE_HEAD, header_style="bold blue")
        overview.add_column("Metric")
        overview.add_column("Value", justify="right")
        overview.add_row("IRR",                f"[{color}][bold]{self.irr:.1%}[/bold][/{color}]")
        overview.add_row("Money-on-money",     f"[{color}][bold]{self.mom:.2f}x[/bold][/{color}]")
        overview.add_row("Hold period",        f"{a.hold_years} years")
        overview.add_row("Entry EV",           f"${self.entry_ev:.0f}M")
        overview.add_row("Exit EV",            f"${self.exit_ev:.0f}M")
        overview.add_row("Equity invested",    f"${self.equity_invested:.0f}M")
        overview.add_row("Equity at exit",     f"${self.equity_at_exit:.0f}M")
        console.print(overview)

        debt_t = Table(title="Debt schedule ($M)", box=box.SIMPLE_HEAD, header_style="bold blue")
        debt_t.add_column("Year")
        for col in self.debt_schedule.columns:
            debt_t.add_column(str(col), justify="right")
        for idx, row in self.debt_schedule.iterrows():
            debt_t.add_row(str(idx), *[f"{v:.1f}" for v in row])
        console.print(debt_t)
        console.print()

    def to_df(self) -> pd.DataFrame:
        return self.income_projections.copy()


class LBO:
    """
    Leveraged Buyout model.

    Models entry, debt paydown, operational improvements, and exit.

    Example
    -------
    >>> from finverse.models.lbo import LBO, LBOAssumptions
    >>> model = LBO(LBOAssumptions(
    ...     entry_ebitda=150.0,
    ...     entry_ev_ebitda=10.0,
    ...     equity_pct=0.40,
    ...     revenue_growth=0.08,
    ...     hold_years=5,
    ...     exit_ev_ebitda=12.0,
    ... ))
    >>> model.run().summary()

    From TickerData:
    >>> model = LBO.from_ticker(data, entry_premium=0.20)
    >>> model.run().summary()
    """

    def __init__(self, assumptions: LBOAssumptions | None = None):
        self._assumptions = assumptions or LBOAssumptions()
        self._results: LBOResults | None = None

    @classmethod
    def from_ticker(cls, data, entry_premium: float = 0.20) -> "LBO":
        """Build LBO from TickerData with a purchase premium."""
        a = LBOAssumptions()

        ebitda_hist = data.ebitda_history
        rev_hist = data.revenue_history

        if not ebitda_hist.empty:
            a.entry_ebitda = float(ebitda_hist.iloc[-1]) * 1000
        if not rev_hist.empty and not ebitda_hist.empty:
            common = rev_hist.index.intersection(ebitda_hist.index)
            if len(common) > 0:
                a.ebitda_margin_entry = float(
                    ebitda_hist.loc[common].iloc[-1] / rev_hist.loc[common].iloc[-1]
                )
        if data.info:
            ev_ebitda = data.info.get("enterpriseToEbitda")
            if ev_ebitda:
                a.entry_ev_ebitda = float(ev_ebitda) * (1 + entry_premium)
                a.exit_ev_ebitda = float(ev_ebitda)

        return cls(a)

    def set(self, **kwargs) -> "LBO":
        for k, v in kwargs.items():
            if hasattr(self._assumptions, k):
                setattr(self._assumptions, k, v)
            else:
                raise ValueError(f"Unknown assumption: '{k}'")
        return self

    def run(self) -> LBOResults:
        from finverse.utils.display import console

        a = self._assumptions
        console.print(f"[dim]Running LBO model ({a.hold_years}yr hold, {a.entry_ev_ebitda:.1f}x entry)...[/dim]")

        entry_ev = a.entry_ebitda * a.entry_ev_ebitda
        total_debt = a.entry_ebitda * (a.senior_leverage + a.sub_leverage)
        equity_invested = entry_ev - total_debt
        senior_debt_0 = a.entry_ebitda * a.senior_leverage
        sub_debt_0 = a.entry_ebitda * a.sub_leverage

        proj_rows = []
        debt_rows = []

        revenue = a.entry_ebitda / a.ebitda_margin_entry
        senior_debt = senior_debt_0
        sub_debt = sub_debt_0

        for yr in range(1, a.hold_years + 1):
            revenue *= (1 + a.revenue_growth)
            margin = min(a.ebitda_margin_entry + a.margin_improvement * yr, 0.60)
            ebitda = revenue * margin
            capex = revenue * a.capex_pct

            senior_interest = senior_debt * a.senior_rate
            sub_interest = sub_debt * a.sub_rate
            total_interest = senior_interest + sub_interest

            ebt = ebitda - total_interest - capex * 0.3
            tax = max(ebt * a.tax_rate, 0)
            net_income = ebt - tax

            fcf = ebitda - total_interest - tax - capex
            debt_paydown = max(fcf * 0.7, 0)

            senior_paydown = min(debt_paydown, senior_debt)
            senior_debt = max(senior_debt - senior_paydown, 0)
            remaining_paydown = debt_paydown - senior_paydown
            sub_debt = max(sub_debt - remaining_paydown, 0)

            proj_rows.append({
                "year": yr,
                "revenue": round(revenue, 1),
                "ebitda": round(ebitda, 1),
                "ebitda_margin": round(margin, 3),
                "interest": round(total_interest, 1),
                "net_income": round(net_income, 1),
                "fcf": round(fcf, 1),
            })

            debt_rows.append({
                "senior_debt": round(senior_debt, 1),
                "sub_debt": round(sub_debt, 1),
                "total_debt": round(senior_debt + sub_debt, 1),
            })

        proj_df = pd.DataFrame(proj_rows).set_index("year")
        debt_df = pd.DataFrame(debt_rows,
                               index=pd.RangeIndex(1, a.hold_years + 1, name="year"))
        debt_df.columns = ["Senior debt", "Sub debt", "Total debt"]

        exit_ebitda = proj_rows[-1]["ebitda"]
        exit_ev = exit_ebitda * a.exit_ev_ebitda * (1 - a.exit_costs)
        total_debt_exit = debt_rows[-1]["total_debt"]
        equity_exit = max(exit_ev - total_debt_exit, 0)

        mom = equity_exit / equity_invested if equity_invested > 0 else 0
        irr = (mom ** (1 / a.hold_years)) - 1 if mom > 0 else -1.0

        console.print(
            f"[green]✓[/green] LBO complete — "
            f"IRR: {irr:.1%}  MoM: {mom:.2f}x  Exit EV: ${exit_ev:.0f}M"
        )

        self._results = LBOResults(
            irr=round(irr, 4),
            mom=round(mom, 3),
            entry_ev=round(entry_ev, 1),
            exit_ev=round(exit_ev, 1),
            equity_invested=round(equity_invested, 1),
            equity_at_exit=round(equity_exit, 1),
            debt_schedule=debt_df,
            income_projections=proj_df,
            assumptions=a,
        )
        return self._results

    def summary(self):
        return self.run().summary()

    @property
    def irr(self) -> float:
        if self._results is None:
            self.run()
        return self._results.irr

    @property
    def mom(self) -> float:
        if self._results is None:
            self.run()
        return self._results.mom

    def __repr__(self):
        a = self._assumptions
        return f"LBO(entry={a.entry_ev_ebitda:.1f}x, hold={a.hold_years}yr, equity={a.equity_pct:.0%})"
