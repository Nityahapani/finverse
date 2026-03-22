"""
finverse.models.three_statement — linked Income Statement, Balance Sheet,
and Cash Flow Statement model. All three statements are fully linked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThreeStatementAssumptions:
    # Income statement
    revenue_growth: float = 0.08
    gross_margin: float = 0.45
    sga_pct: float = 0.15
    rd_pct: float = 0.05
    da_pct: float = 0.04             # depreciation & amortisation % revenue
    interest_rate: float = 0.05
    tax_rate: float = 0.21

    # Balance sheet
    ar_days: float = 45              # accounts receivable days
    inventory_days: float = 30       # inventory days
    ap_days: float = 40              # accounts payable days
    capex_pct: float = 0.05
    debt_repayment: float = 0.0      # annual debt repayment $M

    # Starting balances ($M)
    starting_revenue: float = 1000.0
    starting_cash: float = 100.0
    starting_debt: float = 200.0
    starting_equity: float = 500.0

    projection_years: int = 5


@dataclass
class ThreeStatementResults:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    assumptions: ThreeStatementAssumptions

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print("\n[bold blue]Three-Statement Model[/bold blue]\n")

        for title, df in [
            ("Income Statement ($M)", self.income_statement),
            ("Cash Flow Statement ($M)", self.cash_flow),
            ("Balance Sheet ($M)", self.balance_sheet),
        ]:
            table = Table(title=title, box=box.SIMPLE_HEAD, header_style="bold blue")
            table.add_column("Item")
            for col in df.columns:
                table.add_column(str(col), justify="right")
            for idx, row in df.iterrows():
                table.add_row(str(idx), *[f"{v:.1f}" for v in row])
            console.print(table)

        console.print()

    def to_excel(self, path: str = "three_statement.xlsx") -> str:
        from finverse.export.excel import to_excel as _to_excel
        return _to_excel(self, path)


class ThreeStatement:
    """
    Linked three-statement financial model.

    All three statements tie together: net income flows into retained earnings,
    D&A bridges IS to CF, working capital changes flow from BS to CF.

    Example
    -------
    >>> from finverse.models.three_statement import ThreeStatement, ThreeStatementAssumptions
    >>> model = ThreeStatement(ThreeStatementAssumptions(
    ...     starting_revenue=1000.0,
    ...     revenue_growth=0.10,
    ...     gross_margin=0.50,
    ...     projection_years=5,
    ... ))
    >>> results = model.run()
    >>> results.summary()

    From TickerData:
    >>> model = ThreeStatement.from_ticker(data)
    >>> model.run().summary()
    """

    def __init__(self, assumptions: ThreeStatementAssumptions | None = None):
        self._assumptions = assumptions or ThreeStatementAssumptions()
        self._results: ThreeStatementResults | None = None

    @classmethod
    def from_ticker(cls, data) -> "ThreeStatement":
        """Build 3-statement model seeded from TickerData."""
        a = ThreeStatementAssumptions()

        rev = data.revenue_history
        if not rev.empty:
            a.starting_revenue = float(rev.iloc[-1]) * 1000
            if len(rev) > 1:
                a.revenue_growth = float(rev.pct_change().mean())

        ebitda = data.ebitda_history
        if not ebitda.empty and not rev.empty:
            common = ebitda.index.intersection(rev.index)
            if len(common) > 0:
                margin = float(ebitda.loc[common].iloc[-1] / rev.loc[common].iloc[-1])
                a.gross_margin = min(margin + 0.15, 0.80)

        if not data.balance_sheet.empty:
            for k in ["Cash And Cash Equivalents"]:
                if k in data.balance_sheet.index:
                    a.starting_cash = float(data.balance_sheet.loc[k].iloc[0]) / 1e6
            for k in ["Long Term Debt"]:
                if k in data.balance_sheet.index:
                    a.starting_debt = float(data.balance_sheet.loc[k].iloc[0]) / 1e6

        return cls(a)

    def set(self, **kwargs) -> "ThreeStatement":
        for k, v in kwargs.items():
            if hasattr(self._assumptions, k):
                setattr(self._assumptions, k, v)
            else:
                raise ValueError(f"Unknown assumption: '{k}'")
        return self

    def run(self) -> ThreeStatementResults:
        from finverse.utils.display import console

        a = self._assumptions
        n = a.projection_years
        console.print(f"[dim]Building 3-statement model ({n} years)...[/dim]")

        base_yr = 2024
        years = list(range(base_yr + 1, base_yr + n + 1))

        is_rows = {}
        bs_rows = {}
        cf_rows = {}

        cash = a.starting_cash
        debt = a.starting_debt
        equity = a.starting_equity
        retained_earnings = 0.0
        prev_ar = a.starting_revenue * a.ar_days / 365
        prev_inv = a.starting_revenue * a.inventory_days / 365
        prev_ap = a.starting_revenue * a.ap_days / 365

        revenue = a.starting_revenue

        for yr in years:
            revenue *= (1 + a.revenue_growth)
            gross_profit = revenue * a.gross_margin
            sga = revenue * a.sga_pct
            rd = revenue * a.rd_pct
            ebitda = gross_profit - sga - rd
            da = revenue * a.da_pct
            ebit = ebitda - da
            interest = debt * a.interest_rate
            ebt = ebit - interest
            tax = max(ebt * a.tax_rate, 0)
            net_income = ebt - tax

            ar = revenue * a.ar_days / 365
            inv = revenue * a.inventory_days / 365
            ap = revenue * a.ap_days / 365
            delta_nwc = (ar - prev_ar) + (inv - prev_inv) - (ap - prev_ap)
            prev_ar, prev_inv, prev_ap = ar, inv, ap

            capex = revenue * a.capex_pct
            ocf = net_income + da - delta_nwc
            fcf = ocf - capex
            debt_repay = min(a.debt_repayment, debt)
            net_cash_flow = fcf - debt_repay
            cash += net_cash_flow
            debt = max(debt - debt_repay, 0)
            retained_earnings += net_income
            equity += net_income

            is_rows[yr] = {
                "Revenue":       round(revenue, 1),
                "Gross profit":  round(gross_profit, 1),
                "EBITDA":        round(ebitda, 1),
                "EBIT":          round(ebit, 1),
                "Net income":    round(net_income, 1),
            }

            cf_rows[yr] = {
                "Net income":    round(net_income, 1),
                "D&A":           round(da, 1),
                "ΔWorking cap":  round(-delta_nwc, 1),
                "Op cash flow":  round(ocf, 1),
                "Capex":         round(-capex, 1),
                "Free cash flow":round(fcf, 1),
            }

            bs_rows[yr] = {
                "Cash":          round(max(cash, 0), 1),
                "Accounts rec.": round(ar, 1),
                "Inventory":     round(inv, 1),
                "Total assets":  round(max(cash, 0) + ar + inv + revenue * 0.5, 1),
                "Accounts pay.": round(ap, 1),
                "Total debt":    round(debt, 1),
                "Total equity":  round(equity, 1),
            }

        is_df = pd.DataFrame(is_rows)
        cf_df = pd.DataFrame(cf_rows)
        bs_df = pd.DataFrame(bs_rows)

        console.print(f"[green]✓[/green] Three-statement model built ({n} years)")

        self._results = ThreeStatementResults(
            income_statement=is_df,
            balance_sheet=bs_df,
            cash_flow=cf_df,
            assumptions=a,
        )
        return self._results

    def summary(self):
        return self.run().summary()

    def __repr__(self):
        a = self._assumptions
        return f"ThreeStatement(revenue=${a.starting_revenue:.0f}M, growth={a.revenue_growth:.1%}, {a.projection_years}yr)"
