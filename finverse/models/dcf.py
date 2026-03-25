"""
finverse.models.dcf — Discounted Cash Flow model with ML-assisted inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class DCFAssumptions:
    wacc: float = 0.095
    terminal_growth: float = 0.025
    revenue_growth: Optional[list[float]] = None   # per-year, or None = ML forecast
    ebitda_margin: float = 0.30
    capex_pct_revenue: float = 0.05
    nwc_pct_revenue: float = 0.02
    tax_rate: float = 0.21
    projection_years: int = 5


@dataclass
class DCFResults:
    pv_fcfs: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    implied_price: float
    current_price: Optional[float]
    upside_pct: Optional[float]
    fcf_projections: pd.DataFrame
    assumptions: DCFAssumptions

    def summary(self):
        from finverse.utils.display import console, fmt_currency, fmt_pct, fmt_price
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console.print()

        table = Table(title="DCF Projection", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Year")
        table.add_column("Revenue ($B)", justify="right")
        table.add_column("EBITDA ($B)", justify="right")
        table.add_column("FCF ($B)", justify="right")
        table.add_column("PV of FCF ($B)", justify="right")

        for _, row in self.fcf_projections.iterrows():
            table.add_row(
                str(int(row["year"])),
                fmt_currency(row["revenue"]),
                fmt_currency(row["ebitda"]),
                fmt_currency(row["fcf"]),
                fmt_currency(row["pv_fcf"]),
            )
        console.print(table)

        val_table = Table(title="Valuation Summary", box=box.SIMPLE_HEAD, header_style="bold blue")
        val_table.add_column("Item")
        val_table.add_column("Value ($B)", justify="right")
        val_table.add_row("PV of FCFs", fmt_currency(self.pv_fcfs))
        val_table.add_row("Terminal value (PV)", fmt_currency(self.pv_terminal))
        val_table.add_row("Enterprise value", fmt_currency(self.enterprise_value))
        val_table.add_row("(-) Net debt", fmt_currency(self.net_debt))
        val_table.add_row("[bold]Equity value[/bold]", f"[bold]{fmt_currency(self.equity_value)}[/bold]")
        console.print(val_table)

        color = "green" if (self.upside_pct or 0) > 0 else "red"
        console.print(f"\n  Implied share price:  [bold]{fmt_price(self.implied_price)}[/bold]")
        if self.current_price:
            console.print(f"  Current price:        {fmt_price(self.current_price)}")
            console.print(f"  Upside / downside:    [{color}][bold]{fmt_pct(self.upside_pct)}[/bold][/{color}]")
        console.print(f"\n  WACC: {fmt_pct(self.assumptions.wacc)}  |  Terminal growth: {fmt_pct(self.assumptions.terminal_growth)}")
        console.print()


class DCF:
    """
    Discounted Cash Flow model.

    Can be constructed from a TickerData object (ML fills assumptions)
    or fully manually.

    Parameters
    ----------
    data         : TickerData — from pull.ticker()
    assumptions  : DCFAssumptions — override any defaults

    Example
    -------
    >>> from finverse import pull, DCF
    >>> data = pull.ticker("AAPL")
    >>> model = DCF(data)
    >>> model.run().summary()

    Override assumptions:
    >>> model = DCF(data)
    >>> model.set(wacc=0.10, terminal_growth=0.02)
    >>> model.run().summary()

    Fully manual (no ticker):
    >>> from finverse.models.dcf import DCFAssumptions
    >>> model = DCF.manual(
    ...     base_revenue=383.0,
    ...     assumptions=DCFAssumptions(wacc=0.095, terminal_growth=0.025),
    ...     shares_outstanding=15.4,
    ...     net_debt=-55.0,
    ... )
    >>> model.run().summary()
    """

    def __init__(self, data=None, assumptions: DCFAssumptions | None = None):
        self._data = data
        self._assumptions = assumptions or DCFAssumptions()
        self._base_revenue: float | None = None
        self._shares: float | None = None
        self._net_debt: float | None = None
        self._current_price: float | None = None
        self._results: DCFResults | None = None
        self._ml_used: bool = False

        if data is not None:
            self._infer_from_data()

    def _infer_from_data(self):
        """Pull key inputs from TickerData."""
        d = self._data
        rev = d.revenue_history
        if not rev.empty:
            self._base_revenue = float(rev.iloc[-1])

        self._current_price = d.current_price
        shares = d.shares_outstanding
        self._shares = float(shares) / 1e9 if shares else None

        debt = d.total_debt or 0
        cash = d.cash or 0
        self._net_debt = debt - cash

        ebitda = d.ebitda_history
        rev_h = d.revenue_history
        if not ebitda.empty and not rev_h.empty:
            common = ebitda.index.intersection(rev_h.index)
            if len(common) >= 2:
                margin_series = ebitda.loc[common] / rev_h.loc[common]
                self._assumptions.ebitda_margin = float(margin_series.iloc[-1])

    @classmethod
    def from_ticker(cls, symbol: str, **kwargs) -> "DCF":
        """Convenience: pull data and build DCF in one step."""
        from finverse.pull.ticker import ticker
        data = ticker(symbol)
        return cls(data, **kwargs)

    @classmethod
    def manual(
        cls,
        base_revenue: float,
        shares_outstanding: float,
        net_debt: float = 0.0,
        current_price: float | None = None,
        assumptions: DCFAssumptions | None = None,
    ) -> "DCF":
        """Build a DCF without any ticker data — fully manual inputs."""
        model = cls(data=None, assumptions=assumptions)
        model._base_revenue = base_revenue
        model._shares = shares_outstanding
        model._net_debt = net_debt
        model._current_price = current_price
        return model

    def set(self, **kwargs) -> "DCF":
        """
        Override any assumption. Chainable.

        >>> model.set(wacc=0.10, terminal_growth=0.02, ebitda_margin=0.35)

        revenue_growth accepts either a scalar float (applied uniformly across all
        projection years) or a list of per-year floats.
        """
        for k, v in kwargs.items():
            if hasattr(self._assumptions, k):
                # revenue_growth must be a list[float] or None — coerce scalar
                if k == "revenue_growth" and isinstance(v, (int, float)):
                    v = [float(v)] * self._assumptions.projection_years
                setattr(self._assumptions, k, v)
            else:
                raise ValueError(f"Unknown assumption: '{k}'. Valid: {list(self._assumptions.__dataclass_fields__)}")
        return self

    def use_ml_forecast(self, macro_df: pd.DataFrame | None = None) -> "DCF":
        """
        Use ML to generate revenue growth assumptions instead of a fixed rate.
        Runs ml.forecast.revenue() on the attached TickerData.
        """
        if self._data is None:
            raise ValueError("ML forecast requires TickerData — use DCF(pull.ticker('AAPL'))")

        from finverse.ml.forecast import revenue as forecast_revenue
        fc = forecast_revenue(self._data, n_years=self._assumptions.projection_years, macro_df=macro_df)
        base = self._base_revenue or 1
        growth_rates = [(fc.point[i] - (fc.point[i-1] if i > 0 else base)) / (fc.point[i-1] if i > 0 else base)
                        for i in range(len(fc.point))]
        self._assumptions.revenue_growth = growth_rates
        self._ml_used = True
        return self

    def run(self) -> DCFResults:
        """
        Run the DCF model and return results.

        Returns
        -------
        DCFResults — access .summary() for pretty output or .fcf_projections for raw data
        """
        from finverse.utils.display import console

        if self._base_revenue is None:
            raise ValueError("No base revenue. Use DCF(pull.ticker('AAPL')) or DCF.manual(base_revenue=...)")

        a = self._assumptions
        n = a.projection_years
        wacc = a.wacc
        tg = a.terminal_growth

        console.print(f"[dim]Running DCF (WACC={wacc:.1%}, TG={tg:.1%}, {n}yr)...[/dim]")

        revenue = self._base_revenue
        rows = []
        pv_sum = 0.0

        for yr in range(1, n + 1):
            if a.revenue_growth is not None:
                rg = a.revenue_growth
                if isinstance(rg, (int, float)):
                    g = float(rg)
                elif hasattr(rg, '__len__') and len(rg) >= yr:
                    g = rg[yr - 1]
                else:
                    g = 0.08
            else:
                g = 0.08

            revenue = revenue * (1 + g)
            ebitda = revenue * a.ebitda_margin
            capex = revenue * a.capex_pct_revenue
            nwc_change = revenue * a.nwc_pct_revenue * g
            taxes = ebitda * a.tax_rate
            fcf = ebitda - taxes - capex - nwc_change

            discount = (1 + wacc) ** yr
            pv_fcf = fcf / discount
            pv_sum += pv_fcf

            base_year = 2024
            if self._data is not None:
                rev_h = self._data.revenue_history
                if not rev_h.empty:
                    last = rev_h.index[-1]
                    base_year = (last.year if hasattr(last, "year") else int(str(last)[:4]))

            rows.append({
                "year": base_year + yr,
                "revenue": round(revenue, 2),
                "ebitda": round(ebitda, 2),
                "fcf": round(fcf, 2),
                "pv_fcf": round(pv_fcf, 2),
                "discount_factor": round(discount, 4),
            })

        fcf_df = pd.DataFrame(rows)

        terminal_fcf = rows[-1]["fcf"] * (1 + tg)
        terminal_value = terminal_fcf / (wacc - tg)
        pv_terminal = terminal_value / (1 + wacc) ** n

        ev = pv_sum + pv_terminal
        net_debt = self._net_debt or 0.0
        equity_value = ev - net_debt
        shares = self._shares or 1.0
        implied_price = (equity_value * 1e9) / (shares * 1e9) if shares else None

        upside = None
        if implied_price and self._current_price:
            upside = (implied_price - self._current_price) / self._current_price

        self._results = DCFResults(
            pv_fcfs=round(pv_sum, 2),
            terminal_value=round(terminal_value, 2),
            pv_terminal=round(pv_terminal, 2),
            enterprise_value=round(ev, 2),
            net_debt=round(net_debt, 2),
            equity_value=round(equity_value, 2),
            shares_outstanding=shares or 0,
            implied_price=round(implied_price, 2) if implied_price else 0,
            current_price=self._current_price,
            upside_pct=round(upside, 4) if upside else None,
            fcf_projections=fcf_df,
            assumptions=a,
        )

        ml_note = " [ML forecast]" if self._ml_used else ""
        console.print(f"[green]✓[/green] DCF complete{ml_note} — implied price ${self._results.implied_price:.2f}")
        return self._results

    def summary(self):
        """Run the model and print summary. Shortcut for .run().summary()"""
        return self.run().summary()

    @property
    def implied_price(self) -> float:
        if self._results is None:
            self.run()
        return self._results.implied_price

    @property
    def ev(self) -> float:
        if self._results is None:
            self.run()
        return self._results.enterprise_value

    def __repr__(self):
        name = getattr(self._data, "name", "Manual") if self._data else "Manual"
        return f"DCF('{name}', wacc={self._assumptions.wacc:.1%}, tg={self._assumptions.terminal_growth:.1%})"
