"""
pull.ticker — fetch financial statements and price data via yfinance.
"""
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np

from finverse.utils.validate import clean_ticker

warnings.filterwarnings("ignore", category=FutureWarning)


class TickerData:
    """
    Container for a company's financial data pulled from yfinance.

    Attributes
    ----------
    ticker : str
    income_stmt : pd.DataFrame   — annual income statement
    balance_sheet : pd.DataFrame — annual balance sheet
    cash_flow : pd.DataFrame     — annual cash flow statement
    price_history : pd.DataFrame — daily OHLCV
    info : dict                  — company metadata
    """

    def __init__(self, ticker: str):
        self.ticker = clean_ticker(ticker)
        self.income_stmt: pd.DataFrame = pd.DataFrame()
        self.balance_sheet: pd.DataFrame = pd.DataFrame()
        self.cash_flow: pd.DataFrame = pd.DataFrame()
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.info: dict = {}
        self._metrics: dict = {}

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.info.get("longName", self.ticker)

    @property
    def sector(self) -> str:
        return self.info.get("sector", "Unknown")

    @property
    def market_cap(self) -> float | None:
        return self.info.get("marketCap")

    @property
    def shares_outstanding(self) -> float | None:
        v = self.info.get("sharesOutstanding") or self.info.get("impliedSharesOutstanding")
        return v

    @property
    def current_price(self) -> float | None:
        return self.info.get("currentPrice") or self.info.get("regularMarketPrice")

    @property
    def revenue_history(self) -> pd.Series:
        return self._get_is_item(["Total Revenue", "Revenue"])

    @property
    def ebitda_history(self) -> pd.Series:
        return self._get_is_item(["EBITDA", "Normalized EBITDA"])

    @property
    def ebit_history(self) -> pd.Series:
        return self._get_is_item(["EBIT", "Operating Income"])

    @property
    def net_income_history(self) -> pd.Series:
        return self._get_is_item(["Net Income", "Net Income Common Stockholders"])

    @property
    def fcf_history(self) -> pd.Series:
        ocf = self._get_cf_item(["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        capex = self._get_cf_item(["Capital Expenditure", "Purchase Of PPE"])
        if ocf.empty or capex.empty:
            return pd.Series(dtype=float)
        capex_abs = capex.abs()
        return (ocf - capex_abs).dropna()

    @property
    def total_debt(self) -> float | None:
        try:
            for col in self.balance_sheet.columns[:1]:
                for key in ["Total Debt", "Long Term Debt And Capital Lease Obligation"]:
                    if key in self.balance_sheet.index:
                        return float(self.balance_sheet.loc[key, col]) / 1e9
        except Exception:
            pass
        return None

    @property
    def cash(self) -> float | None:
        try:
            for col in self.balance_sheet.columns[:1]:
                for key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                    if key in self.balance_sheet.index:
                        return float(self.balance_sheet.loc[key, col]) / 1e9
        except Exception:
            pass
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_is_item(self, keys: list[str]) -> pd.Series:
        if self.income_stmt.empty:
            return pd.Series(dtype=float)
        for k in keys:
            if k in self.income_stmt.index:
                s = self.income_stmt.loc[k].dropna().sort_index() / 1e9
                return s
        return pd.Series(dtype=float)

    def _get_cf_item(self, keys: list[str]) -> pd.Series:
        if self.cash_flow.empty:
            return pd.Series(dtype=float)
        for k in keys:
            if k in self.cash_flow.index:
                s = self.cash_flow.loc[k].dropna().sort_index() / 1e9
                return s
        return pd.Series(dtype=float)

    def summary(self):
        from finverse.utils.display import console, fmt_currency, fmt_pct
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]{self.name}[/bold blue] [dim]({self.ticker})[/dim]")
        console.print(f"[dim]{self.sector}[/dim]\n")

        rev = self.revenue_history
        ebitda = self.ebitda_history
        fcf = self.fcf_history

        if not rev.empty:
            table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
            table.add_column("Metric", style="dim")
            for yr in rev.index[-4:]:
                table.add_column(str(yr.year) if hasattr(yr, "year") else str(yr), justify="right")

            def add_row(label, series):
                vals = [fmt_currency(series.get(yr)) for yr in rev.index[-4:]]
                table.add_row(label, *vals)

            add_row("Revenue ($B)", rev)
            if not ebitda.empty:
                add_row("EBITDA ($B)", ebitda)
            if not fcf.empty:
                add_row("FCF ($B)", fcf)

            console.print(table)

        console.print(f"  Market cap:  {fmt_currency(self.market_cap / 1e9 if self.market_cap else None)}")
        console.print(f"  Price:       ${self.current_price:.2f}" if self.current_price else "  Price: —")
        console.print()

    def __repr__(self):
        return f"TickerData(ticker='{self.ticker}', name='{self.name}')"


def ticker(symbol: str, years: int = 5) -> TickerData:
    """
    Pull financial data for a ticker symbol.

    Parameters
    ----------
    symbol : str   — e.g. "AAPL", "MSFT"
    years  : int   — how many years of history to fetch (default 5)

    Returns
    -------
    TickerData

    Example
    -------
    >>> from finverse import pull
    >>> data = pull.ticker("AAPL")
    >>> data.summary()
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    from finverse.utils.display import console

    symbol = clean_ticker(symbol)
    console.print(f"[dim]Fetching {symbol} from yfinance...[/dim]")

    result = TickerData(symbol)

    try:
        t = yf.Ticker(symbol)
        result.info = t.info or {}
        result.income_stmt = t.financials if t.financials is not None else pd.DataFrame()
        result.balance_sheet = t.balance_sheet if t.balance_sheet is not None else pd.DataFrame()
        result.cash_flow = t.cashflow if t.cashflow is not None else pd.DataFrame()

        import datetime
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365 * years)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        result.price_history = hist if hist is not None else pd.DataFrame()

        console.print(f"[green]✓[/green] {result.name} loaded — {len(result.income_stmt.columns)} years of financials")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] partial data for {symbol}: {e}")

    return result
