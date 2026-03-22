"""
finverse.pull — data fetching from free financial data sources.

Sources
-------
- ticker()          yfinance (stocks, financials, price history)
- fred()            Federal Reserve FRED (macro, rates, inflation)
- edgar()           SEC EDGAR (filings, XBRL facts)
- edgar_financials() structured annual financials from EDGAR

Example
-------
    from finverse import pull

    apple = pull.ticker("AAPL")
    macro = pull.fred("GDP", "FEDFUNDS")
    filings = pull.edgar("AAPL", "10-K")
"""

from finverse.pull.ticker import ticker, TickerData
from finverse.pull.fred import fred, macro_snapshot
from finverse.pull.edgar import edgar, edgar_financials

__all__ = [
    "ticker",
    "TickerData",
    "fred",
    "macro_snapshot",
    "edgar",
    "edgar_financials",
]
