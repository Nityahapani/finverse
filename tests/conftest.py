"""
Shared pytest fixtures for finverse tests.
All fixtures use synthetic data — no network calls required.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from finverse.pull.ticker import TickerData


def _make_ticker_data(ticker: str = "AAPL", seed: int = 42) -> TickerData:
    np.random.seed(seed)
    d = TickerData(ticker)
    d.info = {
        "longName": ticker,
        "sector": "Technology",
        "marketCap": 2_800_000_000_000,
        "sharesOutstanding": 15_400_000_000,
        "currentPrice": 185.0,
        "dividendRate": 0.96,
        "revenueGrowth": 0.08,
        "ebitdaMargins": 0.32,
        "profitMargins": 0.21,
        "enterpriseToEbitda": 20.0,
        "enterpriseToRevenue": 6.0,
        "trailingPE": 28.0,
        "debtToEquity": 0.5,
        "returnOnEquity": 0.18,
        "returnOnAssets": 0.12,
        "beta": 1.2,
        "totalDebt": 100_000_000_000,
        "priceToSalesTrailing12Months": 7.0,
    }
    years = pd.date_range("2019", "2024", freq="YE")
    rev = [260e9, 274e9, 365e9, 394e9, 383e9]
    d.income_stmt = pd.DataFrame(
        {
            "Total Revenue": rev,
            "EBITDA": [v * 0.32 for v in rev],
            "Net Income": [v * 0.21 for v in rev],
            "EBIT": [v * 0.25 for v in rev],
        },
        index=years[:5],
    ).T
    d.cash_flow = pd.DataFrame(
        {
            "Operating Cash Flow": [v * 0.28 for v in rev],
            "Capital Expenditure": [-v * 0.05 for v in rev],
        },
        index=years[:5],
    ).T
    d.balance_sheet = pd.DataFrame(
        {
            "Total Assets": [350e9],
            "Cash And Cash Equivalents": [50e9],
            "Long Term Debt": [100e9],
            "Retained Earnings": [150e9],
            "Stockholders Equity": [80e9],
            "Total Current Assets": [100e9],
            "Total Current Liabilities": [60e9],
        },
        index=years[:1],
    ).T
    d.price_history = pd.DataFrame(
        {"Close": np.cumprod(1 + np.random.normal(0.0004, 0.012, 756))},
        index=pd.date_range("2021-01-01", periods=756, freq="B"),
    )
    return d


@pytest.fixture
def apple_data():
    """Standard Apple TickerData fixture."""
    return _make_ticker_data("AAPL", seed=42)


@pytest.fixture
def msft_data():
    """Microsoft TickerData fixture."""
    return _make_ticker_data("MSFT", seed=10)


@pytest.fixture
def googl_data():
    """Alphabet TickerData fixture."""
    return _make_ticker_data("GOOGL", seed=20)


@pytest.fixture
def data_list(apple_data, msft_data, googl_data):
    """List of three TickerData objects for portfolio tests."""
    return [apple_data, msft_data, googl_data]


@pytest.fixture
def simple_dcf(apple_data):
    """Pre-configured DCF model, not yet run."""
    from finverse.models.dcf import DCF
    return DCF(apple_data)


@pytest.fixture
def run_dcf(apple_data):
    """Pre-configured and run DCF model."""
    from finverse.models.dcf import DCF
    model = DCF(apple_data)
    model.run()
    return model


@pytest.fixture
def manual_dcf():
    """Fully manual DCF — no TickerData dependency."""
    from finverse.models.dcf import DCF
    return DCF.manual(
        base_revenue=383.0,
        shares_outstanding=15.4,
        net_debt=50.0,
        current_price=185.0,
    )


@pytest.fixture
def macro_df():
    """Synthetic macro DataFrame (FRED-style)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "GDP": np.cumsum(np.random.normal(0.005, 0.01, n)),
            "UNRATE": np.random.normal(4.5, 0.5, n).clip(2, 10),
            "FEDFUNDS": np.random.normal(3.5, 1.5, n).clip(0, 8),
            "DGS10": np.random.normal(4.0, 0.5, n).clip(1, 8),
            "DGS2": np.random.normal(4.2, 0.6, n).clip(0.5, 8),
            "CPIAUCSL": np.random.normal(2.5, 1.0, n).clip(0, 10),
            "VIXCLS": np.random.normal(18, 8, n).clip(9, 80),
        },
        index=pd.date_range("2000", periods=n, freq="QE"),
    )


# Public helper for direct import in test files
def make_ticker_data(ticker: str = "AAPL", seed: int = 42) -> TickerData:
    """Public wrapper around _make_ticker_data for direct import in test modules."""
    return _make_ticker_data(ticker=ticker, seed=seed)
