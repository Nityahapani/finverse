"""Tests for DCF model — no network calls required."""
import pytest
import pandas as pd
from finverse.models.dcf import DCF, DCFAssumptions
from finverse.pull.ticker import TickerData


def make_mock_data() -> TickerData:
    d = TickerData("AAPL")
    d.info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "marketCap": 2_800_000_000_000,
        "sharesOutstanding": 15_400_000_000,
        "currentPrice": 185.0,
    }
    years = pd.date_range("2019", "2024", freq="YE")
    d.income_stmt = pd.DataFrame(
        {"Revenue": [260e9, 274e9, 365e9, 394e9, 383e9]},
        index=years[:5],
    ).T
    d.income_stmt.index = ["Total Revenue"]

    ebitda_vals = [r * 0.30 for r in [260e9, 274e9, 365e9, 394e9, 383e9]]
    d.income_stmt.loc["EBITDA"] = ebitda_vals

    net_income_vals = [r * 0.21 for r in [260e9, 274e9, 365e9, 394e9, 383e9]]
    d.income_stmt.loc["Net Income"] = net_income_vals

    ocf_vals = [r * 0.28 for r in [260e9, 274e9, 365e9, 394e9, 383e9]]
    capex_vals = [-r * 0.05 for r in [260e9, 274e9, 365e9, 394e9, 383e9]]
    d.cash_flow = pd.DataFrame(
        {"OCF": ocf_vals, "CapEx": capex_vals},
        index=years[:5],
    ).T
    d.cash_flow.index = ["Operating Cash Flow", "Capital Expenditure"]

    bs_data = {"Cash": [50e9], "LTDebt": [100e9]}
    d.balance_sheet = pd.DataFrame(bs_data, index=years[:1]).T
    d.balance_sheet.index = ["Cash And Cash Equivalents", "Long Term Debt"]
    return d


class TestDCFManual:
    def test_basic_run(self):
        model = DCF.manual(
            base_revenue=383.0,
            shares_outstanding=15.4,
            net_debt=50.0,
            current_price=185.0,
        )
        results = model.run()
        assert results.implied_price > 0
        assert results.enterprise_value > 0
        assert results.pv_fcfs > 0
        assert results.pv_terminal > 0
        assert len(results.fcf_projections) == 5

    def test_custom_assumptions(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        model.set(wacc=0.10, terminal_growth=0.02, ebitda_margin=0.25)
        results = model.run()
        assert results.assumptions.wacc == 0.10
        assert results.assumptions.terminal_growth == 0.02

    def test_invalid_assumption_raises(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        with pytest.raises(ValueError):
            model.set(nonexistent_param=0.5)

    def test_upside_calculation(self):
        model = DCF.manual(
            base_revenue=383.0,
            shares_outstanding=15.4,
            net_debt=50.0,
            current_price=100.0,
        )
        results = model.run()
        if results.upside_pct is not None:
            expected = (results.implied_price - 100.0) / 100.0
            assert abs(results.upside_pct - expected) < 0.001

    def test_projection_years(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        model.set(projection_years=3)
        results = model.run()
        assert len(results.fcf_projections) == 3

    def test_to_df(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        results = model.run()
        df = results.fcf_projections
        assert isinstance(df, pd.DataFrame)
        assert "revenue" in df.columns
        assert "fcf" in df.columns
        assert "pv_fcf" in df.columns


class TestDCFFromData:
    def test_from_ticker_data(self):
        data = make_mock_data()
        model = DCF(data)
        results = model.run()
        assert results.implied_price > 0
        assert results.enterprise_value > 0

    def test_ebitda_margin_inferred(self):
        data = make_mock_data()
        model = DCF(data)
        assert abs(model._assumptions.ebitda_margin - 0.30) < 0.05

    def test_chaining(self):
        data = make_mock_data()
        model = DCF(data)
        model.set(wacc=0.10).set(terminal_growth=0.025)
        results = model.run()
        assert results.assumptions.wacc == 0.10


class TestDCFProperties:
    def test_implied_price_property(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        price = model.implied_price
        assert price > 0

    def test_ev_property(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        ev = model.ev
        assert ev > 0

    def test_repr(self):
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        r = repr(model)
        assert "DCF" in r
        assert "wacc" in r
