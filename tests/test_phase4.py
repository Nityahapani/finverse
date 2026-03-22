"""
Tests for Phase 4 models — no network calls required.
Covers: LBO, ThreeStatement, Comps, macro nowcast, model audit.
"""
import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_ticker_data


class TestLBO:
    def test_basic_run(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        model = LBO(LBOAssumptions(
            entry_ebitda=150.0, entry_ev_ebitda=10.0,
            equity_pct=0.40, revenue_growth=0.08,
            hold_years=5, exit_ev_ebitda=12.0,
        ))
        r = model.run()
        assert r.irr > -1 and r.irr < 2
        assert r.mom > 0
        assert r.entry_ev > 0
        assert r.exit_ev > 0
        assert not r.debt_schedule.empty
        assert not r.income_projections.empty
        assert len(r.income_projections) == 5

    def test_irr_increases_with_exit_multiple(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        lo = LBO(LBOAssumptions(entry_ebitda=100, entry_ev_ebitda=10, exit_ev_ebitda=10))
        hi = LBO(LBOAssumptions(entry_ebitda=100, entry_ev_ebitda=10, exit_ev_ebitda=14))
        assert hi.run().irr > lo.run().irr

    def test_properties(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        model = LBO(LBOAssumptions(entry_ebitda=150.0))
        assert isinstance(model.irr, float)
        assert isinstance(model.mom, float)

    def test_set_chainable(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        model = LBO().set(hold_years=3).set(exit_ev_ebitda=12.0)
        r = model.run()
        assert len(r.income_projections) == 3

    def test_from_ticker(self):
        from finverse.models.lbo import LBO
        data = make_ticker_data()
        model = LBO.from_ticker(data)
        r = model.run()
        assert r.irr is not None

    def test_repr(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        r = repr(LBO(LBOAssumptions()))
        assert "LBO" in r


class TestThreeStatement:
    def test_basic_run(self):
        from finverse.models.three_statement import ThreeStatement, ThreeStatementAssumptions
        model = ThreeStatement(ThreeStatementAssumptions(
            starting_revenue=1000.0, revenue_growth=0.10,
            gross_margin=0.50, projection_years=5,
        ))
        r = model.run()
        assert not r.income_statement.empty
        assert not r.balance_sheet.empty
        assert not r.cash_flow.empty
        assert r.income_statement.shape[1] == 5

    def test_revenue_grows(self):
        from finverse.models.three_statement import ThreeStatement, ThreeStatementAssumptions
        model = ThreeStatement(ThreeStatementAssumptions(
            starting_revenue=1000.0, revenue_growth=0.10, projection_years=4
        ))
        r = model.run()
        rev = r.income_statement.loc["Revenue"]
        assert all(rev.diff().dropna() > 0)

    def test_is_linked_to_cf(self):
        from finverse.models.three_statement import ThreeStatement, ThreeStatementAssumptions
        model = ThreeStatement(ThreeStatementAssumptions(starting_revenue=500.0))
        r = model.run()
        assert "Net income" in r.income_statement.index
        assert "Net income" in r.cash_flow.index

    def test_from_ticker(self):
        from finverse.models.three_statement import ThreeStatement
        data = make_ticker_data()
        model = ThreeStatement.from_ticker(data)
        r = model.run()
        assert not r.income_statement.empty

    def test_set_chainable(self):
        from finverse.models.three_statement import ThreeStatement
        model = ThreeStatement()
        model.set(revenue_growth=0.15).set(gross_margin=0.60)
        r = model.run()
        assert r.assumptions.revenue_growth == 0.15


class TestComps:
    def test_basic(self):
        from finverse.models.comps import analyze
        data = make_ticker_data()
        r = analyze(data, peers=["MSFT", "GOOGL", "META"], use_live=False)
        assert not r.comps_table.empty
        assert r.target_ticker == "AAPL"
        assert len(r.comps_table) == 3

    def test_summary_stats(self):
        from finverse.models.comps import analyze
        data = make_ticker_data()
        r = analyze(data, peers=["MSFT", "GOOGL"], use_live=False)
        assert not r.summary_stats.empty

    def test_to_df(self):
        from finverse.models.comps import analyze
        data = make_ticker_data()
        r = analyze(data, peers=["MSFT", "GOOGL"], use_live=False)
        df = r.to_df()
        assert isinstance(df, pd.DataFrame)


class TestMacroNowcast:
    def test_no_data(self):
        from finverse.models.macro import nowcast
        r = nowcast()
        assert -10 < r.gdp_nowcast < 10
        assert 0 <= r.recession_probability <= 1
        assert r.yield_curve_signal in ["inverted", "flat", "normal", "steep"]
        assert r.regime in ["expansion", "slowdown", "contraction", "recovery"]
        assert len(r.inflation_path) == 4
        assert len(r.fed_rate_path) == 4

    def test_with_macro_df(self):
        from finverse.models.macro import nowcast
        macro_df = pd.DataFrame({
            "UNRATE": [4.2], "FEDFUNDS": [5.25],
            "DGS10": [4.5], "DGS2": [4.8],
            "CPIAUCSL": [3.2], "VIXCLS": [18.0],
        }, index=pd.date_range("2024-01-01", periods=1, freq="ME"))
        r = nowcast(macro_df)
        assert r.gdp_nowcast is not None

    def test_inverted_curve_raises_recession_prob(self):
        from finverse.models.macro import nowcast
        inverted = pd.DataFrame({
            "UNRATE": [5.5], "FEDFUNDS": [5.5],
            "DGS10": [3.5], "DGS2": [5.0],
            "CPIAUCSL": [4.5], "VIXCLS": [35.0],
        }, index=pd.date_range("2024-01-01", periods=1, freq="ME"))
        normal = pd.DataFrame({
            "UNRATE": [3.5], "FEDFUNDS": [2.0],
            "DGS10": [4.5], "DGS2": [3.5],
            "CPIAUCSL": [2.0], "VIXCLS": [12.0],
        }, index=pd.date_range("2024-01-01", periods=1, freq="ME"))
        r_inv = nowcast(inverted)
        r_norm = nowcast(normal)
        assert r_inv.recession_probability > r_norm.recession_probability


class TestModelAudit:
    def test_clean_dcf_passes(self):
        from finverse.models.dcf import DCF
        from finverse.audit import audit
        model = DCF.manual(base_revenue=383.0, shares_outstanding=15.4, net_debt=50.0)
        model.run()
        r = audit(model)
        assert r.passed
        assert r.score == 100

    def test_bad_wacc_fails(self):
        from finverse.models.dcf import DCF
        from finverse.audit import audit
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        model.set(wacc=0.50)
        r = audit(model)
        assert not r.passed
        assert len(r.errors) > 0

    def test_tg_exceeds_wacc_flagged(self):
        from finverse.models.dcf import DCF
        from finverse.audit import audit
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        model.set(wacc=0.05, terminal_growth=0.06)
        r = audit(model)
        assert any("terminal_growth" in e.location for e in r.errors)

    def test_lbo_audit(self):
        from finverse.models.lbo import LBO, LBOAssumptions
        from finverse.audit import audit
        model = LBO(LBOAssumptions(entry_ebitda=150.0))
        model.run()
        r = audit(model)
        assert isinstance(r.score, float)
        assert 0 <= r.score <= 100

    def test_three_statement_audit(self):
        from finverse.models.three_statement import ThreeStatement
        from finverse.audit import audit
        model = ThreeStatement()
        model.run()
        r = audit(model)
        assert isinstance(r.score, float)

    def test_audit_to_df(self):
        from finverse.models.dcf import DCF
        from finverse.audit import audit
        model = DCF.manual(base_revenue=100.0, shares_outstanding=10.0)
        r = audit(model)
        df = r.to_df()
        assert isinstance(df, pd.DataFrame)
