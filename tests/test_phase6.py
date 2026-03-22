"""
Tests for Phase 6 — no network calls required.
Covers: EVT, Kelly, DDM, SOTP, EarningsQuality.
"""
import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_ticker_data


class TestEVT:
    def test_basic(self):
        from finverse.risk.evt import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert r.xi is not None
        assert r.var_99 > 0
        assert r.var_999 >= r.var_99
        assert r.var_9999 >= r.var_999
        assert 0 < r.es_99 <= 1
        assert r.n_exceedances > 0

    def test_return_periods(self):
        from finverse.risk.evt import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert len(r.return_periods) > 0
        for loss, period in r.return_periods.items():
            assert period > 0

    def test_from_series(self):
        from finverse.risk.evt import analyze
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0004, 0.015, 756))
        r = analyze(returns)
        assert r.var_99 > 0

    def test_compare_tails(self):
        from finverse.risk.evt import compare_tails
        d1 = make_ticker_data("AAPL", 42)
        d2 = make_ticker_data("MSFT", 10)
        df = compare_tails([d1, d2])
        assert len(df) == 2
        assert "xi (tail index)" in df.columns

    def test_to_df(self):
        from finverse.risk.evt import analyze
        r = analyze(make_ticker_data())
        df = r.to_df()
        assert "xi" in df.columns


class TestKelly:
    def test_from_distribution(self):
        from finverse.risk.kelly import from_distribution
        data = make_ticker_data()
        r = from_distribution(data)
        assert isinstance(r.full_kelly, float)
        assert abs(r.half_kelly - r.full_kelly * 0.5) < 1e-6
        assert abs(r.quarter_kelly - r.full_kelly * 0.25) < 1e-6

    def test_growth_ordering(self):
        from finverse.risk.kelly import from_distribution
        data = make_ticker_data()
        r = from_distribution(data)
        assert r.expected_growth_full >= r.expected_growth_half
        assert r.expected_growth_half >= r.expected_growth_quarter

    def test_simulate(self):
        from finverse.risk.kelly import from_distribution
        data = make_ticker_data()
        r = from_distribution(data)
        paths = r.simulate(n_periods=100, n_paths=50)
        assert "Full Kelly" in paths.columns
        assert len(paths) == 100

    def test_binary_positive_edge(self):
        from finverse.risk.kelly import from_binary
        r = from_binary(win_prob=0.60, win_return=0.10, loss_return=0.08)
        assert r.full_kelly > 0

    def test_binary_negative_edge(self):
        from finverse.risk.kelly import from_binary
        r = from_binary(win_prob=0.40, win_return=0.08, loss_return=0.10)
        assert r.full_kelly <= 0

    def test_multi_asset(self):
        from finverse.risk.kelly import multi_asset
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        fracs = multi_asset(data_list)
        assert len(fracs) == 3

    def test_to_df(self):
        from finverse.risk.kelly import from_distribution
        r = from_distribution(make_ticker_data())
        df = r.to_df()
        assert "full_kelly" in df.columns


class TestDDM:
    def test_gordon_basic(self):
        from finverse.models.ddm import gordon
        r = gordon(dividend=1.84, growth_rate=0.04, cost_of_equity=0.085, current_price=60.0)
        assert r.implied_price > 0
        assert r.upside is not None
        assert r.model == "Gordon Growth Model"

    def test_gordon_ke_must_exceed_g(self):
        from finverse.models.ddm import gordon
        with pytest.raises(ValueError):
            gordon(dividend=1.84, growth_rate=0.10, cost_of_equity=0.08)

    def test_h_model_higher_than_gordon(self):
        from finverse.models.ddm import gordon, h_model
        r_g = gordon(dividend=1.84, growth_rate=0.04, cost_of_equity=0.09)
        r_h = h_model(dividend=1.84, high_growth=0.12, stable_growth=0.04,
                      half_life=5, cost_of_equity=0.09)
        assert r_h.implied_price > r_g.implied_price

    def test_multistage(self):
        from finverse.models.ddm import multistage
        r = multistage(dividend=1.84, stage1_growth=0.15, stage1_years=5,
                       stage2_growth=0.08, stage2_years=5,
                       terminal_growth=0.04, cost_of_equity=0.10)
        assert r.implied_price > 0
        assert r.pv_dividends > 0
        assert r.terminal_value > 0
        assert len(r.dividends_used) == 10

    def test_from_ticker_data(self):
        from finverse.models.ddm import gordon
        data = make_ticker_data()
        r = gordon(data=data, growth_rate=0.04, cost_of_equity=0.085)
        assert r.implied_price > 0

    def test_to_df(self):
        from finverse.models.ddm import gordon
        r = gordon(dividend=1.84, growth_rate=0.04, cost_of_equity=0.09)
        df = r.to_df()
        assert "implied_price" in df.columns


class TestSOTP:
    def test_basic(self):
        from finverse.models.sotp import Segment, analyze
        segs = [
            Segment("Search", metric_value=80000, metric_type="ebitda", multiple=18.0),
            Segment("Cloud",  metric_value=35000, metric_type="revenue", multiple=8.0),
        ]
        r = analyze(segs, ticker="TEST", net_debt=0.0, shares_outstanding=10.0)
        assert r.total_ev > 0
        assert r.equity_value > 0
        assert r.implied_price > 0
        assert not r.segment_values.empty

    def test_pct_sums_to_100(self):
        from finverse.models.sotp import Segment, analyze
        segs = [
            Segment("A", metric_value=50000, metric_type="ebitda", multiple=15.0),
            Segment("B", metric_value=30000, metric_type="ebitda", multiple=12.0),
            Segment("C", metric_value=20000, metric_type="revenue", multiple=5.0),
        ]
        r = analyze(segs, ticker="TEST", shares_outstanding=10.0)
        assert abs(r.segment_values["pct_of_total"].sum() - 100) < 0.1

    def test_largest_segment(self):
        from finverse.models.sotp import Segment, analyze
        segs = [
            Segment("Small", metric_value=10000, metric_type="ebitda", multiple=10.0),
            Segment("Big",   metric_value=80000, metric_type="ebitda", multiple=20.0),
        ]
        r = analyze(segs, ticker="TEST", shares_outstanding=10.0)
        assert r.largest_segment == "Big"

    def test_conglomerate_discount(self):
        from finverse.models.sotp import Segment, analyze
        segs = [Segment("A", metric_value=50000, metric_type="ebitda", multiple=15.0)]
        r_no_disc = analyze(segs, shares_outstanding=10.0)
        r_disc    = analyze(segs, shares_outstanding=10.0, conglomerate_discount=0.20)
        assert r_disc.total_ev < r_no_disc.total_ev

    def test_dcf_segment(self):
        from finverse.models.sotp import Segment, analyze
        segs = [
            Segment("Main",  metric_value=50000, metric_type="ebitda", multiple=15.0),
            Segment("Other", metric_value=5000,  metric_type="dcf_value", dcf_value=5000),
        ]
        r = analyze(segs, shares_outstanding=10.0)
        assert r.total_ev > 0

    def test_from_ticker(self):
        from finverse.models.sotp import from_ticker
        data = make_ticker_data()
        r = from_ticker(data)
        assert r.total_ev > 0

    def test_upside_calculation(self):
        from finverse.models.sotp import Segment, analyze
        segs = [Segment("A", metric_value=50000, metric_type="ebitda", multiple=15.0)]
        r = analyze(segs, shares_outstanding=10.0, current_price=100.0)
        assert r.upside is not None


class TestEarningsQuality:
    def test_basic(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        assert 0 <= r.overall_score <= 100
        assert r.grade in ["A", "B", "C", "D", "F"]
        assert isinstance(r.flags, list)

    def test_ten_signals(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        assert len(r.signals) == 10

    def test_signal_scores_bounded(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        for sig, val in r.signals.items():
            assert 0 <= val <= 1, f"{sig} = {val} out of bounds"

    def test_grade_matches_score(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        if r.overall_score >= 80:
            assert r.grade == "A"
        elif r.overall_score >= 65:
            assert r.grade == "B"
        elif r.overall_score >= 50:
            assert r.grade == "C"
        elif r.overall_score >= 35:
            assert r.grade == "D"
        else:
            assert r.grade == "F"

    def test_to_df(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        df = r.to_df()
        assert "score" in df.columns
        assert "grade" in df.columns

    def test_signal_scores_df(self):
        from finverse.audit.earnings_quality import score
        data = make_ticker_data()
        r = score(data)
        assert not r.signal_scores.empty
        assert "signal" in r.signal_scores.columns
        assert "score" in r.signal_scores.columns
