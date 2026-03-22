"""
Tests for Phase 5 — no network calls required.
Covers: GARCH, cross_sectional, HRP, shrinkage, Merton, Altman,
        real_options, APV, LM dictionary, Benford, VAR, Nelson-Siegel.
"""
import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_ticker_data


class TestGARCH:
    def test_garch11(self):
        from finverse.ml.garch import fit
        data = make_ticker_data()
        r = fit(data, model_type="GARCH(1,1)", forecast_horizon=5)
        assert 0 < r.persistence < 1
        assert r.current_vol > 0
        assert len(r.forecast) == 5
        assert r.gamma is None

    def test_gjr_garch(self):
        from finverse.ml.garch import fit
        data = make_ticker_data()
        r = fit(data, model_type="GJR-GARCH")
        assert r.gamma is not None
        assert r.model == "GJR-GARCH"

    def test_egarch(self):
        from finverse.ml.garch import fit
        data = make_ticker_data()
        r = fit(data, model_type="EGARCH")
        assert r.model == "EGARCH"
        assert isinstance(r.log_likelihood, float)

    def test_from_series(self):
        from finverse.ml.garch import fit
        returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
        r = fit(returns, model_type="GARCH(1,1)")
        assert r.persistence > 0

    def test_compare(self):
        from finverse.ml.garch import compare
        data = make_ticker_data()
        df = compare(data)
        assert len(df) == 3
        assert "aic" in df.columns
        assert "persistence" in df.columns

    def test_to_df(self):
        from finverse.ml.garch import fit
        data = make_ticker_data()
        r = fit(data, model_type="GARCH(1,1)")
        df = r.to_df()
        assert "conditional_vol" in df.columns


class TestCrossSectional:
    def test_basic_forecast(self):
        from finverse.ml.cross_sectional import forecast
        data = make_ticker_data()
        r = forecast(data, target="revenue_growth")
        assert isinstance(r.forecast, float)
        lo, hi = r.confidence_interval
        assert lo <= r.forecast <= hi
        assert 0 <= r.percentile_rank <= 1
        assert r.n_companies_trained > 0

    def test_ebitda_margin(self):
        from finverse.ml.cross_sectional import forecast
        data = make_ticker_data()
        r = forecast(data, target="ebitda_margin")
        assert isinstance(r.forecast, float)

    def test_to_df(self):
        from finverse.ml.cross_sectional import forecast
        data = make_ticker_data()
        r = forecast(data)
        df = r.to_df()
        assert "forecast" in df.columns


class TestHRP:
    def test_weights_sum_to_one(self):
        from finverse.portfolio.hrp import optimize
        tickers = ["AAPL", "MSFT", "GOOGL"]
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(tickers)]
        r = optimize(data_list)
        assert abs(r.weights.sum() - 1.0) < 0.001

    def test_weights_non_negative(self):
        from finverse.portfolio.hrp import optimize
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = optimize(data_list)
        assert all(r.weights >= 0)

    def test_cluster_order(self):
        from finverse.portfolio.hrp import optimize
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = optimize(data_list)
        assert isinstance(r.cluster_order, list)
        assert len(r.cluster_order) == 3

    def test_compare_to_equal_weight(self):
        from finverse.portfolio.hrp import optimize
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = optimize(data_list)
        df = r.compare_to_equal_weight()
        assert "HRP" in df.columns
        assert "Equal weight" in df.columns

    def test_correlation_matrix(self):
        from finverse.portfolio.hrp import optimize
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = optimize(data_list)
        assert not r.correlation_matrix.empty
        assert r.correlation_matrix.shape == (3, 3)


class TestShrinkage:
    def test_basic(self):
        from finverse.portfolio.shrinkage import shrink
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = shrink(data_list)
        assert r.shrunk_cov.shape == (3, 3)
        assert 0 <= r.shrinkage_coefficient <= 1

    def test_to_df(self):
        from finverse.portfolio.shrinkage import shrink
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = shrink(data_list)
        df = r.to_df()
        assert df.shape == (3, 3)

    def test_correlation(self):
        from finverse.portfolio.shrinkage import shrink
        data_list = [make_ticker_data(t, i*10) for i, t in enumerate(["AAPL", "MSFT", "GOOGL"])]
        r = shrink(data_list)
        corr = r.correlation()
        assert corr.shape == (3, 3)
        assert all(abs(corr.values.diagonal() - 1.0) < 1e-6)


class TestMerton:
    def test_basic(self):
        from finverse.credit.merton import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert r.distance_to_default is not None
        assert 0 <= r.prob_default_1y <= 1
        assert r.prob_default_5y >= r.prob_default_1y
        assert r.implied_spread >= 0
        assert isinstance(r.rating_equivalent, str)

    def test_garch_vol_override(self):
        from finverse.credit.merton import analyze
        data = make_ticker_data()
        r = analyze(data, garch_vol=0.35)
        assert r.equity_vol == 0.35

    def test_higher_debt_lower_dd(self):
        from finverse.credit.merton import analyze
        from finverse.pull.ticker import TickerData
        d_lo = make_ticker_data()
        d_lo.info["totalDebt"] = 10e9
        d_hi = make_ticker_data()
        d_hi.info["totalDebt"] = 500e9
        r_lo = analyze(d_lo)
        r_hi = analyze(d_hi)
        assert r_lo.distance_to_default >= r_hi.distance_to_default

    def test_to_df(self):
        from finverse.credit.merton import analyze
        r = analyze(make_ticker_data())
        df = r.to_df()
        assert "distance_to_default" in df.columns


class TestAltman:
    def test_basic(self):
        from finverse.credit.altman import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert isinstance(r.score, float)
        assert r.zone in ["safe", "grey", "distress"]
        assert r.model in ["Z-Score", "Z'-Score", "Z''-Score"]

    def test_safe_company(self):
        from finverse.credit.altman import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert r.zone in ["safe", "grey"]

    def test_to_df(self):
        from finverse.credit.altman import analyze
        r = analyze(make_ticker_data())
        df = r.to_df()
        assert "score" in df.columns
        assert "zone" in df.columns


class TestRealOptions:
    def test_expand(self):
        from finverse.valuation.real_options import expand
        r = expand(project_value=500, expansion_cost=200, sigma=0.30, time_to_expiry=3.0)
        assert r.option_value > 0
        assert r.expanded_npv > r.dcf_npv
        assert 0 <= r.delta <= 1

    def test_abandon(self):
        from finverse.valuation.real_options import abandon
        r = abandon(project_value=300, salvage_value=150, sigma=0.35, time_to_expiry=2.0)
        assert r.option_value > 0
        assert -1 <= r.delta <= 0

    def test_defer(self):
        from finverse.valuation.real_options import defer
        r = defer(project_value=400, investment_cost=350, sigma=0.25, time_to_expiry=2.0)
        assert r.option_value > 0

    def test_expand_increases_with_sigma(self):
        from finverse.valuation.real_options import expand
        lo = expand(500, 200, sigma=0.10, time_to_expiry=3.0)
        hi = expand(500, 200, sigma=0.40, time_to_expiry=3.0)
        assert hi.option_value > lo.option_value

    def test_sensitivity_grid(self):
        from finverse.valuation.real_options import sensitivity_grid, expand
        grid = sensitivity_grid(500, 200, expand, n=4)
        assert grid.shape == (4, 4)

    def test_to_df(self):
        from finverse.valuation.real_options import expand
        r = expand(500, 200, 0.30, 3.0)
        df = r.to_df()
        assert "option_value" in df.columns


class TestAPV:
    def test_basic(self):
        from finverse.valuation.apv import analyze
        r = analyze(base_revenue=383.0, debt=100.0)
        assert r.apv > 0
        assert r.pv_tax_shield >= 0
        assert r.pv_distress_costs >= 0

    def test_from_ticker(self):
        from finverse.valuation.apv import analyze
        data = make_ticker_data()
        r = analyze(data)
        assert r.apv > 0

    def test_higher_debt_higher_tax_shield(self):
        from finverse.valuation.apv import analyze
        lo = analyze(base_revenue=100.0, debt=10.0)
        hi = analyze(base_revenue=100.0, debt=100.0)
        assert hi.pv_tax_shield > lo.pv_tax_shield

    def test_to_df(self):
        from finverse.valuation.apv import analyze
        r = analyze(base_revenue=100.0)
        df = r.to_df()
        assert not df.empty


class TestLoughranMcDonald:
    def test_positive_text(self):
        from finverse.audit.loughran_mcdonald import analyze
        r = analyze("strong growth exceeded guidance confident momentum positive profitable")
        assert r.net_sentiment > 0
        assert r.positive_score > 0

    def test_negative_text(self):
        from finverse.audit.loughran_mcdonald import analyze
        r = analyze("decline loss uncertain challenged risk adverse headwind lawsuit")
        assert r.net_sentiment < 0
        assert r.negative_score > 0

    def test_pos_greater_than_neg(self):
        from finverse.audit.loughran_mcdonald import analyze
        pos = analyze("strong growth exceeded guidance confident momentum positive")
        neg = analyze("decline loss uncertain challenged risk adverse headwind lawsuit")
        assert pos.net_sentiment > neg.net_sentiment

    def test_tone_labels(self):
        from finverse.audit.loughran_mcdonald import analyze
        r = analyze("neutral text about the company operations")
        assert r.tone_label in ["strongly positive", "mildly positive", "neutral",
                                "mildly negative", "strongly negative"]

    def test_to_df(self):
        from finverse.audit.loughran_mcdonald import analyze
        r = analyze("some financial text", source="test")
        df = r.to_df()
        assert "net_sentiment" in df.columns

    def test_compare_filings(self):
        from finverse.audit.loughran_mcdonald import compare_filings
        texts = {
            "2022": "strong growth exceeded guidance confident momentum",
            "2023": "decline loss uncertain challenged risk adverse",
        }
        df = compare_filings(texts)
        assert len(df) == 2


class TestBenford:
    def test_natural_numbers_conform(self):
        from finverse.audit.benford import test as benford_test
        np.random.seed(42)
        natural = np.array([10 ** np.random.uniform(0, 4) for _ in range(500)])
        r = benford_test(natural, source="natural")
        assert r.n_observations == 500
        assert r.conformity in ["close", "acceptable", "nonconforming", "suspicious"]
        assert r.mad < 0.05

    def test_uniform_nonconforming(self):
        from finverse.audit.benford import test as benford_test
        np.random.seed(42)
        natural = np.array([10 ** np.random.uniform(0, 4) for _ in range(500)])
        uniform = np.array([float(f"{np.random.randint(1,10)}{np.random.randint(1000,9999)}")
                           for _ in range(300)])
        r_nat = benford_test(natural)
        r_uni = benford_test(uniform)
        assert r_uni.mad > r_nat.mad

    def test_test_financials(self):
        from finverse.audit.benford import test_financials
        data = make_ticker_data()
        r = test_financials(data)
        assert r.n_observations > 0

    def test_to_df(self):
        from finverse.audit.benford import test as benford_test
        r = benford_test(np.array([1.1, 2.2, 3.3, 1.5, 2.8, 1.9, 3.1, 2.4, 1.7] * 20))
        df = r.to_df()
        assert "observed" in df.columns
        assert "expected" in df.columns


class TestVAR:
    def setup_method(self):
        np.random.seed(42)
        n = 80
        self.data = pd.DataFrame({
            "GDP": np.cumsum(np.random.normal(0.005, 0.01, n)),
            "UNRATE": np.cumsum(np.random.normal(0, 0.005, n)) + 4.5,
            "FEDFUNDS": np.cumsum(np.random.normal(0, 0.02, n)) + 3.5,
        }, index=pd.date_range("2000", periods=n, freq="QE"))

    def test_basic_fit(self):
        from finverse.macro.var_model import fit
        r = fit(self.data, n_lags=2, forecast_horizon=4, irf_horizon=10)
        assert r.n_lags == 2
        assert len(r.variables) == 3
        assert r.forecast.shape == (4, 3)

    def test_irf_length(self):
        from finverse.macro.var_model import fit
        r = fit(self.data, n_lags=1, irf_horizon=10)
        irf = r.irf("FEDFUNDS", "GDP")
        assert len(irf) == 11

    def test_granger_causality(self):
        from finverse.macro.var_model import fit
        r = fit(self.data, n_lags=2)
        assert not r.granger_causality.empty
        assert r.granger_causality.shape == (3, 3)
        assert all(0 <= v <= 1 for v in r.granger_causality.values.flatten())

    def test_aic_bic(self):
        from finverse.macro.var_model import fit
        r = fit(self.data, n_lags=2)
        assert isinstance(r.aic, float)
        assert isinstance(r.bic, float)

    def test_select_lag_order(self):
        from finverse.macro.var_model import select_lag_order
        df = select_lag_order(self.data, max_lags=3)
        assert "AIC" in df.columns
        assert "BIC" in df.columns
        assert len(df) == 3


class TestNelsonSiegel:
    def test_basic_fit(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields)
        assert r.fit_error < 1.0
        assert r.level > 0

    def test_yield_at(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields)
        assert 0 < r.yield_at(5.0) < 0.15
        assert 0 < r.yield_at(10.0) < 0.15

    def test_forward_rate(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields)
        fwd = r.forward_rate(5.0)
        assert fwd > 0

    def test_curve(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields)
        curve = r.curve()
        assert len(curve) == 10
        assert all(v > 0 for v in curve)

    def test_svensson(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields, model="Svensson")
        assert r.beta3 is not None
        assert r.lambda2 is not None

    def test_us_curve(self):
        from finverse.macro.nelson_siegel import us_curve
        r = us_curve()
        assert r.yield_at(10.0) > 0

    def test_to_df(self):
        from finverse.macro.nelson_siegel import fit
        mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
        yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
        r = fit(mats, yields)
        df = r.to_df()
        assert not df.empty
