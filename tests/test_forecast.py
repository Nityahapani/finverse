"""Tests for ML forecast module — no network calls."""
import pytest
import numpy as np
import pandas as pd
from finverse.ml.forecast import revenue, margins, wacc, ForecastResult
from tests.test_dcf import make_mock_data


class TestForecastRevenue:
    def test_from_series(self):
        series = pd.Series([260.0, 274.0, 365.0, 394.0, 383.0])
        result = revenue(series, n_years=3)
        assert isinstance(result, ForecastResult)
        assert len(result.point) == 3
        assert len(result.lower) == 3
        assert len(result.upper) == 3

    def test_confidence_intervals_ordered(self):
        series = pd.Series([260.0, 274.0, 365.0, 394.0, 383.0])
        result = revenue(series, n_years=3)
        for lo, pt, hi in zip(result.lower, result.point, result.upper):
            assert lo <= pt <= hi

    def test_from_ticker_data(self):
        data = make_mock_data()
        result = revenue(data, n_years=3)
        assert isinstance(result, ForecastResult)
        assert result.metric == "Revenue ($B)"

    def test_cagr_calculated(self):
        series = pd.Series([100.0, 110.0, 121.0, 133.0, 146.0])
        result = revenue(series, n_years=3)
        assert isinstance(result.cagr, float)

    def test_to_df(self):
        series = pd.Series([260.0, 274.0, 365.0, 394.0, 383.0])
        result = revenue(series, n_years=3)
        df = result.to_df()
        assert isinstance(df, pd.DataFrame)
        assert "forecast" in df.columns
        assert "lower_80" in df.columns
        assert "upper_80" in df.columns

    def test_short_series_fallback(self):
        series = pd.Series([100.0, 110.0])
        result = revenue(series, n_years=3)
        assert len(result.point) == 3

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            revenue({"bad": "input"})


class TestForecastWACC:
    def test_returns_dict(self):
        data = make_mock_data()
        result = wacc(data)
        assert isinstance(result, dict)
        assert "wacc" in result
        assert "cost_of_equity" in result
        assert "cost_of_debt" in result
        assert "weights" in result

    def test_wacc_in_reasonable_range(self):
        data = make_mock_data()
        result = wacc(data)
        assert 0.04 <= result["wacc"] <= 0.20

    def test_override_risk_free_rate(self):
        data = make_mock_data()
        result = wacc(data, risk_free_rate=0.05)
        assert result["risk_free_rate"] == 0.05
