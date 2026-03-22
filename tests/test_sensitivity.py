"""Tests for sensitivity analysis."""
import pytest
import pandas as pd
from finverse.models.dcf import DCF
from finverse.analysis.sensitivity import sensitivity
from finverse.analysis.scenarios import scenarios


def make_model():
    return DCF.manual(
        base_revenue=383.0,
        shares_outstanding=15.4,
        net_debt=50.0,
        current_price=185.0,
    )


class TestSensitivity:
    def test_returns_dataframe(self):
        model = make_model()
        table = sensitivity(model, rows="wacc", cols="terminal_growth", n=3)
        assert isinstance(table, pd.DataFrame)
        assert table.shape == (3, 3)

    def test_values_positive(self):
        model = make_model()
        table = sensitivity(model, rows="wacc", cols="terminal_growth", n=3)
        assert (table.values > 0).all()

    def test_higher_wacc_lower_price(self):
        model = make_model()
        table = sensitivity(model, rows="wacc", cols="terminal_growth", n=3)
        assert table.iloc[0].mean() > table.iloc[-1].mean()

    def test_custom_range(self):
        model = make_model()
        table = sensitivity(
            model, rows="wacc", cols="terminal_growth",
            row_range=(0.08, 0.12), col_range=(0.01, 0.03), n=3
        )
        assert table.shape == (3, 3)


class TestScenarios:
    def test_returns_dataframe(self):
        model = make_model()
        df = scenarios(
            model,
            bull={"wacc": 0.08, "ebitda_margin": 0.36},
            base={"wacc": 0.095, "ebitda_margin": 0.32},
            bear={"wacc": 0.12, "ebitda_margin": 0.26},
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_bull_gt_bear(self):
        model = make_model()
        df = scenarios(
            model,
            bull={"wacc": 0.08, "ebitda_margin": 0.36},
            base={"wacc": 0.095, "ebitda_margin": 0.32},
            bear={"wacc": 0.12, "ebitda_margin": 0.26},
        )
        bull_price = float(df.loc["Bull", "Implied price"].replace("$", ""))
        bear_price = float(df.loc["Bear", "Implied price"].replace("$", ""))
        assert bull_price > bear_price
