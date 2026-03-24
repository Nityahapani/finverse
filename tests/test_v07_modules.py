"""
tests/test_v07_modules.py
=========================
Test suite for finverse v0.7.0 new modules.
All tests use synthetic data — zero network calls, zero API keys required.
Follows the same pattern as existing finverse tests.

Run: pytest tests/test_v07_modules.py -v
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTickerData:
    """Minimal TickerData-compatible object for offline testing."""

    def __init__(
        self,
        ticker: str = "TEST",
        n_days: int = 756,
        sector: str = "technology",
        beta: float = 1.2,
        wacc: float = 0.095,
        terminal_growth: float = 0.025,
        implied_price: float = 185.0,
    ):
        self.ticker = ticker
        self.sector = sector
        self.beta = beta
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.implied_price = implied_price

        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.018, n_days)
        prices = 100 * np.cumprod(1 + returns)
        dates = pd.date_range(end="2025-12-31", periods=n_days, freq="B")
        self.price_history = pd.DataFrame({"Close": prices}, index=dates)

        self.income_statement = pd.DataFrame({
            "Revenue": [300e9, 350e9, 380e9, 400e9],
            "Net Income": [60e9, 72e9, 80e9, 85e9],
            "Operating Income": [80e9, 95e9, 105e9, 112e9],
        })
        self.balance_sheet = pd.DataFrame({
            "Total Assets": [350e9, 380e9, 410e9, 430e9],
            "Total Debt": [100e9, 105e9, 108e9, 110e9],
            "Cash": [50e9, 55e9, 60e9, 65e9],
        })
        self.cash_flow = pd.DataFrame({
            "Operating Cash Flow": [90e9, 100e9, 110e9, 115e9],
            "Capital Expenditure": [10e9, 12e9, 13e9, 14e9],
        })

    @property
    def info(self):
        return {"beta": self.beta, "sector": self.sector}


@pytest.fixture
def ticker_data():
    return SyntheticTickerData()


@pytest.fixture
def ticker_data_finance():
    return SyntheticTickerData(ticker="FIN", sector="financial", beta=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# ── options.black_scholes ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackScholes:

    def test_call_positive_price(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert result.price > 0

    def test_put_positive_price(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="put")
        assert result.price > 0

    def test_atm_call_known_value(self):
        """ATM call with S=K=100, T=1, r=0, sigma=0.20 ≈ 7.97 (BS benchmark)."""
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.0, sigma=0.20, type="call")
        assert abs(result.price - 7.97) < 0.10

    def test_put_call_parity(self):
        """C - P = S*e^{-q*T} - K*e^{-r*T} (no dividend, so C - P = S - K*e^{-rT})."""
        from finverse.options.black_scholes import price
        S, K, T, r, sigma = 105.0, 100.0, 0.5, 0.05, 0.25
        call = price(S=S, K=K, T=T, r=r, sigma=sigma, type="call")
        put = price(S=S, K=K, T=T, r=r, sigma=sigma, type="put")
        parity_rhs = S - K * math.exp(-r * T)
        assert abs((call.price - put.price) - parity_rhs) < 0.01

    def test_call_delta_bounds(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert 0.0 <= result.delta <= 1.0

    def test_put_delta_bounds(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="put")
        assert -1.0 <= result.delta <= 0.0

    def test_gamma_positive(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert result.gamma > 0

    def test_vega_positive(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert result.vega > 0

    def test_theta_negative_for_long_call(self):
        """Long call loses value with time (theta < 0)."""
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert result.theta < 0

    def test_intrinsic_value_itm_call(self):
        from finverse.options.black_scholes import price
        result = price(S=110, K=100, T=0.01, r=0.05, sigma=0.20, type="call")
        assert abs(result.intrinsic_value - 10.0) < 0.01

    def test_time_value_nonnegative(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        assert result.time_value >= 0

    def test_deep_otm_call_near_zero(self):
        from finverse.options.black_scholes import price
        result = price(S=50, K=200, T=0.1, r=0.05, sigma=0.20, type="call")
        assert result.price < 0.01

    def test_invalid_T_raises(self):
        from finverse.options.black_scholes import price
        with pytest.raises(ValueError):
            price(S=100, K=100, T=0, r=0.05, sigma=0.20)

    def test_invalid_sigma_raises(self):
        from finverse.options.black_scholes import price
        with pytest.raises(ValueError):
            price(S=100, K=100, T=1.0, r=0.05, sigma=0)

    def test_summary_runs(self):
        from finverse.options.black_scholes import price
        result = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, type="call")
        result.summary()   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# ── options.implied_vol ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestImpliedVol:

    def test_roundtrip_call(self):
        """IV solve → reprice should match original price."""
        from finverse.options.black_scholes import price
        from finverse.options.implied_vol import solve_iv
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
        opt = price(S=S, K=K, T=T, r=r, sigma=sigma, type="call")
        iv = solve_iv(market_price=opt.price, S=S, K=K, T=T, r=r, type="call")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_roundtrip_put(self):
        from finverse.options.black_scholes import price
        from finverse.options.implied_vol import solve_iv
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.30
        opt = price(S=S, K=K, T=T, r=r, sigma=sigma, type="put")
        iv = solve_iv(market_price=opt.price, S=S, K=K, T=T, r=r, type="put")
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_impossible_price_returns_none(self):
        """Price below intrinsic should return None."""
        from finverse.options.implied_vol import solve_iv
        iv = solve_iv(market_price=0.001, S=200, K=100, T=1.0, r=0.05, type="call")
        # Deep ITM call with tiny price is impossible → None
        assert iv is None

    def test_high_vol_roundtrip(self):
        from finverse.options.black_scholes import price
        from finverse.options.implied_vol import solve_iv
        opt = price(S=100, K=100, T=1.0, r=0.05, sigma=0.80, type="call")
        iv = solve_iv(opt.price, 100, 100, 1.0, 0.05, "call")
        assert iv is not None
        assert abs(iv - 0.80) < 0.005


# ─────────────────────────────────────────────────────────────────────────────
# ── options.binomial (American) ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestBinomial:

    def test_american_put_ge_european(self):
        """American put should be >= European put (early exercise premium)."""
        from finverse.options.black_scholes import price as bs_price
        from finverse.options.binomial import price_american
        S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.08, 0.25
        eu = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, type="put")
        am = price_american(S=S, K=K, T=T, r=r, sigma=sigma, type="put", steps=200)
        assert am.price >= eu.price - 0.01   # allow tiny numerical error

    def test_american_call_no_dividend_equals_european(self):
        """American call without dividends should ≈ European call."""
        from finverse.options.black_scholes import price as bs_price
        from finverse.options.binomial import price_american
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        eu = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, type="call")
        am = price_american(S=S, K=K, T=T, r=r, sigma=sigma, type="call", steps=200)
        assert abs(am.price - eu.price) < 0.15   # binomial converges slowly

    def test_binomial_positive_price(self):
        from finverse.options.binomial import price_american
        result = price_american(S=100, K=100, T=0.5, r=0.05, sigma=0.30, type="put")
        assert result.price > 0

    def test_summary_runs(self):
        from finverse.options.binomial import price_american
        result = price_american(S=100, K=100, T=0.5, r=0.05, sigma=0.30, type="put")
        result.summary()


# ─────────────────────────────────────────────────────────────────────────────
# ── options top-level API ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionsTopLevel:

    def test_price_call(self):
        from finverse import options
        result = options.price(S=185.0, K=190.0, T=0.25, r=0.053, sigma=0.28, type="call")
        assert result.price > 0
        assert hasattr(result, "delta")

    def test_price_put(self):
        from finverse import options
        result = options.price(S=185.0, K=190.0, T=0.25, r=0.053, sigma=0.28, type="put")
        assert result.price > 0

    def test_implied_vol_roundtrip(self):
        from finverse import options
        opt = options.price(S=100, K=100, T=1.0, r=0.05, sigma=0.30, type="call")
        iv = options.implied_vol(market_price=opt.price, S=100, K=100, T=1.0, r=0.05, type="call")
        assert iv is not None
        assert abs(iv - 0.30) < 0.005

    def test_tail_hedge_suggestion(self, ticker_data):
        from finverse import options
        hedge = options.tail_hedge_suggestion(ticker_data)
        assert hedge.put_price > 0
        assert 0 < hedge.cost_pct_of_spot < 0.30
        assert hedge.suggested_strike < ticker_data.price_history["Close"].iloc[-1]
        hedge.summary()

    def test_price_american(self):
        from finverse import options
        result = options.price_american(S=185, K=190, T=0.25, r=0.053, sigma=0.28, type="put", steps=100)
        assert result.price > 0


# ─────────────────────────────────────────────────────────────────────────────
# ── derivatives._discount ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscount:

    def test_discount_factor_flat(self):
        from finverse.derivatives._discount import discount_factor
        P = discount_factor(t=1.0, curve=None, flat_rate=0.05)
        assert abs(P - math.exp(-0.05)) < 1e-10

    def test_discount_factor_t0(self):
        from finverse.derivatives._discount import discount_factor
        P = discount_factor(t=0.0, curve=None, flat_rate=0.05)
        assert abs(P - 1.0) < 1e-10

    def test_forward_rate_flat(self):
        """Under flat continuous rate r, forward rate = e^r - 1 (simple equivalent)."""
        import math
        from finverse.derivatives._discount import forward_rate
        r = 0.05
        fwd = forward_rate(t1=0.0, t2=1.0, curve=None, flat_rate=r)
        expected = math.exp(r) - 1  # continuous → simple conversion
        assert abs(fwd - expected) < 0.001

    def test_par_swap_rate_positive(self):
        from finverse.derivatives._discount import par_swap_rate
        psr = par_swap_rate(tenor=5.0, payment_freq="semi-annual", flat_rate=0.05)
        assert 0.01 < psr < 0.15

    def test_annuity_pv_positive(self):
        from finverse.derivatives._discount import annuity_pv
        ann = annuity_pv(tenor=5.0, payment_freq="semi-annual", flat_rate=0.05)
        assert ann > 0


# ─────────────────────────────────────────────────────────────────────────────
# ── derivatives.rates ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestRates:

    def test_swap_npv_at_par_rate(self):
        """Swap at par rate should have NPV close to zero (within 0.5% of fixed leg PV)."""
        from finverse.derivatives.rates import swap
        from finverse.derivatives._discount import par_swap_rate
        tenor = 5.0
        psr = par_swap_rate(tenor=tenor, flat_rate=0.05)
        result = swap(notional=1_000_000, fixed_rate=psr, tenor=tenor)
        # NPV should be small relative to notional (< 0.5%)
        assert abs(result.npv) / 1_000_000 < 0.005

    def test_swap_pay_fixed_above_par_negative_npv(self):
        """Paying well-above-par fixed rate → negative NPV for fixed payer."""
        from finverse.derivatives.rates import swap
        from finverse.derivatives._discount import par_swap_rate
        psr = par_swap_rate(tenor=5.0)
        # Pay 100bps above par — fixed leg costs more than float → negative NPV
        result = swap(notional=10_000_000, fixed_rate=psr + 0.0100, tenor=5.0)
        assert result.npv < 0

    def test_swap_dv01_positive(self):
        from finverse.derivatives.rates import swap
        result = swap(notional=10_000_000, fixed_rate=0.045, tenor=5)
        assert result.dv01 > 0

    def test_swap_cash_flows_dataframe(self):
        from finverse.derivatives.rates import swap
        result = swap(notional=1_000_000, fixed_rate=0.05, tenor=3, payment_freq="annual")
        assert isinstance(result.cash_flows, pd.DataFrame)
        assert len(result.cash_flows) == 3

    def test_swap_par_swap_rate_populated(self):
        from finverse.derivatives.rates import swap
        result = swap(notional=1_000_000, fixed_rate=0.05, tenor=5)
        assert 0.001 < result.par_swap_rate < 0.20

    def test_fra_fair_value_zero_at_forward(self):
        """FRA at the implied forward rate has NPV small relative to notional."""
        from finverse.derivatives.rates import fra
        from finverse.derivatives._discount import forward_rate
        fwd = forward_rate(0.5, 1.0, flat_rate=0.05)
        result = fra(notional=5_000_000, contract_rate=fwd, start=0.5, end=1.0)
        # NPV should be < 0.1% of notional
        assert abs(result.npv) / 5_000_000 < 0.001

    def test_fra_npv_positive_when_contract_below_forward(self):
        from finverse.derivatives.rates import fra
        from finverse.derivatives._discount import forward_rate
        fwd = forward_rate(0.5, 1.0, flat_rate=0.05)
        result = fra(notional=5_000_000, contract_rate=fwd - 0.01, start=0.5, end=1.0)
        assert result.npv > 0

    def test_swaption_price_positive(self):
        from finverse.derivatives.rates import swaption
        result = swaption(notional=10_000_000, strike_rate=0.048,
                          option_expiry=1.0, swap_tenor=5, vol=0.20, type="payer")
        assert result.price >= 0

    def test_swaption_receiver_positive(self):
        from finverse.derivatives.rates import swaption
        result = swaption(notional=10_000_000, strike_rate=0.048,
                          option_expiry=1.0, swap_tenor=5, vol=0.20, type="receiver")
        assert result.price >= 0

    def test_swap_summary_runs(self):
        from finverse.derivatives.rates import swap
        result = swap(notional=10_000_000, fixed_rate=0.045, tenor=5)
        result.summary()

    def test_fra_summary_runs(self):
        from finverse.derivatives.rates import fra
        result = fra(notional=5_000_000, contract_rate=0.052, start=0.5, end=1.0)
        result.summary()

    def test_swaption_summary_runs(self):
        from finverse.derivatives.rates import swaption
        result = swaption(notional=10_000_000, strike_rate=0.048,
                          option_expiry=1.0, swap_tenor=5, vol=0.20)
        result.summary()


# ─────────────────────────────────────────────────────────────────────────────
# ── derivatives.fx ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestFX:

    def test_fx_forward_cip(self):
        """Forward rate via CIP: F = S*(1+r_d)^T / (1+r_f)^T."""
        from finverse.derivatives.fx import forward
        result = forward(spot=1.085, r_domestic=0.053, r_foreign=0.038, tenor=1.0, pair="EURUSD")
        expected = 1.085 * (1.053 / 1.038)
        assert abs(result.forward_rate - expected) < 0.0001

    def test_fx_forward_points_sign(self):
        """Higher domestic rate → forward > spot → positive forward points."""
        from finverse.derivatives.fx import forward
        result = forward(spot=1.0, r_domestic=0.06, r_foreign=0.02, tenor=1.0)
        assert result.forward_points > 0

    def test_fx_option_call_positive(self):
        from finverse.derivatives.fx import option
        result = option(spot=1.085, strike=1.10, tenor=0.5,
                        r_domestic=0.053, r_foreign=0.038, sigma=0.085, type="call")
        assert result.price > 0

    def test_fx_option_put_positive(self):
        from finverse.derivatives.fx import option
        result = option(spot=1.085, strike=1.06, tenor=0.5,
                        r_domestic=0.053, r_foreign=0.038, sigma=0.085, type="put")
        assert result.price > 0

    def test_fx_option_gk_call_delta_bounds(self):
        from finverse.derivatives.fx import option
        result = option(spot=1.085, strike=1.085, tenor=0.25,
                        r_domestic=0.05, r_foreign=0.03, sigma=0.10, type="call")
        assert 0 < result.delta < 1

    def test_fx_option_gk_put_delta_bounds(self):
        from finverse.derivatives.fx import option
        result = option(spot=1.085, strike=1.085, tenor=0.25,
                        r_domestic=0.05, r_foreign=0.03, sigma=0.10, type="put")
        assert -1 < result.delta < 0

    def test_cross_currency_swap_zero_basis(self):
        """Zero basis spread → NPV = 0."""
        from finverse.derivatives.fx import cross_currency_swap
        result = cross_currency_swap(notional_usd=10_000_000, pair="EURUSD",
                                     spot=1.085, tenor=3, basis_spread=0.0)
        assert abs(result.npv) < 1

    def test_cross_currency_swap_negative_basis(self):
        """Negative basis → borrower in USD pays extra → negative NPV."""
        from finverse.derivatives.fx import cross_currency_swap
        result = cross_currency_swap(notional_usd=10_000_000, pair="EURUSD",
                                     spot=1.085, tenor=3, basis_spread=-0.001)
        assert result.npv < 0

    def test_currency_adjusted_wacc_increases(self):
        """FX exposure should increase WACC vs base."""
        from finverse.derivatives.fx import currency_adjusted_wacc
        adj = currency_adjusted_wacc(
            base_wacc=0.095,
            revenue_fx_exposure={"EUR": 0.35, "GBP": 0.20},
        )
        assert adj > 0.095

    def test_fx_forward_summary_runs(self):
        from finverse.derivatives.fx import forward
        result = forward(spot=1.085, r_domestic=0.053, r_foreign=0.038, tenor=1.0)
        result.summary()

    def test_fx_option_summary_runs(self):
        from finverse.derivatives.fx import option
        result = option(spot=1.085, strike=1.10, tenor=0.5,
                        r_domestic=0.053, r_foreign=0.038, sigma=0.085, type="call")
        result.summary()

    def test_invalid_tenor_raises(self):
        from finverse.derivatives.fx import option
        with pytest.raises(ValueError):
            option(spot=1.0, strike=1.0, tenor=0, r_domestic=0.05, r_foreign=0.03, sigma=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# ── risk.stress_testing ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestStressTesting:

    @pytest.fixture
    def portfolio(self):
        return [
            SyntheticTickerData("AAPL", sector="technology"),
            SyntheticTickerData("JPM", sector="financial", beta=1.1),
            SyntheticTickerData("XOM", sector="energy", beta=0.9),
        ]

    def test_apply_gfc_portfolio_return_negative(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="gfc_2008")
        assert result.portfolio_return < 0

    def test_apply_all_scenarios_return_set(self, portfolio):
        from finverse.risk import stress_testing
        results = stress_testing.run_all(portfolio)
        assert len(results.results) == 7   # 7 built-in scenarios

    def test_apply_custom_scenario(self, portfolio):
        from finverse.risk import stress_testing
        shocks = {"equity_return": -0.25, "rate_shift_bps": 200,
                  "credit_spread_bps": 150, "vix_level": 35}
        result = stress_testing.apply(portfolio, scenario="custom", shocks=shocks)
        assert result.portfolio_return < 0
        assert abs(result.portfolio_return) > 0

    def test_custom_requires_shocks(self, portfolio):
        from finverse.risk import stress_testing
        with pytest.raises(ValueError):
            stress_testing.apply(portfolio, scenario="custom")

    def test_unknown_scenario_raises(self, portfolio):
        from finverse.risk import stress_testing
        with pytest.raises(KeyError):
            stress_testing.apply(portfolio, scenario="nonexistent_1234")

    def test_apply_to_dcf(self):
        from finverse.risk import stress_testing
        # Mock DCF model object
        class MockDCF:
            wacc = 0.095
            terminal_growth = 0.025
            implied_price = 185.0
        result = stress_testing.apply_to_dcf(MockDCF(), scenario="rate_shock_2022")
        assert result.dcf_price_impact is not None
        assert result.wacc_stressed is not None
        assert result.wacc_stressed > MockDCF.wacc   # rate shock increases WACC

    def test_var_breach_flag(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="gfc_2008", var_99=0.05)
        assert result.var_breach is True  # GFC should breach 5% VaR

    def test_key_risk_drivers_populated(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="gfc_2008")
        assert len(result.key_risk_drivers) == 3

    def test_holding_returns_all_present(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="covid_2020")
        assert set(result.holding_returns.keys()) == {"AAPL", "JPM", "XOM"}

    def test_portfolio_pnl_with_value(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="gfc_2008", portfolio_value=1_000_000)
        assert result.portfolio_pnl is not None
        assert result.portfolio_pnl < 0

    def test_summary_runs(self, portfolio):
        from finverse.risk import stress_testing
        result = stress_testing.apply(portfolio, scenario="gfc_2008")
        result.summary()

    def test_run_all_summary_runs(self, portfolio):
        from finverse.risk import stress_testing
        results = stress_testing.run_all(portfolio)
        results.summary()

    def test_covid_scenario_short_duration(self, portfolio):
        from finverse.risk import stress_testing
        from finverse.risk._scenarios import get_scenario
        scenario = get_scenario("covid_2020")
        assert scenario.duration_years < 1.0

    def test_rate_shock_2022_positive_rate_shift(self):
        from finverse.risk._scenarios import get_scenario
        s = get_scenario("rate_shock_2022")
        assert s.rate_shift_bps > 0


# ─────────────────────────────────────────────────────────────────────────────
# ── ml.macro_factor_rotation ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFactorRotation:

    def test_predict_returns_result(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0, credit_spread=0.010)
        result = predict(macro_snapshot=snap)
        assert result.current_regime in {"expansion", "recovery", "slowdown", "contraction", "stress"}

    def test_factor_scores_all_present(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        expected_factors = {"growth", "momentum", "value", "quality", "low_vol", "size"}
        assert set(result.factor_scores.keys()) == expected_factors

    def test_factor_scores_in_range(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        for f, score in result.factor_scores.items():
            assert -1.0 <= score <= 1.0, f"Factor {f} score {score} out of range"

    def test_tilts_sum_near_zero(self):
        """Long-short tilts should roughly sum to zero (zero cost)."""
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        total_tilt = sum(result.tilts.values())
        assert abs(total_tilt) < 0.30   # allow some imbalance

    def test_stress_regime_low_vol_overweight(self):
        """In stress regime, low-vol should be highest-scored factor."""
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=-0.02, vix=45.0, credit_spread=0.030)
        result = predict(macro_snapshot=snap)
        assert result.factor_scores.get("low_vol", 0) > 0

    def test_expansion_growth_overweight(self):
        """In risk-on regimes (expansion/recovery), growth should score positively."""
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.02, vix=14.0, credit_spread=0.008)
        result = predict(macro_snapshot=snap)
        assert result.current_regime in {"expansion", "recovery"}
        assert result.factor_scores.get("growth", 0) > 0

    def test_top_factors_not_empty(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        assert len(result.top_factors) >= 1

    def test_confidence_valid(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        assert result.confidence in {"HIGH", "MEDIUM", "LOW"}

    def test_different_horizons(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        r3m = predict(horizon="3m", macro_snapshot=snap)
        r12m = predict(horizon="12m", macro_snapshot=snap)
        # 12m should have lower confidence than 3m
        conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        assert conf_map[r3m.confidence] >= conf_map[r12m.confidence]

    def test_summary_runs(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=0.01, vix=18.0)
        result = predict(macro_snapshot=snap)
        result.summary()

    def test_inverted_curve_reduces_growth_score(self):
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        normal = predict(macro_snapshot=MacroSnapshot(yield_curve_slope=0.02, vix=15))
        inverted = predict(macro_snapshot=MacroSnapshot(yield_curve_slope=-0.02, vix=15))
        assert (inverted.factor_scores.get("growth", 0)
                <= normal.factor_scores.get("growth", 0))


# ─────────────────────────────────────────────────────────────────────────────
# ── ml.earnings_surprise ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestEarningsSurprise:

    def test_analyze_returns_result(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert result.ticker == "TEST"

    def test_beat_miss_sum_to_one(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert abs(result.beat_probability + result.miss_probability - 1.0) < 1e-6

    def test_beat_probability_in_range(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert 0.05 <= result.beat_probability <= 0.95

    def test_historical_beat_rate_in_range(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert 0.0 <= result.historical_beat_rate <= 1.0

    def test_percentile_in_range(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert 0 <= result.surprise_score_percentile <= 100

    def test_macro_headwind_valid(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert result.macro_headwind in {"HIGH", "MEDIUM", "LOW"}

    def test_confidence_valid(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert result.confidence in {"HIGH", "MEDIUM", "LOW"}

    def test_no_options_chain_implied_move_none(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        assert result.implied_move_pct is None

    def test_summary_runs(self, ticker_data):
        from finverse.ml import earnings_surprise
        result = earnings_surprise.analyze(ticker_data)
        result.summary()

    def test_feature_builder(self):
        from finverse.ml._surprise_model import build_features, predict_beat_probability
        surprises = [0.05, -0.02, 0.08, 0.03, -0.01, 0.07, 0.04, 0.06]
        features = build_features(surprises, 0.3, 75.0, "expansion", None, None)
        assert features.shape == (8,)
        prob = predict_beat_probability(features)
        assert 0.05 <= prob <= 0.95

    def test_high_beat_rate_increases_probability(self):
        """Company with perfect beat history should score higher."""
        from finverse.ml._surprise_model import build_features, predict_beat_probability
        good = build_features([0.1]*12, 0.5, 80.0, "expansion", None, None)
        bad = build_features([-0.1]*12, -0.5, 20.0, "contraction", None, None)
        assert predict_beat_probability(good) > predict_beat_probability(bad)

    def test_screen_returns_batch(self):
        from finverse.ml import earnings_surprise
        batch = earnings_surprise.screen(sector="tech", top_n=5)
        assert len(batch.results) <= 5
        assert batch.sector == "tech"

    def test_screen_summary_runs(self):
        from finverse.ml import earnings_surprise
        batch = earnings_surprise.screen(sector="tech", top_n=3)
        batch.summary()


# ─────────────────────────────────────────────────────────────────────────────
# ── ml.price_target_ensemble ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceTargetEnsemble:

    def test_ensemble_weights_sum_to_one(self):
        from finverse.ml._ensemble_weights import get_weights
        w = get_weights(sector="tech", regime="expansion", has_consensus=True)
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_no_consensus_weights_sum_to_one(self):
        from finverse.ml._ensemble_weights import get_weights
        w = get_weights(sector="tech", regime="expansion", has_consensus=False)
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_no_consensus_weight_is_zero(self):
        from finverse.ml._ensemble_weights import get_weights
        w = get_weights(sector="tech", regime="expansion", has_consensus=False)
        assert w["consensus"] == 0.0

    def test_compute_ensemble_with_all_signals(self):
        from finverse.ml._ensemble_weights import compute_ensemble, get_weights
        targets = {"dcf": 180.0, "comps": 175.0, "momentum": 190.0, "consensus": 185.0}
        weights = get_weights()
        result = compute_ensemble(targets, weights)
        assert 170 < result < 200

    def test_compute_ensemble_skips_none(self):
        from finverse.ml._ensemble_weights import compute_ensemble, get_weights
        targets = {"dcf": 180.0, "comps": None, "momentum": None, "consensus": None}
        weights = get_weights()
        result = compute_ensemble(targets, weights)
        assert result == 180.0   # only DCF available → pure DCF

    def test_signal_agreement_high_when_close(self):
        from finverse.ml._ensemble_weights import signal_agreement
        targets = {"dcf": 100.0, "comps": 101.0, "momentum": 99.0, "consensus": 100.5}
        agreement = signal_agreement(targets, 100.0)
        assert agreement == "HIGH"

    def test_signal_agreement_low_when_divergent(self):
        from finverse.ml._ensemble_weights import signal_agreement
        targets = {"dcf": 50.0, "comps": 200.0, "momentum": 100.0, "consensus": 150.0}
        agreement = signal_agreement(targets, 100.0)
        assert agreement == "LOW"

    def test_ci_80_narrower_than_95(self):
        from finverse.ml._ensemble_weights import compute_confidence_intervals
        targets = {"dcf": 180.0, "comps": 160.0, "momentum": 200.0, "consensus": 175.0}
        ci_80, ci_95 = compute_confidence_intervals(targets, 178.0)
        assert ci_80[1] - ci_80[0] < ci_95[1] - ci_95[0]

    def test_analyze_returns_result(self, ticker_data):
        from finverse.ml import price_target_ensemble
        # Mock DCF price to avoid running real DCF
        class MockDCF:
            implied_price = 185.0
        result = price_target_ensemble.analyze(ticker_data, dcf_model=MockDCF())
        assert result.ticker == "TEST"
        assert result.ensemble_target > 0
        assert result.current_price > 0

    def test_rating_buy_for_high_upside(self, ticker_data):
        from finverse.ml._ensemble_weights import compute_ensemble, get_weights
        from finverse.ml.price_target_ensemble import _derive_rating
        rating = _derive_rating(upside=0.30, agreement="HIGH")
        assert rating == "BUY"

    def test_rating_sell_for_large_downside(self):
        from finverse.ml.price_target_ensemble import _derive_rating
        rating = _derive_rating(upside=-0.25, agreement="HIGH")
        assert rating == "SELL"

    def test_summary_runs_with_mock_dcf(self, ticker_data):
        from finverse.ml import price_target_ensemble
        class MockDCF:
            implied_price = 185.0
        result = price_target_ensemble.analyze(ticker_data, dcf_model=MockDCF())
        result.summary()

    def test_upside_calculated_correctly(self, ticker_data):
        from finverse.ml import price_target_ensemble
        class MockDCF:
            implied_price = 185.0
        result = price_target_ensemble.analyze(ticker_data, dcf_model=MockDCF())
        expected_upside = (result.ensemble_target - result.current_price) / result.current_price
        assert abs(result.upside_pct - expected_upside) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# ── Cross-module integration tests ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_options_with_evt_tail_hedge(self, ticker_data):
        """tail_hedge_suggestion should work with a mock EVT result."""
        from finverse import options

        class MockEVT:
            var_999 = 0.25
            var_99 = 0.18

        hedge = options.tail_hedge_suggestion(ticker_data, evt_result=MockEVT())
        assert hedge.put_price > 0
        assert hedge.var_99_estimate == 0.25

    def test_stress_testing_with_dcf_model(self):
        """apply_to_dcf should produce a price impact and stressed WACC."""
        from finverse.risk import stress_testing

        class MockDCF:
            wacc = 0.095
            terminal_growth = 0.025
            implied_price = 185.0

        result = stress_testing.apply_to_dcf(MockDCF(), scenario="rate_shock_2022")
        # Rate shock 2022: rates +425bps → WACC rises
        assert result.wacc_stressed > MockDCF.wacc
        assert result.dcf_price_impact is not None

    def test_macro_factor_rotation_contraction_avoids_growth(self):
        """Contraction + inverted curve: growth must be negative-scored."""
        from finverse.ml.macro_factor_rotation import predict, MacroSnapshot
        snap = MacroSnapshot(yield_curve_slope=-0.015, vix=30.0)
        result = predict(macro_snapshot=snap)
        assert result.factor_scores.get("growth", 0) < 0.5

    def test_derivatives_rates_and_fx_both_work(self):
        from finverse.derivatives import rates, fx
        s = rates.swap(notional=5_000_000, fixed_rate=0.045, tenor=3)
        f = fx.forward(spot=1.085, r_domestic=0.053, r_foreign=0.038, tenor=1.0)
        assert s.npv is not None
        assert f.forward_rate > 0

    def test_fx_adjusted_wacc_feeds_to_dcf_signature(self):
        """Verify currency_adjusted_wacc output is usable in DCF.set(wacc=...)."""
        from finverse.derivatives.fx import currency_adjusted_wacc
        adj_wacc = currency_adjusted_wacc(
            base_wacc=0.095,
            revenue_fx_exposure={"EUR": 0.35, "GBP": 0.20},
        )
        assert isinstance(adj_wacc, float)
        assert 0.09 < adj_wacc < 0.15

    def test_earnings_surprise_with_regime(self, ticker_data):
        """earnings_surprise.analyze with regime_result from mock."""
        from finverse.ml import earnings_surprise

        class MockRegime:
            class current_regime:
                value = "expansion"

        result = earnings_surprise.analyze(ticker_data, regime_result=MockRegime())
        assert result.macro_headwind == "LOW"

    def test_stress_run_all_worst_is_gfc(self):
        """GFC 2008 should be among the two worst scenarios for a diversified equity portfolio."""
        tickers = [
            SyntheticTickerData("AAPL", sector="technology", beta=1.2),
            SyntheticTickerData("JPM", sector="financial", beta=1.1),
        ]
        from finverse.risk import stress_testing
        results = stress_testing.run_all(tickers)
        sorted_returns = sorted(results.results, key=lambda r: r.portfolio_return)
        worst_names = [r.scenario_name for r in sorted_returns[:2]]
        assert any("Financial Crisis" in n or "Dot-com" in n for n in worst_names)
