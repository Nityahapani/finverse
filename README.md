# finverse

**Institutional-grade financial modeling for Python.**

[![PyPI version](https://img.shields.io/pypi/v/finverse?color=blue&label=PyPI)](https://pypi.org/project/finverse/)
[![Python](https://img.shields.io/pypi/pyversions/finverse)](https://pypi.org/project/finverse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/finverse?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/finverse)
[![CI](https://github.com/Nityahapani/finverse/actions/workflows/ci.yml/badge.svg)](https://github.com/Nityahapani/finverse/actions)

DCF · LBO · Credit · Options · Bonds · Derivatives · ML Forecasting · Portfolio · Macro · Audit

**No Bloomberg. No API keys. Pure Python.**

---

```bash
pip install finverse
```

```python
from finverse import pull, DCF, sensitivity, regime_dcf, options
from finverse.ml import garch, cross_sectional, earnings_surprise, macro_factor_rotation
from finverse.credit import merton, altman
from finverse.risk import evt, kelly, stress_testing
from finverse.portfolio import hrp, black_litterman
from finverse.audit import earnings_quality, manipulation
from finverse.models.options import call, put, implied_vol
from finverse.models.bonds import price as bond_price
from finverse.macro import nelson_siegel
from finverse.derivatives import rates, fx

data = pull.ticker("AAPL")

# ML-powered DCF
model = DCF(data).use_ml_forecast()
model.run().summary()
sensitivity(model, rows="wacc", cols="terminal_growth")

# Regime-conditional DCF — adjusts every assumption to the macro regime
regime_dcf(data).summary()

# GARCH vol → Merton credit model
vol = garch.fit(data, model_type="GJR-GARCH")
merton.analyze(data, garch_vol=vol.current_vol).summary()
altman.analyze(data).summary()

# Manipulation fingerprint — 40+ accounting signals
manipulation.fingerprint(data).summary()

# Tail risk and Kelly sizing
evt.analyze(data).summary()
kelly.from_distribution(data).summary()

# Portfolio with analyst views
stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]
black_litterman.optimize(stocks, views=[
    BLView(["AAPL"], [1.0], expected_ret=0.15, confidence=0.8),
]).summary()

# Options — full pricing layer
from finverse import options
opt = options.price(S=185, K=190, T=0.25, r=0.053, sigma=0.28, type="call")
opt.summary()                                        # price, all Greeks
iv = options.implied_vol(market_price=9.03, S=185, K=190, T=0.25, r=0.053)
chain = options.chain(data)                          # live chain via yfinance
chain.vol_surface().plot()                           # implied vol surface
options.scan_arbitrage(chain).summary()              # put-call parity violations

# Derivatives
curve = nelson_siegel.us_curve()
swap = rates.swap(notional=10_000_000, fixed_rate=0.045, tenor=5, curve=curve)
swap.summary()
fwd = fx.forward(spot=1.085, r_domestic=0.053, r_foreign=0.038, tenor=1.0)
fwd.summary()

# Earnings beat probability before earnings
result = earnings_surprise.analyze(data)
result.summary()

# Factor rotation — which factors to overweight given the macro regime
rotation = macro_factor_rotation.predict()
rotation.summary()

# Stress testing — historical shock scenarios
stress_testing.apply(stocks, scenario="gfc_2008").summary()
stress_testing.run_all(stocks).summary()
stress_testing.apply_to_dcf(model, scenario="rate_shock_2022").summary()

# Options and bonds
call(spot=185, strike=190, sigma=0.28, maturity=0.25).summary()
bond_price(face=1000, coupon_rate=0.05, ytm=0.06, maturity=10).summary()

# Yield curve
nelson_siegel.us_curve().summary()
```

---

## What's inside

**58 modules** across 13 layers. Everything runs fully offline except `pull.*` data functions.

---

### Data — `finverse.pull`

| Function | Source | Key needed |
|---|---|---|
| `pull.ticker("AAPL")` | yfinance — financials, price history | None |
| `pull.edgar("AAPL", "10-K")` | SEC EDGAR — filings, XBRL facts | None |
| `pull.fred("GDP", "UNRATE")` | Federal Reserve macro data | Free at fred.stlouisfed.org |
| `pull.macro_snapshot()` | Rates, inflation, VIX, credit spreads | Free |

---

### Valuation — `finverse.models`

| Model | How to use | What it does |
|---|---|---|
| `DCF` | `DCF(data).use_ml_forecast().run()` | Discounted cash flow with ML-estimated assumptions |
| `LBO` | `LBO(LBOAssumptions(...)).run()` | Full buyout: debt schedule, IRR, MoM |
| `ThreeStatement` | `ThreeStatement(data).run()` | Linked IS / BS / CF |
| `comps` | `comps(data, peers=["MSFT","GOOGL"])` | Comparable company analysis, implied price range |
| `regime_dcf` | `regime_dcf(data)` | DCF with per-regime assumptions, probability-weighted price |
| `synthetic_peers` | `build_peers(data, {"software":0.6,"hardware":0.4})` | Peer multiples for companies with no clean peer set |
| `gordon` | `gordon(dividend=1.84, growth_rate=0.04, ke=0.085)` | Gordon Growth Model |
| `h_model` | `h_model(dividend=1.84, high_growth=0.15, ...)` | H-Model DDM |
| `multistage` | `multistage(dividend=1.84, stage1_growth=0.15, ...)` | Multistage DDM |
| `sotp` | `sotp([Segment(...), Segment(...)])` | Sum of the Parts |
| `apv` | `apv.analyze(data)` | Adjusted Present Value (Modigliani-Miller) |
| `real_options` | `real_options.expand(500, 200, 0.30, 3.0)` | Expand, abandon, defer (Black-Scholes) |
| `macro.nowcast` | `macro.nowcast()` | GDP nowcast, recession probability, regime |
| `options` | `call(185, 190, 0.28, 0.25)` | European call/put, all 5 Greeks, IV solver |
| `bonds` | `bond_price(1000, 0.05, 0.06, 10)` | Price, YTM, duration, convexity, DV01 |

```python
from finverse import pull, DCF, sensitivity, scenarios, regime_dcf
from finverse.models.lbo import LBO, LBOAssumptions
from finverse.models.ddm import gordon, h_model, multistage
from finverse.models.sotp import Segment, analyze as sotp
from finverse.models.synthetic_peers import build_peers
from finverse.models.options import call, put, implied_vol, vol_surface
from finverse.models.bonds import price as bond_price, ytm_from_price

data = pull.ticker("AAPL")

# --- DCF ---
model = DCF(data).use_ml_forecast()
model.run().summary()
sensitivity(model, rows="wacc", cols="terminal_growth")
scenarios(model,
    bull={"wacc": 0.085, "revenue_growth": 0.12},
    base={"wacc": 0.095, "revenue_growth": 0.08},
    bear={"wacc": 0.115, "revenue_growth": 0.03},
)

# --- Regime-Conditional DCF ---
result = regime_dcf(data)
result.summary()
print(f"Current regime:    {result.current_regime}")
print(f"Regime-weighted:   ${result.weighted_price:.2f}")
print(f"Static DCF:        ${result.static_price:.2f}")
print(f"Regime adjustment: {result.regime_discount_pct:+.1%}")

# --- LBO ---
result = LBO(LBOAssumptions(
    entry_ebitda=150, entry_ev_ebitda=10.0,
    equity_pct=0.40, hold_years=5, exit_ev_ebitda=12.0,
)).run()
print(f"IRR: {result.irr:.1%}  MoM: {result.mom:.2f}x")

# --- Options (models layer) ---
c = call(spot=185, strike=190, sigma=0.28, maturity=0.25)
c.summary()
p = put(spot=185, strike=190, sigma=0.28, maturity=0.25)
iv = implied_vol(market_price=9.03, spot=185, strike=190, maturity=0.25)

# --- Bonds ---
b = bond_price(face=1000, coupon_rate=0.05, ytm=0.06, maturity=10)
b.summary()
b2 = ytm_from_price(market_price=950, coupon_rate=0.05, maturity=10)
print(f"YTM: {b2.ytm:.4%}")
```

---

### Options — `finverse.options` ✦ new in v0.7.0

Full options pricing library: Black-Scholes, American binomial tree, implied vol surface, live chain integration, and put-call parity arbitrage scanner.

| Function | What it does |
|---|---|
| `options.price(S, K, T, r, sigma, type)` | European call/put — price, intrinsic, time value, all 5 Greeks |
| `options.price_american(...)` | American option via CRR binomial tree (500 steps) |
| `options.implied_vol(market_price, ...)` | Implied vol via Brent's method — always converges |
| `options.chain(data)` | Live options chain from yfinance |
| `options.scan_arbitrage(chain)` | Put-call parity violation scanner |
| `options.tail_hedge_suggestion(data, evt_result)` | EVT-linked put hedge sizing |

```python
from finverse import options
from finverse.risk import evt

# Price a European option
opt = options.price(S=185.0, K=190.0, T=0.25, r=0.053, sigma=0.28, type="call")
opt.summary()
# → price, intrinsic, time value, delta, gamma, theta, vega, rho, breakeven

# American put via binomial tree
opt = options.price_american(S=185, K=190, T=0.25, r=0.053, sigma=0.28,
                              type="put", steps=500)
opt.summary()

# Implied volatility from market price
iv = options.implied_vol(market_price=7.50, S=185, K=190, T=0.25, r=0.053)
print(f"IV: {iv:.2%}")

# Live chain + vol surface (requires yfinance)
chain = options.chain(data)
chain.summary()
chain.vol_surface().plot()

# Arbitrage scanner
options.scan_arbitrage(chain).summary()

# Tail hedge linked to EVT tail risk
tail = evt.analyze(data)
hedge = options.tail_hedge_suggestion(data, evt_result=tail)
hedge.summary()
# → suggested put strike, cost as % of spot, delta, EVT VaR used
```

---

### Derivatives — `finverse.derivatives` ✦ new in v0.7.0

Interest rate and FX derivatives pricing. Integrates with `macro.nelson_siegel` for discount factors and with `DCF` for currency-adjusted WACC.

| Submodule | What it covers |
|---|---|
| `derivatives.rates` | Fixed/float IR swaps, FRAs, European swaptions (Black's model) |
| `derivatives.fx` | FX forwards (CIP), cross-currency basis swaps, FX options (Garman-Kohlhagen) |

```python
from finverse.derivatives import rates, fx
from finverse.macro import nelson_siegel

curve = nelson_siegel.us_curve()

# Interest rate swap — NPV, par rate, DV01, cash flow schedule
swap = rates.swap(notional=10_000_000, fixed_rate=0.045, tenor=5,
                  payment_freq="semi-annual", curve=curve)
swap.summary()

# Forward Rate Agreement
fra = rates.fra(notional=5_000_000, contract_rate=0.052,
                start=0.5, end=1.0, curve=curve)
fra.summary()

# Swaption — option to enter a swap
swaption = rates.swaption(notional=10_000_000, strike_rate=0.048,
                          option_expiry=1.0, swap_tenor=5,
                          vol=0.20, curve=curve, type="payer")
swaption.summary()

# FX forward — covered interest parity
fwd = fx.forward(spot=1.085, r_domestic=0.053, r_foreign=0.038,
                 tenor=1.0, pair="EURUSD")
fwd.summary()
# → forward rate, forward points, CIP-implied foreign rate

# FX option — Garman-Kohlhagen
opt = fx.option(spot=1.085, strike=1.10, tenor=0.5,
                r_domestic=0.053, r_foreign=0.038,
                sigma=0.085, type="call", pair="EURUSD")
opt.summary()

# Cross-currency basis swap
ccs = fx.cross_currency_swap(notional_usd=10_000_000, pair="EURUSD",
                              spot=1.085, tenor=3, basis_spread=-0.0010)
ccs.summary()

# Currency-adjusted WACC for multinational DCF
wacc_adj = fx.currency_adjusted_wacc(
    base_wacc=0.095,
    revenue_fx_exposure={"EUR": 0.35, "GBP": 0.20},
)
dcf = DCF.manual(base_revenue=383.0, ...).set(wacc=wacc_adj).run()
```

---

### ML — `finverse.ml`

| Module | Algorithm | What it does |
|---|---|---|
| `ml.forecast` | XGBoost + bootstrap | Revenue/margin forecasts with 80% confidence intervals |
| `ml.cross_sectional` | GBM on universe | Train on 80+ companies simultaneously, forecast any ticker |
| `ml.garch` | MLE via scipy | GARCH(1,1), EGARCH, GJR-GARCH volatility — no arch package needed |
| `ml.factor` | Rolling OLS | Fama-French factor decomposition (market, value, momentum, quality) |
| `ml.regime` | Hidden Markov Model | Regime detection — expansion, slowdown, contraction, recovery |
| `ml.nlp` | Lexicon | Financial text sentiment (no external model) |
| `ml.cluster` | KMeans / DBSCAN | ML peer group detection from financial ratios |
| `ml.anomaly` | Isolation Forest + Beneish | Earnings anomaly detection |
| `ml.causal` | Granger causality | Which macro variables drive earnings |
| `ml.earnings_surprise` | GBM + Platt scaling | Beat/miss probability before earnings ✦ new in v0.7.0 |
| `ml.price_target_ensemble` | Adaptive ensemble | ML-weighted DCF + comps + momentum + consensus target ✦ new in v0.7.0 |
| `ml.macro_factor_rotation` | Regime-conditional | Factor tilt predictions (value/growth/momentum/quality/low-vol/size) ✦ new in v0.7.0 |

```python
from finverse.ml import (
    forecast, garch, cross_sectional, factor, regime,
    earnings_surprise, price_target_ensemble, macro_factor_rotation,
)

# Revenue forecast with confidence intervals
fc = forecast.revenue(data, n_years=5)
fc.summary()

# GARCH volatility
vol = garch.fit(data, model_type="GJR-GARCH")
vol.summary()
garch.compare(data)

# Earnings beat/miss probability before earnings
result = earnings_surprise.analyze(data)
result.summary()
# → beat probability (calibrated), historical beat rate, revision momentum,
#   implied move vs historical move, edge ratio, macro headwind

# With options chain for implied move comparison
chain = options.chain(data)
result = earnings_surprise.analyze(data, options_chain=chain)

# Sector screen — top 20 stocks by beat probability
earnings_surprise.screen(sector="tech", top_n=20).summary()

# ML-weighted ensemble price target
result = price_target_ensemble.analyze(data)
result.summary()
# → ensemble target, upside %, 80%/95% CI, signal breakdown by weight,
#   BUY/HOLD/SELL rating, signal agreement (HIGH/MEDIUM/LOW)

# Supply your own DCF model
from finverse import DCF
dcf = DCF(data).use_ml_forecast().run()
result = price_target_ensemble.analyze(data, dcf_model=dcf)

# Factor rotation — which factors to overweight in current regime
rotation = macro_factor_rotation.predict()
rotation.summary()
# → regime, factor scores (−1 to +1), recommended tilts, top/avoid factors,
#   historical accuracy, rationale

# Plug directly into portfolio optimizer
from finverse.portfolio import optimizer
optimizer.optimize(stocks, factor_tilts=rotation.tilts).summary()

# Different horizons
macro_factor_rotation.predict(horizon="6m").summary()
```

---

### Risk — `finverse.risk`

| Module | Method | What it does |
|---|---|---|
| `risk.monte_carlo` | Monte Carlo | 10k DCF scenarios, price distribution, P(upside) |
| `risk.var` | Historical + parametric | VaR(95/99%), CVaR, max drawdown, stress scenarios |
| `risk.evt` | Peaks-Over-Threshold (GPD) | Tail VaR at 99%, 99.9%, 99.99% — beyond normal distribution |
| `risk.kelly` | Continuous + binary + multi-asset | Optimal position sizing, wealth path simulation |
| `risk.stress_testing` | Historical scenario library | Named shock scenarios applied to portfolios and DCF models ✦ new in v0.7.0 |

```python
from finverse.risk import monte_carlo, var, evt, kelly, stress_testing

# Monte Carlo
mc = monte_carlo.simulate(model, n_simulations=10_000)
mc.summary()
mc.plot()

# Extreme Value Theory
tail = evt.analyze(data)
tail.summary()
evt.compare_tails([pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL"]])

# Kelly criterion
kelly.from_distribution(data).summary()
kelly.from_binary(win_prob=0.55, win_return=0.10, loss_return=0.08).summary()
kelly.multi_asset([pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL"]])

# Stress testing — 7 built-in historical scenarios
stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]

stress_testing.apply(stocks, scenario="gfc_2008").summary()
# → portfolio return, $ P&L, worst/best holding, VaR breach, key risk drivers

stress_testing.run_all(stocks).summary()
# → all 7 scenarios ranked by severity

# Apply to a DCF model — stresses WACC and growth assumptions
stress_testing.apply_to_dcf(model, scenario="rate_shock_2022").summary()
# → stressed WACC, implied price under scenario vs base case

# Custom scenario
stress_testing.apply(stocks, scenario="custom", shocks={
    "equity_return": -0.25, "rate_shift_bps": 200,
    "credit_spread_bps": 150, "vix_level": 35,
}).summary()
```

**Built-in scenario library:**

| Scenario ID | Event | Equity | Rates | Credit spreads |
|---|---|---|---|---|
| `gfc_2008` | Global Financial Crisis | −55% | −300bps | +500bps |
| `covid_2020` | COVID Crash | −34% | −150bps | +300bps |
| `dotcom_2000` | Dot-com Bust | −49% | −525bps | +200bps |
| `rate_shock_1994` | Fed Rate Shock 1994 | −10% | +300bps | +100bps |
| `rate_shock_2022` | Rate Shock 2022 | −20% | +425bps | +120bps |
| `asian_crisis_1997` | Asian Financial Crisis | −15% | −75bps | +180bps |
| `russia_default_1998` | Russia Default / LTCM | −20% | −75bps | +300bps |

---

### Portfolio — `finverse.portfolio`

| Module | Method | What it does |
|---|---|---|
| `portfolio.optimizer` | Mean-variance | Max Sharpe, min vol, risk parity, equal weight, efficient frontier |
| `portfolio.hrp` | Hierarchical clustering | Hierarchical Risk Parity — no matrix inversion, more stable |
| `portfolio.shrinkage` | Ledoit-Wolf | Shrinks covariance toward constant correlation target |
| `portfolio.black_litterman` | Bayesian updating | Blends CAPM equilibrium with analyst views |
| `portfolio.cvar_opt` | Linear programming | Minimises CVaR directly — tail-risk optimal weights |

```python
from finverse.portfolio import optimizer, hrp, shrinkage
from finverse.portfolio.black_litterman import optimize as bl_optimize, BLView
from finverse.portfolio.cvar_opt import optimize as cvar_optimize
from finverse.ml import macro_factor_rotation

stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]

optimizer.optimize(stocks, method="max_sharpe").summary()
hrp.optimize(stocks).summary()

# Black-Litterman with analyst views
views = [
    BLView(["AAPL"], [1.0], expected_ret=0.15, confidence=0.8),
    BLView(["MSFT", "GOOGL"], [1.0, -1.0], expected_ret=0.03, confidence=0.6),
]
bl_optimize(stocks, views=views).summary()

# CVaR optimization — minimises tail loss
cvar_optimize(stocks, confidence=0.95).summary()

# Regime-aware optimizer — apply factor tilts from macro_factor_rotation
rotation = macro_factor_rotation.predict()
optimizer.optimize(stocks, factor_tilts=rotation.tilts).summary()
```

---

### Credit — `finverse.credit`

```python
from finverse.credit import merton, altman
from finverse.ml import garch

vol = garch.fit(data)
merton.analyze(data, garch_vol=vol.current_vol).summary()
# → asset value, asset vol, distance-to-default, PD(1y), PD(5y),
#   implied credit spread (bps), rating (AAA → D)

altman.analyze(data).summary()
# → score, zone (safe / grey / distress), component ratios
```

---

### Audit — `finverse.audit`

| Module | Method | What it does |
|---|---|---|
| `audit()` | Rule-based | Model health check: bad assumptions, broken logic, 0–100 score |
| `audit.manipulation` | 40-signal Random Forest | Accounting manipulation probability, top drivers |
| `audit.earnings_quality` | 10-factor composite | Accruals, OCF/NI, persistence, smoothness, FCF — A–F grade |
| `audit.benford` | Chi-square + MAD | Benford's Law test for number manipulation |
| `audit.loughran_mcdonald` | LM dictionary | Financial text: negative, uncertainty, litigious, modal scores |

```python
from finverse import audit
from finverse.audit import manipulation, earnings_quality, benford, loughran_mcdonald

audit(model).summary()
manipulation.fingerprint(data).summary()
earnings_quality.score(data).summary()
benford.test_financials(data).summary()
loughran_mcdonald.analyze(filing_text, source="AAPL 10-K 2024").summary()
loughran_mcdonald.compare_filings({"2022": t1, "2023": t2, "2024": t3})
```

---

### Macro — `finverse.macro`

```python
from finverse.macro import var_model, nelson_siegel
from finverse.models.macro import nowcast

nowcast().summary()
# → GDP nowcast, recession probability (12M), regime,
#   4-quarter inflation + fed rate paths

curve = nelson_siegel.us_curve()
curve.summary()
print(f"10Y yield:       {curve.yield_at(10):.3%}")
print(f"5Y forward rate: {curve.forward_rate(5):.3%}")

# Auto-used as discount curve by derivatives.rates and derivatives.fx
```

---

### Analysis — `finverse.analysis`

```python
from finverse import sensitivity, scenarios
from finverse.screen import screener
from finverse import backtest

sensitivity(model, rows="wacc", cols="terminal_growth")
scenarios(model,
    bull={"wacc": 0.085, "ebitda_margin": 0.36, "revenue_growth": 0.12},
    base={"wacc": 0.095, "ebitda_margin": 0.32, "revenue_growth": 0.08},
    bear={"wacc": 0.115, "ebitda_margin": 0.26, "revenue_growth": 0.03},
)

screener.undervalued(sector="tech").summary()
backtest.dcf_signal(model, data).summary()
```

---

### Export — `finverse.export`

```python
from finverse.export import to_excel, to_report

to_excel(model, "aapl_dcf.xlsx")
# → banker-formatted Excel: blue = formulas, green = outputs, gray = headers

to_report(model, "aapl_summary.txt")
# → plain text one-pager with all assumptions and outputs
```

---

## Full module list

```
finverse/
├── pull/              ticker, edgar, fred
├── models/            dcf, lbo, three_statement, comps, regime_dcf,
│                      synthetic_peers, ddm, sotp, options, bonds,
│                      real_options, apv, macro
├── ml/                forecast, cross_sectional, garch, factor, regime,
│                      nlp, cluster, anomaly, causal,
│                      earnings_surprise ✦, price_target_ensemble ✦,
│                      macro_factor_rotation ✦
├── risk/              monte_carlo, var, evt, kelly,
│                      stress_testing ✦
├── options/           black_scholes, implied_vol, binomial,      ✦ new layer
│                      chain, vol_surface
├── derivatives/       rates, fx                                  ✦ new layer
├── portfolio/         optimizer, hrp, shrinkage,
│                      black_litterman, cvar_opt
├── credit/            merton, altman
├── valuation/         real_options, apv
├── macro/             var_model, nelson_siegel
├── audit/             model_audit, manipulation, earnings_quality,
│                      benford, loughran_mcdonald
├── analysis/          sensitivity, scenarios
├── screen/            screener
├── backtest/          engine
└── export/            excel, report

✦ added in v0.7.0
```

---

## API key requirements

| Feature | Key needed |
|---|---|
| `pull.fred()`, `pull.macro_snapshot()` | Free FRED key — fred.stlouisfed.org |
| `options.chain(data)` | None — uses yfinance |
| All pricing, ML, derivatives, stress testing | None — fully offline |

```bash
export FRED_API_KEY=your_key_here   # only needed for pull.fred()
```

---

## Install

```bash
pip install finverse
pip install finverse[full]    # seaborn, hmmlearn, reportlab
pip install finverse[dev]     # pytest, black, ruff, mypy
```

**Requirements:** Python 3.9+, numpy, pandas, scikit-learn, scipy, xgboost, yfinance, rich, openpyxl, matplotlib

---

## Tests

```bash
pip install finverse[dev]
pytest tests/ -v
```

All tests use synthetic data — no network calls, no API keys required. Tested against Python 3.9–3.12.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
# Portfolio with analyst views
stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]
black_litterman.optimize(stocks, views=[
    BLView(["AAPL"], [1.0], expected_ret=0.15, confidence=0.8),
]).summary()

# Options and bonds
call(spot=185, strike=190, sigma=0.28, maturity=0.25).summary()
bond_price(face=1000, coupon_rate=0.05, ytm=0.06, maturity=10).summary()

# Yield curve
nelson_siegel.us_curve().summary()
```

---

## What's inside

**51 modules** across 11 layers. Everything runs fully offline except `pull.*` data functions.

---

### Data — `finverse.pull`

| Function | Source | Key needed |
|---|---|---|
| `pull.ticker("AAPL")` | yfinance — financials, price history | None |
| `pull.edgar("AAPL", "10-K")` | SEC EDGAR — filings, XBRL facts | None |
| `pull.fred("GDP", "UNRATE")` | Federal Reserve macro data | Free at fred.stlouisfed.org |
| `pull.macro_snapshot()` | Rates, inflation, VIX, credit spreads | Free |

---

### Valuation — `finverse.models`

| Model | How to use | What it does |
|---|---|---|
| `DCF` | `DCF(data).use_ml_forecast().run()` | Discounted cash flow with ML-estimated assumptions |
| `LBO` | `LBO(LBOAssumptions(...)).run()` | Full buyout: debt schedule, IRR, MoM |
| `ThreeStatement` | `ThreeStatement(data).run()` | Linked IS / BS / CF |
| `comps` | `comps(data, peers=["MSFT","GOOGL"])` | Comparable company analysis, implied price range |
| `regime_dcf` | `regime_dcf(data)` | DCF with per-regime assumptions, probability-weighted price |
| `synthetic_peers` | `build_peers(data, {"software":0.6,"hardware":0.4})` | Peer multiples for companies with no clean peer set |
| `gordon` | `gordon(dividend=1.84, growth_rate=0.04, ke=0.085)` | Gordon Growth Model |
| `h_model` | `h_model(dividend=1.84, high_growth=0.15, ...)` | H-Model DDM |
| `multistage` | `multistage(dividend=1.84, stage1_growth=0.15, ...)` | Multistage DDM |
| `sotp` | `sotp([Segment(...), Segment(...)])` | Sum of the Parts |
| `apv` | `apv.analyze(data)` | Adjusted Present Value (Modigliani-Miller) |
| `real_options` | `real_options.expand(500, 200, 0.30, 3.0)` | Expand, abandon, defer (Black-Scholes) |
| `macro.nowcast` | `macro.nowcast()` | GDP nowcast, recession probability, regime |
| `options` | `call(185, 190, 0.28, 0.25)` | European call/put, all 5 Greeks, IV solver |
| `bonds` | `bond_price(1000, 0.05, 0.06, 10)` | Price, YTM, duration, convexity, DV01 |

```python
from finverse import pull, DCF, sensitivity, scenarios, regime_dcf
from finverse.models.lbo import LBO, LBOAssumptions
from finverse.models.ddm import gordon, h_model, multistage
from finverse.models.sotp import Segment, analyze as sotp
from finverse.models.synthetic_peers import build_peers
from finverse.models.options import call, put, implied_vol, vol_surface
from finverse.models.bonds import price as bond_price, ytm_from_price

data = pull.ticker("AAPL")

# --- DCF ---
model = DCF(data).use_ml_forecast()
model.run().summary()
sensitivity(model, rows="wacc", cols="terminal_growth")
scenarios(model,
    bull={"wacc": 0.085, "revenue_growth": 0.12},
    base={"wacc": 0.095, "revenue_growth": 0.08},
    bear={"wacc": 0.115, "revenue_growth": 0.03},
)

# --- Regime-Conditional DCF ---
# Adjusts WACC, growth, margins per detected macro regime
# Returns probability-weighted implied price across all regimes
result = regime_dcf(data)
result.summary()
print(f"Current regime:    {result.current_regime}")
print(f"Regime-weighted:   ${result.weighted_price:.2f}")
print(f"Static DCF:        ${result.static_price:.2f}")
print(f"Regime adjustment: {result.regime_discount_pct:+.1%}")

# --- LBO ---
result = LBO(LBOAssumptions(
    entry_ebitda=150, entry_ev_ebitda=10.0,
    equity_pct=0.40, hold_years=5, exit_ev_ebitda=12.0,
)).run()
print(f"IRR: {result.irr:.1%}  MoM: {result.mom:.2f}x")

# --- Synthetic Peers ---
# For conglomerates or niche companies with no clean peer set
result = build_peers(data, segment_weights={
    "hardware": 0.55,
    "software": 0.25,
    "consumer_staples": 0.20,
})
result.summary()
p25, median, p75 = result.implied_price_range

# --- Options ---
c = call(spot=185, strike=190, sigma=0.28, maturity=0.25)
c.summary()
# → price, delta, gamma, theta, vega, rho

p = put(spot=185, strike=190, sigma=0.28, maturity=0.25)

# Solve implied vol from market price
iv = implied_vol(market_price=9.03, spot=185, strike=190, maturity=0.25)
print(f"Implied vol: {iv.implied_vol:.2%}")

# Vol surface across strikes and maturities
surface = vol_surface(spot=185, sigma=0.28)

# --- Bonds ---
b = bond_price(face=1000, coupon_rate=0.05, ytm=0.06, maturity=10)
b.summary()
# → clean price, dirty price, YTM, Macaulay duration,
#   modified duration, convexity, DV01, yield scenario table

# Solve YTM from market price
b2 = ytm_from_price(market_price=950, coupon_rate=0.05, maturity=10)
print(f"YTM: {b2.ytm:.4%}")
```

---

### ML — `finverse.ml`

| Module | Algorithm | What it does |
|---|---|---|
| `ml.forecast` | XGBoost + bootstrap | Revenue/margin forecasts with 80% confidence intervals |
| `ml.cross_sectional` | GBM on universe | Train on 80+ companies simultaneously, forecast any ticker |
| `ml.garch` | MLE via scipy | GARCH(1,1), EGARCH, GJR-GARCH volatility — no arch package needed |
| `ml.factor` | Rolling OLS | Fama-French factor decomposition (market, value, momentum, quality) |
| `ml.regime` | Hidden Markov Model | Regime detection — expansion, slowdown, contraction, recovery |
| `ml.nlp` | Lexicon | Financial text sentiment (no external model) |
| `ml.cluster` | KMeans / DBSCAN | ML peer group detection from financial ratios |
| `ml.anomaly` | Isolation Forest + Beneish | Earnings anomaly detection |
| `ml.causal` | Granger causality | Which macro variables drive earnings |

```python
from finverse.ml import forecast, garch, cross_sectional, factor, regime

# Revenue forecast with confidence intervals
fc = forecast.revenue(data, n_years=5)
fc.summary()
# → point estimates, 80% CI, implied CAGR, key drivers

# GARCH volatility — picks up leverage effect (bad news = more vol)
vol = garch.fit(data, model_type="GJR-GARCH")
vol.summary()
garch.compare(data)  # compare all 3 models by AIC

# Cross-sectional — trained on universe, not just the company's own history
cs = cross_sectional.forecast(data, target="revenue_growth")
cs.summary()
# → forecast, 80% CI, percentile rank vs 80-company universe, feature importance

# Factor decomposition
factors = factor.decompose(data, window="3y")
factors.summary()

# Regime detection — feeds into regime_dcf automatically
r = regime.detect(data.price_history["Close"])
print(f"Current regime: {r.current_regime.value}")
adjusted_wacc = r.adjust_wacc(0.095)
```

---

### Risk — `finverse.risk`

| Module | Method | What it does |
|---|---|---|
| `risk.monte_carlo` | Monte Carlo | 10k DCF scenarios, price distribution, P(upside) |
| `risk.var` | Historical + parametric | VaR(95/99%), CVaR, max drawdown, stress scenarios |
| `risk.evt` | Peaks-Over-Threshold (GPD) | Tail VaR at 99%, 99.9%, 99.99% — beyond normal distribution |
| `risk.kelly` | Continuous + binary + multi-asset | Optimal position sizing, wealth path simulation |

```python
from finverse.risk import monte_carlo, var, evt, kelly

# Monte Carlo
mc = monte_carlo.simulate(model, n_simulations=10_000)
mc.summary()
mc.plot()

# Extreme Value Theory — models the tail that normal distribution misses
tail = evt.analyze(data)
tail.summary()
# → ξ (tail index), VaR(99%), VaR(99.9%), VaR(99.99%), return periods
# e.g. "a 15% daily loss is expected once every 47 years"
evt.compare_tails([pull.ticker(t) for t in ["AAPL","MSFT","GOOGL"]])

# Kelly criterion
k = kelly.from_distribution(data)      # continuous: f* = μ/σ²
k.summary()
# → full Kelly, half Kelly, quarter Kelly, expected geometric growth
paths = k.simulate(n_periods=252)       # wealth path simulation

k2 = kelly.from_binary(win_prob=0.55, win_return=0.10, loss_return=0.08)
k3 = kelly.multi_asset([pull.ticker(t) for t in ["AAPL","MSFT","GOOGL"]])
```

---

### Portfolio — `finverse.portfolio`

| Module | Method | What it does |
|---|---|---|
| `portfolio.optimizer` | Mean-variance | Max Sharpe, min vol, risk parity, equal weight, efficient frontier |
| `portfolio.hrp` | Hierarchical clustering | Hierarchical Risk Parity — no matrix inversion, more stable |
| `portfolio.shrinkage` | Ledoit-Wolf | Shrinks covariance toward constant correlation target |
| `portfolio.black_litterman` | Bayesian updating | Blends CAPM equilibrium with analyst views |
| `portfolio.cvar_opt` | Linear programming | Minimises CVaR directly — tail-risk optimal weights |

```python
from finverse.portfolio import optimizer, hrp, shrinkage
from finverse.portfolio.black_litterman import optimize as bl_optimize, BLView
from finverse.portfolio.cvar_opt import optimize as cvar_optimize

stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]

# Standard MVO
optimizer.optimize(stocks, method="max_sharpe").summary()
optimizer.optimize(stocks, method="min_vol").summary()
optimizer.optimize(stocks, method="risk_parity").summary()

# HRP — no matrix inversion, works with correlated assets
hrp.optimize(stocks).summary()
hrp.optimize(stocks).compare_to_equal_weight()

# Ledoit-Wolf shrinkage
cov = shrinkage.shrink(stocks, method="constant_correlation")
cov.summary()

# Black-Litterman — combine equilibrium with views
views = [
    BLView(["AAPL"], [1.0], expected_ret=0.15, confidence=0.8),
    # "AAPL returns 15%, high confidence"
    BLView(["MSFT", "GOOGL"], [1.0, -1.0], expected_ret=0.03, confidence=0.6),
    # "MSFT outperforms GOOGL by 3%"
]
bl = bl_optimize(stocks, views=views)
bl.summary()
# → posterior returns, how much views shifted each asset, optimal weights

# CVaR optimization — minimises tail loss via linear programming
cvar = cvar_optimize(stocks, confidence=0.95)
cvar.summary()
# → weights that minimise expected loss in worst 5% of days
```

---

### Credit — `finverse.credit`

```python
from finverse.credit import merton, altman
from finverse.ml import garch

# Merton structural credit model
vol = garch.fit(data)
r = merton.analyze(data, garch_vol=vol.current_vol)
r.summary()
# → asset value, asset vol, distance-to-default (σ)
#   PD(1y), PD(5y), implied credit spread (bps), rating (AAA → D)

# Altman Z-Score
r = altman.analyze(data)                     # auto-selects variant
r = altman.analyze(data, model="Z-Score")    # public manufacturers
r = altman.analyze(data, model="Z'-Score")   # private companies
r = altman.analyze(data, model="Z''-Score")  # non-manufacturers
r.summary()
# → score, zone (safe / grey / distress), component ratios
```

---

### Audit — `finverse.audit`

| Module | Method | What it does |
|---|---|---|
| `audit()` | Rule-based | Model health check: bad assumptions, broken logic, 0–100 score |
| `audit.manipulation` | 40-signal Random Forest | Accounting manipulation probability, Beneish M-Score, top drivers |
| `audit.earnings_quality` | 10-factor composite | Accruals, OCF/NI, persistence, smoothness, FCF — A–F grade |
| `audit.benford` | Chi-square + MAD | Benford's Law test for number manipulation |
| `audit.loughran_mcdonald` | LM dictionary | Financial text: negative, uncertainty, litigious, modal scores |

```python
from finverse import audit
from finverse.audit import manipulation, earnings_quality, benford, loughran_mcdonald

# Model health check
audit(model).summary()
# → score, errors, warnings, specific fix suggestions

# Manipulation fingerprint — 40+ signals, not just Beneish
result = manipulation.fingerprint(data)
result.summary()
# → probability 0–1, risk level, top 8 risk drivers with scores
# e.g. "AR growth exceeding revenue growth (0.18), SGA inflation (0.14)"

# Earnings quality
earnings_quality.score(data).summary()
# → 0–100 score, A–F grade, 10 signals scored individually

# Benford's Law
benford.test_financials(data).summary()
# → MAD, chi-square p-value, conformity, flagged digits

# LM sentiment on filing text
result = loughran_mcdonald.analyze(filing_text, source="AAPL 10-K 2024")
result.summary()
loughran_mcdonald.compare_filings({"2022": t1, "2023": t2, "2024": t3})
```

---

### Macro — `finverse.macro`

```python
from finverse.macro import var_model, nelson_siegel
from finverse.models.macro import nowcast

# Macro nowcast
result = nowcast()
result.summary()
# → GDP nowcast (%), recession probability (12M), yield curve signal,
#   regime, 4-quarter inflation path, 4-quarter fed rate path

# VAR with impulse response functions
macro = pull.fred("UNRATE", "FEDFUNDS", "CPIAUCSL")
quarterly = macro.resample("QE").last().pct_change().dropna()
result = var_model.fit(quarterly, n_lags=2, forecast_horizon=8)
result.summary()
result.plot_irf("FEDFUNDS")                    # shock to fed funds
result.irf("FEDFUNDS", "UNRATE")               # specific pair
var_model.select_lag_order(quarterly, max_lags=6)

# Nelson-Siegel yield curve
curve = nelson_siegel.us_curve()
curve.summary()
print(f"10Y yield:       {curve.yield_at(10):.3%}")
print(f"5Y forward rate: {curve.forward_rate(5):.3%}")
print(f"Level (β₀):      {curve.level:.3%}")
print(f"Slope (β₁):      {curve.slope:.3%}")
curve.curve().plot()

# Svensson (two humps)
curve = nelson_siegel.fit(maturities, yields, model="Svensson")
```

---

### Analysis — `finverse.analysis`

```python
from finverse import sensitivity, scenarios
from finverse.screen import screener
from finverse import backtest

# 2-variable sensitivity heatmap
sensitivity(model, rows="wacc", cols="terminal_growth")
sensitivity(model, rows="ebitda_margin", cols="revenue_growth", n=7)

# Bull / base / bear scenarios
scenarios(model,
    bull={"wacc": 0.085, "ebitda_margin": 0.36, "revenue_growth": 0.12},
    base={"wacc": 0.095, "ebitda_margin": 0.32, "revenue_growth": 0.08},
    bear={"wacc": 0.115, "ebitda_margin": 0.26, "revenue_growth": 0.03},
)

# ML stock screener
screener.undervalued(sector="tech").summary()
screener.by_criteria(["AAPL","MSFT","GOOGL"], min_revenue_growth=0.05, max_pe=40)

# Signal backtesting
prices = data.price_history["Close"]
signal = prices.pct_change(63).shift(1)
result = backtest.run(signal, prices, "3M Momentum")
result.summary()
result.plot()
backtest.dcf_signal(model, data).summary()
```

---

### Export — `finverse.export`

```python
from finverse.export import to_excel, to_report

to_excel(model, "aapl_dcf.xlsx")
# → banker-formatted Excel: blue = formulas, green = outputs, gray = headers

to_report(model, "aapl_summary.txt")
# → plain text one-pager with all assumptions and outputs
```

---

## Full module list

```
finverse/
├── pull/              ticker, edgar, fred
├── models/            dcf, lbo, three_statement, comps, regime_dcf,
│                      synthetic_peers, ddm, sotp, options, bonds,
│                      real_options, apv, macro
├── ml/                forecast, cross_sectional, garch, factor,
│                      regime, nlp, cluster, anomaly, causal
├── risk/              monte_carlo, var, evt, kelly
├── portfolio/         optimizer, hrp, shrinkage,
│                      black_litterman, cvar_opt
├── credit/            merton, altman
├── valuation/         real_options, apv
├── macro/             var_model, nelson_siegel
├── audit/             model_audit, manipulation, earnings_quality,
│                      benford, loughran_mcdonald
├── analysis/          sensitivity, scenarios
├── screen/            screener
├── backtest/          engine
└── export/            excel, report
```

---

## Install

```bash
pip install finverse
pip install finverse[full]    # seaborn, hmmlearn, reportlab
pip install finverse[dev]     # pytest, black, ruff, mypy
```

**Requirements:** Python 3.9+, numpy, pandas, scikit-learn, scipy, xgboost, yfinance, rich, openpyxl, matplotlib

---

## FRED API key

Only needed for `pull.fred()` and `pull.macro_snapshot()`. All other features work without any keys.

```bash
export FRED_API_KEY=your_key_here
```

Free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — no credit card, 30 seconds.

---

## Tests

```bash
pip install finverse[dev]
pytest tests/ -v
```

All tests use synthetic data — no network calls, no API keys required. Tested against Python 3.9–3.12.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
