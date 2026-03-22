<div align="center">

# finverse

**The ML-powered financial modeling toolkit for Python.**

[![PyPI version](https://badge.fury.io/py/finverse.svg)](https://pypi.org/project/finverse/)
[![Python](https://img.shields.io/pypi/pyversions/finverse.svg)](https://pypi.org/project/finverse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/yourusername/finverse/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/finverse/actions)

Build institutional-grade financial models in Python.<br>
DCF · LBO · Credit · Portfolio · ML Forecasting · Yield Curves · Tail Risk<br>
**No Bloomberg. No API keys. Just Python.**

</div>

---

```python
from finverse import pull, DCF, sensitivity
from finverse.ml import forecast, garch
from finverse.credit import merton, altman
from finverse.risk import evt, kelly
from finverse.portfolio import hrp
from finverse.macro import nelson_siegel

data = pull.ticker("AAPL")

# ML-powered DCF in 3 lines
model = DCF(data).use_ml_forecast()
model.run().summary()
sensitivity(model, rows="wacc", cols="terminal_growth")

# Credit analysis
vol = garch.fit(data)                           # GJR-GARCH volatility
merton.analyze(data, garch_vol=vol.current_vol) # distance-to-default, PD, spread
altman.analyze(data).summary()                  # Z-Score distress prediction

# Tail risk and position sizing
evt.analyze(data).summary()                     # GPD tail VaR at 99.9%
kelly.from_distribution(data).summary()         # optimal position sizing

# Portfolio construction
stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]
hrp.optimize(stocks).summary()                  # Hierarchical Risk Parity

# Yield curve
curve = nelson_siegel.us_curve()
print(f"10Y yield: {curve.yield_at(10):.3%}")
print(f"10Y-2Y spread: {curve.yield_at(10) - curve.yield_at(2):+.3%}")
```

---

## Install

```bash
pip install finverse
```

```bash
pip install finverse[full]   # adds seaborn, hmmlearn, reportlab
pip install finverse[dev]    # adds pytest, black, ruff, mypy
```

**Requirements:** Python 3.9+, numpy, pandas, scikit-learn, scipy, xgboost, yfinance, rich, openpyxl, matplotlib

---

## What's inside

44 modules across 10 layers. Everything runs offline except `pull.*` functions.

### Data layer — `finverse.pull`

```python
from finverse import pull

data  = pull.ticker("AAPL")           # financials + price history via yfinance (free, no key)
filings = pull.edgar("AAPL", "10-K")  # SEC EDGAR filings + XBRL facts (free, no key)
macro = pull.fred("GDP", "UNRATE")    # Federal Reserve macro data (free key at fred.stlouisfed.org)
snap  = pull.macro_snapshot()         # rates, inflation, VIX, credit spreads
```

### Valuation models — `finverse.models`

| Model | Class / function | Description |
|---|---|---|
| DCF | `DCF(data)` | Discounted cash flow, ML-assisted assumptions, monte carlo |
| LBO | `LBO(assumptions)` | Full buyout with senior/sub debt, IRR, MoM |
| Three-statement | `ThreeStatement(data)` | Linked IS / BS / CF model |
| Comparable comps | `comps(data, peers=[...])` | Auto peer detection + implied price range |
| Dividend models | `gordon()`, `h_model()`, `multistage()` | Gordon Growth, H-Model, Multistage DDM |
| Sum of Parts | `sotp(segments, ...)` | Segment-by-segment EV aggregation |
| Macro nowcast | `macro.nowcast()` | GDP nowcast, recession probability, yield curve signal |
| APV | `valuation.apv.analyze(data)` | Adjusted Present Value (Modigliani-Miller) |
| Real options | `valuation.real_options.expand/abandon/defer()` | Black-Scholes corporate options |

```python
from finverse import DCF, LBO, ThreeStatement, comps
from finverse.models.lbo import LBOAssumptions
from finverse.models.ddm import gordon, h_model, multistage
from finverse.models.sotp import Segment, analyze as sotp
from finverse.valuation import real_options, apv

# DCF — manual or ML-assisted
model = DCF.manual(base_revenue=383.0, shares_outstanding=15.4, net_debt=50.0)
model.set(wacc=0.095, terminal_growth=0.025)
model.run().summary()
print(model.implied_price, model.ev)

# LBO
result = LBO(LBOAssumptions(
    entry_ebitda=150, entry_ev_ebitda=10.0,
    equity_pct=0.40, hold_years=5,
    exit_ev_ebitda=12.0, revenue_growth=0.08,
)).run()
print(f"IRR: {result.irr:.1%}  MoM: {result.mom:.2f}x")

# Dividend Discount Models
r = gordon(dividend=1.84, growth_rate=0.04, cost_of_equity=0.085)
r = h_model(dividend=1.84, high_growth=0.15, stable_growth=0.04, half_life=7)
r = multistage(dividend=1.84, stage1_growth=0.15, stage1_years=5,
               stage2_growth=0.08, stage2_years=5, terminal_growth=0.04)

# Sum of the Parts
result = sotp([
    Segment("Search",  metric_value=80000, metric_type="ebitda", multiple=18.0),
    Segment("Cloud",   metric_value=35000, metric_type="revenue", multiple=8.0),
    Segment("YouTube", metric_value=12000, metric_type="ebitda", multiple=20.0),
], ticker="GOOGL", net_debt=-100.0, shares_outstanding=12.8)

# Real options
real_options.expand(project_value=500, expansion_cost=200, sigma=0.30, time_to_expiry=3).summary()
real_options.abandon(project_value=300, salvage_value=150, sigma=0.35, time_to_expiry=2).summary()
real_options.defer(project_value=400, investment_cost=350, sigma=0.25, time_to_expiry=2).summary()
```

### ML layer — `finverse.ml`

| Module | Algorithm | What it does |
|---|---|---|
| `ml.forecast` | XGBoost + bootstrap | Per-company revenue/margin forecasts with 80% CI |
| `ml.cross_sectional` | GBM on universe | Train on 80+ companies, forecast any target ticker |
| `ml.garch` | MLE (scipy) | GARCH(1,1), EGARCH, GJR-GARCH volatility modeling |
| `ml.factor` | Rolling OLS | Fama-French factor decomposition (market, value, momentum, quality) |
| `ml.regime` | Hidden Markov Model | Market regime detection (expansion/contraction/stress/recovery) |
| `ml.nlp` | Lexicon-based | Financial text sentiment (no external model needed) |
| `ml.cluster` | KMeans / DBSCAN | ML peer group detection from financial ratios |
| `ml.anomaly` | Isolation Forest + Beneish | Earnings anomaly detection |
| `ml.causal` | Granger causality | Which macro variables actually drive earnings |

```python
from finverse.ml import forecast, garch, cross_sectional, factor, regime, anomaly, causal

# Revenue forecast with confidence intervals
fc = forecast.revenue(data, n_years=5)
fc.summary()
# → point estimates, 80% CI, implied CAGR, key drivers

# GARCH volatility
vol = garch.fit(data, model_type="GJR-GARCH")  # also "GARCH(1,1)", "EGARCH"
vol.summary()
# → ω, α, β, γ, persistence, current vol, multi-step forecast

# Compare all GARCH models by AIC
garch.compare(data)

# Cross-sectional: trained on universe of companies
cs = cross_sectional.forecast(data, target="revenue_growth")
cs.summary()
# → forecast with CI, percentile rank vs universe, feature importance

# Factor decomposition
factors = factor.decompose(data, window="3y")
factors.summary()
# → market β, value, momentum, quality, low-vol, size loadings

# Market regime detection
r = regime.detect(data.price_history["Close"])
r.summary()
print(f"Current regime: {r.current_regime.value}")
print(f"WACC adjustment: {r.wacc_adjustment:+.1%}")
adjusted_wacc = r.adjust_wacc(0.095)

# Macro → earnings causality
result = causal.analyze(data)
result.summary()
# → ranks GDP, rates, inflation etc. by causal strength on earnings
```

### Risk — `finverse.risk`

```python
from finverse.risk import monte_carlo, var, evt, kelly

# Monte Carlo over DCF assumptions
mc = monte_carlo.simulate(model, n_simulations=10_000)
mc.summary()
mc.plot()
# → price distribution, P(upside), 5th–95th percentile range

# VaR and CVaR
r = var.var(data, confidence=0.95, method="historical")
r.summary()
# → VaR(95%), VaR(99%), CVaR(95%), CVaR(99%), max drawdown, stress scenarios

# Extreme Value Theory — tail risk beyond normal distribution
tail = evt.analyze(data)
tail.summary()
# → GPD parameters (ξ, σ), VaR(99%), VaR(99.9%), VaR(99.99%), return periods
evt.compare_tails([apple, msft, googl])  # rank stocks by tail heaviness

# Kelly criterion — optimal position sizing
k = kelly.from_distribution(data)        # continuous Kelly f* = μ/σ²
k = kelly.from_binary(win_prob=0.55, win_return=0.10, loss_return=0.08)
k.summary()
# → full/half/quarter Kelly fractions, expected geometric growth rates
paths = k.simulate(n_periods=252)        # wealth path simulation
kelly.multi_asset([apple, msft, googl]) # covariance-matrix Kelly
```

### Portfolio — `finverse.portfolio`

```python
from finverse.portfolio import optimizer, hrp, shrinkage

stocks = [pull.ticker(t) for t in ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]]

# Mean-variance optimization
optimizer.optimize(stocks, method="max_sharpe").summary()
optimizer.optimize(stocks, method="min_vol").summary()
optimizer.optimize(stocks, method="risk_parity").summary()
optimizer.optimize(stocks, method="equal_weight").summary()

# Hierarchical Risk Parity — no matrix inversion, more stable
hrp.optimize(stocks).summary()
hrp.optimize(stocks).compare_to_equal_weight()

# Ledoit-Wolf covariance shrinkage — better conditioned matrix
cov = shrinkage.shrink(stocks, method="constant_correlation")
cov.summary()
# → shrinkage coefficient, condition number before/after

# Efficient frontier
ef = optimizer.frontier(stocks)
```

### Credit — `finverse.credit`

```python
from finverse.credit import merton, altman
from finverse.ml import garch

# Merton structural model — equity as call option on firm assets
vol = garch.fit(data)
r = merton.analyze(data, garch_vol=vol.current_vol)
r.summary()
# → asset value, asset vol, distance-to-default, PD(1y), PD(5y),
#   implied credit spread (bps), approximate rating (AAA → D)

# Altman Z-Score family — financial distress prediction
r = altman.analyze(data)                    # auto-selects model
r = altman.analyze(data, model="Z-Score")   # public manufacturers
r = altman.analyze(data, model="Z'-Score")  # private companies
r = altman.analyze(data, model="Z''-Score") # non-manufacturers / services
r.summary()
# → score, zone (safe / grey / distress), component ratios
```

### Macro — `finverse.macro`

```python
from finverse.macro import var_model, nelson_siegel
from finverse.models.macro import nowcast

# GDP nowcast + recession probability
macro = nowcast()
macro.summary()
# → GDP nowcast (%), recession probability (12M), yield curve signal,
#   regime (expansion/slowdown/contraction/recovery),
#   4-quarter inflation + fed rate path

# Vector Autoregression with impulse response functions
import pandas as pd
macro_data = pull.fred("UNRATE", "FEDFUNDS", "CPIAUCSL")
quarterly = macro_data.resample("QE").last().pct_change().dropna()
result = var_model.fit(quarterly, n_lags=2, forecast_horizon=8)
result.summary()
result.plot_irf("FEDFUNDS")   # impulse response to rate shock
result.irf("FEDFUNDS", "UNRATE")  # specific variable pair
var_model.select_lag_order(quarterly, max_lags=6)  # AIC/BIC lag selection

# Nelson-Siegel yield curve
maturities = [0.25, 1, 2, 5, 10, 30]
yields     = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
curve = nelson_siegel.fit(maturities, yields)
curve = nelson_siegel.us_curve()         # uses typical current levels
curve.summary()
print(curve.yield_at(7))                 # interpolate any maturity
print(curve.forward_rate(5))             # instantaneous forward rate
print(curve.level, curve.slope)          # β₀ and β₁ factors
curve.curve().plot()                     # full fitted curve as pd.Series

# Svensson extension (two humps)
curve = nelson_siegel.fit(maturities, yields, model="Svensson")
```

### Audit — `finverse.audit`

```python
from finverse import audit
from finverse.audit import earnings_quality, benford, loughran_mcdonald

# Model health check — catches bad assumptions, broken logic
audit(model).summary()         # DCF
audit(lbo_model).summary()     # LBO
audit(ts_model).summary()      # ThreeStatement
audit(excel_path="model.xlsx") # Excel file — flags hardcoded numbers in formulas
# → 0–100 score, errors / warnings / info, specific suggestions

# Earnings quality — 10-factor composite score
r = earnings_quality.score(data)
r.summary()
# → 0–100 score, A–F grade, 10 individual signals:
#   accruals ratio, OCF/NI coverage, revenue cash conversion,
#   earnings persistence (AR1), smoothness, loss avoidance pattern,
#   asset growth signal, margin stability, WC efficiency, FCF consistency

# Benford's Law — statistical test for data manipulation
benford.test_financials(data).summary()     # from TickerData
benford.test(income_statement_values).summary()
# → MAD, chi-square p-value, conformity rating, flagged digits

# Loughran-McDonald financial sentiment dictionary
result = loughran_mcdonald.analyze(filing_text, source="AAPL 10-K 2024")
result.summary()
# → positive/negative/uncertainty/litigious/modal scores
#   net sentiment, tone label, top positive/negative words
loughran_mcdonald.compare_filings({"2022": t1, "2023": t2, "2024": t3})
```

### Analysis — `finverse.analysis`

```python
from finverse import sensitivity, scenarios
from finverse.screen import screener
from finverse import backtest

# 2-variable sensitivity heatmap
sensitivity(model, rows="wacc", cols="terminal_growth")           # color-coded
sensitivity(model, rows="ebitda_margin", cols="revenue_growth", n=7)

# Bull / base / bear scenario engine
scenarios(model,
    bull={"wacc": 0.085, "ebitda_margin": 0.36, "revenue_growth": 0.12},
    base={"wacc": 0.095, "ebitda_margin": 0.32, "revenue_growth": 0.08},
    bear={"wacc": 0.115, "ebitda_margin": 0.26, "revenue_growth": 0.03},
)

# ML stock screener — composite score across sector
screener.undervalued(sector="tech").summary()      # also "finance", "healthcare", "energy"
screener.by_criteria(
    ["AAPL", "MSFT", "GOOGL"],
    min_revenue_growth=0.05, max_pe=40,
)

# Signal-based backtesting
prices = data.price_history["Close"]
signal = prices.pct_change(63).shift(1)   # 3-month momentum signal
result = backtest.run(signal, prices, "Momentum")
result.summary()
result.plot()
backtest.momentum(data, lookback=252).summary()
backtest.dcf_signal(model, data).summary()
```

### Export — `finverse.export`

```python
from finverse.export import to_excel, to_report

to_excel(model, "aapl_dcf.xlsx")
# → banker-formatted Excel: blue cells = formulas, green = outputs,
#   gray = section headers, percentage/currency formats

to_report(model, "aapl_summary.txt")
# → plain text one-pager with all assumptions and valuation outputs
```

---

## FRED API key

Only needed for `pull.fred()` and `pull.macro_snapshot()`. Everything else works with no keys.

```bash
export FRED_API_KEY=your_key_here
```

Get one free at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) — no credit card, takes 30 seconds.

---

## Running tests

```bash
pip install finverse[dev]
pytest tests/ -v
```

All 138 tests use synthetic data — no network calls, no API keys required. Tests run against Python 3.9–3.12.

---

## Project structure

```
finverse/
├── finverse/
│   ├── pull/           # data: yfinance, EDGAR, FRED
│   ├── models/         # DCF, LBO, ThreeStatement, comps, DDM, SOTP, macro
│   ├── ml/             # forecast, garch, cross_sectional, factor, regime, nlp, cluster, anomaly, causal
│   ├── risk/           # monte_carlo, var, evt, kelly
│   ├── portfolio/      # optimizer, hrp, shrinkage
│   ├── credit/         # merton, altman
│   ├── valuation/      # real_options, apv
│   ├── macro/          # var_model, nelson_siegel
│   ├── analysis/       # sensitivity, scenarios
│   ├── audit/          # model_audit, earnings_quality, benford, loughran_mcdonald
│   ├── backtest/       # engine
│   ├── screen/         # screener
│   └── export/         # excel, report
├── tests/              # 138 tests, all synthetic data
├── examples/           # quickstart.py, credit_analysis.py, finverse_demo.ipynb
├── pyproject.toml
└── README.md
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome — especially for new data sources, additional ML models, and more valuation frameworks.

---

## License

MIT — see [LICENSE](LICENSE).
