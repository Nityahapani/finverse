# Changelog

All notable changes to finverse are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.7.0] — Phase 7: Advanced Valuation, Portfolio, Audit, Options, Derivatives, Stress Testing, ML Signals

### Added — Valuation
- **`models.regime_dcf`** — Regime-Conditional DCF. Wires the HMM regime detector directly into the DCF engine. Each macro regime (expansion, slowdown, contraction, recovery, stress) gets its own WACC, revenue growth, EBITDA margin, and terminal growth assumptions. Output is a probability-weighted implied price distribution rather than a single point estimate. Regime discount typically ranges ±10–20% vs static DCF.
- **`models.synthetic_peers`** — Synthetic Peer Construction. For companies with no clean peer set (conglomerates, niche sectors, early-stage), builds implied multiples by blending sector benchmarks weighted by segment mix. A 60/40 software/hardware company gets `0.6 × software_multiples + 0.4 × hardware_multiples`. Covers 14 sector benchmarks across 10 multiple types. Returns implied price range at 25th, median, and 75th percentile.
- **`models.options`** — European options pricing with full Greeks. Black-Scholes call and put pricing, delta, gamma, theta, vega, rho. Implied volatility solver via Brent's method (guaranteed convergence, no initial guess needed). Volatility surface across strikes and maturities. Put-call parity verification. Pure numpy/scipy — no external options library.
- **`models.bonds`** — Fixed-rate bond pricing and risk measures. Clean price, dirty price, accrued interest, yield-to-maturity, current yield, Macaulay duration, modified duration, convexity, DV01. YTM solver from market price. Price-yield relationship table. Yield shock scenario analysis (±25, ±50, ±100 bps) built into every result.

### Added — Portfolio
- **`portfolio.black_litterman`** — Black-Litterman model. Combines CAPM market equilibrium returns with analyst views using Bayesian updating. Views can be absolute ("AAPL returns 15%") or relative ("MSFT outperforms GOOGL by 3%"). Confidence parameter controls how far posterior moves from equilibrium. Uses Ledoit-Wolf shrinkage on the covariance matrix automatically. Significantly more stable than raw MVO — doesn't amplify estimation errors.
- **`portfolio.cvar_opt`** — CVaR (Conditional Value at Risk) minimisation. Solves for weights that minimise expected tail loss directly via Rockafellar-Uryasev linear programming (scipy HiGHS solver). Unlike mean-variance which penalises all variance symmetrically, CVaR optimisation only penalises downside tail losses. Supports confidence levels (95%, 99%), max/min weight constraints, and optional minimum return constraint.

### Added — Audit
- **`audit.manipulation`** — Accounting Manipulation Fingerprinting. 40+ accounting signals combined into a weighted manipulation probability score (0–1). Goes well beyond the classic 8-signal Beneish M-Score by adding revenue-cash divergence, SGA inflation relative to revenue, AR growth vs revenue growth, asset quality index, six interaction terms, and multi-signal pattern detection. Trained on a synthetic universe calibrated to known manipulation patterns. Returns Beneish M-Score alongside the composite score for comparison.

### Added — Options layer (new top-level package)
- **`finverse.options`** — Full options pricing library as a dedicated package. Separate from `models.options` (which prices single European options). This layer adds live chain integration, volatility surface construction, American option pricing, and arbitrage detection.
  - **`options.black_scholes`** — Black-Scholes closed-form pricing for European calls and puts. Full Greeks: delta, gamma, theta ($/day), vega (per 1% vol), rho (per 1% rate). Intrinsic value, time value, breakeven. Validates inputs and raises on invalid parameters.
  - **`options.implied_vol`** — Implied volatility solver using Brent's method via scipy. Guaranteed convergence, no initial guess needed. Returns `None` for arbitrage-violating inputs rather than crashing.
  - **`options.binomial`** — Cox-Ross-Rubinstein (CRR) binomial tree for American option pricing with early exercise. Configurable steps (default 500). Delta computed via two-step finite difference.
  - **`options.chain`** — yfinance live options chain wrapper. Fetches calls and puts across the nearest 6 expiries. Computes time to expiry `T` automatically. `scan_arbitrage()` detects put-call parity violations above a configurable threshold.
  - **`options.vol_surface`** — Implied volatility surface by expiry and moneyness bucket. `build_surface()` takes any list of `{expiry, moneyness, iv}` records. `.plot()` renders the surface via matplotlib.
  - **`options.tail_hedge_suggestion(data, evt_result)`** — Integrates with `risk.evt` to recommend a put strike, expiry, and cost as % of spot based on the GPD tail loss estimate.

### Added — Derivatives layer (new top-level package)
- **`finverse.derivatives`** — Interest rate and FX derivatives pricing. Integrates with `macro.nelson_siegel` for discount factors and with `DCF` for currency-adjusted WACC.
  - **`derivatives.rates`** — Interest rate derivatives.
    - `rates.swap()` — Fixed/float vanilla IR swap. NPV from fixed-payer perspective, par swap rate, DV01, breakeven rate shift, full cash flow schedule (DataFrame). Accepts Nelson-Siegel curve or flat rate fallback.
    - `rates.fra()` — Forward Rate Agreement. NPV, implied forward rate, settlement amount at start date.
    - `rates.swaption()` — European swaption via Black's model. Price, delta, vega, par swap rate. Payer and receiver variants.
  - **`derivatives.fx`** — FX derivatives.
    - `fx.forward()` — FX forward via covered interest parity (CIP). Forward rate, forward points (pips), CIP-implied foreign rate.
    - `fx.option()` — FX option via Garman-Kohlhagen (Black-Scholes treating foreign rate as continuous dividend yield). Price, delta, gamma, vega, breakeven.
    - `fx.cross_currency_swap()` — Cross-currency basis swap NPV. Basis spread impact, FX delta.
    - `fx.currency_adjusted_wacc()` — Multinational DCF helper. Adjusts base WACC for hedging costs weighted by revenue FX exposure buckets. Output feeds directly into `DCF.set(wacc=...)`.

### Added — Risk
- **`risk.stress_testing`** — Historical stress scenario library. Apply named shocks to any portfolio of TickerData objects or to a DCF/LBO model directly. Beta-adjusted equity impact, sector multipliers (tech, EM), per-holding breakdown, VaR breach detection, and plain-English commentary.
  - **7 built-in scenarios:** `gfc_2008`, `covid_2020`, `dotcom_2000`, `rate_shock_1994`, `rate_shock_2022`, `asian_crisis_1997`, `russia_default_1998`.
  - **`stress_testing.apply(stocks, scenario)`** — single scenario on a portfolio.
  - **`stress_testing.run_all(stocks)`** — all 7 scenarios ranked by severity.
  - **`stress_testing.apply_to_dcf(model, scenario)`** — stresses WACC and terminal growth; returns implied price change.
  - **`stress_testing.apply(..., scenario="custom", shocks={...})`** — user-defined shock vector.

### Added — ML
- **`ml.earnings_surprise`** — Beat/miss probability engine. Combines historical surprise patterns, analyst estimate revision momentum, earnings quality score (from `audit.earnings_quality`), and macro regime context into a calibrated probability using a gradient-boosted classifier with Platt scaling. Optional integration with `options.chain()` for implied move vs historical move edge ratio. `screen(sector, top_n)` screens an entire sector.
  - Output fields: `beat_probability`, `miss_probability`, `surprise_score_percentile`, `historical_beat_rate`, `avg_surprise_magnitude`, `revision_momentum`, `implied_move_pct`, `historical_move_pct`, `edge_ratio`, `macro_headwind`, `confidence`.
- **`ml.price_target_ensemble`** — ML-weighted ensemble price target. Combines DCF implied price, comparable comps implied price, 12-month momentum signal, and analyst consensus into a single target. Weights are adaptive by sector and regime — learned from historical signal predictive accuracy rather than set manually. If consensus data is unavailable (small-cap), weight redistributes to DCF and comps automatically.
  - Output fields: `ensemble_target`, `upside_pct`, `confidence_interval_80`, `confidence_interval_95`, `dcf_target`, `comps_target`, `momentum_target`, `consensus_target`, `weights`, `signal_agreement`, `rating` (BUY/HOLD/SELL).
- **`ml.macro_factor_rotation`** — Regime-conditional factor tilt predictions. Maps the current macro regime (from `ml.regime`), yield curve shape, and VIX level to recommended overweights and underweights across six factors: value, growth, momentum, quality, low-vol, size. Historical accuracy tracked per regime. Output tilts plug directly into `portfolio.optimizer(factor_tilts=...)`.
  - Output fields: `current_regime`, `factor_scores` (−1 to +1 per factor), `tilts` (portfolio weight adjustments), `top_factors`, `avoid_factors`, `historical_accuracy`, `macro_context`, `confidence`, `rationale`.
  - Supports `horizon` parameter: `"3m"`, `"6m"`, `"12m"` — longer horizons reduce confidence.

### Tests
- Added `tests/test_v07_modules.py` — 115 tests covering all 7 new modules. All use synthetic data, zero network calls, zero API keys. Covers unit tests, boundary conditions, `.summary()` output, and 7 cross-module integration tests (e.g. `options + evt`, `stress_testing + DCF`, `macro_factor_rotation + earnings_surprise`).
- Total test count: **253 tests** (138 existing + 115 new).

---

## [0.6.0] — Phase 6: Risk, DDM, SOTP, Earnings Quality

### Added — Risk
- **`risk.evt`** — Extreme Value Theory via Peaks-Over-Threshold method. Fits Generalised Pareto Distribution to tail losses using MLE. Computes VaR and Expected Shortfall at 99%, 99.9%, 99.99% — levels where normal distribution fails entirely. Return period analysis (how often does a loss this large occur?). `compare_tails()` ranks stocks by tail heaviness (ξ parameter).
- **`risk.kelly`** — Kelly Criterion in three forms: continuous (`f* = μ/σ²`), binary (`f* = (pb−q)/b`), and multi-asset (covariance-matrix Kelly). Returns full/half/quarter Kelly fractions with expected geometric growth rates. `.simulate()` generates wealth path simulations under each sizing strategy.

### Added — Valuation
- **`models.ddm`** — Dividend Discount Model family. Gordon Growth Model (`P = D₁/(ke−g)`), H-Model (linear growth transition), and Multistage DDM (explicit high-growth and transition phases). All three work from TickerData or manual inputs.
- **`models.sotp`** — Sum of the Parts valuation. Each segment valued by EV/EBITDA, EV/Revenue, P/E, or direct DCF value. Optional conglomerate discount. `from_ticker()` auto-builds from consolidated financials when no segment breakdown exists.

### Added — Audit
- **`audit.earnings_quality`** — 10-factor composite earnings quality score (0–100, A–F grade). Signals: accruals ratio, OCF/NI coverage, revenue cash conversion, earnings persistence (AR1), smoothness (σNI/σOCF), loss avoidance pattern, asset growth signal, margin stability, working capital efficiency, FCF consistency. Weighted composite with individual signal scores.

---

## [0.5.0] — Phase 5: GARCH, Credit, Macro, NLP, Portfolio

### Added — ML
- **`ml.garch`** — GARCH family volatility modeling from scratch via scipy MLE. GARCH(1,1), EGARCH, and GJR-GARCH (captures leverage effect — bad news creates more volatility than good news). Returns conditional volatility time series, multi-step forecasts, persistence, long-run vol. `garch.compare()` fits all three and ranks by AIC/BIC.
- **`ml.cross_sectional`** — Universe-level cross-sectional ML forecasting. Trains Gradient Boosting on 80+ companies simultaneously rather than on a single company's history. Bootstrap confidence intervals. Returns percentile rank vs universe and feature importance.

### Added — Portfolio
- **`portfolio.hrp`** — Hierarchical Risk Parity (Lopez de Prado 2016). Hierarchical clustering on correlation matrix + recursive bisection. No matrix inversion — more stable with correlated assets. `compare_to_equal_weight()` shows deviation from naive diversification.
- **`portfolio.shrinkage`** — Ledoit-Wolf covariance shrinkage toward constant correlation target. Reduces condition number of covariance matrix, producing more stable portfolio weights. Returns shrinkage coefficient and condition number before/after.

### Added — Credit
- **`credit.merton`** — Merton structural credit model. Solves system of equations to extract implied asset value and asset volatility from equity market data. Returns distance-to-default, PD(1y), PD(5y), implied credit spread in bps, and approximate rating (AAA→D). Accepts GARCH vol directly.
- **`credit.altman`** — Altman Z-Score family. Z-Score (public manufacturers), Z'-Score (private), Z''-Score (non-manufacturers/services). Auto-selects model based on company type. Returns score, zone (safe/grey/distress), and all component ratios.

### Added — Valuation
- **`valuation.real_options`** — Corporate real options via Black-Scholes. Expand, abandon, and defer options with full results. Sensitivity grid over sigma and time. `defer()` uses Merton's continuous dividend yield formulation for opportunity cost of waiting.
- **`valuation.apv`** — Adjusted Present Value (Modigliani-Miller). Separates unlevered firm value, tax shield, financial distress costs, and debt issuance costs. Compares to WACC-DCF and shows the difference explicitly.

### Added — Macro
- **`macro.var_model`** — Vector Autoregression with impulse response functions. VAR(p) fitted by OLS equation-by-equation. Orthogonalised IRF via Cholesky decomposition. Pairwise Granger causality table. `select_lag_order()` compares AIC/BIC across lag lengths.
- **`macro.nelson_siegel`** — Nelson-Siegel (1987) and Svensson (1994) yield curve fitting via L-BFGS-B with multiple restarts. `yield_at(m)` interpolates at any maturity. `forward_rate(m)` extracts instantaneous forward rates. `us_curve()` fits to current typical Treasury levels.

### Added — Audit
- **`audit.loughran_mcdonald`** — Loughran-McDonald (2011) financial-domain sentiment dictionary. Purpose-built for financial text — avoids misclassifications from general NLP (e.g. "liability" is not negative in finance). Scores negative, positive, uncertainty, litigious, strong modal, weak modal, and constraining language. `compare_filings()` tracks tone over time.
- **`audit.benford`** — Benford's Law test with chi-square statistic and Mean Absolute Deviation. `test_financials()` runs directly on TickerData. Flags specific digits that deviate most. Conformity levels: close, acceptable, nonconforming, suspicious.

---

## [0.4.0] — Phase 4: LBO, Three-Statement, Comps, Macro Nowcast, Audit

### Added
- **`models.lbo`** — Full LBO model. Senior and subordinated debt tranches, annual debt paydown from FCF, margin improvement schedule, exit returns. IRR, money-on-money, debt schedule, income projections. `from_ticker()` seeds from company data.
- **`models.three_statement`** — Linked three-statement financial model. Net income flows to retained earnings, D&A bridges IS to CF, working capital changes (AR/inventory/AP days) flow from BS to CF. Every line ties.
- **`models.comps`** — Comparable company analysis. Auto-detects peers via `ml.cluster` if none provided. Pulls multiples (EV/EBITDA, P/E, EV/Revenue), builds comps table, derives implied price range at 25th, median, and 75th percentile.
- **`models.macro`** (`nowcast`) — GDP nowcast using macro indicator composite. Recession probability (12-month), yield curve signal, regime classification, 4-quarter forward paths for inflation and fed funds rate.
- **`audit`** (model health checker) — Audits DCF, LBO, ThreeStatement, and Excel files. Checks WACC bounds, terminal growth vs WACC, margin sanity, terminal value dominance, leverage limits, equity cushion, FCF sustainability, and hardcoded numbers in Excel formulas. Returns 0–100 health score with specific fix suggestions.

---

## [0.3.0] — Phase 3: Risk, Screening, Backtesting, Portfolio

### Added
- **`risk.monte_carlo`** — Monte Carlo over DCF assumptions. 10k scenarios, vectorised numpy. Returns price distribution, P(upside), percentile range. `.plot()` histogram.
- **`risk.var`** — Historical and parametric VaR. VaR(95/99%), CVaR/Expected Shortfall(95/99%), max drawdown, annualised vol, 7 pre-built stress scenarios (COVID, GFC, dot-com, Black Monday, 2022 rate shock).
- **`screen.screener`** — ML composite stock scoring. 30% DCF upside + 25% revenue growth + 20% quality + 15% momentum + 10% anomaly. Sector universes: tech (15), finance (10), healthcare (10), energy (10).
- **`backtest.engine`** — Signal-based backtesting. Signal-to-position (long when signal > median), transaction costs, equity curve. Metrics: total/annualised return, vol, Sharpe, max drawdown, Calmar, win rate, trade log. `momentum()` and `dcf_signal()` ready-made strategies.
- **`portfolio.optimizer`** — Mean-variance optimisation. Max Sharpe (Monte Carlo 5k portfolios), min vol, risk parity (inverse vol), equal weight. Efficient frontier. Constraints: max/min weight.

---

## [0.2.0] — Phase 2: ML Layer

### Added
- **`ml.forecast`** — XGBoost revenue and margin forecasting with 50-bootstrap confidence intervals. `revenue()`, `margins()`, `wacc()`. Returns ForecastResult with point/lower/upper/CAGR/drivers.
- **`ml.factor`** — Rolling OLS factor decomposition. Market, value, momentum, quality, low-vol, size loadings. R², alpha, residual vol.
- **`ml.regime`** — HMM market regime detection with KMeans fallback. Expansion, contraction, recovery, stress regimes. `adjust_wacc()` applies regime-appropriate WACC delta.
- **`ml.nlp`** — Lexicon-based financial text sentiment. Guidance signal detection, risk flag extraction (10 patterns), topic extraction. `analyze_filings()` trend analysis across multiple filings.
- **`ml.cluster`** — KMeans/DBSCAN peer detection from financial ratios. Cosine similarity peer ranking. Synthetic universe of 24 stocks across 4 sectors.
- **`ml.anomaly`** — Isolation Forest on financial ratios + Beneish M-Score (threshold −1.78) + accrual ratio (threshold 0.10). Composite anomaly score.
- **`ml.causal`** — Granger causality via OLS F-statistic proxy. Ranks macro variables by causal strength on earnings with direction and lag.

---

## [0.1.0] — Phase 1: Foundation

### Added
- **`pull.ticker`** — yfinance wrapper. Revenue, EBITDA, net income, FCF history. Balance sheet, cash flow, price history. Properties: `market_cap`, `current_price`, `shares_outstanding`, `total_debt`, `cash`.
- **`pull.edgar`** — SEC EDGAR filings + XBRL facts. Free, no key.
- **`pull.fred`** — Federal Reserve macro data via fredapi.
- **`models.dcf`** — Discounted cash flow. ML-assisted assumptions, manual override, `.use_ml_forecast()`. Properties: `implied_price`, `ev`. `DCF.manual()` constructor for fully manual inputs.
- **`analysis.sensitivity`** — 2-variable sensitivity heatmap. Color-coded rich table output.
- **`analysis.scenarios`** — Bull/base/bear scenario engine. Side-by-side comparison.
- **`export.excel`** — Banker-formatted Excel. Blue = formulas, green = outputs, gray = headers.
- **`export.report`** — Plain text one-pager.opt.summary()                                        # price, all Greeks
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

MIT — see [LICENSE](LICENSE).- **`models.sotp`** — Sum of the Parts valuation. Each segment valued by EV/EBITDA, EV/Revenue, P/E, or direct DCF value. Optional conglomerate discount. `from_ticker()` auto-builds from consolidated financials when no segment breakdown exists.

### Added — Audit
- **`audit.earnings_quality`** — 10-factor composite earnings quality score (0–100, A–F grade). Signals: accruals ratio, OCF/NI coverage, revenue cash conversion, earnings persistence (AR1), smoothness (σNI/σOCF), loss avoidance pattern, asset growth signal, margin stability, working capital efficiency, FCF consistency. Weighted composite with individual signal scores.

---

## [0.5.0] — Phase 5: GARCH, Credit, Macro, NLP, Portfolio

### Added — ML
- **`ml.garch`** — GARCH family volatility modeling from scratch via scipy MLE. GARCH(1,1), EGARCH, and GJR-GARCH (captures leverage effect — bad news creates more volatility than good news). Returns conditional volatility time series, multi-step forecasts, persistence, long-run vol. `garch.compare()` fits all three and ranks by AIC/BIC.
- **`ml.cross_sectional`** — Universe-level cross-sectional ML forecasting. Trains Gradient Boosting on 80+ companies simultaneously rather than on a single company's history. Bootstrap confidence intervals. Returns percentile rank vs universe and feature importance.

### Added — Portfolio
- **`portfolio.hrp`** — Hierarchical Risk Parity (Lopez de Prado 2016). Hierarchical clustering on correlation matrix + recursive bisection. No matrix inversion — more stable with correlated assets. `compare_to_equal_weight()` shows deviation from naive diversification.
- **`portfolio.shrinkage`** — Ledoit-Wolf covariance shrinkage toward constant correlation target. Reduces condition number of covariance matrix, producing more stable portfolio weights. Returns shrinkage coefficient and condition number before/after.

### Added — Credit
- **`credit.merton`** — Merton structural credit model. Solves system of equations to extract implied asset value and asset volatility from equity market data. Returns distance-to-default, PD(1y), PD(5y), implied credit spread in bps, and approximate rating (AAA→D). Accepts GARCH vol directly.
- **`credit.altman`** — Altman Z-Score family. Z-Score (public manufacturers), Z'-Score (private), Z''-Score (non-manufacturers/services). Auto-selects model based on company type. Returns score, zone (safe/grey/distress), and all component ratios.

### Added — Valuation
- **`valuation.real_options`** — Corporate real options via Black-Scholes. Expand, abandon, and defer options with full results. Sensitivity grid over sigma and time. `defer()` uses Merton's continuous dividend yield formulation for opportunity cost of waiting.
- **`valuation.apv`** — Adjusted Present Value (Modigliani-Miller). Separates unlevered firm value, tax shield, financial distress costs, and debt issuance costs. Compares to WACC-DCF and shows the difference explicitly.

### Added — Macro
- **`macro.var_model`** — Vector Autoregression with impulse response functions. VAR(p) fitted by OLS equation-by-equation. Orthogonalised IRF via Cholesky decomposition. Pairwise Granger causality table. `select_lag_order()` compares AIC/BIC across lag lengths.
- **`macro.nelson_siegel`** — Nelson-Siegel (1987) and Svensson (1994) yield curve fitting via L-BFGS-B with multiple restarts. `yield_at(m)` interpolates at any maturity. `forward_rate(m)` extracts instantaneous forward rates. `us_curve()` fits to current typical Treasury levels.

### Added — Audit
- **`audit.loughran_mcdonald`** — Loughran-McDonald (2011) financial-domain sentiment dictionary. Purpose-built for financial text — avoids misclassifications from general NLP (e.g. "liability" is not negative in finance). Scores negative, positive, uncertainty, litigious, strong modal, weak modal, and constraining language. `compare_filings()` tracks tone over time.
- **`audit.benford`** — Benford's Law test with chi-square statistic and Mean Absolute Deviation. `test_financials()` runs directly on TickerData. Flags specific digits that deviate most. Conformity levels: close, acceptable, nonconforming, suspicious.

---

## [0.4.0] — Phase 4: LBO, Three-Statement, Comps, Macro Nowcast, Audit

### Added
- **`models.lbo`** — Full LBO model. Senior and subordinated debt tranches, annual debt paydown from FCF, margin improvement schedule, exit returns. IRR, money-on-money, debt schedule, income projections. `from_ticker()` seeds from company data.
- **`models.three_statement`** — Linked three-statement financial model. Net income flows to retained earnings, D&A bridges IS to CF, working capital changes (AR/inventory/AP days) flow from BS to CF. Every line ties.
- **`models.comps`** — Comparable company analysis. Auto-detects peers via `ml.cluster` if none provided. Pulls multiples (EV/EBITDA, P/E, EV/Revenue), builds comps table, derives implied price range at 25th, median, and 75th percentile.
- **`models.macro`** (`nowcast`) — GDP nowcast using macro indicator composite. Recession probability (12-month), yield curve signal, regime classification, 4-quarter forward paths for inflation and fed funds rate.
- **`audit`** (model health checker) — Audits DCF, LBO, ThreeStatement, and Excel files. Checks WACC bounds, terminal growth vs WACC, margin sanity, terminal value dominance, leverage limits, equity cushion, FCF sustainability, and hardcoded numbers in Excel formulas. Returns 0–100 health score with specific fix suggestions.

---

## [0.3.0] — Phase 3: Risk, Screening, Backtesting, Portfolio

### Added
- **`risk.monte_carlo`** — Monte Carlo over DCF assumptions. 10k scenarios, vectorised numpy. Returns price distribution, P(upside), percentile range. `.plot()` histogram.
- **`risk.var`** — Historical and parametric VaR. VaR(95/99%), CVaR/Expected Shortfall(95/99%), max drawdown, annualised vol, 7 pre-built stress scenarios (COVID, GFC, dot-com, Black Monday, 2022 rate shock).
- **`screen.screener`** — ML composite stock scoring. 30% DCF upside + 25% revenue growth + 20% quality + 15% momentum + 10% anomaly. Sector universes: tech (15), finance (10), healthcare (10), energy (10).
- **`backtest.engine`** — Signal-based backtesting. Signal-to-position (long when signal > median), transaction costs, equity curve. Metrics: total/annualised return, vol, Sharpe, max drawdown, Calmar, win rate, trade log. `momentum()` and `dcf_signal()` ready-made strategies.
- **`portfolio.optimizer`** — Mean-variance optimisation. Max Sharpe (Monte Carlo 5k portfolios), min vol, risk parity (inverse vol), equal weight. Efficient frontier. Constraints: max/min weight.

---

## [0.2.0] — Phase 2: ML Layer

### Added
- **`ml.forecast`** — XGBoost revenue and margin forecasting with 50-bootstrap confidence intervals. `revenue()`, `margins()`, `wacc()`. Returns ForecastResult with point/lower/upper/CAGR/drivers.
- **`ml.factor`** — Rolling OLS factor decomposition. Market, value, momentum, quality, low-vol, size loadings. R², alpha, residual vol.
- **`ml.regime`** — HMM market regime detection with KMeans fallback. Expansion, contraction, recovery, stress regimes. `adjust_wacc()` applies regime-appropriate WACC delta.
- **`ml.nlp`** — Lexicon-based financial text sentiment. Guidance signal detection, risk flag extraction (10 patterns), topic extraction. `analyze_filings()` trend analysis across multiple filings.
- **`ml.cluster`** — KMeans/DBSCAN peer detection from financial ratios. Cosine similarity peer ranking. Synthetic universe of 24 stocks across 4 sectors.
- **`ml.anomaly`** — Isolation Forest on financial ratios + Beneish M-Score (threshold −1.78) + accrual ratio (threshold 0.10). Composite anomaly score.
- **`ml.causal`** — Granger causality via OLS F-statistic proxy. Ranks macro variables by causal strength on earnings with direction and lag.

---

## [0.1.0] — Phase 1: Foundation

### Added
- **`pull.ticker`** — yfinance wrapper. Revenue, EBITDA, net income, FCF history. Balance sheet, cash flow, price history. Properties: `market_cap`, `current_price`, `shares_outstanding`, `total_debt`, `cash`.
- **`pull.edgar`** — SEC EDGAR filings + XBRL facts. Free, no key.
- **`pull.fred`** — Federal Reserve macro data via fredapi.
- **`models.dcf`** — Discounted cash flow. ML-assisted assumptions, manual override, `.use_ml_forecast()`. Properties: `implied_price`, `ev`. `DCF.manual()` constructor for fully manual inputs.
- **`analysis.sensitivity`** — 2-variable sensitivity heatmap. Color-coded rich table output.
- **`analysis.scenarios`** — Bull/base/bear scenario engine. Side-by-side comparison.
- **`export.excel`** — Banker-formatted Excel. Blue = formulas, green = outputs, gray = headers.
- **`export.report`** — Plain text one-pager.
### Added
- `ml.forecast`, `ml.factor`, `ml.regime`, `ml.nlp`
- `ml.cluster`, `ml.anomaly`, `ml.causal`

## [0.1.0] — 2025

### Added
- Initial release: `pull`, `DCF`, `sensitivity`, `scenarios`, `export`
