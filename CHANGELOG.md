# Changelog

All notable changes to finverse are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.7.0] — Phase 7: Advanced Valuation, Portfolio, and Audit

### Added — Valuation
- **`models.regime_dcf`** — Regime-Conditional DCF. Wires the HMM regime detector directly into the DCF engine. Each macro regime (expansion, slowdown, contraction, recovery, stress) gets its own WACC, revenue growth, EBITDA margin, and terminal growth assumptions. Output is a probability-weighted implied price distribution rather than a single point estimate. Regime discount typically ranges ±10–20% vs static DCF.
- **`models.synthetic_peers`** — Synthetic Peer Construction. For companies with no clean peer set (conglomerates, niche sectors, early-stage), builds implied multiples by blending sector benchmarks weighted by segment mix. A 60/40 software/hardware company gets `0.6 × software_multiples + 0.4 × hardware_multiples`. Covers 14 sector benchmarks across 10 multiple types. Returns implied price range at 25th, median, and 75th percentile.
- **`models.options`** — European options pricing with full Greeks. Black-Scholes call and put pricing, delta, gamma, theta, vega, rho. Implied volatility solver via Brent's method (guaranteed convergence, no initial guess needed). Volatility surface across strikes and maturities. Put-call parity verification. Pure numpy/scipy — no external options library.
- **`models.bonds`** — Fixed-rate bond pricing and risk measures. Clean price, dirty price, accrued interest, yield-to-maturity, current yield, Macaulay duration, modified duration, convexity, DV01. YTM solver from market price. Price-yield relationship table. Yield shock scenario analysis (±25, ±50, ±100 bps) built into every result.

### Added — Portfolio
- **`portfolio.black_litterman`** — Black-Litterman model. Combines CAPM market equilibrium returns with analyst views using Bayesian updating. Views can be absolute ("AAPL returns 15%") or relative ("MSFT outperforms GOOGL by 3%"). Confidence parameter controls how far posterior moves from equilibrium. Uses Ledoit-Wolf shrinkage on the covariance matrix automatically. Significantly more stable than raw MVO — doesn't amplify estimation errors.
- **`portfolio.cvar_opt`** — CVaR (Conditional Value at Risk) minimisation. Solves for weights that minimise expected tail loss directly via Rockafellar-Uryasev linear programming (scipy HiGHS solver). Unlike mean-variance which penalises all variance symmetrically, CVaR optimisation only penalises downside tail losses. Supports confidence levels (95%, 99%), max/min weight constraints, and optional minimum return constraint.

### Added — Audit
- **`audit.manipulation`** — Accounting Manipulation Fingerprinting. 40+ accounting signals combined into a weighted manipulation probability score (0–1). Goes well beyond the classic 8-signal Beneish M-Score by adding revenue-cash divergence, SGA inflation relative to revenue, AR growth vs revenue growth, asset quality index, six interaction terms, and multi-signal pattern detection. Trained on a synthetic universe calibrated to known manipulation patterns (Enron, WorldCom-style). Returns Beneish M-Score alongside the composite score for comparison.

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
- **`export.report`** — Plain text one-pager.
### Added
- `ml.forecast`, `ml.factor`, `ml.regime`, `ml.nlp`
- `ml.cluster`, `ml.anomaly`, `ml.causal`

## [0.1.0] — 2025

### Added
- Initial release: `pull`, `DCF`, `sensitivity`, `scenarios`, `export`
