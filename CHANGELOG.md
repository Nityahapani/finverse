# Changelog

All notable changes are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.6.0] — 2025

### Added
- `risk.evt` — Extreme Value Theory (Peaks-Over-Threshold, GPD) for tail VaR at 99.9%+
- `risk.kelly` — Kelly criterion: continuous, binary, multi-asset
- `models.ddm` — Dividend Discount Models: Gordon Growth, H-Model, Multistage
- `models.sotp` — Sum of the Parts valuation
- `audit.earnings_quality` — 10-factor composite earnings quality score (0–100)

## [0.5.0] — 2025

### Added
- `ml.garch` — GARCH(1,1), EGARCH, GJR-GARCH (pure scipy MLE, no arch package)
- `ml.cross_sectional` — Universe-level cross-sectional ML forecasting
- `portfolio.hrp` — Hierarchical Risk Parity (Lopez de Prado 2016)
- `portfolio.shrinkage` — Ledoit-Wolf covariance shrinkage
- `credit.merton` — Merton distance-to-default, PD, credit spread, rating
- `credit.altman` — Z-Score, Z'-Score, Z''-Score
- `valuation.real_options` — Expand, abandon, defer options (Black-Scholes)
- `valuation.apv` — Adjusted Present Value (Modigliani-Miller)
- `audit.loughran_mcdonald` — Financial-domain sentiment dictionary
- `audit.benford` — Benford's Law test for data manipulation
- `macro.var_model` — VAR(p), impulse response functions, Granger causality
- `macro.nelson_siegel` — Nelson-Siegel and Svensson yield curve fitting

## [0.4.0] — 2025

### Added
- `models.lbo` — Full LBO with debt schedule, IRR, MoM
- `models.three_statement` — Linked IS/BS/CF model
- `models.comps` — Comparable company analysis
- `models.macro` — GDP nowcast, recession probability
- `audit` — Model health checker

## [0.3.0] — 2025

### Added
- `risk.monte_carlo`, `risk.var`
- `screen.screener`, `backtest.engine`
- `portfolio.optimizer`

## [0.2.0] — 2025

### Added
- `ml.forecast`, `ml.factor`, `ml.regime`, `ml.nlp`
- `ml.cluster`, `ml.anomaly`, `ml.causal`

## [0.1.0] — 2025

### Added
- Initial release: `pull`, `DCF`, `sensitivity`, `scenarios`, `export`
