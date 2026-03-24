"""
finverse.ml — machine learning layer for financial modeling.

Modules
-------
forecast              — per-company revenue/margin/WACC forecasting (XGBoost)
cross_sectional       — universe-level cross-sectional ML forecasting
garch                 — GARCH(1,1), EGARCH, GJR-GARCH volatility modeling
factor                — Fama-French style factor decomposition
regime                — Hidden Markov Model market regime detection
nlp                   — sentiment analysis on financial text
cluster               — ML peer group detection
anomaly               — Isolation Forest + Beneish M-Score anomaly detection
causal                — Granger causality: macro → earnings
macro_factor_rotation — regime-conditional factor tilt predictions   [v0.7.0]
earnings_surprise     — beat/miss probability before earnings         [v0.7.0]
price_target_ensemble — ML-weighted ensemble price target             [v0.7.0]
"""
from finverse.ml import forecast
from finverse.ml import cross_sectional
from finverse.ml import garch
from finverse.ml import factor
from finverse.ml import regime
from finverse.ml import nlp
from finverse.ml import cluster
from finverse.ml import anomaly
from finverse.ml import causal
from finverse.ml import macro_factor_rotation
from finverse.ml import earnings_surprise
from finverse.ml import price_target_ensemble

__all__ = [
    "forecast", "cross_sectional", "garch",
    "factor", "regime", "nlp", "cluster", "anomaly", "causal",
    "macro_factor_rotation", "earnings_surprise", "price_target_ensemble",
]
