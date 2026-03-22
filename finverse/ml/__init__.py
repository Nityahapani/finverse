"""
finverse.ml — machine learning layer for financial modeling.

Modules
-------
forecast          — per-company revenue/margin/WACC forecasting (XGBoost)
cross_sectional   — universe-level cross-sectional ML forecasting
garch             — GARCH(1,1), EGARCH, GJR-GARCH volatility modeling
factor            — Fama-French style factor decomposition
regime            — Hidden Markov Model market regime detection
nlp               — sentiment analysis on financial text
cluster           — ML peer group detection
anomaly           — Isolation Forest + Beneish M-Score anomaly detection
causal            — Granger causality: macro → earnings
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

__all__ = [
    "forecast", "cross_sectional", "garch",
    "factor", "regime", "nlp", "cluster", "anomaly", "causal",
]
