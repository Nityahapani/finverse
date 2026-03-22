"""
finverse.risk — risk analysis tools.

Modules
-------
monte_carlo  — Monte Carlo simulation over DCF assumptions
var          — Value at Risk, CVaR, stress testing
evt          — Extreme Value Theory (GPD, tail VaR, return periods)
kelly        — Kelly criterion and optimal position sizing
"""
from finverse.risk import monte_carlo
from finverse.risk import var
from finverse.risk import evt
from finverse.risk import kelly

__all__ = ["monte_carlo", "var", "evt", "kelly"]
