"""
finverse.risk — risk analysis tools.

Modules
-------
monte_carlo    — Monte Carlo simulation over DCF assumptions
var            — Value at Risk, CVaR, stress testing
evt            — Extreme Value Theory (GPD, tail VaR, return periods)
kelly          — Kelly criterion and optimal position sizing
stress_testing — historical shock scenario library (GFC, COVID, etc.)  
"""
from finverse.risk import monte_carlo
from finverse.risk import var
from finverse.risk import evt
from finverse.risk import kelly
from finverse.risk import stress_testing

__all__ = ["monte_carlo", "var", "evt", "kelly", "stress_testing"]
