"""
finverse.risk._scenarios
Built-in historical stress scenario shock vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioShocks:
    """Shock vector for a single stress scenario."""
    scenario_id: str
    name: str
    description: str
    equity_return: float          # total equity market return (e.g. -0.55)
    rate_shift_bps: float         # parallel shift in risk-free rates (bps)
    credit_spread_bps: float      # change in IG credit spreads (bps)
    vix_level: float              # peak VIX level
    oil_return: float             # crude oil price return
    usd_return: float             # USD index return (+ = USD strengthens)
    tech_multiplier: float        # additional tech sector multiplier vs market
    em_multiplier: float          # EM vs developed market multiplier
    duration_years: float         # scenario duration in years


SCENARIOS: dict[str, ScenarioShocks] = {
    "gfc_2008": ScenarioShocks(
        scenario_id="gfc_2008",
        name="Global Financial Crisis (2008-09)",
        description="Lehman collapse, credit freeze, global recession",
        equity_return=-0.55,
        rate_shift_bps=-300,
        credit_spread_bps=500,
        vix_level=80,
        oil_return=-0.70,
        usd_return=0.12,
        tech_multiplier=1.0,
        em_multiplier=1.2,
        duration_years=1.5,
    ),
    "covid_2020": ScenarioShocks(
        scenario_id="covid_2020",
        name="COVID Crash (Feb-Mar 2020)",
        description="Fastest bear market in history; 34% drawdown in 33 days",
        equity_return=-0.34,
        rate_shift_bps=-150,
        credit_spread_bps=300,
        vix_level=85,
        oil_return=-0.60,
        usd_return=0.08,
        tech_multiplier=0.8,
        em_multiplier=1.1,
        duration_years=0.2,
    ),
    "dotcom_2000": ScenarioShocks(
        scenario_id="dotcom_2000",
        name="Dot-com Bust (2000-02)",
        description="Tech bubble collapse; NASDAQ -78%, S&P -49%",
        equity_return=-0.49,
        rate_shift_bps=-525,
        credit_spread_bps=200,
        vix_level=45,
        oil_return=0.10,
        usd_return=0.05,
        tech_multiplier=1.8,
        em_multiplier=0.9,
        duration_years=2.5,
    ),
    "rate_shock_1994": ScenarioShocks(
        scenario_id="rate_shock_1994",
        name="Fed Rate Shock (1994)",
        description="Fed doubled rates in 12 months; bond market crash",
        equity_return=-0.10,
        rate_shift_bps=300,
        credit_spread_bps=100,
        vix_level=23,
        oil_return=0.05,
        usd_return=-0.08,
        tech_multiplier=0.9,
        em_multiplier=1.3,
        duration_years=1.0,
    ),
    "rate_shock_2022": ScenarioShocks(
        scenario_id="rate_shock_2022",
        name="Rate Shock (2022)",
        description="Fed hiked 425bps; bonds and growth stocks crushed",
        equity_return=-0.20,
        rate_shift_bps=425,
        credit_spread_bps=120,
        vix_level=36,
        oil_return=0.40,
        usd_return=0.15,
        tech_multiplier=1.6,
        em_multiplier=1.2,
        duration_years=1.0,
    ),
    "asian_crisis_1997": ScenarioShocks(
        scenario_id="asian_crisis_1997",
        name="Asian Financial Crisis (1997)",
        description="EM currency collapse; contagion across Asia",
        equity_return=-0.15,
        rate_shift_bps=-75,
        credit_spread_bps=180,
        vix_level=38,
        oil_return=-0.30,
        usd_return=0.10,
        tech_multiplier=0.7,
        em_multiplier=2.5,
        duration_years=1.5,
    ),
    "russia_default_1998": ScenarioShocks(
        scenario_id="russia_default_1998",
        name="Russia Default / LTCM (1998)",
        description="Sovereign default + hedge fund systemic risk",
        equity_return=-0.20,
        rate_shift_bps=-75,
        credit_spread_bps=300,
        vix_level=45,
        oil_return=-0.35,
        usd_return=-0.05,
        tech_multiplier=0.8,
        em_multiplier=2.0,
        duration_years=0.5,
    ),
}


def get_scenario(scenario_id: str) -> ScenarioShocks:
    """Retrieve a named scenario or raise KeyError."""
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {list(SCENARIOS.keys())} + 'custom'"
        )
    return SCENARIOS[scenario_id]


def list_scenarios() -> list[str]:
    return list(SCENARIOS.keys())
