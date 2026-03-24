"""
finverse.risk.stress_testing
============================
Apply historical stress scenarios to portfolios, DCF models, LBOs, and positions.

Usage
-----
from finverse.risk import stress_testing

# Single scenario on a portfolio
stocks = [pull.ticker(t) for t in ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']]
result = stress_testing.apply(stocks, scenario='gfc_2008')
result.summary()

# Run all scenarios ranked by severity
results = stress_testing.run_all(stocks)
results.summary()

# Apply to a DCF model
from finverse import DCF
model = DCF(data).use_ml_forecast().run()
result = stress_testing.apply_to_dcf(model, scenario='rate_shock_2022')
result.summary()

# Custom scenario
shock = {'equity_return': -0.25, 'rate_shift_bps': 200,
         'credit_spread_bps': 150, 'vix_level': 35}
result = stress_testing.apply(stocks, scenario='custom', shocks=shock)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from finverse.risk._scenarios import (
    ScenarioShocks,
    SCENARIOS,
    get_scenario,
    list_scenarios,
)
from finverse.risk._stress_engine import (
    compute_portfolio_impact,
    compute_dcf_impact,
    identify_key_risk_drivers,
    build_commentary,
)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class StressResult:
    scenario_name: str
    portfolio_return: float
    portfolio_pnl: float | None
    worst_holding: str
    best_holding: str
    var_breach: bool
    dcf_price_impact: float | None
    wacc_stressed: float | None
    key_risk_drivers: list[str]
    commentary: str
    holding_returns: dict[str, float] = field(default_factory=dict)

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()

            color = "red" if self.portfolio_return < -0.20 else "yellow" if self.portfolio_return < -0.10 else "green"
            t = Table(title=f"Stress Test — {self.scenario_name}")
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", justify="right")

            t.add_row("Portfolio Return", f"[{color}]{self.portfolio_return:.2%}[/{color}]")
            if self.portfolio_pnl is not None:
                t.add_row("Portfolio P&L ($)", f"${self.portfolio_pnl:,.0f}")
            t.add_row("Worst Holding", self.worst_holding)
            t.add_row("Best Holding", self.best_holding)
            t.add_row("VaR(99%) Breach", "⚠ YES" if self.var_breach else "✓ No")
            if self.dcf_price_impact is not None:
                t.add_row("DCF Price Impact", f"{self.dcf_price_impact:.2%}")
            if self.wacc_stressed is not None:
                t.add_row("Stressed WACC", f"{self.wacc_stressed:.3%}")
            t.add_row("Key Risk Drivers", " | ".join(self.key_risk_drivers))
            console.print(t)
            console.print(f"\n[italic]{self.commentary}[/italic]\n")

            if self.holding_returns:
                ht = Table(title="Per-Holding Estimated Returns")
                ht.add_column("Ticker", style="bold")
                ht.add_column("Estimated Return", justify="right")
                for ticker, ret in sorted(self.holding_returns.items(), key=lambda x: x[1]):
                    c = "red" if ret < -0.20 else "yellow" if ret < 0 else "green"
                    ht.add_row(ticker, f"[{c}]{ret:.2%}[/{c}]")
                console.print(ht)

        except ImportError:
            print(f"Stress [{self.scenario_name}]: portfolio_return={self.portfolio_return:.2%}")
            for k, v in self.holding_returns.items():
                print(f"  {k}: {v:.2%}")


@dataclass
class StressResultSet:
    results: list[StressResult]

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            # Sort by severity
            sorted_results = sorted(self.results, key=lambda r: r.portfolio_return)
            t = Table(title="Stress Test Summary — All Scenarios (ranked by severity)")
            t.add_column("Scenario", style="bold cyan")
            t.add_column("Portfolio Return", justify="right")
            t.add_column("Worst Holding")
            t.add_column("VaR Breach")
            t.add_column("Top Risk Driver")
            for r in sorted_results:
                color = "red" if r.portfolio_return < -0.20 else "yellow" if r.portfolio_return < -0.10 else "green"
                t.add_row(
                    r.scenario_name,
                    f"[{color}]{r.portfolio_return:.2%}[/{color}]",
                    r.worst_holding,
                    "⚠" if r.var_breach else "✓",
                    r.key_risk_drivers[0] if r.key_risk_drivers else "—",
                )
            console.print(t)
        except ImportError:
            for r in sorted(self.results, key=lambda x: x.portfolio_return):
                print(f"{r.scenario_name}: {r.portfolio_return:.2%}")


# ── Public API ────────────────────────────────────────────────────────────────

def apply(
    holdings: list[Any],
    scenario: str = "gfc_2008",
    shocks: dict | None = None,
    portfolio_value: float | None = None,
    var_99: float | None = None,
) -> StressResult:
    """
    Apply a stress scenario to a list of TickerData holdings.

    Parameters
    ----------
    holdings : list of TickerData objects (or any object with .ticker attribute)
    scenario : str — scenario ID or 'custom'
    shocks : dict — required if scenario='custom'; keys: equity_return,
                    rate_shift_bps, credit_spread_bps, vix_level
    portfolio_value : float — optional total portfolio value for $ P&L
    var_99 : float — optional VaR(99%) threshold for breach check (e.g. 0.15)
    """
    if scenario == "custom":
        if shocks is None:
            raise ValueError("shocks dict required when scenario='custom'")
        scenario_shocks = ScenarioShocks(
            scenario_id="custom",
            name="Custom Scenario",
            description="User-defined shock vector",
            equity_return=shocks.get("equity_return", -0.20),
            rate_shift_bps=shocks.get("rate_shift_bps", 0),
            credit_spread_bps=shocks.get("credit_spread_bps", 0),
            vix_level=shocks.get("vix_level", 25),
            oil_return=shocks.get("oil_return", 0.0),
            usd_return=shocks.get("usd_return", 0.0),
            tech_multiplier=shocks.get("tech_multiplier", 1.0),
            em_multiplier=shocks.get("em_multiplier", 1.0),
            duration_years=shocks.get("duration_years", 1.0),
        )
    else:
        scenario_shocks = get_scenario(scenario)

    impact = compute_portfolio_impact(holdings, scenario_shocks)
    drivers = identify_key_risk_drivers(scenario_shocks)
    commentary = build_commentary(scenario_shocks, impact["portfolio_return"])

    pnl = portfolio_value * impact["portfolio_return"] if portfolio_value else None
    var_breach = abs(impact["portfolio_return"]) > (var_99 or 0.15)

    return StressResult(
        scenario_name=scenario_shocks.name,
        portfolio_return=impact["portfolio_return"],
        portfolio_pnl=pnl,
        worst_holding=impact["worst_holding"],
        best_holding=impact["best_holding"],
        var_breach=var_breach,
        dcf_price_impact=None,
        wacc_stressed=None,
        key_risk_drivers=drivers,
        commentary=commentary,
        holding_returns=impact["holding_returns"],
    )


def run_all(
    holdings: list[Any],
    portfolio_value: float | None = None,
) -> StressResultSet:
    """Run all built-in scenarios and return ranked results."""
    results = []
    for scenario_id in list_scenarios():
        r = apply(holdings, scenario=scenario_id, portfolio_value=portfolio_value)
        results.append(r)
    return StressResultSet(results=results)


def apply_to_dcf(
    dcf_model: Any,
    scenario: str = "gfc_2008",
    shocks: dict | None = None,
) -> StressResult:
    """
    Stress a DCF model — adjusts WACC and growth assumptions.

    Parameters
    ----------
    dcf_model : a finverse DCF result object (must have .wacc, .terminal_growth, .implied_price)
    scenario : str — scenario ID or 'custom'
    shocks : dict — required if scenario='custom'
    """
    if scenario == "custom":
        if shocks is None:
            raise ValueError("shocks dict required when scenario='custom'")
        scenario_shocks = ScenarioShocks(
            scenario_id="custom",
            name="Custom Scenario",
            description="User-defined shock vector",
            equity_return=shocks.get("equity_return", -0.20),
            rate_shift_bps=shocks.get("rate_shift_bps", 0),
            credit_spread_bps=shocks.get("credit_spread_bps", 0),
            vix_level=shocks.get("vix_level", 25),
            oil_return=shocks.get("oil_return", 0.0),
            usd_return=shocks.get("usd_return", 0.0),
            tech_multiplier=shocks.get("tech_multiplier", 1.0),
            em_multiplier=shocks.get("em_multiplier", 1.0),
            duration_years=shocks.get("duration_years", 1.0),
        )
    else:
        scenario_shocks = get_scenario(scenario)

    dcf_impact = compute_dcf_impact(dcf_model, scenario_shocks)
    drivers = identify_key_risk_drivers(scenario_shocks)
    port_return = dcf_impact.get("dcf_price_impact", scenario_shocks.equity_return)
    commentary = build_commentary(scenario_shocks, port_return or 0.0)

    return StressResult(
        scenario_name=scenario_shocks.name,
        portfolio_return=port_return or 0.0,
        portfolio_pnl=None,
        worst_holding="DCF Model",
        best_holding="—",
        var_breach=abs(port_return or 0.0) > 0.20,
        dcf_price_impact=dcf_impact.get("dcf_price_impact"),
        wacc_stressed=dcf_impact.get("wacc_stressed"),
        key_risk_drivers=drivers,
        commentary=commentary,
    )
