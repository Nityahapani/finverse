"""
finverse.ml.macro_factor_rotation
==================================
Predict which factors (value, growth, momentum, quality, low-vol, size)
are likely to outperform given the current macro regime, yield curve, and VIX.
Integrates with finverse.portfolio.optimizer via factor_tilts.

Usage
-----
from finverse.ml import macro_factor_rotation

result = macro_factor_rotation.predict()
result.summary()

# Apply tilts directly to portfolio optimizer
from finverse.portfolio import optimizer
stocks = [pull.ticker(t) for t in ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']]
result = macro_factor_rotation.predict()
optimizer.optimize(stocks, factor_tilts=result.tilts).summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from finverse.ml._factor_regime_model import (
    compute_factor_scores,
    scores_to_tilts,
    get_top_and_avoid,
    build_rationale,
    REGIME_HISTORICAL_ACCURACY,
)

Horizon = Literal["3m", "6m", "12m"]

# Horizon confidence adjustment (longer horizon = lower confidence)
HORIZON_CONFIDENCE: dict[str, float] = {"3m": 1.0, "6m": 0.85, "12m": 0.70}


@dataclass
class MacroSnapshot:
    """Lightweight macro context snapshot."""
    yield_curve_slope: float | None = None   # 10Y - 2Y spread
    vix: float | None = None
    credit_spread: float | None = None
    fed_funds_rate: float | None = None
    inflation_rate: float | None = None


@dataclass
class FactorRotationResult:
    current_regime: str
    factor_scores: dict[str, float]
    tilts: dict[str, float]
    top_factors: list[str]
    avoid_factors: list[str]
    historical_accuracy: float
    macro_context: MacroSnapshot
    confidence: str
    rationale: str
    horizon: str

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()

            console.print(f"\n[bold cyan]Macro Factor Rotation — {self.horizon} Horizon[/bold cyan]")
            console.print(f"Regime: [bold]{self.current_regime.upper()}[/bold]  |  "
                          f"Confidence: [bold]{self.confidence}[/bold]  |  "
                          f"Historical Accuracy: {self.historical_accuracy:.0%}")

            t = Table(title="Factor Scores & Recommended Tilts")
            t.add_column("Factor", style="bold")
            t.add_column("Score (-1 to +1)", justify="right")
            t.add_column("Portfolio Tilt", justify="right")
            t.add_column("Signal")

            for factor in ["growth", "momentum", "value", "quality", "low_vol", "size"]:
                score = self.factor_scores.get(factor, 0.0)
                tilt = self.tilts.get(factor, 0.0)
                if score > 0.4:
                    signal = "[green]OVERWEIGHT ↑[/green]"
                elif score < -0.4:
                    signal = "[red]UNDERWEIGHT ↓[/red]"
                else:
                    signal = "[yellow]NEUTRAL →[/yellow]"
                t.add_row(factor.replace("_", "-").title(), f"{score:+.2f}", f"{tilt:+.2%}", signal)

            console.print(t)
            console.print(f"\n[italic]{self.rationale}[/italic]\n")

            # Macro context
            mc = self.macro_context
            ctx = Table(title="Macro Context Inputs")
            ctx.add_column("Indicator")
            ctx.add_column("Value", justify="right")
            if mc.yield_curve_slope is not None:
                ctx.add_row("Yield Curve (10Y-2Y)", f"{mc.yield_curve_slope*100:+.0f}bps")
            if mc.vix is not None:
                ctx.add_row("VIX", f"{mc.vix:.1f}")
            if mc.credit_spread is not None:
                ctx.add_row("IG Credit Spread", f"{mc.credit_spread*100:.0f}bps")
            if mc.fed_funds_rate is not None:
                ctx.add_row("Fed Funds Rate", f"{mc.fed_funds_rate:.2%}")
            console.print(ctx)

        except ImportError:
            print(f"Factor Rotation [{self.current_regime}]  top={self.top_factors}  avoid={self.avoid_factors}")
            for f, s in self.factor_scores.items():
                print(f"  {f}: {s:+.2f}  tilt={self.tilts.get(f, 0):+.2%}")


def _detect_regime_from_snapshot(snapshot: MacroSnapshot) -> str:
    """
    Simple heuristic regime detection from macro snapshot.
    Complements finverse.ml.regime (which uses price-based HMM).
    """
    slope = snapshot.yield_curve_slope
    vix = snapshot.vix or 18
    credit = snapshot.credit_spread or 0.010

    if vix > 35 or (credit and credit > 0.025):
        return "stress"
    if slope is not None and slope < -0.005:
        return "contraction"
    if slope is not None and slope < 0.005:
        return "slowdown"
    if slope is not None and slope > 0.010:
        return "recovery"
    return "expansion"


def _load_macro_snapshot() -> MacroSnapshot:
    """
    Try to load current macro snapshot from finverse.pull.macro_snapshot().
    Falls back to reasonable defaults if unavailable.
    """
    try:
        from finverse import pull
        snap = pull.macro_snapshot()
        return MacroSnapshot(
            yield_curve_slope=getattr(snap, "yield_curve_slope", None),
            vix=getattr(snap, "vix", None),
            credit_spread=getattr(snap, "ig_credit_spread", None),
            fed_funds_rate=getattr(snap, "fed_funds_rate", None),
            inflation_rate=getattr(snap, "inflation_rate", None),
        )
    except Exception:
        # Fallback neutral defaults
        return MacroSnapshot(
            yield_curve_slope=0.005,
            vix=18.0,
            credit_spread=0.010,
            fed_funds_rate=0.053,
            inflation_rate=0.030,
        )


def predict(
    horizon: Horizon = "3m",
    regime_result: Any | None = None,
    macro_snapshot: MacroSnapshot | None = None,
) -> FactorRotationResult:
    """
    Predict factor tilts for the given horizon given current macro regime.

    Parameters
    ----------
    horizon : '3m', '6m', or '12m'
    regime_result : RegimeResult from ml.regime.detect() (optional)
    macro_snapshot : MacroSnapshot (optional; auto-loaded from pull.macro_snapshot if not provided)
    """
    # Load macro context
    if macro_snapshot is None:
        macro_snapshot = _load_macro_snapshot()

    # Determine regime
    if regime_result is not None and hasattr(regime_result, "current_regime"):
        regime_str = str(regime_result.current_regime.value
                         if hasattr(regime_result.current_regime, "value")
                         else regime_result.current_regime).lower()
    else:
        regime_str = _detect_regime_from_snapshot(macro_snapshot)

    # Compute factor scores with macro adjustments
    factor_scores = compute_factor_scores(
        regime=regime_str,
        yield_curve_slope=macro_snapshot.yield_curve_slope,
        vix=macro_snapshot.vix,
        credit_spread=macro_snapshot.credit_spread,
    )

    tilts = scores_to_tilts(factor_scores, scale=0.10)
    top_factors, avoid_factors = get_top_and_avoid(factor_scores)
    rationale = build_rationale(
        regime_str, top_factors, avoid_factors,
        macro_snapshot.yield_curve_slope, macro_snapshot.vix,
    )

    base_accuracy = REGIME_HISTORICAL_ACCURACY.get(regime_str, 0.63)
    horizon_adj = HORIZON_CONFIDENCE.get(horizon, 1.0)
    accuracy = base_accuracy * horizon_adj

    if accuracy > 0.68:
        confidence = "HIGH"
    elif accuracy > 0.58:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return FactorRotationResult(
        current_regime=regime_str,
        factor_scores=factor_scores,
        tilts=tilts,
        top_factors=top_factors,
        avoid_factors=avoid_factors,
        historical_accuracy=accuracy,
        macro_context=macro_snapshot,
        confidence=confidence,
        rationale=rationale,
        horizon=horizon,
    )
