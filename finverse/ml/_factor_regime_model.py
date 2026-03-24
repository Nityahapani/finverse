"""
finverse.ml._factor_regime_model
Regime → factor tilt mapping model.
"""
from __future__ import annotations

import numpy as np

# ── Regime → Factor score table ──────────────────────────────────────────────
# Scores in [-1, +1]; positive = overweight, negative = underweight
# Based on academic literature and historical regime-factor performance

REGIME_FACTOR_SCORES: dict[str, dict[str, float]] = {
    "expansion": {
        "growth":    +0.80,
        "momentum":  +0.70,
        "value":     +0.10,
        "quality":   +0.20,
        "low_vol":   -0.50,
        "size":      +0.40,
    },
    "slowdown": {
        "growth":    -0.30,
        "momentum":  -0.10,
        "value":     +0.30,
        "quality":   +0.70,
        "low_vol":   +0.60,
        "size":      -0.40,
    },
    "contraction": {
        "growth":    -0.70,
        "momentum":  -0.60,
        "value":     +0.40,
        "quality":   +0.65,
        "low_vol":   +0.80,
        "size":      -0.50,
    },
    "recovery": {
        "growth":    +0.50,
        "momentum":  +0.60,
        "value":     +0.80,
        "quality":   +0.10,
        "low_vol":   -0.40,
        "size":      +0.75,
    },
    "stress": {
        "growth":    -0.80,
        "momentum":  -0.70,
        "value":     +0.20,
        "quality":   +0.75,
        "low_vol":   +0.90,
        "size":      -0.65,
    },
}

# Historical accuracy of regime-conditional factor calls (from academic studies)
REGIME_HISTORICAL_ACCURACY: dict[str, float] = {
    "expansion": 0.67,
    "slowdown": 0.61,
    "contraction": 0.72,
    "recovery": 0.65,
    "stress": 0.74,
}

# Yield curve adjustments: when curve is inverted, reduce growth/momentum further
YIELD_CURVE_FACTOR_ADJUSTMENTS: dict[str, float] = {
    "growth":   -0.15,
    "momentum": -0.10,
    "value":    +0.05,
    "quality":  +0.10,
    "low_vol":  +0.15,
    "size":     -0.10,
}

# High VIX (>30) adjustments
HIGH_VIX_FACTOR_ADJUSTMENTS: dict[str, float] = {
    "growth":   -0.10,
    "momentum": -0.15,
    "value":    +0.05,
    "quality":  +0.15,
    "low_vol":  +0.20,
    "size":     -0.15,
}


def compute_factor_scores(
    regime: str,
    yield_curve_slope: float | None = None,   # 10Y - 2Y spread
    vix: float | None = None,
    credit_spread: float | None = None,
) -> dict[str, float]:
    """
    Compute adjusted factor scores given regime and macro context.

    Returns dict of factor → score in [-1, +1].
    """
    base = REGIME_FACTOR_SCORES.get(regime.lower(), REGIME_FACTOR_SCORES["expansion"]).copy()

    # Yield curve adjustment: inverted = additional headwind for growth
    if yield_curve_slope is not None and yield_curve_slope < 0:
        magnitude = min(abs(yield_curve_slope) / 0.02, 1.0)  # scale by inversion depth
        for factor, adj in YIELD_CURVE_FACTOR_ADJUSTMENTS.items():
            base[factor] = np.clip(base[factor] + adj * magnitude, -1.0, 1.0)

    # VIX adjustment
    if vix is not None and vix > 30:
        magnitude = min((vix - 30) / 30, 1.0)
        for factor, adj in HIGH_VIX_FACTOR_ADJUSTMENTS.items():
            base[factor] = np.clip(base[factor] + adj * magnitude, -1.0, 1.0)

    return base


def scores_to_tilts(factor_scores: dict[str, float], scale: float = 0.10) -> dict[str, float]:
    """
    Convert factor scores to portfolio tilts (additive weight adjustments).

    scale : max tilt per factor (default ±10%)
    """
    return {f: round(score * scale, 4) for f, score in factor_scores.items()}


def get_top_and_avoid(
    factor_scores: dict[str, float],
    top_n: int = 3,
    avoid_n: int = 2,
) -> tuple[list[str], list[str]]:
    sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
    top = [f for f, s in sorted_factors[:top_n] if s > 0]
    avoid = [f for f, s in sorted(factor_scores.items(), key=lambda x: x[1])[:avoid_n] if s < 0]
    return top, avoid


def build_rationale(
    regime: str,
    top_factors: list[str],
    avoid_factors: list[str],
    yield_curve_slope: float | None,
    vix: float | None,
) -> str:
    top_str = ", ".join(top_factors) if top_factors else "none"
    avoid_str = ", ".join(avoid_factors) if avoid_factors else "none"
    parts = [
        f"In a {regime} regime, the model recommends overweighting {top_str} "
        f"and underweighting {avoid_str}."
    ]
    if yield_curve_slope is not None and yield_curve_slope < 0:
        parts.append(f"Inverted yield curve ({yield_curve_slope*100:+.0f}bps) adds headwind to growth/momentum.")
    if vix is not None and vix > 30:
        parts.append(f"Elevated VIX ({vix:.0f}) increases weighting toward low-vol and quality.")
    return " ".join(parts)
