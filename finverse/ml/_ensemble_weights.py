"""
finverse.ml._ensemble_weights
Adaptive weight learning for price target ensemble.
Weights are sector/size/regime conditioned.
"""
from __future__ import annotations

import numpy as np

# Default weights by sector — learned from historical predictive accuracy
# (DCF, Comps, Momentum, Consensus)
BASE_WEIGHTS: dict[str, list[float]] = {
    "tech":        [0.30, 0.20, 0.22, 0.28],
    "technology":  [0.30, 0.20, 0.22, 0.28],
    "finance":     [0.25, 0.30, 0.18, 0.27],
    "financial":   [0.25, 0.30, 0.18, 0.27],
    "healthcare":  [0.28, 0.25, 0.15, 0.32],
    "energy":      [0.22, 0.28, 0.25, 0.25],
    "consumer":    [0.27, 0.27, 0.20, 0.26],
    "utilities":   [0.32, 0.25, 0.12, 0.31],
    "default":     [0.30, 0.25, 0.20, 0.25],
}

# Regime adjustments: in stress/contraction, momentum weight drops, DCF increases
REGIME_WEIGHT_ADJUSTMENTS: dict[str, list[float]] = {
    "expansion":   [ 0.00,  0.00,  0.03, -0.03],
    "recovery":    [-0.02,  0.02,  0.05, -0.05],
    "slowdown":    [ 0.03,  0.02, -0.03, -0.02],
    "contraction": [ 0.05,  0.03, -0.06, -0.02],
    "stress":      [ 0.07,  0.02, -0.07, -0.02],
}


def get_weights(
    sector: str = "default",
    regime: str = "expansion",
    has_consensus: bool = True,
) -> dict[str, float]:
    """
    Return adaptive ensemble weights for (DCF, Comps, Momentum, Consensus).

    Parameters
    ----------
    sector : str
    regime : str
    has_consensus : bool — if False, redistribute consensus weight to DCF/Comps

    Returns dict with keys: dcf, comps, momentum, consensus
    """
    base = BASE_WEIGHTS.get(sector.lower(), BASE_WEIGHTS["default"]).copy()
    adj = REGIME_WEIGHT_ADJUSTMENTS.get(regime.lower(), [0, 0, 0, 0])
    weights = [b + a for b, a in zip(base, adj)]

    if not has_consensus:
        consensus_w = weights[3]
        weights[3] = 0.0
        # Redistribute to DCF and Comps
        weights[0] += consensus_w * 0.55
        weights[1] += consensus_w * 0.45

    # Normalize to sum to 1
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    return {
        "dcf": round(weights[0], 4),
        "comps": round(weights[1], 4),
        "momentum": round(weights[2], 4),
        "consensus": round(weights[3], 4),
    }


def compute_ensemble(
    targets: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    """
    Compute weighted ensemble target from available signals.

    targets : dict with keys dcf, comps, momentum, consensus (None = missing)
    weights : dict from get_weights()
    """
    total_w = 0.0
    weighted_sum = 0.0
    for key, val in targets.items():
        if val is not None:
            w = weights.get(key, 0.0)
            weighted_sum += val * w
            total_w += w
    if total_w == 0:
        return 0.0
    return weighted_sum / total_w


def signal_agreement(targets: dict[str, float | None], ensemble: float) -> str:
    """
    Assess how much the individual signals agree.
    Returns 'HIGH', 'MEDIUM', or 'LOW'.
    """
    valid = [v for v in targets.values() if v is not None]
    if len(valid) < 2:
        return "LOW"
    deviations = [abs(v - ensemble) / max(abs(ensemble), 1.0) for v in valid]
    avg_dev = np.mean(deviations)
    if avg_dev < 0.08:
        return "HIGH"
    if avg_dev < 0.18:
        return "MEDIUM"
    return "LOW"


def compute_confidence_intervals(
    targets: dict[str, float | None],
    ensemble: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute 80% and 95% confidence intervals from signal dispersion.
    Uses the cross-signal standard deviation scaled by a coverage factor.
    """
    valid = [v for v in targets.values() if v is not None]
    if len(valid) < 2:
        # Use ±10% and ±18% defaults
        return (ensemble * 0.90, ensemble * 1.10), (ensemble * 0.82, ensemble * 1.18)

    std = float(np.std(valid))
    ci_80 = (ensemble - 1.28 * std, ensemble + 1.28 * std)
    ci_95 = (ensemble - 1.96 * std, ensemble + 1.96 * std)
    return ci_80, ci_95
