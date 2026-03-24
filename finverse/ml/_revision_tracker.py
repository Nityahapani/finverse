"""
finverse.ml._revision_tracker
Estimate revision momentum computation.
"""
from __future__ import annotations

from typing import Any


def compute_revision_momentum(data: Any) -> float:
    """
    Compute analyst estimate revision momentum.

    Returns a float in [-1, +1]:
      +1 = strong upward revisions
       0 = flat / no data
      -1 = strong downward revisions

    Tries to pull revision trend from yfinance analyst data.
    Falls back to 0.0 if unavailable.
    """
    try:
        import yfinance as yf
        ticker = getattr(data, "ticker", str(data))
        yf_ticker = yf.Ticker(ticker)

        # analyst_price_targets gives current/low/high/mean
        recs = yf_ticker.recommendations_summary
        if recs is not None and not recs.empty:
            # Use buy vs sell ratio as proxy for revision direction
            if "strongBuy" in recs.columns and "strongSell" in recs.columns:
                row = recs.iloc[-1]
                total = float(row.sum()) if row.sum() > 0 else 1.0
                buy_score = (row.get("strongBuy", 0) + row.get("buy", 0)) / total
                sell_score = (row.get("strongSell", 0) + row.get("sell", 0)) / total
                return float(buy_score - sell_score)
    except Exception:
        pass
    return 0.0


def classify_momentum(momentum: float) -> str:
    if momentum > 0.3:
        return "Strong upward revisions"
    if momentum > 0.1:
        return "Mild upward revisions"
    if momentum < -0.3:
        return "Strong downward revisions"
    if momentum < -0.1:
        return "Mild downward revisions"
    return "Flat / no clear revision trend"
