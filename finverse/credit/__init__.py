"""
finverse.credit — credit analysis and distress prediction.

Modules
-------
merton   — Merton structural credit model (distance-to-default, PD, credit spread)
altman   — Altman Z-Score family (public, private, non-manufacturer)
"""
from finverse.credit import merton
from finverse.credit import altman

__all__ = ["merton", "altman"]
