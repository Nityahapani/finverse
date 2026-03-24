"""
finverse.options.vol_surface
Implied volatility surface construction: term structure + smile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


@dataclass
class VolSurface:
    """Implied volatility surface by expiry and strike."""
    ticker: str
    surface_df: pd.DataFrame  # index=expiry, columns=moneyness buckets, values=IV
    atm_vols: dict[str, float] = field(default_factory=dict)   # expiry → ATM IV

    def summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            console.print(f"\n[bold cyan]Implied Vol Surface — {self.ticker}[/bold cyan]")
            t = Table(title="ATM Implied Volatility by Expiry")
            t.add_column("Expiry", style="bold")
            t.add_column("ATM IV", justify="right")
            for exp, iv in self.atm_vols.items():
                t.add_row(exp, f"{iv:.2%}")
            console.print(t)
        except ImportError:
            print(f"Vol Surface [{self.ticker}]")
            for exp, iv in self.atm_vols.items():
                print(f"  {exp}: ATM IV = {iv:.2%}")

    def plot(self) -> None:
        """Plot the vol surface (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            if self.surface_df.empty:
                print("No surface data to plot.")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in self.surface_df.columns:
                ax.plot(self.surface_df.index, self.surface_df[col], marker="o", label=str(col))
            ax.set_title(f"Implied Vol Surface — {self.ticker}")
            ax.set_xlabel("Expiry")
            ax.set_ylabel("Implied Volatility")
            ax.legend(title="Moneyness")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib required for plot()")


def build_surface(chain_data: list[dict], ticker: str = "") -> VolSurface:
    """
    Build a VolSurface from a list of option records.

    Each record must have: expiry (str), moneyness (float), iv (float).
    moneyness = strike / spot, so 1.0 = ATM.
    """
    if not chain_data:
        return VolSurface(ticker=ticker, surface_df=pd.DataFrame(), atm_vols={})

    df = pd.DataFrame(chain_data)
    # Pivot: rows = expiry, cols = moneyness bucket
    moneyness_bins = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    df["m_bucket"] = pd.cut(df["moneyness"], bins=moneyness_bins, labels=[f"{m:.2f}" for m in moneyness_bins[:-1]])
    pivot = df.groupby(["expiry", "m_bucket"])["iv"].mean().unstack()

    atm_vols = {}
    for exp in pivot.index:
        # Closest to 1.00
        row = pivot.loc[exp].dropna()
        if not row.empty:
            closest = row.index[np.argmin(np.abs([float(c) - 1.0 for c in row.index]))]
            atm_vols[str(exp)] = float(row[closest])

    return VolSurface(ticker=ticker, surface_df=pivot, atm_vols=atm_vols)
