"""
pull.fred — fetch macro series from the Federal Reserve (FRED).
Requires a free API key from fred.stlouisfed.org
"""
from __future__ import annotations

import os
import pandas as pd

FRED_KEY_ENV = "FRED_API_KEY"

COMMON_SERIES = {
    "GDP": "GDP",
    "UNRATE": "Unemployment rate",
    "CPIAUCSL": "CPI (inflation)",
    "FEDFUNDS": "Fed funds rate",
    "DGS10": "10Y Treasury yield",
    "DGS2": "2Y Treasury yield",
    "T10Y2Y": "Yield curve (10Y-2Y)",
    "VIXCLS": "VIX",
    "DCOILWTICO": "WTI crude oil",
    "DEXUSEU": "USD/EUR exchange rate",
    "HOUST": "Housing starts",
    "INDPRO": "Industrial production",
    "UMCSENT": "Consumer sentiment",
    "BAMLH0A0HYM2": "High yield spread",
}


def fred(*series_ids: str, start: str = "2000-01-01", api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch one or more FRED series.

    Parameters
    ----------
    *series_ids : str   — FRED series IDs, e.g. "GDP", "UNRATE", "FEDFUNDS"
    start       : str   — start date (default "2000-01-01")
    api_key     : str   — FRED API key (or set env var FRED_API_KEY)
                          Get free key at fred.stlouisfed.org

    Returns
    -------
    pd.DataFrame with one column per series, DatetimeIndex

    Example
    -------
    >>> from finverse import pull
    >>> macro = pull.fred("GDP", "UNRATE", "FEDFUNDS")
    >>> macro.tail()
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi is required: pip install fredapi")

    from finverse.utils.display import console

    key = api_key or os.environ.get(FRED_KEY_ENV)
    if not key:
        raise ValueError(
            "FRED API key required.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            f"Then set: export {FRED_KEY_ENV}=your_key_here\n"
            "Or pass: pull.fred('GDP', api_key='your_key')"
        )

    f = Fred(api_key=key)
    dfs = {}

    for sid in series_ids:
        console.print(f"[dim]Fetching FRED: {sid} ({COMMON_SERIES.get(sid, sid)})...[/dim]")
        try:
            s = f.get_series(sid, observation_start=start)
            s.name = sid
            dfs[sid] = s
            console.print(f"[green]✓[/green] {sid} — {len(s)} observations")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] could not fetch {sid}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def macro_snapshot(api_key: str | None = None) -> pd.DataFrame:
    """
    Pull a standard macro dashboard: rates, inflation, growth, sentiment.

    Returns
    -------
    pd.DataFrame with key macro series from 2010 onwards

    Example
    -------
    >>> from finverse import pull
    >>> snap = pull.macro_snapshot()
    """
    key_series = ["FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "UNRATE", "VIXCLS", "BAMLH0A0HYM2"]
    return fred(*key_series, start="2010-01-01", api_key=api_key)
