"""
finverse.analysis.sensitivity — 2-variable sensitivity heatmap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy


def sensitivity(
    model,
    rows: str = "wacc",
    cols: str = "terminal_growth",
    row_range: tuple | None = None,
    col_range: tuple | None = None,
    n: int = 5,
    output: str = "implied_price",
) -> pd.DataFrame:
    """
    Build a sensitivity table for any two DCF assumptions.

    Parameters
    ----------
    model       : DCF model instance (already configured)
    rows        : assumption name for rows (default "wacc")
    cols        : assumption name for cols (default "terminal_growth")
    row_range   : (min, max) for row variable — auto-set if None
    col_range   : (min, max) for col variable — auto-set if None
    n           : number of steps per axis (default 5)
    output      : "implied_price" or "ev" (default "implied_price")

    Returns
    -------
    pd.DataFrame — heatmap table, printed with color in terminal/Jupyter

    Example
    -------
    >>> from finverse import pull, DCF, sensitivity
    >>> data = pull.ticker("AAPL")
    >>> model = DCF(data)
    >>> table = sensitivity(model, rows="wacc", cols="terminal_growth")
    >>> print(table)
    """
    from finverse.utils.display import console

    DEFAULTS = {
        "wacc": (0.07, 0.13),
        "terminal_growth": (0.01, 0.04),
        "ebitda_margin": (0.20, 0.45),
        "revenue_growth": (0.02, 0.15),
        "capex_pct_revenue": (0.02, 0.10),
        "tax_rate": (0.15, 0.28),
    }

    row_vals = np.linspace(*(row_range or DEFAULTS.get(rows, (0.05, 0.15))), n)
    col_vals = np.linspace(*(col_range or DEFAULTS.get(cols, (0.01, 0.05))), n)

    console.print(f"[dim]Running sensitivity: {rows} × {cols} ({n}×{n} = {n*n} scenarios)...[/dim]")

    results = {}
    for rv in row_vals:
        row_results = {}
        for cv in col_vals:
            m_copy = deepcopy(model)
            m_copy.set(**{rows: rv, cols: cv})
            try:
                res = m_copy.run()
                val = res.implied_price if output == "implied_price" else res.enterprise_value
            except Exception:
                val = np.nan
            row_results[round(cv, 4)] = round(val, 2) if val else np.nan
        results[round(rv, 4)] = row_results

    df = pd.DataFrame(results).T
    df.index.name = rows
    df.columns.name = cols

    _fmt_row = lambda x: f"{x:.1%}" if x < 1 else f"{x:.2f}"
    df.index = [_fmt_row(v) for v in row_vals]
    df.columns = [_fmt_row(v) for v in col_vals]

    console.print(f"[green]✓[/green] Sensitivity table complete\n")
    _print_sensitivity(df, rows, cols, output)
    return df


def _print_sensitivity(df: pd.DataFrame, row_label: str, col_label: str, output: str):
    from finverse.utils.display import console
    from rich.table import Table
    from rich import box

    table = Table(
        title=f"Sensitivity: {output} | {row_label} (rows) × {col_label} (cols)",
        box=box.SIMPLE_HEAD,
        header_style="bold blue",
    )
    table.add_column(f"{row_label} \\ {col_label}")
    for col in df.columns:
        table.add_column(str(col), justify="right")

    vals = df.values.flatten()
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    for idx, row in df.iterrows():
        styled = []
        for v in row:
            if np.isnan(v):
                styled.append("—")
            else:
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                if norm > 0.66:
                    styled.append(f"[bold green]${v:.2f}[/bold green]")
                elif norm > 0.33:
                    styled.append(f"${v:.2f}")
                else:
                    styled.append(f"[red]${v:.2f}[/red]")
        table.add_row(str(idx), *styled)

    console.print(table)
