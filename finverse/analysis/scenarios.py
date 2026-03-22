"""
finverse.analysis.scenarios — bull / base / bear scenario engine.
"""
from __future__ import annotations

from copy import deepcopy
import pandas as pd


def scenarios(model, bull: dict, base: dict, bear: dict) -> pd.DataFrame:
    """
    Run bull / base / bear scenarios on a DCF model.

    Parameters
    ----------
    model : DCF
    bull  : dict of assumption overrides for bull case
    base  : dict of assumption overrides for base case
    bear  : dict of assumption overrides for bear case

    Returns
    -------
    pd.DataFrame with implied price + EV per scenario

    Example
    -------
    >>> from finverse.analysis.scenarios import scenarios
    >>> table = scenarios(model,
    ...     bull={"revenue_growth": 0.12, "wacc": 0.085, "ebitda_margin": 0.36},
    ...     base={"revenue_growth": 0.08, "wacc": 0.095, "ebitda_margin": 0.32},
    ...     bear={"revenue_growth": 0.03, "wacc": 0.115, "ebitda_margin": 0.26},
    ... )
    """
    from finverse.utils.display import console
    from rich.table import Table
    from rich import box

    cases = {"Bull": bull, "Base": base, "Bear": bear}
    rows = []

    for name, overrides in cases.items():
        m = deepcopy(model)
        m.set(**overrides)
        try:
            res = m.run()
            rows.append({
                "Scenario": name,
                "Implied price": f"${res.implied_price:.2f}",
                "EV ($B)": f"{res.enterprise_value:.1f}",
                "Upside": f"{res.upside_pct:.1%}" if res.upside_pct else "—",
                **{k: f"{v:.1%}" if isinstance(v, float) and v < 2 else v
                   for k, v in overrides.items()},
            })
        except Exception as e:
            rows.append({"Scenario": name, "Error": str(e)})

    df = pd.DataFrame(rows).set_index("Scenario")

    table = Table(title="Scenario Analysis", box=box.SIMPLE_HEAD, header_style="bold blue")
    for col in df.columns:
        table.add_column(col, justify="right" if col != "Scenario" else "left")

    colors = {"Bull": "green", "Base": "white", "Bear": "red"}
    for idx, row in df.iterrows():
        color = colors.get(str(idx), "white")
        table.add_row(*[f"[{color}]{v}[/{color}]" if color != "white" else str(v) for v in row])

    console.print()
    console.print(table)
    console.print()
    return df
