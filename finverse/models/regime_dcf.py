"""
finverse.models.regime_dcf — Regime-Conditional Valuation.

Wires the HMM regime detector directly into the DCF engine.
Instead of a single implied price, outputs a regime-weighted
probability distribution of intrinsic values.

Each macro regime gets its own set of DCF assumptions:
  - Expansion:   higher growth, tighter WACC, better margins
  - Slowdown:    moderate growth, normal WACC
  - Contraction: haircut growth, wider WACC, margin compression
  - Recovery:    rebounding growth, normalising WACC

The final implied price is the probability-weighted average
across all regimes — a much more honest valuation than a
single-point DCF with fixed assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Regime assumption adjustments (deltas on top of base model) ────────────
REGIME_ADJUSTMENTS = {
    "EXPANSION": {
        "wacc_delta":           -0.010,   # WACC tightens 100bps
        "revenue_growth_delta": +0.030,   # growth 3pp above base
        "ebitda_margin_delta":  +0.020,   # margins expand 2pp
        "terminal_growth_delta":+0.005,   # higher long-run growth
        "label":                "Expansion",
        "color":                "green",
    },
    "SLOWDOWN": {
        "wacc_delta":            0.000,
        "revenue_growth_delta":  0.000,
        "ebitda_margin_delta":   0.000,
        "terminal_growth_delta": 0.000,
        "label":                "Slowdown",
        "color":                "yellow",
    },
    "CONTRACTION": {
        "wacc_delta":           +0.020,   # WACC widens 200bps
        "revenue_growth_delta": -0.040,   # growth 4pp below base
        "ebitda_margin_delta":  -0.030,   # margins compress 3pp
        "terminal_growth_delta":-0.010,
        "label":                "Contraction",
        "color":                "red",
    },
    "RECOVERY": {
        "wacc_delta":           -0.005,
        "revenue_growth_delta": +0.015,
        "ebitda_margin_delta":  +0.010,
        "terminal_growth_delta": 0.000,
        "label":                "Recovery",
        "color":                "blue",
    },
    "STRESS": {
        "wacc_delta":           +0.035,
        "revenue_growth_delta": -0.070,
        "ebitda_margin_delta":  -0.050,
        "terminal_growth_delta":-0.015,
        "label":                "Stress",
        "color":                "red",
    },
}


@dataclass
class RegimeDCFResult:
    ticker: str
    current_regime: str
    regime_probabilities: dict[str, float]
    regime_prices: dict[str, float]
    regime_assumptions: dict[str, dict]
    weighted_price: float
    static_price: float
    regime_discount: float
    regime_discount_pct: float
    current_price: float | None

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]Regime-Conditional DCF — {self.ticker}[/bold blue]")
        console.print(
            f"[dim]Current regime: "
            f"[bold]{self.current_regime}[/bold] "
            f"({self.regime_probabilities.get(self.current_regime, 0):.0%} probability)[/dim]\n"
        )

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Regime")
        table.add_column("Prob", justify="right")
        table.add_column("WACC", justify="right")
        table.add_column("Rev growth", justify="right")
        table.add_column("EBITDA margin", justify="right")
        table.add_column("Implied price", justify="right")
        table.add_column("Contribution", justify="right")

        color_map = {r: v["color"] for r, v in REGIME_ADJUSTMENTS.items()}

        for regime, prob in self.regime_probabilities.items():
            if prob < 0.001:
                continue
            price  = self.regime_prices.get(regime, 0)
            assum  = self.regime_assumptions.get(regime, {})
            contrib = prob * price
            c = color_map.get(regime, "white")
            table.add_row(
                f"[{c}]{regime.capitalize()}[/{c}]",
                f"{prob:.0%}",
                f"{assum.get('wacc', 0):.1%}",
                f"{assum.get('revenue_growth', 0):.1%}",
                f"{assum.get('ebitda_margin', 0):.1%}",
                f"[{c}]${price:.2f}[/{c}]",
                f"${contrib:.2f}",
            )

        console.print(table)

        disc_color = "red" if self.regime_discount < 0 else "green"
        console.print(f"\n  Static DCF price:           ${self.static_price:.2f}")
        console.print(
            f"  Regime-weighted price:      [bold]${self.weighted_price:.2f}[/bold]"
        )
        console.print(
            f"  Regime adjustment:          "
            f"[{disc_color}]{self.regime_discount:+.2f} "
            f"({self.regime_discount_pct:+.1%})[/{disc_color}]"
        )
        if self.current_price:
            upside = (self.weighted_price - self.current_price) / self.current_price
            u_color = "green" if upside > 0 else "red"
            console.print(
                f"  Current price:              ${self.current_price:.2f}  "
                f"([{u_color}]{upside:+.1%} upside[/{u_color}])"
            )
        console.print()

    def to_df(self) -> pd.DataFrame:
        rows = []
        for regime, prob in self.regime_probabilities.items():
            rows.append({
                "regime":        regime,
                "probability":   prob,
                "implied_price": self.regime_prices.get(regime, 0),
                "contribution":  prob * self.regime_prices.get(regime, 0),
                **self.regime_assumptions.get(regime, {}),
            })
        return pd.DataFrame(rows).set_index("regime").round(4)


def analyze(
    data,
    base_model=None,
    regime_result=None,
) -> RegimeDCFResult:
    """
    Run a regime-conditional DCF.

    Detects the current macro regime, assigns probability weights to
    each regime, adjusts DCF assumptions per regime, and returns a
    probability-weighted implied price distribution.

    Parameters
    ----------
    data          : TickerData from pull.ticker()
    base_model    : DCF instance (optional — built from data if not provided)
    regime_result : RegimeResult from ml.regime.detect() (optional — auto-detected)

    Returns
    -------
    RegimeDCFResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.models.regime_dcf import analyze as regime_dcf
    >>> data = pull.ticker("AAPL")
    >>> result = regime_dcf(data)
    >>> result.summary()

    With pre-built models:
    >>> from finverse.models.dcf import DCF
    >>> from finverse.ml.regime import detect
    >>> base = DCF(data)
    >>> base.run()
    >>> regime = detect(data.price_history["Close"])
    >>> result = regime_dcf(data, base_model=base, regime_result=regime)
    """
    from finverse.utils.display import console
    from finverse.models.dcf import DCF

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Running regime-conditional DCF for {ticker}...[/dim]")

    # ── Build base DCF ────────────────────────────────────────────────────
    if base_model is None:
        base_model = DCF(data)
        base_model.run()
    elif base_model._results is None:
        base_model.run()

    base_a    = base_model._assumptions
    static_px = float(base_model._results.implied_price)

    # ── Detect regime ─────────────────────────────────────────────────────
    regime_probs: dict[str, float] = {}
    current_regime = "SLOWDOWN"

    if regime_result is not None:
        current_regime = str(regime_result.current_regime.value).upper() \
            if hasattr(regime_result.current_regime, "value") \
            else str(regime_result.current_regime).upper()
        if hasattr(regime_result, "regime_probs"):
            regime_probs = {
                str(k).upper(): float(v)
                for k, v in regime_result.regime_probs.items()
            }
    else:
        try:
            from finverse.ml.regime import detect
            if not data.price_history.empty:
                r = detect(data.price_history["Close"])
                current_regime = str(r.current_regime.value).upper() \
                    if hasattr(r.current_regime, "value") \
                    else str(r.current_regime).upper()
                if hasattr(r, "regime_probs"):
                    regime_probs = {
                        str(k).upper(): float(v)
                        for k, v in r.regime_probs.items()
                    }
        except Exception:
            pass

    # Default uniform probs if not available
    if not regime_probs:
        default_probs = {
            "EXPANSION":   0.25,
            "SLOWDOWN":    0.30,
            "CONTRACTION": 0.30,
            "RECOVERY":    0.15,
        }
        # Upweight current regime
        for k in default_probs:
            default_probs[k] = 0.10
        default_probs[current_regime] = 0.70
        regime_probs = default_probs

    # Normalise
    total = sum(regime_probs.values())
    regime_probs = {k: v / total for k, v in regime_probs.items()}

    # ── Per-regime DCF ────────────────────────────────────────────────────
    regime_prices: dict[str, float]     = {}
    regime_assumptions: dict[str, dict] = {}

    for regime_key, prob in regime_probs.items():
        if prob < 0.001:
            regime_prices[regime_key] = static_px
            continue

        adj = REGIME_ADJUSTMENTS.get(regime_key, REGIME_ADJUSTMENTS["SLOWDOWN"])

        new_wacc   = float(np.clip(base_a.wacc   + adj["wacc_delta"],   0.04, 0.25))
        rg = base_a.revenue_growth
        if rg is None:
            base_rg = 0.08
        elif isinstance(rg, (int, float)):
            base_rg = float(rg)
        else:
            arr = [x for x in rg if x is not None]
            base_rg = float(np.mean(arr)) if arr else 0.08
        new_growth = float(np.clip(base_rg + adj["revenue_growth_delta"], -0.20, 0.50))
        new_margin = float(np.clip(
            base_a.ebitda_margin + adj["ebitda_margin_delta"], 0.02, 0.80
        ))
        new_tg     = float(np.clip(
            base_a.terminal_growth + adj["terminal_growth_delta"], 0.005, 0.04
        ))

        # Guard: terminal_growth must be < wacc
        if new_tg >= new_wacc:
            new_tg = new_wacc - 0.01

        try:
            m = DCF.manual(
                base_revenue=base_model._base_revenue or 100.0,
                shares_outstanding=base_model._shares or 1.0,
                net_debt=base_model._net_debt or 0.0,
                current_price=data.current_price,
            )
            m.set(
                wacc=new_wacc,
                terminal_growth=new_tg,
                revenue_growth=new_growth,
                ebitda_margin=new_margin,
                projection_years=base_a.projection_years,
            )
            r = m.run()
            price = float(np.clip(r.implied_price, 0, static_px * 5))
        except Exception:
            price = static_px

        regime_prices[regime_key] = round(price, 2)
        regime_assumptions[regime_key] = {
            "wacc":           round(new_wacc, 4),
            "revenue_growth": round(new_growth, 4),
            "ebitda_margin":  round(new_margin, 4),
            "terminal_growth":round(new_tg, 4),
        }

    # ── Weighted price ────────────────────────────────────────────────────
    weighted = sum(
        regime_probs.get(r, 0) * regime_prices.get(r, static_px)
        for r in regime_probs
    )
    weighted       = round(float(weighted), 2)
    regime_disc    = round(weighted - static_px, 2)
    regime_disc_pct= round(regime_disc / static_px, 4) if static_px else 0

    console.print(
        f"[green]✓[/green] Regime-conditional DCF — "
        f"current regime: {current_regime} | "
        f"weighted price: ${weighted:.2f} | "
        f"static: ${static_px:.2f} | "
        f"adjustment: {regime_disc:+.2f}"
    )

    return RegimeDCFResult(
        ticker=ticker,
        current_regime=current_regime,
        regime_probabilities={k: round(v, 4) for k, v in regime_probs.items()},
        regime_prices=regime_prices,
        regime_assumptions=regime_assumptions,
        weighted_price=weighted,
        static_price=static_px,
        regime_discount=regime_disc,
        regime_discount_pct=regime_disc_pct,
        current_price=data.current_price if hasattr(data, "current_price") else None,
    )
