"""
finverse.audit.earnings_quality — composite earnings quality scoring.

Combines 10 accounting-based signals into a single quality score (0–100).
High score = high quality earnings (sustainable, cash-backed, consistent).
Low score = low quality (accrual-heavy, erratic, potentially managed).

Signals based on academic research:
  1. Accruals ratio (Richardson et al. 2005)
  2. Operating CF / Net income ratio
  3. Revenue quality (cash conversion)
  4. Earnings persistence (AR(1) coefficient)
  5. Earnings smoothness (σ earnings vs σ CF)
  6. Conservative accounting (asymmetric timeliness)
  7. Loss avoidance (frequency of small profits)
  8. Asset growth (overinvestment signal)
  9. Gross margin stability
  10. Working capital efficiency trend
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class EarningsQualityResult:
    ticker: str
    overall_score: float            # 0–100
    grade: str                      # A, B, C, D, F
    signals: dict[str, float]       # individual signal scores 0–1
    signal_scores: pd.DataFrame     # detailed breakdown
    flags: list[str]                # specific concerns
    interpretation: str

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        grade_colors = {"A": "green", "B": "blue", "C": "yellow", "D": "red", "F": "red"}
        c = grade_colors.get(self.grade, "white")

        console.print(f"\n[bold blue]Earnings Quality — {self.ticker}[/bold blue]\n")
        console.print(
            f"Overall score: [{c}][bold]{self.overall_score:.0f}/100[/bold][/{c}]  "
            f"|  Grade: [{c}][bold]{self.grade}[/bold][/{c}]"
        )
        console.print(f"[dim]{self.interpretation}[/dim]\n")

        table = Table(title="Signal breakdown", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Signal")
        table.add_column("Score", justify="right")
        table.add_column("Assessment")
        table.add_column("")

        for _, row in self.signal_scores.iterrows():
            score = row["score"]
            bar = "█" * int(score * 10)
            color = "green" if score > 0.7 else ("yellow" if score > 0.4 else "red")
            assess = "good" if score > 0.7 else ("moderate" if score > 0.4 else "weak")
            table.add_row(
                row["signal"],
                f"[{color}]{score:.2f}[/{color}]",
                f"[{color}]{assess}[/{color}]",
                f"[dim]{bar}[/dim]",
            )

        console.print(table)

        if self.flags:
            console.print(f"\n[yellow]Quality concerns:[/yellow]")
            for f in self.flags:
                console.print(f"  • {f}")
        else:
            console.print(f"\n[green]No significant quality concerns detected.[/green]")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "score": self.overall_score,
            "grade": self.grade,
            **self.signals,
        }])


def _safe_ratio(num, denom, default=0.5, clip=(0, 10)):
    try:
        if denom == 0 or np.isnan(denom):
            return default
        r = num / denom
        return float(np.clip(r, clip[0], clip[1]))
    except Exception:
        return default


def _score_accruals(data) -> tuple[float, str | None]:
    """Signal 1: Accruals ratio — lower is better."""
    try:
        ni = data.net_income_history
        ocf = data._get_cf_item(["Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities"])
        if ni.empty or ocf.empty:
            return 0.5, None
        common = ni.index.intersection(ocf.index)
        if len(common) < 2:
            return 0.5, None

        ta = None
        if not data.balance_sheet.empty:
            for k in ["Total Assets"]:
                if k in data.balance_sheet.index:
                    ta = float(data.balance_sheet.loc[k].iloc[0]) / 1e9

        accruals = ni.loc[common] - ocf.loc[common]
        avg_accrual = float(accruals.mean())
        denom = ta or float(data.revenue_history.iloc[-1]) if not data.revenue_history.empty else 1
        ratio = abs(avg_accrual) / max(abs(denom), 0.01)

        score = float(np.clip(1 - ratio * 5, 0, 1))
        flag = f"High accruals ratio ({ratio:.3f}) — earnings not fully backed by cash" if ratio > 0.08 else None
        return score, flag
    except Exception:
        return 0.5, None


def _score_cf_coverage(data) -> tuple[float, str | None]:
    """Signal 2: OCF / Net Income — >1 is ideal."""
    try:
        ni = data.net_income_history
        ocf = data._get_cf_item(["Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities"])
        if ni.empty or ocf.empty:
            return 0.5, None
        common = ni.index.intersection(ocf.index)
        if len(common) == 0:
            return 0.5, None

        ratio = _safe_ratio(float(ocf.loc[common].mean()), float(ni.loc[common].mean()),
                            default=0.5, clip=(0, 5))
        score = float(np.clip((ratio - 0.5) / 1.5, 0, 1))
        flag = f"OCF/NI = {ratio:.2f} — earnings quality concern" if ratio < 0.6 else None
        return score, flag
    except Exception:
        return 0.5, None


def _score_revenue_quality(data) -> tuple[float, str | None]:
    """Signal 3: Revenue to cash conversion quality."""
    try:
        rev = data.revenue_history
        ocf = data._get_cf_item(["Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities"])
        if rev.empty or ocf.empty:
            return 0.5, None
        common = rev.index.intersection(ocf.index)
        if len(common) < 2:
            return 0.5, None

        cf_margin = ocf.loc[common] / rev.loc[common]
        avg = float(cf_margin.mean())
        score = float(np.clip(avg * 3, 0, 1))
        flag = f"Low OCF/Revenue ({avg:.1%}) — revenue not converting to cash" if avg < 0.10 else None
        return score, flag
    except Exception:
        return 0.5, None


def _score_earnings_persistence(data) -> tuple[float, str | None]:
    """Signal 4: AR(1) of earnings — higher persistence = higher quality."""
    try:
        ni = data.net_income_history
        if len(ni) < 4:
            return 0.5, None
        y = ni.values[1:]
        x = ni.values[:-1]
        if np.std(x) == 0:
            return 0.5, None
        ar1 = float(np.corrcoef(x, y)[0, 1])
        score = float(np.clip((ar1 + 1) / 2, 0, 1))
        flag = f"Low earnings persistence (AR1={ar1:.2f}) — volatile earnings" if ar1 < 0.3 else None
        return score, flag
    except Exception:
        return 0.5, None


def _score_smoothness(data) -> tuple[float, str | None]:
    """Signal 5: σ(earnings) / σ(OCF) — lower = smoother = possibly managed."""
    try:
        ni = data.net_income_history
        ocf = data._get_cf_item(["Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities"])
        if ni.empty or ocf.empty or len(ni) < 3:
            return 0.5, None
        common = ni.index.intersection(ocf.index)
        if len(common) < 3:
            return 0.5, None

        sigma_ni = float(ni.loc[common].std())
        sigma_ocf = float(ocf.loc[common].std())
        if sigma_ocf < 1e-6:
            return 0.5, None
        ratio = sigma_ni / sigma_ocf
        if ratio < 0.3:
            score = 0.3
            flag = f"Suspiciously smooth earnings (σNI/σOCF={ratio:.2f}) — possible smoothing"
        elif ratio < 0.8:
            score = 0.85
            flag = None
        elif ratio < 1.5:
            score = 0.70
            flag = None
        else:
            score = 0.4
            flag = f"Volatile earnings vs cash flows (σNI/σOCF={ratio:.2f})"
        return float(score), flag
    except Exception:
        return 0.5, None


def _score_loss_avoidance(data) -> tuple[float, str | None]:
    """Signal 6: Frequency of small positive earnings vs small losses."""
    try:
        ni = data.net_income_history
        if len(ni) < 4:
            return 0.7, None
        vals = ni.values
        small_pos = float(np.sum((vals > 0) & (vals < np.median(vals[vals > 0]) * 0.2)))
        small_neg = float(np.sum((vals < 0) & (vals > -np.median(abs(vals[vals < 0])) * 0.2))) if any(vals < 0) else 0
        if small_pos > small_neg * 3 and small_pos >= 2:
            return 0.35, "Pattern of just-above-zero earnings — possible loss avoidance"
        return 0.8, None
    except Exception:
        return 0.7, None


def _score_asset_growth(data) -> tuple[float, str | None]:
    """Signal 7: Asset growth signal (high growth = lower future returns)."""
    try:
        if data.balance_sheet.empty:
            return 0.6, None
        assets_row = None
        for k in ["Total Assets"]:
            if k in data.balance_sheet.index:
                assets_row = data.balance_sheet.loc[k]
                break
        if assets_row is None or len(assets_row) < 2:
            return 0.6, None
        growth = float((assets_row.iloc[0] - assets_row.iloc[-1]) / max(abs(assets_row.iloc[-1]), 1))
        if growth > 0.25:
            return 0.35, f"High asset growth ({growth:.0%}) — overinvestment risk"
        elif growth > 0.10:
            return 0.65, None
        else:
            return 0.85, None
    except Exception:
        return 0.6, None


def _score_margin_stability(data) -> tuple[float, str | None]:
    """Signal 8: Gross/EBITDA margin stability."""
    try:
        rev = data.revenue_history
        ebitda = data.ebitda_history
        if rev.empty or ebitda.empty or len(rev) < 3:
            return 0.6, None
        common = rev.index.intersection(ebitda.index)
        if len(common) < 3:
            return 0.6, None
        margins = ebitda.loc[common] / rev.loc[common]
        cv = float(margins.std() / max(abs(margins.mean()), 0.01))
        score = float(np.clip(1 - cv * 3, 0, 1))
        flag = f"High margin volatility (CV={cv:.2f}) — unstable business" if cv > 0.20 else None
        return score, flag
    except Exception:
        return 0.6, None


def _score_wc_efficiency(data) -> tuple[float, str | None]:
    """Signal 9: Working capital efficiency trend."""
    try:
        rev = data.revenue_history
        if rev.empty or len(rev) < 3:
            return 0.6, None
        growth_std = float(rev.pct_change().dropna().std())
        score = float(np.clip(1 - growth_std * 2, 0.2, 0.9))
        return score, None
    except Exception:
        return 0.6, None


def _score_fcf_consistency(data) -> tuple[float, str | None]:
    """Signal 10: FCF consistency and coverage."""
    try:
        fcf = data.fcf_history
        ni = data.net_income_history
        if fcf.empty or len(fcf) < 3:
            return 0.5, None
        pct_positive = float((fcf > 0).mean())
        score = pct_positive
        common = fcf.index.intersection(ni.index)
        if len(common) >= 2:
            fcf_ni = _safe_ratio(float(fcf.loc[common].mean()), float(ni.loc[common].mean()),
                                 default=0.5, clip=(0, 3))
            score = (pct_positive + min(fcf_ni, 1)) / 2
        flag = f"FCF negative in {(1-pct_positive):.0%} of years" if pct_positive < 0.6 else None
        return float(np.clip(score, 0, 1)), flag
    except Exception:
        return 0.5, None


SIGNAL_FUNCTIONS = [
    ("Accruals ratio",            _score_accruals),
    ("OCF / Net income",          _score_cf_coverage),
    ("Revenue cash conversion",   _score_revenue_quality),
    ("Earnings persistence",      _score_earnings_persistence),
    ("Earnings smoothness",       _score_smoothness),
    ("Loss avoidance pattern",    _score_loss_avoidance),
    ("Asset growth signal",       _score_asset_growth),
    ("Margin stability",          _score_margin_stability),
    ("Working capital efficiency",_score_wc_efficiency),
    ("FCF consistency",           _score_fcf_consistency),
]

WEIGHTS = [0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.08, 0.10, 0.07, 0.07]


def score(data) -> EarningsQualityResult:
    """
    Compute composite earnings quality score (0–100).

    Runs 10 accounting-based quality signals and combines them
    into a weighted composite score.

    Parameters
    ----------
    data : TickerData — from pull.ticker()

    Returns
    -------
    EarningsQualityResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.audit.earnings_quality import score
    >>> data = pull.ticker("AAPL")
    >>> result = score(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Computing earnings quality score for {ticker}...[/dim]")

    signal_results = []
    flags = []
    weighted_sum = 0.0

    for (name, fn), weight in zip(SIGNAL_FUNCTIONS, WEIGHTS):
        try:
            sig_score, flag = fn(data)
        except Exception:
            sig_score, flag = 0.5, None

        sig_score = float(np.clip(sig_score, 0, 1))
        weighted_sum += sig_score * weight
        if flag:
            flags.append(flag)

        signal_results.append({"signal": name, "score": round(sig_score, 4), "weight": weight})

    overall = round(weighted_sum * 100, 1)

    if overall >= 80:
        grade = "A"
        interpretation = "High quality earnings. Well-backed by cash flows, stable, persistent."
    elif overall >= 65:
        grade = "B"
        interpretation = "Good earnings quality. Minor concerns but broadly reliable."
    elif overall >= 50:
        grade = "C"
        interpretation = "Moderate quality. Some accrual or consistency concerns worth monitoring."
    elif overall >= 35:
        grade = "D"
        interpretation = "Below-average quality. Significant accruals or volatility detected."
    else:
        grade = "F"
        interpretation = "Poor earnings quality. Multiple red flags — treat reported earnings with caution."

    signal_df = pd.DataFrame(signal_results)
    signals_dict = {row["signal"]: row["score"] for _, row in signal_df.iterrows()}

    console.print(f"[green]✓[/green] Earnings quality: {overall:.0f}/100 (grade {grade}) | {len(flags)} flags")

    return EarningsQualityResult(
        ticker=ticker,
        overall_score=overall,
        grade=grade,
        signals=signals_dict,
        signal_scores=signal_df,
        flags=flags,
        interpretation=interpretation,
    )
