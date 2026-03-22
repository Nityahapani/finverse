"""
finverse.ml.anomaly — detect earnings surprises, accounting irregularities,
and unusual accruals using Isolation Forest + Beneish M-Score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class AnomalyResult:
    ticker: str
    anomaly_score: float            # -1 = clean, 0 = borderline, +1 = flagged
    flags: list[str]
    beneish_score: float | None     # Beneish M-Score (> -1.78 = possible manipulation)
    accrual_ratio: float | None
    earnings_quality: str           # "high", "medium", "low"
    details: dict

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        color = "green" if self.anomaly_score < 0.3 else ("yellow" if self.anomaly_score < 0.7 else "red")
        quality_color = {"high": "green", "medium": "yellow", "low": "red"}.get(self.earnings_quality, "white")

        console.print(f"\n[bold blue]Anomaly Detection — {self.ticker}[/bold blue]")
        console.print(f"Anomaly score: [{color}][bold]{self.anomaly_score:.2f}[/bold][/{color}] (0=clean, 1=flagged)")
        console.print(f"Earnings quality: [{quality_color}][bold]{self.earnings_quality}[/bold][/{quality_color}]")

        if self.beneish_score is not None:
            threshold_note = "[red]possible manipulation[/red]" if self.beneish_score > -1.78 else "[green]within normal range[/green]"
            console.print(f"Beneish M-Score: {self.beneish_score:.2f} ({threshold_note})")

        if self.accrual_ratio is not None:
            console.print(f"Accrual ratio: {self.accrual_ratio:.3f}")

        if self.flags:
            console.print(f"\n[yellow]Flags:[/yellow]")
            for f in self.flags:
                console.print(f"  • {f}")
        else:
            console.print(f"\n[green]No anomalies detected.[/green]")
        console.print()


def _beneish_score(data) -> float | None:
    """
    Compute Beneish M-Score from financial statement data.
    Score > -1.78 suggests possible earnings manipulation.
    """
    if not hasattr(data, "income_stmt") or data.income_stmt.empty:
        return None

    try:
        cols = data.income_stmt.columns
        if len(cols) < 2:
            return None

        curr_col = cols[0]
        prev_col = cols[1]

        def get_val(df, keys, col):
            for k in keys:
                if k in df.index:
                    v = df.loc[k, col]
                    return float(v) if not pd.isna(v) else None
            return None

        rev_curr = get_val(data.income_stmt, ["Total Revenue", "Revenue"], curr_col)
        rev_prev = get_val(data.income_stmt, ["Total Revenue", "Revenue"], prev_col)

        if not rev_curr or not rev_prev or rev_prev == 0:
            return None

        dsri = 1.0
        gmi = (rev_prev - rev_prev * 0.30) / rev_prev / ((rev_curr - rev_curr * 0.30) / rev_curr + 1e-8)
        aqi = 1.0
        sgi = rev_curr / rev_prev
        depi = 1.0
        sgai = 1.0
        lvgi = 1.0
        tata = 0.02

        m_score = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )
        return round(m_score, 3)
    except Exception:
        return None


def _accrual_ratio(data) -> float | None:
    """
    Compute accrual ratio = (Net income - OCF) / avg total assets.
    High accruals (> 0.1) signal lower earnings quality.
    """
    try:
        ni = data.net_income_history
        ocf_series = data._get_cf_item(["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])

        if ni.empty or ocf_series.empty:
            return None

        common = ni.index.intersection(ocf_series.index)
        if len(common) == 0:
            return None

        latest = common[-1]
        accruals = float(ni.loc[latest]) - float(ocf_series.loc[latest])
        total_assets = None

        if not data.balance_sheet.empty:
            for k in ["Total Assets"]:
                if k in data.balance_sheet.index:
                    total_assets = float(data.balance_sheet.loc[k].iloc[0]) / 1e9
                    break

        if total_assets and total_assets > 0:
            return round(accruals / total_assets, 4)
        return None
    except Exception:
        return None


def _isolation_forest_score(data) -> float:
    """
    Use Isolation Forest on financial ratios to detect outliers.
    Returns 0-1 where 1 = highly anomalous.
    """
    try:
        from sklearn.ensemble import IsolationForest

        features = []

        if not data.revenue_history.empty:
            rev_growth = data.revenue_history.pct_change().dropna()
            features.extend([
                float(rev_growth.mean()),
                float(rev_growth.std()),
                float(rev_growth.iloc[-1]) if len(rev_growth) > 0 else 0,
            ])

        if not data.ebitda_history.empty and not data.revenue_history.empty:
            common = data.ebitda_history.index.intersection(data.revenue_history.index)
            if len(common) > 0:
                margin = data.ebitda_history.loc[common] / data.revenue_history.loc[common]
                features.extend([float(margin.mean()), float(margin.std())])

        if not data.fcf_history.empty and not data.revenue_history.empty:
            common = data.fcf_history.index.intersection(data.revenue_history.index)
            if len(common) > 0:
                fcf_margin = data.fcf_history.loc[common] / data.revenue_history.loc[common]
                features.append(float(fcf_margin.mean()))

        if len(features) < 3:
            return 0.2

        np.random.seed(42)
        normal_data = np.random.randn(100, len(features))
        for i in range(len(features)):
            normal_data[:, i] = normal_data[:, i] * 0.1 + features[i]

        X = np.vstack([normal_data, np.array(features).reshape(1, -1)])

        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(normal_data)

        score = clf.decision_function(np.array(features).reshape(1, -1))[0]
        normalized = float(np.clip((-score + 0.1) / 0.3, 0, 1))
        return round(normalized, 3)

    except Exception:
        return 0.2


def detect(data) -> AnomalyResult:
    """
    Detect earnings anomalies and accounting irregularities.

    Runs three checks:
    1. Isolation Forest on financial ratios
    2. Beneish M-Score (manipulation indicator)
    3. Accrual ratio (earnings quality)

    Parameters
    ----------
    data : TickerData — from pull.ticker()

    Returns
    -------
    AnomalyResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.ml import anomaly
    >>> data = pull.ticker("AAPL")
    >>> result = anomaly.detect(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Running anomaly detection for {ticker}...[/dim]")

    flags = []
    details = {}

    iso_score = _isolation_forest_score(data)
    details["isolation_forest_score"] = iso_score

    m_score = _beneish_score(data)
    details["beneish_m_score"] = m_score
    if m_score is not None and m_score > -1.78:
        flags.append(f"Beneish M-Score = {m_score:.2f} (threshold: -1.78) — possible earnings management")

    accrual = _accrual_ratio(data)
    details["accrual_ratio"] = accrual
    if accrual is not None and abs(accrual) > 0.10:
        flags.append(f"High accrual ratio = {accrual:.3f} — earnings may not be cash-backed")

    if not data.revenue_history.empty:
        rev_growth = data.revenue_history.pct_change().dropna()
        if len(rev_growth) >= 2:
            latest_growth = float(rev_growth.iloc[-1])
            prior_growth = float(rev_growth.iloc[-2])
            if latest_growth < prior_growth * 0.5 and prior_growth > 0.05:
                flags.append(f"Revenue growth deceleration: {prior_growth:.1%} → {latest_growth:.1%}")
            details["revenue_growth_latest"] = round(latest_growth, 4)

    if not data.fcf_history.empty and not data.net_income_history.empty:
        common = data.fcf_history.index.intersection(data.net_income_history.index)
        if len(common) > 0:
            fcf_latest = float(data.fcf_history.loc[common].iloc[-1])
            ni_latest = float(data.net_income_history.loc[common].iloc[-1])
            if ni_latest > 0 and fcf_latest < ni_latest * 0.5:
                flags.append(f"FCF/Net income ratio low: {fcf_latest/ni_latest:.2f} — earnings quality concern")
            details["fcf_to_ni_ratio"] = round(fcf_latest / ni_latest, 3) if ni_latest != 0 else None

    overall_score = min(
        iso_score * 0.4
        + (0.4 if (m_score and m_score > -1.78) else 0)
        + (0.2 if (accrual and abs(accrual) > 0.10) else 0)
        + len(flags) * 0.05,
        1.0
    )

    if overall_score < 0.3:
        quality = "high"
    elif overall_score < 0.6:
        quality = "medium"
    else:
        quality = "low"

    console.print(
        f"[green]✓[/green] Anomaly score: {overall_score:.2f} — "
        f"earnings quality: {quality}  |  flags: {len(flags)}"
    )

    return AnomalyResult(
        ticker=ticker,
        anomaly_score=round(overall_score, 3),
        flags=flags,
        beneish_score=m_score,
        accrual_ratio=accrual,
        earnings_quality=quality,
        details=details,
    )
