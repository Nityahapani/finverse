"""
finverse.credit.altman — Altman Z-Score and Z'-Score (private firms).

Original Z-Score (1968) for public manufacturers.
Z'-Score for private companies.
Z''-Score for non-manufacturers / service firms.

All pure math — no API needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AltmanResult:
    ticker: str
    model: str                     # "Z-Score", "Z'-Score", "Z''-Score"
    score: float
    zone: str                      # "safe", "grey", "distress"
    ratios: dict[str, float]
    interpretation: str
    safe_threshold: float
    distress_threshold: float

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        zone_color = {"safe": "green", "grey": "yellow", "distress": "red"}[self.zone]

        console.print(f"\n[bold blue]Altman {self.model} — {self.ticker}[/bold blue]\n")
        console.print(
            f"Score: [{zone_color}][bold]{self.score:.2f}[/bold][/{zone_color}]  "
            f"|  Zone: [{zone_color}][bold]{self.zone.upper()}[/bold][/{zone_color}]"
        )
        console.print(f"[dim]{self.interpretation}[/dim]\n")

        table = Table(title="Component ratios", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Ratio")
        table.add_column("Value", justify="right")
        table.add_column("Definition")

        definitions = {
            "X1": "Working capital / Total assets",
            "X2": "Retained earnings / Total assets",
            "X3": "EBIT / Total assets",
            "X4": "Market cap / Total liabilities",
            "X4_prime": "Book equity / Total liabilities",
            "X5": "Revenue / Total assets",
        }
        for k, v in self.ratios.items():
            table.add_row(k, f"{v:.4f}", definitions.get(k, ""))
        console.print(table)
        console.print(
            f"\n  Safe zone: > {self.safe_threshold}  |  "
            f"Distress zone: < {self.distress_threshold}\n"
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "model": self.model,
            "score": self.score,
            "zone": self.zone,
            **self.ratios,
        }])


def _get_ratio(df: pd.DataFrame, keys: list[str], col) -> float:
    for k in keys:
        if k in df.index:
            v = df.loc[k, col]
            return float(v) if not (np.isnan(float(v)) if isinstance(v, float) else False) else 0.0
    return 0.0


def analyze(
    data,
    model: str = "auto",
) -> AltmanResult:
    """
    Compute Altman Z-Score for financial distress prediction.

    Parameters
    ----------
    data  : TickerData — needs income_stmt, balance_sheet, info, price_history
    model : "Z-Score" (public manufacturers), "Z'-Score" (private),
            "Z''-Score" (non-manufacturers), "auto" (default — picks best fit)

    Returns
    -------
    AltmanResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.credit import altman
    >>> data = pull.ticker("AAPL")
    >>> result = altman.analyze(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Computing Altman Z-Score for {ticker}...[/dim]")

    is_df = data.income_stmt if hasattr(data, "income_stmt") and not data.income_stmt.empty else pd.DataFrame()
    bs_df = data.balance_sheet if hasattr(data, "balance_sheet") and not data.balance_sheet.empty else pd.DataFrame()

    if is_df.empty or bs_df.empty:
        console.print("[yellow]Warning: insufficient financial data — using estimates[/yellow]")
        total_assets = 350.0
        revenue = 383.0
        ebit = 120.0
        retained_earnings = 150.0
        working_capital = 50.0
        total_liabilities = 200.0
        market_cap = (data.market_cap or 2.8e12) / 1e9
        book_equity = 80.0
    else:
        curr_col = bs_df.columns[0] if not bs_df.empty else None

        total_assets = _get_ratio(bs_df, [
            "Total Assets", "TotalAssets",
        ], curr_col) / 1e9 if curr_col else 350.0

        total_liabilities = _get_ratio(bs_df, [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities", "TotalLiabilitiesNetMinorityInterest",
        ], curr_col) / 1e9 if curr_col else 200.0

        current_assets = _get_ratio(bs_df, [
            "Current Assets", "Total Current Assets", "CurrentAssets",
        ], curr_col) / 1e9 if curr_col else 100.0

        current_liab = _get_ratio(bs_df, [
            "Current Liabilities", "Total Current Liabilities", "CurrentLiabilities",
        ], curr_col) / 1e9 if curr_col else 60.0

        retained = _get_ratio(bs_df, [
            "Retained Earnings", "RetainedEarnings",
        ], curr_col) / 1e9 if curr_col else 150.0

        book_equity = _get_ratio(bs_df, [
            "Stockholders Equity", "Common Stock Equity",
            "Total Stockholders Equity", "StockholdersEquity",
        ], curr_col) / 1e9 if curr_col else 80.0

        is_col = is_df.columns[0] if not is_df.empty else None
        revenue = _get_ratio(is_df, ["Total Revenue", "Revenue"], is_col) / 1e9 if is_col else 383.0
        ebit = _get_ratio(is_df, ["EBIT", "Operating Income", "Ebit"], is_col) / 1e9 if is_col else 120.0

        working_capital = current_assets - current_liab
        retained_earnings = retained
        market_cap = (data.market_cap or 2.8e12) / 1e9

        if total_assets <= 0:
            total_assets = 350.0
        if total_liabilities <= 0:
            total_liabilities = 200.0

    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_cap / max(total_liabilities, 0.01)
    X4p = book_equity / max(total_liabilities, 0.01)
    X5 = revenue / total_assets

    sector = (data.info.get("sector", "") if hasattr(data, "info") and data.info else "")
    is_manufacturer = sector.lower() in ["industrials", "materials", "consumer discretionary"]
    is_public = data.market_cap is not None and data.market_cap > 0

    if model == "auto":
        if is_public and is_manufacturer:
            model = "Z-Score"
        elif not is_public:
            model = "Z'-Score"
        else:
            model = "Z''-Score"

    if model == "Z-Score":
        score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        safe_threshold = 2.99
        distress_threshold = 1.81
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}
    elif model == "Z'-Score":
        score = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4p + 0.998*X5
        safe_threshold = 2.9
        distress_threshold = 1.23
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4_prime": X4p, "X5": X5}
    else:  # Z''-Score
        score = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4p
        safe_threshold = 2.6
        distress_threshold = 1.1
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4_prime": X4p}

    if score > safe_threshold:
        zone = "safe"
        interpretation = "Low distress risk. Financially healthy."
    elif score > distress_threshold:
        zone = "grey"
        interpretation = "Grey zone. Some financial stress indicators present — monitor closely."
    else:
        zone = "distress"
        interpretation = "High distress risk. Potential financial difficulties ahead."

    # Flag negative book equity — Altman ratios become unreliable in this case
    # (common for mature companies with aggressive buyback programs, e.g. Apple)
    if book_equity < 0:
        interpretation += (
            " ⚠ Note: negative book equity detected — Altman scores are unreliable "
            "for companies with aggressive buyback programs (e.g. Apple). "
            "The grey/distress classification may be misleading; use Merton model instead."
        )

    console.print(
        f"[green]✓[/green] Altman {model}: score={score:.2f}, zone={zone}"
    )

    return AltmanResult(
        ticker=ticker,
        model=model,
        score=round(score, 4),
        zone=zone,
        ratios={k: round(v, 4) for k, v in ratios.items()},
        interpretation=interpretation,
        safe_threshold=safe_threshold,
        distress_threshold=distress_threshold,
    )            f"Score: [{zone_color}][bold]{self.score:.2f}[/bold][/{zone_color}]  "
            f"|  Zone: [{zone_color}][bold]{self.zone.upper()}[/bold][/{zone_color}]"
        )
        console.print(f"[dim]{self.interpretation}[/dim]\n")

        table = Table(title="Component ratios", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Ratio")
        table.add_column("Value", justify="right")
        table.add_column("Definition")

        definitions = {
            "X1": "Working capital / Total assets",
            "X2": "Retained earnings / Total assets",
            "X3": "EBIT / Total assets",
            "X4": "Market cap / Total liabilities",
            "X4_prime": "Book equity / Total liabilities",
            "X5": "Revenue / Total assets",
        }
        for k, v in self.ratios.items():
            table.add_row(k, f"{v:.4f}", definitions.get(k, ""))
        console.print(table)
        console.print(
            f"\n  Safe zone: > {self.safe_threshold}  |  "
            f"Distress zone: < {self.distress_threshold}\n"
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": self.ticker,
            "model": self.model,
            "score": self.score,
            "zone": self.zone,
            **self.ratios,
        }])


def _get_ratio(df: pd.DataFrame, keys: list[str], col) -> float:
    for k in keys:
        if k in df.index:
            v = df.loc[k, col]
            return float(v) if not (np.isnan(float(v)) if isinstance(v, float) else False) else 0.0
    return 0.0


def analyze(
    data,
    model: str = "auto",
) -> AltmanResult:
    """
    Compute Altman Z-Score for financial distress prediction.

    Parameters
    ----------
    data  : TickerData — needs income_stmt, balance_sheet, info, price_history
    model : "Z-Score" (public manufacturers), "Z'-Score" (private),
            "Z''-Score" (non-manufacturers), "auto" (default — picks best fit)

    Returns
    -------
    AltmanResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.credit import altman
    >>> data = pull.ticker("AAPL")
    >>> result = altman.analyze(data)
    >>> result.summary()
    """
    from finverse.utils.display import console

    ticker = data.ticker if hasattr(data, "ticker") else "UNKNOWN"
    console.print(f"[dim]Computing Altman Z-Score for {ticker}...[/dim]")

    is_df = data.income_stmt if hasattr(data, "income_stmt") and not data.income_stmt.empty else pd.DataFrame()
    bs_df = data.balance_sheet if hasattr(data, "balance_sheet") and not data.balance_sheet.empty else pd.DataFrame()

    if is_df.empty or bs_df.empty:
        console.print("[yellow]Warning: insufficient financial data — using estimates[/yellow]")
        total_assets = 350.0
        revenue = 383.0
        ebit = 120.0
        retained_earnings = 150.0
        working_capital = 50.0
        total_liabilities = 200.0
        market_cap = (data.market_cap or 2.8e12) / 1e9
        book_equity = 80.0
    else:
        curr_col = bs_df.columns[0] if not bs_df.empty else None

        total_assets = _get_ratio(bs_df, ["Total Assets"], curr_col) / 1e9 if curr_col else 350.0
        total_liabilities = _get_ratio(bs_df, ["Total Liabilities Net Minority Interest",
                                                 "Total Liabilities"], curr_col) / 1e9 if curr_col else 200.0
        current_assets = _get_ratio(bs_df, ["Current Assets", "Total Current Assets"], curr_col) / 1e9 if curr_col else 100.0
        current_liab = _get_ratio(bs_df, ["Current Liabilities", "Total Current Liabilities"], curr_col) / 1e9 if curr_col else 60.0
        retained = _get_ratio(bs_df, ["Retained Earnings"], curr_col) / 1e9 if curr_col else 150.0
        book_equity = _get_ratio(bs_df, ["Stockholders Equity", "Common Stock Equity"], curr_col) / 1e9 if curr_col else 80.0

        is_col = is_df.columns[0] if not is_df.empty else None
        revenue = _get_ratio(is_df, ["Total Revenue", "Revenue"], is_col) / 1e9 if is_col else 383.0
        ebit = _get_ratio(is_df, ["EBIT", "Operating Income"], is_col) / 1e9 if is_col else 120.0

        working_capital = current_assets - current_liab
        retained_earnings = retained
        market_cap = (data.market_cap or 2.8e12) / 1e9

        if total_assets <= 0:
            total_assets = 350.0
        if total_liabilities <= 0:
            total_liabilities = 200.0

    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_cap / max(total_liabilities, 0.01)
    X4p = book_equity / max(total_liabilities, 0.01)
    X5 = revenue / total_assets

    sector = (data.info.get("sector", "") if hasattr(data, "info") and data.info else "")
    is_manufacturer = sector.lower() in ["industrials", "materials", "consumer discretionary"]
    is_public = data.market_cap is not None and data.market_cap > 0

    if model == "auto":
        if is_public and is_manufacturer:
            model = "Z-Score"
        elif not is_public:
            model = "Z'-Score"
        else:
            model = "Z''-Score"

    if model == "Z-Score":
        score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        safe_threshold = 2.99
        distress_threshold = 1.81
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}
    elif model == "Z'-Score":
        score = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4p + 0.998*X5
        safe_threshold = 2.9
        distress_threshold = 1.23
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4_prime": X4p, "X5": X5}
    else:  # Z''-Score
        score = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4p
        safe_threshold = 2.6
        distress_threshold = 1.1
        ratios = {"X1": X1, "X2": X2, "X3": X3, "X4_prime": X4p}

    if score > safe_threshold:
        zone = "safe"
        interpretation = "Low distress risk. Financially healthy."
    elif score > distress_threshold:
        zone = "grey"
        interpretation = "Grey zone. Some financial stress indicators present — monitor closely."
    else:
        zone = "distress"
        interpretation = "High distress risk. Potential financial difficulties ahead."

    console.print(
        f"[green]✓[/green] Altman {model}: score={score:.2f}, zone={zone}"
    )

    return AltmanResult(
        ticker=ticker,
        model=model,
        score=round(score, 4),
        zone=zone,
        ratios={k: round(v, 4) for k, v in ratios.items()},
        interpretation=interpretation,
        safe_threshold=safe_threshold,
        distress_threshold=distress_threshold,
    )
