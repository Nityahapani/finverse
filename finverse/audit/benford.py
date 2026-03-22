"""
finverse.audit.benford — Benford's Law test for financial data quality.

Benford's Law states that in naturally occurring numerical data,
the leading digit d appears with probability log10(1 + 1/d).

Significant deviations from Benford's distribution can signal:
- Data manipulation or fabrication
- Rounding or truncation artifacts
- Errors in data entry

Used by forensic accountants and the IRS for fraud detection.
Pure math — no external dependencies beyond scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import chi2, kstest


BENFORD_PROBS = np.array([
    np.log10(1 + 1/d) for d in range(1, 10)
])

BENFORD_LABELS = [str(d) for d in range(1, 10)]


@dataclass
class BenfordResult:
    source: str
    n_observations: int
    observed_freq: dict[str, float]     # observed frequencies by leading digit
    expected_freq: dict[str, float]     # Benford expected frequencies
    chi2_statistic: float
    chi2_p_value: float
    mad: float                          # Mean Absolute Deviation
    conformity: str                     # "close", "acceptable", "nonconforming", "suspicious"
    flagged_digits: list[str]
    interpretation: str

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        conformity_colors = {
            "close": "green",
            "acceptable": "blue",
            "nonconforming": "yellow",
            "suspicious": "red",
        }
        c = conformity_colors.get(self.conformity, "white")

        console.print(f"\n[bold blue]Benford's Law Test — {self.source}[/bold blue]")
        console.print(f"[dim]{self.n_observations:,} observations[/dim]\n")
        console.print(
            f"Conformity: [{c}][bold]{self.conformity.upper()}[/bold][/{c}]  |  "
            f"MAD: {self.mad:.4f}  |  "
            f"χ²={self.chi2_statistic:.2f} (p={self.chi2_p_value:.4f})"
        )

        table = Table(title="Leading digit distribution", box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Digit")
        table.add_column("Observed", justify="right")
        table.add_column("Expected (Benford)", justify="right")
        table.add_column("Difference", justify="right")
        table.add_column("")

        for d in range(1, 10):
            label = str(d)
            obs = self.observed_freq.get(label, 0)
            exp = self.expected_freq.get(label, 0)
            diff = obs - exp
            bar_obs = "█" * int(obs * 50)
            bar_exp = "░" * int(exp * 50)
            flag = " ◄" if label in self.flagged_digits else ""
            diff_color = "red" if abs(diff) > 0.03 else ("yellow" if abs(diff) > 0.015 else "green")
            table.add_row(
                label,
                f"{obs:.3f}",
                f"{exp:.3f}",
                f"[{diff_color}]{diff:+.3f}[/{diff_color}]",
                f"[dim]{bar_obs}[/dim][blue]{bar_exp}[/blue]{flag}",
            )

        console.print(table)
        console.print(f"\n  {self.interpretation}")

        if self.flagged_digits:
            console.print(f"  [yellow]Flagged digits:[/yellow] {', '.join(self.flagged_digits)}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed_freq,
            "expected": self.expected_freq,
            "difference": {k: self.observed_freq.get(k, 0) - self.expected_freq.get(k, 0)
                          for k in self.expected_freq},
        })


def _leading_digit(x: float) -> int | None:
    """Extract the leading digit from a number."""
    if x <= 0 or np.isnan(x) or np.isinf(x):
        return None
    s = f"{abs(x):.10e}"
    for ch in s:
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def test(
    numbers: list | np.ndarray | pd.Series,
    source: str = "financial data",
    significance: float = 0.05,
) -> BenfordResult:
    """
    Test a series of numbers against Benford's Law.

    Parameters
    ----------
    numbers      : numeric data (financial figures, transaction amounts, etc.)
    source       : str — description for display
    significance : float — p-value threshold for flagging (default 0.05)

    Returns
    -------
    BenfordResult

    Example
    -------
    >>> from finverse.audit.benford import test as benford_test
    >>> import pandas as pd
    >>> # Test income statement figures
    >>> data = pull.ticker("AAPL")
    >>> is_df = data.income_stmt
    >>> all_values = is_df.values.flatten()
    >>> result = benford_test(all_values, source="Apple income statement")
    >>> result.summary()
    """
    from finverse.utils.display import console
    console.print(f"[dim]Running Benford's Law test on {source}...[/dim]")

    if isinstance(numbers, pd.Series):
        arr = numbers.dropna().values
    elif isinstance(numbers, pd.DataFrame):
        arr = numbers.values.flatten()
    else:
        arr = np.array(numbers, dtype=float)

    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]

    leading_digits = [_leading_digit(abs(x)) for x in arr]
    leading_digits = [d for d in leading_digits if d is not None]
    n = len(leading_digits)

    if n < 50:
        console.print(f"[yellow]Warning: only {n} observations — Benford test requires 50+[/yellow]")

    observed_counts = np.zeros(9)
    for d in leading_digits:
        if 1 <= d <= 9:
            observed_counts[d - 1] += 1

    observed_freq = observed_counts / max(n, 1)
    expected_freq = BENFORD_PROBS

    observed_dict = {str(d): round(observed_freq[d-1], 6) for d in range(1, 10)}
    expected_dict = {str(d): round(expected_freq[d-1], 6) for d in range(1, 10)}

    expected_counts = expected_freq * n
    valid = expected_counts >= 5
    if valid.sum() >= 7:
        chi2_stat = float(np.sum((observed_counts[valid] - expected_counts[valid])**2 / expected_counts[valid]))
        chi2_pval = float(1 - chi2.cdf(chi2_stat, df=valid.sum() - 1))
    else:
        chi2_stat = 0.0
        chi2_pval = 1.0

    mad = float(np.mean(np.abs(observed_freq - expected_freq)))

    if mad < 0.006:
        conformity = "close"
        interpretation = "Data closely conforms to Benford's Law. No manipulation detected."
    elif mad < 0.012:
        conformity = "acceptable"
        interpretation = "Acceptable conformity. Minor deviations within expected range."
    elif mad < 0.015:
        conformity = "nonconforming"
        interpretation = "Nonconforming. Investigate data collection and rounding practices."
    else:
        conformity = "suspicious"
        interpretation = "Significant deviation from Benford's Law. Warrants forensic review."

    flagged = [
        str(d) for d in range(1, 10)
        if abs(observed_freq[d-1] - expected_freq[d-1]) > 0.03
    ]

    if chi2_pval < significance and n >= 50:
        if conformity in ["close", "acceptable"]:
            conformity = "nonconforming"
            interpretation += " Chi-square test rejects Benford conformity."

    console.print(
        f"[green]✓[/green] Benford test — "
        f"MAD={mad:.4f}, conformity={conformity}"
        + (f", flagged digits: {', '.join(flagged)}" if flagged else "")
    )

    return BenfordResult(
        source=source,
        n_observations=n,
        observed_freq=observed_dict,
        expected_freq=expected_dict,
        chi2_statistic=round(chi2_stat, 4),
        chi2_p_value=round(chi2_pval, 6),
        mad=round(mad, 6),
        conformity=conformity,
        flagged_digits=flagged,
        interpretation=interpretation,
    )


def test_financials(data, source: str | None = None) -> BenfordResult:
    """
    Run Benford's test on all financial statement values from TickerData.

    Parameters
    ----------
    data   : TickerData — from pull.ticker()
    source : str — optional label

    Returns
    -------
    BenfordResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.audit.benford import test_financials
    >>> data = pull.ticker("AAPL")
    >>> result = test_financials(data)
    >>> result.summary()
    """
    label = source or f"{getattr(data, 'ticker', 'company')} financial statements"
    all_values = []

    for attr in ["income_stmt", "balance_sheet", "cash_flow"]:
        df = getattr(data, attr, pd.DataFrame())
        if not df.empty:
            vals = df.values.flatten()
            all_values.extend([abs(v) for v in vals if not np.isnan(v) and v != 0])

    if not all_values:
        from finverse.utils.display import console
        console.print("[yellow]No financial data available for Benford test[/yellow]")
        return test(np.array([1.0, 2.0, 3.0]), source=label)

    return test(np.array(all_values), source=label)
