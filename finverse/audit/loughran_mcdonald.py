"""
finverse.audit.loughran_mcdonald — Loughran-McDonald (2011) financial
sentiment dictionary.

Purpose-built for financial text — avoids misclassifications that plague
general-purpose dictionaries (e.g. "liability" is negative in Harvard IV
but neutral in finance; "risks" appears constantly in filings).

Computes:
- Positive / Negative tone scores
- Uncertainty score
- Litigious score
- Modal verb scores (strong/weak)
- Constraining language score

Reference: Loughran & McDonald (2011), Journal of Finance.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass


NEGATIVE_WORDS = {
    "abandoned", "abnormal", "abolish", "abrupt", "absence", "abuse",
    "adverse", "adversely", "allegation", "alleged", "ambiguous",
    "ambiguity", "bankruptcy", "breach", "burden", "cease", "challenged",
    "claim", "complaints", "concern", "concerns", "constrain", "contraction",
    "controversy", "crisis", "damage", "damages", "decline", "declining",
    "default", "deficiency", "deficient", "delay", "deteriorate",
    "deterioration", "difficult", "difficulties", "discontinue", "dispute",
    "disruption", "doubt", "downturn", "dramatically", "exposure",
    "failed", "failure", "falling", "forfeit", "fraud", "guilty",
    "halt", "hampered", "harm", "harmful", "headwind", "impair",
    "impairment", "inability", "inadequate", "incident", "inferior",
    "injunction", "insufficient", "investigation", "judgment", "lawsuit",
    "layoff", "liability", "limitation", "litigation", "loss", "losses",
    "lower", "materially", "misconduct", "miss", "negative", "negligent",
    "noncompliance", "obstacle", "opposition", "penalty", "poor",
    "problem", "problems", "questioned", "reduction", "restructure",
    "restatement", "risk", "risks", "shortfall", "significant",
    "slower", "substandard", "suspension", "terminated", "termination",
    "unable", "uncertainty", "underperform", "unfavorable", "unlawful",
    "unprofitable", "unreliable", "unstable", "violation", "volatile",
    "vulnerability", "warning", "weak", "weakness", "worsen",
}

POSITIVE_WORDS = {
    "above", "accomplish", "accomplished", "achievement", "advancement",
    "advantage", "affirmative", "appreciable", "best", "better",
    "breakthrough", "capabilities", "capitalize", "celebrate", "commitment",
    "competitive", "confident", "consistent", "continue", "contribute",
    "deliver", "demonstrates", "distinguished", "diversified", "dominant",
    "earned", "effective", "efficiently", "enhanced", "exceed",
    "excellent", "exceptional", "expanding", "favorable", "flexibility",
    "growth", "improved", "improvement", "increasing", "innovative",
    "leading", "momentum", "opportunity", "outstanding", "outperform",
    "positive", "profitability", "profitable", "progress", "robust",
    "significant", "strengthen", "strong", "superior", "sustained",
    "unique", "unprecedented", "value", "worldwide",
}

UNCERTAINTY_WORDS = {
    "approximate", "approximately", "appear", "appears", "arguably",
    "assume", "assumed", "assumption", "believe", "believed", "cautious",
    "conceivably", "conditional", "contingent", "could", "depend",
    "depends", "doubt", "dubious", "estimate", "estimated", "eventually",
    "expect", "expected", "feels", "flexible", "hope", "imprecise",
    "indefinite", "indefinitely", "likely", "may", "maybe", "might",
    "necessarily", "pending", "perhaps", "possible", "possibly",
    "predict", "probable", "probably", "roughly", "seek", "seems",
    "suggests", "suppose", "tentative", "uncertain", "uncertainly",
    "uncertainty", "unclear", "unconditional", "unknown", "unlikely",
    "unresolved", "unsettled", "vague", "whenever", "whether",
}

LITIGIOUS_WORDS = {
    "abuse", "accused", "allegation", "allege", "alleged", "amend",
    "anti-competitive", "antitrust", "arbitration", "attorney",
    "claim", "class-action", "complainant", "complaint", "contempt",
    "counterclaim", "damages", "deceptive", "defendant", "defense",
    "dispute", "enforcement", "federal", "fraud", "guilty", "indemnify",
    "injunction", "lawsuit", "legal", "legislation", "liable",
    "litigation", "misconduct", "negligence", "penalty", "plaintiff",
    "proceedings", "prosecution", "regulatory", "restitution",
    "sanction", "settlement", "statute", "subpoena", "sue", "sued",
    "suing", "trial", "tribunal", "unlawful", "verdict", "violation",
}

STRONG_MODAL = {"will", "must", "require", "requires", "required", "always", "certainly"}
WEAK_MODAL = {"could", "may", "might", "possible", "possibly", "perhaps", "generally",
              "usually", "often", "sometimes"}
CONSTRAINING = {"must", "prohibit", "prohibits", "require", "requires", "required",
                "cannot", "shall", "should", "obligation", "obligated", "restrict",
                "restriction", "covenant", "covenant", "covenant"}


@dataclass
class LMResult:
    text_source: str
    word_count: int
    negative_score: float
    positive_score: float
    uncertainty_score: float
    litigious_score: float
    strong_modal_score: float
    weak_modal_score: float
    constraining_score: float
    net_sentiment: float          # positive - negative
    tone_label: str
    top_negative_words: list[str]
    top_positive_words: list[str]

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        tone_color = "green" if self.net_sentiment > 0.01 else ("red" if self.net_sentiment < -0.01 else "yellow")
        console.print(f"\n[bold blue]Loughran-McDonald Analysis — {self.text_source}[/bold blue]")
        console.print(f"[dim]{self.word_count:,} words analyzed[/dim]\n")
        console.print(f"Net sentiment: [{tone_color}][bold]{self.net_sentiment:+.4f}[/bold][/{tone_color}] ({self.tone_label})")

        table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
        table.add_column("Dimension")
        table.add_column("Score", justify="right")
        table.add_column("Benchmark", justify="right")

        benchmarks = {
            "Negative":      ("1.5–3.0%", self.negative_score),
            "Positive":      ("0.5–1.5%", self.positive_score),
            "Uncertainty":   ("1.0–2.5%", self.uncertainty_score),
            "Litigious":     ("0.5–1.5%", self.litigious_score),
            "Strong modal":  ("0.1–0.5%", self.strong_modal_score),
            "Weak modal":    ("1.0–2.0%", self.weak_modal_score),
            "Constraining":  ("0.5–1.5%", self.constraining_score),
        }
        for label, (bench, score) in benchmarks.items():
            table.add_row(label, f"{score:.3%}", bench)

        console.print(table)

        if self.top_negative_words:
            console.print(f"\n  [dim]Top negative words:[/dim] {', '.join(self.top_negative_words[:8])}")
        if self.top_positive_words:
            console.print(f"  [dim]Top positive words:[/dim] {', '.join(self.top_positive_words[:8])}")
        console.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "source": self.text_source,
            "word_count": self.word_count,
            "negative": self.negative_score,
            "positive": self.positive_score,
            "uncertainty": self.uncertainty_score,
            "litigious": self.litigious_score,
            "net_sentiment": self.net_sentiment,
            "tone": self.tone_label,
        }])


def analyze(
    text: str,
    source: str = "document",
) -> LMResult:
    """
    Analyze financial text using the Loughran-McDonald dictionary.

    Parameters
    ----------
    text   : str — financial text (filing, transcript, press release)
    source : str — label for display (e.g. "10-K 2024", "Q3 earnings call")

    Returns
    -------
    LMResult

    Example
    -------
    >>> from finverse.audit.loughran_mcdonald import analyze as lm_analyze
    >>> with open("10k.txt") as f:
    ...     text = f.read()
    >>> result = lm_analyze(text, source="Apple 10-K 2024")
    >>> result.summary()

    Or combine with pull.edgar:
    >>> from finverse import pull
    >>> filing = pull.edgar("AAPL", "10-K", n=1)
    >>> text = filing.get("text", "")
    >>> result = lm_analyze(text, source="AAPL 10-K")
    """
    from finverse.utils.display import console
    console.print(f"[dim]Running LM analysis on {len(text)} chars ({source})...[/dim]")

    words = re.findall(r"\b[a-z]+\b", text.lower())
    n = len(words)

    if n == 0:
        console.print("[yellow]Warning: no words found in text[/yellow]")
        return LMResult(source, 0, 0, 0, 0, 0, 0, 0, 0, 0, "neutral", [], [])

    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    def score(dictionary):
        return sum(word_freq.get(w, 0) for w in dictionary) / n

    neg_score = score(NEGATIVE_WORDS)
    pos_score = score(POSITIVE_WORDS)
    unc_score = score(UNCERTAINTY_WORDS)
    lit_score = score(LITIGIOUS_WORDS)
    strong_score = score(STRONG_MODAL)
    weak_score = score(WEAK_MODAL)
    const_score = score(CONSTRAINING)

    net = pos_score - neg_score

    if net > 0.015:
        tone = "strongly positive"
    elif net > 0.005:
        tone = "mildly positive"
    elif net < -0.015:
        tone = "strongly negative"
    elif net < -0.005:
        tone = "mildly negative"
    else:
        tone = "neutral"

    top_neg = sorted(
        [(w, word_freq.get(w, 0)) for w in NEGATIVE_WORDS if w in word_freq],
        key=lambda x: -x[1]
    )[:8]
    top_pos = sorted(
        [(w, word_freq.get(w, 0)) for w in POSITIVE_WORDS if w in word_freq],
        key=lambda x: -x[1]
    )[:8]

    console.print(
        f"[green]✓[/green] LM analysis — "
        f"net sentiment: {net:+.3%} ({tone})  |  "
        f"negative: {neg_score:.2%}  positive: {pos_score:.2%}  "
        f"uncertainty: {unc_score:.2%}"
    )

    return LMResult(
        text_source=source,
        word_count=n,
        negative_score=round(neg_score, 6),
        positive_score=round(pos_score, 6),
        uncertainty_score=round(unc_score, 6),
        litigious_score=round(lit_score, 6),
        strong_modal_score=round(strong_score, 6),
        weak_modal_score=round(weak_score, 6),
        constraining_score=round(const_score, 6),
        net_sentiment=round(net, 6),
        tone_label=tone,
        top_negative_words=[w for w, _ in top_neg],
        top_positive_words=[w for w, _ in top_pos],
    )


def compare_filings(texts: dict[str, str]) -> pd.DataFrame:
    """
    Compare LM sentiment across multiple filings over time.

    Parameters
    ----------
    texts : dict of {label: text} — e.g. {"2022": text_2022, "2023": text_2023}

    Returns
    -------
    pd.DataFrame with sentiment trends

    Example
    -------
    >>> results = compare_filings({"2022": text_22, "2023": text_23, "2024": text_24})
    >>> print(results[["negative", "positive", "net_sentiment"]])
    """
    rows = []
    for label, text in texts.items():
        r = analyze(text, source=label)
        rows.append(r.to_df().assign(period=label))

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True).set_index("period")
    return df
