import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str], source: str = "data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[finverse] {source} is missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )


def require_positive(value: float, name: str):
    if value <= 0:
        raise ValueError(f"[finverse] {name} must be positive, got {value}")


def require_range(value: float, name: str, lo: float, hi: float):
    if not (lo <= value <= hi):
        raise ValueError(f"[finverse] {name} must be between {lo} and {hi}, got {value}")


def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()
