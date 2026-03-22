"""
pull.edgar — fetch SEC filings from EDGAR (free, no key required).
"""
from __future__ import annotations

import time
import requests
import pandas as pd

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "finverse research@finverse.io"}


def _get_cik(ticker: str) -> str:
    """Resolve ticker to CIK number."""
    url = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&forms=10-K".format(ticker)
    r = requests.get(
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={}&type=10-K&dateb=&owner=include&count=1&search_text=&output=atom".format(ticker),
        headers=HEADERS,
        timeout=10,
    )
    import re
    match = re.search(r"CIK=(\d+)", r.text)
    if not match:
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=HEADERS, timeout=10)
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Could not find CIK for ticker '{ticker}'. Check the symbol.")
    return match.group(1).zfill(10)


def edgar(ticker: str, form: str = "10-K", n: int = 5) -> dict:
    """
    Fetch recent SEC filings for a company.

    Parameters
    ----------
    ticker : str   — stock ticker, e.g. "AAPL"
    form   : str   — filing type: "10-K", "10-Q", "8-K" (default "10-K")
    n      : int   — number of recent filings to return (default 5)

    Returns
    -------
    dict with keys:
        "ticker"   : str
        "cik"      : str
        "filings"  : pd.DataFrame — filing index with dates and URLs
        "facts"    : dict         — raw XBRL financial facts (if available)

    Example
    -------
    >>> from finverse import pull
    >>> filings = pull.edgar("AAPL", "10-K", n=3)
    >>> filings["filings"]
    """
    from finverse.utils.display import console
    from finverse.utils.validate import clean_ticker

    ticker = clean_ticker(ticker)
    console.print(f"[dim]Fetching EDGAR filings for {ticker} ({form})...[/dim]")

    result = {"ticker": ticker, "cik": None, "filings": pd.DataFrame(), "facts": {}}

    try:
        cik = _get_cik(ticker)
        result["cik"] = cik
        time.sleep(0.1)

        submissions_url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = requests.get(submissions_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        rows = []
        for f, d, acc, doc in zip(forms, dates, accessions, primary_docs):
            if f == form:
                acc_clean = acc.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
                rows.append({"form": f, "date": d, "accession": acc, "url": url})
                if len(rows) >= n:
                    break

        result["filings"] = pd.DataFrame(rows)

        time.sleep(0.1)
        facts_url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        facts_resp = requests.get(facts_url, headers=HEADERS, timeout=15)
        if facts_resp.status_code == 200:
            result["facts"] = facts_resp.json()

        console.print(f"[green]✓[/green] {len(rows)} {form} filings found for {ticker}")

    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] EDGAR fetch error for {ticker}: {e}")

    return result


def edgar_financials(ticker: str) -> pd.DataFrame:
    """
    Extract structured annual financials from EDGAR XBRL facts.

    Returns a DataFrame with revenue, net income, assets etc. per year.

    Example
    -------
    >>> from finverse import pull
    >>> df = pull.edgar_financials("AAPL")
    >>> df.head()
    """
    from finverse.utils.display import console

    data = edgar(ticker, "10-K")
    facts = data.get("facts", {})

    if not facts:
        console.print("[yellow]No XBRL facts available.[/yellow]")
        return pd.DataFrame()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    WANTED = {
        "Revenues": "Revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
        "NetIncomeLoss": "Net Income",
        "OperatingIncomeLoss": "Operating Income",
        "EarningsPerShareBasic": "EPS Basic",
        "Assets": "Total Assets",
        "CashAndCashEquivalentsAtCarryingValue": "Cash",
        "LongTermDebt": "Long Term Debt",
        "NetCashProvidedByUsedInOperatingActivities": "Operating CF",
        "PaymentsToAcquirePropertyPlantAndEquipment": "Capex",
    }

    rows = {}
    for gaap_key, label in WANTED.items():
        if gaap_key not in us_gaap:
            continue
        units = us_gaap[gaap_key].get("units", {})
        unit_data = units.get("USD", units.get("USD/shares", []))
        for entry in unit_data:
            if entry.get("form") == "10-K" and entry.get("fp") == "FY":
                yr = entry.get("end", "")[:4]
                val = entry.get("val", 0)
                if yr not in rows:
                    rows[yr] = {}
                if label not in rows[yr]:
                    rows[yr][label] = val / 1e9

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    df.index.name = "Year"
    df.sort_index(inplace=True)
    return df.round(2)
