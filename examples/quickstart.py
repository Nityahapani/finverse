"""
finverse quickstart — run this file to see the full toolkit in action.
No API keys required for most features.

Usage:
    python examples/quickstart.py

For FRED data (macro module), set:
    export FRED_API_KEY=your_free_key_from_fred.stlouisfed.org
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("  finverse — ML-powered financial modeling toolkit")
print("=" * 60)


# ── 1. Pull data ─────────────────────────────────────────────
print("\n[1/10] Pulling data from yfinance...")
from finverse import pull

data = pull.ticker("AAPL")
data.summary()


# ── 2. DCF model ─────────────────────────────────────────────
print("\n[2/10] Building DCF model...")
from finverse import DCF

model = DCF(data)
model.set(wacc=0.095, terminal_growth=0.025)
results = model.run()
results.summary()


# ── 3. ML revenue forecast ───────────────────────────────────
print("\n[3/10] ML revenue forecast...")
from finverse.ml import forecast

fc = forecast.revenue(data, n_years=5)
fc.summary()


# ── 4. Sensitivity table ─────────────────────────────────────
print("\n[4/10] Sensitivity analysis...")
from finverse import sensitivity

sensitivity(model, rows="wacc", cols="terminal_growth", n=5)


# ── 5. Monte Carlo ───────────────────────────────────────────
print("\n[5/10] Monte Carlo simulation (1,000 scenarios)...")
from finverse.risk import monte_carlo

mc = monte_carlo.simulate(model, n_simulations=1_000)
mc.summary()


# ── 6. GARCH volatility ──────────────────────────────────────
print("\n[6/10] GARCH volatility model...")
from finverse.ml import garch

garch_result = garch.fit(data, model_type="GJR-GARCH")
garch_result.summary()


# ── 7. Merton credit model ───────────────────────────────────
print("\n[7/10] Merton distance-to-default...")
from finverse.credit import merton

credit = merton.analyze(data, garch_vol=garch_result.current_vol)
credit.summary()


# ── 8. Altman Z-Score ────────────────────────────────────────
print("\n[8/10] Altman Z-Score...")
from finverse.credit import altman

z = altman.analyze(data)
z.summary()


# ── 9. Portfolio — HRP ───────────────────────────────────────
print("\n[9/10] Hierarchical Risk Parity portfolio...")
from finverse.portfolio import hrp

tickers = ["AAPL", "MSFT", "GOOGL"]
data_list = [pull.ticker(t) for t in tickers]
portfolio = hrp.optimize(data_list)
portfolio.summary()


# ── 10. Earnings quality audit ───────────────────────────────
print("\n[10/10] Earnings quality score...")
from finverse.audit.earnings_quality import score

eq = score(data)
eq.summary()


# ── Bonus: Nelson-Siegel yield curve ─────────────────────────
print("\n[Bonus] Nelson-Siegel yield curve...")
from finverse.macro import nelson_siegel

mats   = [0.25, 1, 2, 5, 10, 30]
yields = [0.053, 0.052, 0.048, 0.044, 0.044, 0.045]
curve = nelson_siegel.fit(mats, yields)
curve.summary()


print("\n" + "=" * 60)
print("  All done! See README.md for full API documentation.")
print("=" * 60)
