"""
finverse example: full credit analysis workflow.

Demonstrates: GARCH vol → Merton DD → Altman Z → EVT tail risk → Kelly sizing.
"""
from finverse import pull
from finverse.ml import garch
from finverse.credit import merton, altman
from finverse.risk import evt, kelly
from finverse.audit.earnings_quality import score as eq_score

print("Credit Analysis — Apple Inc.")
print("=" * 50)

data = pull.ticker("AAPL")

print("\n1. GARCH volatility (feeds into Merton)...")
g = garch.fit(data, model_type="GJR-GARCH")
print(f"   Current conditional vol: {g.current_vol:.2%}")
print(f"   Long-run vol:            {g.long_run_vol:.2%}")
print(f"   Persistence:             {g.persistence:.4f}")

print("\n2. Merton distance-to-default...")
credit = merton.analyze(data, garch_vol=g.current_vol)
credit.summary()

print("\n3. Altman Z-Score...")
z = altman.analyze(data)
z.summary()

print("\n4. Extreme Value Theory — tail risk...")
tail = evt.analyze(data)
tail.summary()

print("\n5. Kelly criterion — optimal position size...")
k = kelly.from_distribution(data)
k.summary()

print("\n6. Earnings quality...")
eq = eq_score(data)
eq.summary()

print("\nSummary:")
print(f"  Merton rating:      {credit.rating_equivalent}")
print(f"  Altman zone:        {z.zone} ({z.score:.2f})")
print(f"  EVT VaR 99.9%:      {tail.var_999:.2%}")
print(f"  Earnings quality:   {eq.overall_score:.0f}/100 (grade {eq.grade})")
print(f"  Kelly (half):       {k.half_kelly:.1%}")
