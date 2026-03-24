"""
finverse.derivatives
====================
Interest rate and FX derivatives pricing.

Submodules
----------
rates  — IR swaps, FRAs, and swaptions via Black's model     [v0.7.0]
fx     — FX forwards, cross-currency swaps, FX options (GK)  [v0.7.0]

Usage
-----
from finverse.derivatives import rates, fx
from finverse.macro import nelson_siegel

# Interest rate swap
curve = nelson_siegel.us_curve()
swap = rates.swap(notional=10_000_000, fixed_rate=0.045, tenor=5,
                  payment_freq='semi-annual', curve=curve)
swap.summary()

# Forward Rate Agreement
fra = rates.fra(notional=5_000_000, contract_rate=0.052,
                start=0.5, end=1.0, curve=curve)
fra.summary()

# Swaption — option to enter a swap
swaption = rates.swaption(notional=10_000_000, strike_rate=0.048,
                          option_expiry=1.0, swap_tenor=5,
                          vol=0.20, curve=curve, type='payer')
swaption.summary()

# FX forward — covered interest parity
fwd = fx.forward(spot=1.085, r_domestic=0.053, r_foreign=0.038,
                 tenor=1.0, pair='EURUSD')
fwd.summary()

# FX option — Garman-Kohlhagen
opt = fx.option(spot=1.085, strike=1.10, tenor=0.5,
                r_domestic=0.053, r_foreign=0.038,
                sigma=0.085, type='call', pair='EURUSD')
opt.summary()

# Cross-currency basis swap
ccs = fx.cross_currency_swap(notional_usd=10_000_000, pair='EURUSD',
                              spot=1.085, tenor=3, basis_spread=-0.0010)
ccs.summary()

# Currency-adjusted WACC for multinational DCF
from finverse import DCF
wacc_adj = fx.currency_adjusted_wacc(base_wacc=0.095,
                                      revenue_fx_exposure={'EUR': 0.35, 'GBP': 0.20},
                                      tenor=5)
dcf = DCF.manual(base_revenue=383.0, ...).set(wacc=wacc_adj).run()
"""
from finverse.derivatives import rates
from finverse.derivatives import fx

__all__ = ["rates", "fx"]
