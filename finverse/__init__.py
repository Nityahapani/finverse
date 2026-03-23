"""
finverse — The ML-powered financial modeling toolkit.
v0.7.0 — Phase 7: Regime-Conditional DCF, Manipulation Fingerprinting,
          Synthetic Peers, Black-Litterman, CVaR Optimization,
          Options Pricing, Bond Pricing
"""
__version__ = "0.7.0"
__author__ = "finverse"

from finverse.models.dcf import DCF
from finverse.models.lbo import LBO
from finverse.models.three_statement import ThreeStatement
from finverse.models.comps import analyze as comps
from finverse.models.ddm import gordon as ddm_gordon, h_model, multistage as ddm_multistage
from finverse.models.sotp import Segment, analyze as sotp
from finverse.models.regime_dcf import analyze as regime_dcf
from finverse.models.synthetic_peers import build_peers as synthetic_peers
from finverse.models.options import call as option_call, put as option_put
from finverse.models.bonds import price as bond_price, ytm_from_price
from finverse.models import macro as macro_nowcast
from finverse import pull, ml, risk, screen, backtest, portfolio, audit, credit, valuation, macro
from finverse.analysis.sensitivity import sensitivity
from finverse.analysis.scenarios import scenarios

__all__ = [
    "DCF", "LBO", "ThreeStatement", "comps", "sotp", "Segment",
    "regime_dcf", "synthetic_peers",
    "ddm_gordon", "h_model", "ddm_multistage",
    "option_call", "option_put",
    "bond_price", "ytm_from_price",
    "pull", "ml", "risk", "screen", "backtest",
    "portfolio", "audit", "credit", "valuation", "macro",
    "sensitivity", "scenarios",
    "__version__",
]
