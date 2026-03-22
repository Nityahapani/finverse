"""
finverse — The ML-powered financial modeling toolkit.

Phase 1: pull, DCF, sensitivity, scenarios, export
Phase 2: ml.forecast, ml.factor, ml.regime, ml.nlp, ml.cluster, ml.anomaly, ml.causal
Phase 3: risk, screen, backtest, portfolio
Phase 4: LBO, ThreeStatement, comps, macro nowcast, audit
Phase 5: ml.garch, ml.cross_sectional, portfolio.hrp, portfolio.shrinkage,
         credit.merton, credit.altman, valuation.real_options, valuation.apv,
         audit.loughran_mcdonald, audit.benford, macro.var_model, macro.nelson_siegel
Phase 6: risk.evt, risk.kelly, models.ddm, models.sotp,
         audit.earnings_quality
"""

__version__ = "0.6.0"
__author__ = "finverse"

from finverse.models.dcf import DCF
from finverse.models.lbo import LBO
from finverse.models.three_statement import ThreeStatement
from finverse.models.comps import analyze as comps
from finverse.models.ddm import gordon as ddm_gordon, h_model, multistage as ddm_multistage
from finverse.models.sotp import Segment, analyze as sotp
from finverse.models import macro as macro_nowcast
from finverse import pull
from finverse import ml
from finverse import risk
from finverse import screen
from finverse import backtest
from finverse import portfolio
from finverse import audit
from finverse import credit
from finverse import valuation
from finverse import macro
from finverse.analysis.sensitivity import sensitivity
from finverse.analysis.scenarios import scenarios

__all__ = [
    "DCF", "LBO", "ThreeStatement",
    "comps", "sotp", "Segment",
    "ddm_gordon", "h_model", "ddm_multistage",
    "pull", "ml", "risk", "screen", "backtest",
    "portfolio", "audit", "credit", "valuation", "macro",
    "sensitivity", "scenarios",
    "__version__",
]
