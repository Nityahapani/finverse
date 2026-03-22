"""
finverse.macro — advanced macro modeling.

Modules
-------
var_model      — Vector Autoregression with impulse response functions
nelson_siegel  — Nelson-Siegel / Svensson yield curve fitting
"""
from finverse.macro import var_model
from finverse.macro import nelson_siegel

__all__ = ["var_model", "nelson_siegel"]
