"""
finverse.backtest — strategy backtesting engine.

Modules
-------
engine  — signal-based backtesting, momentum, DCF signal strategies
"""
from finverse.backtest import engine
from finverse.backtest.engine import run, momentum, dcf_signal

__all__ = ["engine", "run", "momentum", "dcf_signal"]
