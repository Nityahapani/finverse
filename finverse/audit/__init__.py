"""
finverse.audit — model health checks, NLP, and data validation.

Functions / Modules
-------------------
audit()           — audit DCF, LBO, ThreeStatement, or Excel files
loughran_mcdonald — financial text sentiment (LM dictionary)
benford           — Benford's Law test for data manipulation
earnings_quality  — composite 10-factor earnings quality score
"""
from finverse.audit.model_audit import audit, AuditResult, AuditFlag
from finverse.audit import loughran_mcdonald
from finverse.audit import benford
from finverse.audit import earnings_quality

__all__ = ["audit", "AuditResult", "AuditFlag",
           "loughran_mcdonald", "benford", "earnings_quality"]
