"""Shared records emitted by quantitative research agents."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class EvidenceRef:
    feature: str
    start_date: date
    end_date: date
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class AgentSignal:
    agent_id: str
    version: str
    symbol: str
    decision_date: date
    horizon_sessions: int
    score: float
    input_cutoff: date
    evidence: EvidenceRef
