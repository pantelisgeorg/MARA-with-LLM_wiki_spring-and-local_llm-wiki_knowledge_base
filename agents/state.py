"""State schema for the multi-agent research graph."""

from __future__ import annotations
from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    research_results: Annotated[list[dict], operator.add]  # accumulated across parallel researchers
    critique: str
    gaps: list[str]
    gap_research: Annotated[list[dict], operator.add]
    final_report: str
    iteration: int
    max_iterations: int
    status: str  # current node label for UI
