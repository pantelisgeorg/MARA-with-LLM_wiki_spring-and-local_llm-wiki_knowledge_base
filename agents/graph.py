"""LangGraph multi-agent research graph."""

from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.nodes import planner, researcher, critic, gap_researcher, writer, should_continue


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("critic", critic)
    g.add_node("gap_researcher", gap_researcher)
    g.add_node("writer", writer)

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", should_continue, {
        "gap_researcher": "gap_researcher",
        "writer": "writer",
    })
    g.add_edge("gap_researcher", "critic")  # loop back
    g.add_edge("writer", END)

    return g.compile()


graph = build_graph()
