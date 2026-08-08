"""Streamlit UI for M.A.R.A. - Multi-Agent Research Analyst."""

import os
import sys

# Ensure the script's own directory is on sys.path (Windows + Streamlit fix)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# Also add CWD as fallback
_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents.graph import graph  # noqa: E402

### Page config ###############################
st.set_page_config(page_title="M.A.R.A.", layout="wide")

AGENT_META = {
    "planner":        ("Planner",        "Breaking topic into sub-questions"),
    "researcher":     ("Researcher",      "Searching the web and synthesizing"),
    "critic":         ("Critic",          "Reviewing for gaps and quality"),
    "gap_researcher": ("Gap Researcher",  "Filling knowledge gaps"),
    "writer":         ("Writer",          "Composing the final report"),
}

### Sidebar ###
with st.sidebar:
    st.title("M.A.R.A.")
    st.caption("Multi-Agent Research Analyst - powered by LangGraph")
    st.divider()
    topic = st.text_area("Enter a research topic", height=100,
                         placeholder="e.g. Impact of AI on healthcare in 2025")
    max_iter = st.slider("Max refinement loops", 1, 3, 2)
    run = st.button("Run Research", type="primary", use_container_width=True)
    st.divider()
    st.markdown(
        "**Agent Pipeline**\n\n"
        "Planner > Researcher > Critic > Gap Researcher > Writer"
    )

### Main ####
if run and topic:
    col_status, col_report = st.columns([1, 2])

    with col_status:
        st.subheader("Agent Activity")
        status_container = st.container()

    with col_report:
        st.subheader("Research Report")
        report_placeholder = st.empty()
        report_placeholder.info("Agents are working ...")

    initial_state = {
        "topic": topic,
        "sub_questions": [],
        "research_results": [],
        "critique": "",
        "gaps": [],
        "gap_research": [],
        "final_report": "",
        "iteration": 0,
        "max_iterations": max_iter,
        "status": "",
    }

    # Stream graph execution node-by-node
    final = None
    with status_container:
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                label, desc = AGENT_META.get(node_name, (node_name, ""))
                with st.expander(f"{label}", expanded=True):
                    st.caption(desc)
                    if node_name == "planner":
                        for i, q in enumerate(node_output.get("sub_questions", []), 1):
                            st.markdown(f"**{i}.** {q}")
                    elif node_name in ("researcher", "gap_researcher"):
                        key = "research_results" if node_name == "researcher" else "gap_research"
                        for r in node_output.get(key, []):
                            st.markdown(f"**Q:** {r['question']}")
                            st.markdown(r["findings"][:300] + " ...")
                    elif node_name == "critic":
                        st.markdown(node_output.get("critique", ""))
                        gaps = node_output.get("gaps", [])
                        if gaps:
                            st.warning(f"Gaps found: {len(gaps)} - looping back")
                        else:
                            st.success("No gaps - proceeding to write")
                    elif node_name == "writer":
                        final = node_output.get("final_report", "")

    if final:
        report_placeholder.empty()
        with col_report:
            st.markdown(final)
            st.download_button("Download Report", final, file_name="research_report.md",
                               mime="text/markdown", use_container_width=True)
elif run:
    st.warning("Please enter a research topic.")
