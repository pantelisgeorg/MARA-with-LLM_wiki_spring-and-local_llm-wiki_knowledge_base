"""Node functions for the multi-agent research graph."""

from __future__ import annotations
import os
import re
import json
import urllib.request
import urllib.parse
from langchain_openai import ChatOpenAI
from agents.state import ResearchState

# LLM endpoint — defaults to llama.cpp on :8083. Override via .env.
# For ollama, set LLM_API_BASE=http://localhost:11434/v1.
# For cloud providers (OpenAI, Gemini), set LLM_API_BASE and LLM_API_KEY accordingly.
#
# Thinking models (e.g. Gemma 4) emit a separate `reasoning_content` that can
# consume the whole token budget and leave `content` empty. Thinking is disabled
# by default for custom endpoints (llama.cpp / ollama) so the answer lands in
# `content`. Override with LLM_ENABLE_THINKING=true.
_custom_endpoint = bool(os.getenv("LLM_API_BASE"))
_disable_thinking = _custom_endpoint and os.getenv("LLM_ENABLE_THINKING", "false").lower() != "true"
_extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if _disable_thinking else None
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "unsloth/gemma-4-12b-it-GGUF:Q4_K_M"),
    temperature=0.1,
    openai_api_key=os.getenv("LLM_API_KEY", "local"),
    openai_api_base=os.getenv("LLM_API_BASE", "http://localhost:8083/v1"),
    extra_body=_extra_body,
)

# Wiki knowledge base endpoint (LLM_wiki_spring).
# Retrieval = semantic search over entities/concepts/sources pages (index.md /
# log.md are excluded by the embedding service) + one-hop expansion along the
# consolidation cross-link graph.
WIKI_BASE = os.getenv("WIKI_SEARCH_URL", "http://localhost:8080").rstrip("/")


def _http_json(method: str, path: str, body: dict | None = None,
               params: dict | None = None):
    url = WIKI_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


_wiki_graph_cache: dict | None = None
_wiki_graph_loaded = False


def _wiki_graph() -> dict | None:
    """Fetch the cross-link graph ({nodes, edges}) once, cached. Built by LinkGraph
    from [[wiki/...]] refs — the consolidation relational layer over the wiki."""
    global _wiki_graph_cache, _wiki_graph_loaded
    if _wiki_graph_loaded:
        return _wiki_graph_cache
    _wiki_graph_loaded = True
    try:
        _wiki_graph_cache = _http_json(
            "GET", "/api/wiki/graph", params={"includeSources": "true"})
    except Exception:
        _wiki_graph_cache = None
    return _wiki_graph_cache


def _wiki_adjacency() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (label_by_path, adjacency) from the cached cross-link graph."""
    g = _wiki_graph()
    if not isinstance(g, dict):
        return {}, {}
    labels = {n.get("id", ""): (n.get("label") or n.get("id", ""))
              for n in g.get("nodes", [])}
    adj: dict[str, list[str]] = {}
    for n in g.get("nodes", []):
        adj.setdefault(n.get("id", ""), [])
    for e in g.get("edges", []):
        f, t = e.get("from", ""), e.get("to", "")
        if f and t:
            adj.setdefault(f, []).append(t)
    return labels, adj


def _wiki_graph_summary() -> str | None:
    """Render the graph as a compact 'page -> linked pages' adjacency list."""
    labels, adj = _wiki_adjacency()
    if not adj:
        return None
    lines = []
    for src in adj:
        targets = adj[src]
        if not targets:
            continue
        src_label = labels.get(src, src)
        tgt_labels = [labels.get(t, t) for t in targets]
        lines.append(f"- {src_label} -> {', '.join(tgt_labels)}")
    return "\n".join(lines) if lines else None


def _wiki_retrieve(question: str, limit: int = 6, max_pages: int = 8) -> tuple[str, list[str]]:
    """Semantic-search the llm-wiki, then expand one hop along the cross-link graph,
    returning (full page text for seed pages + graph neighbors, list of page paths)."""
    try:
        resp = _http_json("POST", "/api/qmd/query",
                          body={"query": question, "limit": limit})
    except Exception as e:
        return f"[wiki search error: {e}]", []
    hits = resp.get("results", []) if isinstance(resp, dict) else []
    if not hits:
        return "[no matching pages found in the knowledge base for this question]", []

    seed_paths = [h.get("file", "") for h in hits if h.get("file")]
    score_by_path = {h.get("file", ""): float(h.get("score", 0.0)) for h in hits}

    _, adj = _wiki_adjacency()
    paths = list(seed_paths)
    for p in seed_paths:
        for nb in adj.get(p, []):
            if nb and nb not in paths:
                paths.append(nb)
    paths = paths[:max_pages]

    chunks = []
    for path in paths:
        page_text = ""
        try:
            got = _http_json("GET", "/api/qmd/get", params={"file": path})
            page_text = got.get("text", "") if isinstance(got, dict) else ""
        except Exception:
            page_text = ""
        role = (f"semantic match, score {score_by_path[path]:.3f}"
                if path in score_by_path else "graph neighbor")
        chunks.append(f"[{path}] ({role})\n{page_text}")
    return "\n\n---\n\n".join(chunks), paths


_wiki_index_cache: str | None = None
_wiki_index_loaded = False


def _wiki_index() -> str | None:
    """Fetch wiki/index.md (catalog of available topics) once, cached."""
    global _wiki_index_cache, _wiki_index_loaded
    if _wiki_index_loaded:
        return _wiki_index_cache or None
    _wiki_index_loaded = True
    try:
        got = _http_json("GET", "/api/qmd/get", params={"file": "wiki/index.md"})
        _wiki_index_cache = got.get("text", "") if isinstance(got, dict) else ""
    except Exception:
        _wiki_index_cache = ""
    return _wiki_index_cache or None


def _clean(text: str) -> str:
    """Strip Gemma thinking tags from output."""
    return re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Pull JSON out of text, stripping ```json ... ``` markdown fences that
    local models (e.g. Gemma 4) tend to wrap around structured output."""
    t = text.strip()
    m = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    return t


def _normalize(s: str) -> str:
    """Normalize text for fuzzy gap dedup: lowercase, alnum only."""
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


# ### Planner ######

def planner(state: ResearchState) -> dict:
    """Break the topic into 3-5 focused sub-questions."""
    index_text = _wiki_index()
    graph_text = _wiki_graph_summary()
    prompt = (
        "You are a research planner. The research must be answered strictly "
        "from a knowledge base. Below is the knowledge base's catalog of "
        "available topics and its graph of cross-links between them. Given the "
        "topic, produce 3-5 specific sub-questions that, when answered together, "
        "form a comprehensive report — but keep every sub-question answerable "
        "from the listed topics, and prefer questions that trace the shown "
        "relationships. Return ONLY a JSON array of strings. No thinking tags.\n\n"
        f"Topic: {state['topic']}"
    )
    if index_text:
        prompt += f"\n\nKnowledge base catalog:\n{index_text}"
    if graph_text:
        prompt += f"\n\nRelationship graph (page -> linked pages):\n{graph_text}"
    resp = llm.invoke(prompt)
    content = _extract_json(_clean(resp.content))
    try:
        questions = json.loads(content)
    except json.JSONDecodeError:
        questions = []
        for q in content.splitlines():
            q = q.strip().strip(",").strip('"').strip("'").strip()
            if q and q not in ("[", "]", "```json", "```"):
                questions.append(q)
    return {"sub_questions": questions, "status": "Planning complete"}


# ### Researcher ###

def _research_question(question: str) -> dict:
    """Search the knowledge base and summarize findings for a question."""
    raw, sources = _wiki_retrieve(question)
    prompt = (
        f"You are a research analyst working strictly from a knowledge base. "
        f"Using ONLY the sources below, write a concise 2-3 paragraph summary "
        f"answering the question. Do not use any prior knowledge — every claim "
        f"must be supported by the provided sources. If the sources do not cover "
        f"part of the question, say so explicitly and omit that part rather than "
        f"speculating. Write clean prose; do not include citations. No thinking tags.\n\n"
        f"Question: {question}\n\nKnowledge base sources:\n{raw}"
    )
    summary = llm.invoke(prompt)
    return {"question": question, "findings": _clean(summary.content), "sources": sources}


def researcher(state: ResearchState) -> dict:
    """Research all sub-questions."""
    results = [_research_question(q) for q in state["sub_questions"]]
    return {"research_results": results, "status": "Research complete"}


# ### Gap Researcher (runs only on iteration > 0) ##############################

def gap_researcher(state: ResearchState) -> dict:
    """Research gaps identified by the critic."""
    results = [_research_question(g) for g in state.get("gaps", [])]
    return {"gap_research": results, "status": "Gap research complete"}


# ### Critic ######

def critic(state: ResearchState) -> dict:
    """Review research for gaps and quality."""
    all_findings = "\n\n".join(
        f"Q: {r['question']}\n{r['findings']}"
        for r in state["research_results"] + state.get("gap_research", [])
    )
    researched = list(state.get("sub_questions", [])) + [
        r.get("question", "") for r in state.get("gap_research", [])
    ]
    researched_str = "\n".join(f"- {q}" for q in researched if q) or "(none yet)"
    resp = llm.invoke(
        f"You are a research critic. Review the findings below for the topic "
        f'"{state["topic"]}".\n\n'
        f"Findings:\n{all_findings}\n\n"
        f"Already-researched questions (do not repeat these):\n{researched_str}\n\n"
        f"The knowledge base is FIXED — do not flag gaps about topics it cannot "
        f"cover. The questions listed above have ALREADY been researched; do NOT "
        f"re-flag any of them or anything already covered by the findings. Only "
        f"flag a gap if it is essential to the topic AND likely answerable from "
        f"the knowledge base. Return at most 2 gaps. If the findings adequately "
        f"cover the topic given what the knowledge base contains, return an empty "
        f"gaps array.\n\n"
        f"Respond in JSON with two keys:\n"
        f'  "critique": a short paragraph assessing quality,\n'
        f'  "gaps": a JSON array of strings.\n'
        f"No thinking tags. Return only valid JSON."
    )
    content = _extract_json(_clean(resp.content))
    try:
        parsed = json.loads(content)
        critique = parsed.get("critique", content)
        gaps = parsed.get("gaps", [])
    except json.JSONDecodeError:
        critique = content
        gaps = []
    if not isinstance(gaps, list):
        gaps = []
    seen_norm = {_normalize(q) for q in researched if q}
    deduped: list[str] = []
    for g in gaps:
        if not isinstance(g, str):
            continue
        n = _normalize(g)
        if n and n not in seen_norm:
            deduped.append(g)
            seen_norm.add(n)
        if len(deduped) >= 2:
            break
    return {
        "critique": critique,
        "gaps": deduped,
        "iteration": state.get("iteration", 0) + 1,
        "status": "Critique complete",
    }


# ### Writer ######

def writer(state: ResearchState) -> dict:
    """Produce the final markdown report."""
    all_findings = "\n\n".join(
        f"Q: {r['question']}\n{r['findings']}"
        for r in state["research_results"] + state.get("gap_research", [])
    )
    resp = llm.invoke(
        f"You are a senior research writer. Using the findings and critique below, "
        f"write a polished, well-structured markdown report on the topic "
        f'"{state["topic"]}".\n\n'
        f"Include: title, executive summary, sections for each key area, and a "
        f"conclusion. Write using ONLY the findings below — do not introduce any "
        f"fact not present in the findings, and do not use prior knowledge. Write "
        f"clean prose without inline citations; do not add a sources section. "
        f"Do not include any thinking tags.\n\n"
        f"Findings:\n{all_findings}\n\nCritique:\n{state['critique']}"
    )
    report = _clean(resp.content)
    seen: list[str] = []
    for r in state["research_results"] + state.get("gap_research", []):
        for p in r.get("sources", []) or []:
            if p and p not in seen:
                seen.append(p)
    if seen:
        report = report.rstrip() + "\n\n## Sources\n" + "\n".join(
            f"- [[{p}]]" for p in seen)
    return {"final_report": report, "status": "Report complete"}


# ### Routing ######

def should_continue(state: ResearchState) -> str:
    """Decide whether to loop back for gap research or proceed to writing."""
    if state.get("gaps") and state.get("iteration", 0) < state.get("max_iterations", 2):
        return "gap_researcher"
    return "writer"
