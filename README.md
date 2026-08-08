# M.A.R.A. — Multi-Agent Research Analyst (Local Wiki Edition)

A self-correcting, multi-agent research system that answers questions **strictly from a private knowledge base** — no cloud APIs required for inference.

M.A.R.A. uses five specialized LangGraph agents (Planner, Researcher, Critic, Gap Researcher, Writer) orchestrated through a stateful graph with conditional routing. Instead of searching the open web, it queries a local knowledge base via semantic search, expands results along a cross-link graph, and synthesizes findings into a polished markdown report.

---

## Architecture

```
                          +-------------+
                          |  llm-wiki/  |  Knowledge base
                          |  wiki/*.md  |  (entities, concepts, sources)
                          +------+------+
                                 |
                    semantic search  +  cross-link expansion
                    (embeddings)     |  (consolidation graph)
                                 |
                    +------------v------------+
                    |  LLM_wiki_spring :8080   |  Spring Boot wiki server
                    |  /api/qmd/query          |
                    |  /api/qmd/get            |
                    |  /api/wiki/graph         |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |     M.A.R.A. agents      |
                    |  Planner   → Researcher  |
                    |     ↑          ↓         |
                    |  Gap Researcher ← Critic |
                    |               ↓          |
                    |            Writer        |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |  llama.cpp / ollama      |
                    |  :8083/v1 (OpenAI API)   |
                    |  Gemma 4 / any GGUF      |
                    +-------------------------+
```

The **Critic → Gap Researcher** loop is the key differentiator. M.A.R.A. evaluates its own findings against the knowledge base, identifies what is missing, and loops back to fill those gaps — up to a configurable iteration limit.

Each agent is a LangGraph node with typed state passing via `ResearchState`. Conditional edges control the flow, making the graph self-correcting by design.

---

## How It Works

1. **Planner** reads the knowledge base catalog (`wiki/index.md`) and cross-link graph, then generates 3–5 sub-questions constrained to topics the KB can answer.

2. **Researcher** performs semantic search over the wiki (OpenAI `text-embedding-3-small` embeddings, cosine similarity), expands results one hop along the cross-link graph, fetches full page text, and synthesizes findings with the local LLM.

3. **Critic** reviews all findings, identifies gaps (max 2 per loop), deduplicates against already-researched questions, and routes back to research if gaps remain.

4. **Gap Researcher** fills remaining gaps using the same wiki pipeline, then returns to the Critic for re-evaluation.

5. **Writer** composes a structured markdown report from accumulated findings, appending a deterministic `## Sources` section with `[[wiki/...]]` provenance links — no LLM hallucinated citations.

### Fallback mode

If `WIKI_SEARCH_URL` is left empty, M.A.R.A. falls back to **DuckDuckGo web search** and can use any OpenAI-compatible LLM (Google AI Studio, Together, etc.).

---

## Project Structure

```
.
├── agents/
│   ├── __init__.py    # Package marker
│   ├── state.py       # ResearchState TypedDict (shared graph state)
│   ├── nodes.py       # Agent logic, LLM calls, wiki retrieval, routing
│   └── graph.py       # LangGraph compilation with conditional edges
├── screenshots/       # Demo screenshots
├── app.py             # Streamlit UI with real-time agent activity panel
├── requirements.txt
├── .env               # LLM endpoint + wiki URL (not committed)
├── .gitignore
└── README.md
```

---

## Dependencies

This project depends on three external services:

| Service | Purpose | Repo / Source |
|---------|---------|---------------|
| **llm-wiki** | The knowledge base — a folder of markdown pages (`wiki/entities/`, `wiki/concepts/`, `wiki/sources/`) with an `index.md` catalog and cross-links via `[[wiki/...]]` references | Your own knowledge |
| **LLM_wiki_spring** | Spring Boot app that serves the wiki over HTTP with semantic search (embeddings), full-text fetch, and graph traversal | `~/Desktop/LLM_wiki_spring/` |
| **llama.cpp** or **ollama** | Local LLM inference server with an OpenAI-compatible `/v1` endpoint | [llama.cpp](https://github.com/ggml-org/llama.cpp) or [ollama](https://ollama.com) |

---

## Quick Start

### 1. Start the wiki server

```bash
cd ~/Desktop/LLM_wiki_spring
./mvnw spring-boot:run     # or java -jar target/*.jar
```

The wiki API will be available at `http://localhost:8080`.

### 2. Start the local LLM

**Option A — llama.cpp:**
```bash
llama-server \
  -m /path/to/gemma-4-12b-it-Q4_K_M.gguf \
  --port 8083 \
  --ctx-size 8192 \
  --threads 8 \
  --n-gpu-layers 99
```

**Option B — ollama:**
```bash
ollama serve                    # if not already running
ollama pull gemma3:12b          # or any model
# ollama exposes OpenAI-compatible API at http://localhost:11434/v1
```

### 3. Configure environment

```bash
# Create .env manually with the variables below
```

For **llama.cpp**:
```
LLM_API_BASE=http://localhost:8083/v1
LLM_MODEL=unsloth/gemma-4-12b-it-GGUF:Q4_K_M
LLM_API_KEY=local
WIKI_SEARCH_URL=http://localhost:8080
```

For **ollama**, change `LLM_API_BASE` to `http://localhost:11434/v1` and `LLM_MODEL` to the ollama model name (e.g. `gemma3:12b`).

For **cloud LLMs** (Google AI Studio, OpenAI, Together), leave `LLM_API_BASE` empty or point it at the provider's endpoint, set the appropriate API key, and optionally leave `WIKI_SEARCH_URL` empty to fall back to DuckDuckGo web search.

### 4. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter a topic covered by your knowledge base, and click **Run Research**.

---

## Agents

| Agent | Role |
|-------|------|
| **Planner** | Reads the KB catalog + cross-link graph, decomposes the topic into 3–5 sub-questions |
| **Researcher** | Semantic search → graph expansion → full-text fetch → LLM synthesis per question |
| **Critic** | Reviews findings against the KB, identifies gaps (max 2), deduplicates |
| **Gap Researcher** | Fills gaps found by the Critic, then loops back for re-evaluation |
| **Writer** | Composes a polished markdown report with provenance-tagged sources |

---

## Configuration Reference

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `LLM_API_BASE` | Yes | `https://generativelanguage.googleapis.com/v1beta/openai/` | LLM endpoint (llama.cpp, ollama, or cloud) |
| `LLM_MODEL` | Yes | `gemini-2.5-flash` | Model name passed to the endpoint |
| `LLM_API_KEY` | Yes* | `GOOGLE_API_KEY` | API key (any string for local servers) |
| `WIKI_SEARCH_URL` | No | (empty) | Wiki server base URL; unset = DuckDuckGo fallback |
| `LLM_ENABLE_THINKING` | No | `false` | Set to `true` if using Gemma 4 thinking models |

\* Required for cloud providers. llama.cpp and ollama accept any non-empty string.

---

## Acknowledgments

Built on the **[M.A.R.A. framework](https://github.com/MuditNautiyal-21/MARA)** by Mudit Nautiyal — a LangGraph-based multi-agent research architecture that was extended here with local wiki retrieval, graph-based expansion, and local LLM inference.

---

## License

MIT
