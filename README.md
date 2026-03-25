<div align="center">

<br/>

```
 ██████╗ ██╗  ██╗███████╗███╗   ███╗ ██████╗ ███████╗
██╔════╝ ██║  ██║██╔════╝████╗ ████║██╔═══██╗██╔════╝
██║      ███████║█████╗  ██╔████╔██║██║   ██║███████╗
██║      ██╔══██║██╔══╝  ██║╚██╔╝██║██║   ██║╚════██║
╚██████╗ ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝███████║
 ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝
```

# ⚗️ CHEMOS — Chemical Hybrid Expert Multi-Agent Orchestration System

### *Validation-Augmented, Context-Aware RAG for Chemical Engineering Intelligence*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrated-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-00e5c3?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-52b7ff?style=flat-square)](CONTRIBUTING.md)

<br/>

> **CHEMOS** is a production-grade, multi-agent Retrieval-Augmented Generation system purpose-built for the chemical engineering domain. It combines hybrid dense + sparse retrieval, LLM-powered validation, context-persistent query refinement, and a conditional generation pipeline — all orchestrated as a directed graph via LangGraph — to deliver reliable, grounded, and traceable answers to complex domain queries.

<br/>

</div>

---

## CLI WORKING DEMO VIDEO : https://youtu.be/XWAgALZmd28

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Agent Pipeline](#-agent-pipeline)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Frontend UI](#-frontend-ui)
- [Evaluation](#-evaluation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

Chemical engineering knowledge is dense, highly specialised, and operationally critical. Generic RAG systems fail in this domain because they lack:

- The ability to resolve ambiguous follow-up questions ("explain that", "what's its use?") against prior conversation context
- A principled mechanism to **refuse generation** when retrieved context is insufficient or irrelevant
- Domain-aware hybrid retrieval that balances semantic similarity with keyword precision

**CHEMOS** addresses all three gaps through a six-node agentic pipeline, persistent chat-history-aware query refinement, and a strict validation gate that prevents hallucinated or ungrounded responses from ever reaching the user.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CHEMOS  |  LangGraph DAG                        │
│                                                                          │
│   User Query + Chat History                                              │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────────┐                                                    │
│   │  Query Refiner  │  ← Resolves ambiguity against full chat history   │
│   └────────┬────────┘                                                    │
│            │  refined_query                                              │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │  Query Analyzer │  ← Extracts intent, entities, filters, strategy   │
│   └────────┬────────┘                                                    │
│            │  structured_query                                           │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │ Hybrid Retrieval│  ← Dense (vector) + Sparse (BM25) + re-ranking    │
│   └────────┬────────┘                                                    │
│            │  retrieval_package  (docs + formatted_context)             │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │    Validator    │  ← Scores relevance + sufficiency; gates output   │
│   └────────┬────────┘                                                    │
│            │                                                             │
│     ┌──────┴──────┐                                                      │
│     │  APPROVED?  │                                                      │
│     └──┬──────┬───┘                                                      │
│      YES      NO                                                         │
│        │       │                                                         │
│        ▼       ▼                                                         │
│  ┌──────────┐ ┌───────────────┐                                          │
│  │Generator │ │Refusal Handler│                                          │
│  └────┬─────┘ └──────┬────────┘                                          │
│       │               │                                                  │
│       └───────┬───────┘                                                  │
│               ▼                                                          │
│          final_answer                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### State Object

All nodes communicate exclusively through a typed `AgentState` dictionary — there are no shared globals or side-channel dependencies between agents.

```python
class AgentState(TypedDict):
    user_query:        str
    chat_history:      list[dict[str, str]]   # {"role": ..., "content": ...}
    refined_query:     Optional[str]
    structured_query:  Optional[dict]
    retrieval_package: Optional[dict]
    retrieval_count:   Optional[int]
    validation_result: Optional[dict]
    final_answer:      Optional[str]
```

---

## 🤖 Agent Pipeline

### Node 1 — Query Refiner

The first and most critical node in the pipeline. Converts ambiguous, context-dependent queries into fully self-contained search strings before any retrieval occurs.

**Handles 11 resolution scenarios:**

| Scenario | Input | Refined Output |
|---|---|---|
| No history | `"What is Bernoulli's principle?"` | *(unchanged)* |
| Single topic in history | `"explain that"` | `"Explain the Navier-Stokes equations"` |
| Multi-topic, ambiguous reference | `"what are its limitations?"` | `"What are the limitations of GPT-4?"` *(most recent topic)* |
| User references earlier topic | `"back to the first one"` | `"Explain CSTR reactor design"` *(oldest topic)* |
| Follow-up dimension | `"what about at high pressure?"` | `"How does distillation perform at high pressure?"` |
| Explicit topic switch | `"now tell me about heat exchangers"` | *(unchanged — fresh query)* |
| Long history, stale context | `"explain the process"` | Uses only the most recent coherent sub-conversation |
| Partial entity reference | `"the second method you mentioned"` | Maps to specific named method from history |
| Compound query | `"explain both"` | `"Explain [topic A] and [topic B]"` |
| Clarification request | `"what do you mean?"` | `"Clarify the explanation of [last concept]"` |
| Implicit scope | `"show me the code"` | `"Show Python code for [active algorithm]"` |

**Optimisation:** If `chat_history` is empty, the LLM call is skipped entirely and the query passes through unchanged — zero latency overhead on first turns.

---

### Node 2 — Query Analyzer

Parses the refined query into a structured representation used to guide retrieval strategy selection.

**Outputs:**

```json
{
  "intent":      "explanation",
  "entities":    ["Nusselt number", "convective heat transfer"],
  "filters":     {"domain": "heat_transfer", "doc_type": "textbook"},
  "strategy":    "hybrid",
  "complexity":  "moderate"
}
```

---

### Node 3 — Hybrid Retrieval

Executes a two-stage retrieval combining dense and sparse methods, followed by cross-encoder re-ranking.

| Stage | Method | Purpose |
|---|---|---|
| Dense Retrieval | Vector similarity (embeddings) | Semantic relevance |
| Sparse Retrieval | BM25 keyword search | Precise term matching |
| Re-ranking | Cross-encoder scoring | Final top-k selection |
| Packaging | Context formatting | Structured output for downstream nodes |

---

### Node 4 — Validator

Acts as a quality gate. Evaluates the retrieval package against two independent axes before any answer is generated.

```
Relevance Score   → Is the context topically related to the query?
Sufficiency Score → Does the context contain enough information to answer?
```

**Decision logic:**

```
IF relevance_score ≥ threshold AND sufficiency_score ≥ threshold:
    validation_decision = "APPROVED"  →  routes to Generator
ELSE:
    validation_decision = "REJECTED"  →  routes to Refusal Handler
```

This node is the primary hallucination-prevention mechanism in the system.

---

### Node 5a — Generator

Synthesises a grounded, referenced answer using only the validated context. The generator is explicitly constrained to the retrieved documents and cannot introduce external knowledge.

---

### Node 5b — Refusal Handler

Returns a structured, user-friendly refusal message when the validator rejects the retrieval package, including the specific reasoning so the user can refine their query.

---

## ✨ Key Features

- **Context-Persistent Query Refinement** — Full `chat_history` flows through every pipeline invocation; the Refiner resolves pronouns, demonstratives, and implicit references across arbitrarily long conversations
- **Validation-Gated Generation** — No answer is generated unless retrieval quality meets configurable thresholds; prevents hallucination at the architectural level
- **Hybrid Retrieval** — Dense + sparse + cross-encoder re-ranking for both semantic and lexical coverage
- **Conditional Graph Routing** — LangGraph conditional edges cleanly separate the approval and rejection paths without any if/else control flow in node functions
- **Fully Typed State** — `TypedDict`-enforced state contract; all node I/O is explicit and statically analysable
- **Singleton Retriever** — `HybridRetrievalAgent` is instantiated once at module load time; no per-query initialisation overhead
- **Structured Logging** — Every node emits timestamped, levelled log output for observability and debugging
- **Provider-Agnostic LLM Interface** — A single line change in `_call_llm_for_refinement` swaps between any LangChain-compatible ChatModel
- **Streamlit UI with Live Pipeline Visualisation** — Real-time animated node progression, per-answer metadata, and session statistics
- **Interactive CLI** — Persistent-history command-line mode for testing and evaluation without the UI

---

## 📁 Project Structure

```
chemos/
│
├── app.py                          # Streamlit front-end (VAG-RAG UI)
├── final_multi_agent_rag_system.py # LangGraph orchestration & public API
│
├── agents/
│   ├── query_analyzer_agent.py     # Intent / entity extraction
│   ├── hybrid_rag_agent.py         # Dense + sparse retrieval + re-ranking
│   ├── validator_agent.py          # Relevance & sufficiency scoring
│   └── generator_agent.py         # Grounded answer synthesis
│
├── config/
│   ├── settings.py                 # Thresholds, model names, index paths
│   └── prompts.py                  # Centralised prompt templates
│
├── data/
│   ├── raw/                        # Source documents (PDFs, text, etc.)
│   ├── processed/                  # Chunked & cleaned documents
│   └── indices/                    # Vector store & BM25 index artefacts
│
├── evaluation/
│   ├── eval_pipeline.py            # End-to-end evaluation harness
│   ├── metrics.py                  # Faithfulness, relevance, RAGAS wrappers
│   └── testsets/                   # Domain-specific Q&A ground truth
│
├── notebooks/
│   └── pipeline_exploration.ipynb  # Interactive experimentation
│
├── tests/
│   ├── test_refiner.py
│   ├── test_retrieval.py
│   └── test_validator.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- An OpenAI-compatible API key (or any LangChain-supported LLM provider)
- A configured vector store (Chroma, Pinecone, Qdrant, FAISS, etc.)

### 1. Clone the repository

```bash
git clone https://github.com/Satyam-Singh-x/MultiAgent_HybridRAG_ChemicalEngineering.git
cd MultiAgent_HybridRAG_ChemicalEngineering
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys and index paths
```

---

## 🔧 Configuration

Create a `.env` file at the project root based on `.env.example`:

```dotenv
# ── LLM Provider ────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini                  # used by Query Refiner & Generator

# ── Vector Store ────────────────────────────────────────────────
VECTOR_STORE_TYPE=chroma                  # chroma | pinecone | qdrant | faiss
VECTOR_STORE_PATH=./data/indices/chroma
EMBEDDING_MODEL=text-embedding-3-small

# ── Retrieval ────────────────────────────────────────────────────
RETRIEVAL_TOP_K=8
BM25_INDEX_PATH=./data/indices/bm25.pkl
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── Validator Thresholds ─────────────────────────────────────────
VALIDATOR_RELEVANCE_THRESHOLD=0.65
VALIDATOR_SUFFICIENCY_THRESHOLD=0.60

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL=INFO
```

---

## 🚀 Usage

### Streamlit Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the full VAG-RAG interface with live pipeline visualisation.

### Python API

```python
from final_multi_agent_rag_system import run_multi_agent_system

# First turn — no history
result = run_multi_agent_system(
    query="What is the significance of the Reynolds number in fluid flow?"
)

print(result["final_answer"])
print(result["refined_query"])        # same as input on first turn
print(result["retrieval_count"])      # number of chunks retrieved
print(result["validation_result"])    # {"validation_decision": "APPROVED", ...}

# Subsequent turn — with chat history
chat_history = [
    {"role": "user",      "content": "What is the Reynolds number?"},
    {"role": "assistant", "content": result["final_answer"]},
]

result2 = run_multi_agent_system(
    query="What are its practical implications in pipe design?",
    chat_history=chat_history,
)

# refined_query will be:
# "What are the practical implications of the Reynolds number in pipe design?"
print(result2["refined_query"])
```

### Interactive CLI

```bash
python final_multi_agent_rag_system.py
```

The CLI maintains a persistent `chat_history` across turns so the Query Refiner correctly resolves all follow-up references throughout the session.

```
╔══════════════════════════════════════════╗
║   Multi-Agent Agentic RAG  (LangGraph)   ║
║   Type 'exit' or Ctrl-C to quit.         ║
╚══════════════════════════════════════════╝

You › What is the Clausius-Clapeyron equation?
Assistant › The Clausius-Clapeyron equation describes the relationship between …

You › explain its derivation
[Refined Query]  Explain the derivation of the Clausius-Clapeyron equation
Assistant › The derivation begins from the condition of thermodynamic equilibrium …
```

---

## 📖 API Reference

### `run_multi_agent_system(query, chat_history=None)`

The primary public interface. Executes the full six-node pipeline and returns a structured result dictionary.

**Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | ✅ | User's latest message. May be ambiguous or context-dependent. |
| `chat_history` | `list[dict]` | ❌ | Prior conversation turns. Each dict: `{"role": "user"\|"assistant", "content": str}`. Defaults to `[]`. |

**Returns**

| Key | Type | Description |
|---|---|---|
| `final_answer` | `str` | Generated or refusal answer |
| `refined_query` | `str` | Query after context resolution (identical to input if no history) |
| `retrieval_count` | `int \| None` | Number of document chunks retrieved |
| `formatted_context` | `str` | Human-readable representation of retrieved context |
| `validation_result` | `dict \| None` | Full validator output including `validation_decision`, `reasoning`, `relevance_score`, `sufficiency_score` |

**Example response**

```python
{
    "final_answer":      "The Nusselt number is a dimensionless parameter …",
    "refined_query":     "Explain the physical interpretation of the Nusselt number",
    "retrieval_count":   6,
    "formatted_context": "Source 1: Heat Transfer (Cengel, 8th ed.) …",
    "validation_result": {
        "validation_decision": "APPROVED",
        "reasoning":           "Retrieved context directly addresses the query …",
        "relevance_score":     0.91,
        "sufficiency_score":   0.84,
    }
}
```

---

## 🖥️ Frontend UI

The Streamlit application (`app.py`) provides a production-grade interface designed around the chemical engineering domain.

**Interface Highlights**

| Feature | Description |
|---|---|
| Live Pipeline Visualiser | Animated 5-node strip that illuminates each agent as it runs |
| Refined Query Chip | Displays the resolved query when the Refiner modifies the input |
| Validator Banner | Colour-coded APPROVED / REJECTED status with glowing indicator |
| Score Pills | Inline display of `Chunks`, `Relevance`, and `Sufficiency` scores |
| Collapsible Context | Full retrieved context visible per message |
| Session Statistics | Running totals for queries, approvals, rejections, and chunks |
| Quick-Start Questions | Sidebar shortcuts for common chemical engineering topics |
| Persistent History | `chat_history` is maintained across the entire session |

---

## 📊 Evaluation

Run the built-in evaluation harness against the domain test set:

```bash
python evaluation/eval_pipeline.py \
    --testset evaluation/testsets/chem_eng_qa.jsonl \
    --output  evaluation/results/run_001.json \
    --metrics faithfulness,answer_relevancy,context_precision
```

**Metrics computed**

| Metric | Tool | Description |
|---|---|---|
| Faithfulness | RAGAS | Measures factual consistency of answer with retrieved context |
| Answer Relevancy | RAGAS | Measures how well the answer addresses the question |
| Context Precision | RAGAS | Measures signal-to-noise ratio in retrieved chunks |
| Validation Accuracy | Custom | Measures correctness of APPROVED / REJECTED decisions |
| Refusal Rate | Custom | Fraction of queries that triggered the refusal path |

---

## 🗺️ Roadmap

- [ ] **Streaming responses** — token-level streaming from Generator to UI via LangGraph streaming API
- [ ] **Multi-document source attribution** — inline citation markers linking answer sentences to specific source documents
- [ ] **Retrieval feedback loop** — let users mark unhelpful retrievals to improve future queries
- [ ] **Domain fine-tuned embeddings** — custom embedding model trained on chemical engineering literature
- [ ] **REST API layer** — FastAPI wrapper around `run_multi_agent_system` for service deployment
- [ ] **Docker + Compose setup** — containerised deployment with persistent vector store volume
- [ ] **Evaluation dashboard** — live RAGAS metrics panel inside the Streamlit UI
- [ ] **Multi-modal support** — process P&ID diagrams, reaction scheme images, and process flow charts

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branching strategy, code style, and the pull request process.

```bash
# Development setup
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest tests/ -v --cov=.
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full terms.

---

<div align="center">

<br/>

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) · [LangChain](https://langchain.com) · [Streamlit](https://streamlit.io)

<br/>

*"The goal of engineering is not just to build things, but to build things that work reliably."*

</div>
