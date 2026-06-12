# Polypharmacy Safety Agent

AI-powered multi-agent system that detects dangerous drug interactions for patients consulting multiple specialist doctors simultaneously.

---

## The Problem

When a patient sees a cardiologist, a rheumatologist, and an endocrinologist independently, each doctor prescribes without visibility into the others' medications. This information asymmetry at the point of prescribing is a leading cause of preventable adverse drug events — polypharmacy affects an estimated 40% of patients over 65 and contributes to over 125,000 deaths annually in the US alone.

---

## Quick Start

```bash
# 1. Copy .env.example to .env and fill in your API keys
cp .env.example .env

# 2. Start the backend
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# 3. Open the frontend at http://localhost:3000 or open ui/index.html directly in a browser
npm start
```

> **Setup check:** run `python setup_check.py` first to verify Redis, ChromaDB, and sentence-transformers are all reachable.

---

## Architecture

- **ProfileBuilderAgent** — parses FHIR-lite JSON, normalises brand names to generics via a 60-entry synonym map, and writes the canonical medication list to Redis
- **DrugInteractionAuditorAgent** — five-step pipeline: allergy check → deterministic rule engine (15 curated rules) → ChromaDB RAG retrieval → LLM semantic check for uncovered pairs → CRITICAL-wins deduplication
- **ConflictReportGeneratorAgent** — generates three audience-tailored reports (patient / care coordinator / physician) in a single LLM call using XML-delimited output; pre-templated fallback ensures reports are never empty
- **LangGraph StateGraph** — typed `AgentState`, conditional routing (`CRITICAL` → immediate alert, `MODERATE` → `interrupt_before` human checkpoint, `NONE` → safe confirm), `MemorySaver` checkpointer enables graph resumption after human review
- **Memory layer** — Redis (medications / conflicts / allergies per patient), ChromaDB (16 FDA interaction paragraphs embedded with `all-MiniLM-L6-v2`), SQLite (compliance audit trail with severity index for every node execution)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph 0.1 (`StateGraph`, `MemorySaver`, `interrupt_before`) |
| **LLM** | Configurable via `LLM_PROVIDER` env var: Google Gemini, Groq, Cerebras, or Ollama; tenacity retry (3×, exponential 2–16 s) |
| **Patient Memory** | Redis / Upstash (`redis-py 5`) |
| **Knowledge RAG** | ChromaDB 0.5 + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **API** | FastAPI 0.111 + Uvicorn; WebSocket per-patient broadcast for real-time agent trace |
| **Frontend** | React 18 (CDN), Tailwind CSS (CDN), Babel standalone — zero build step |
| **Evaluation** | Custom eval runner; micro-F1 conflict detection, critical recall gate, brand normalisation accuracy |

---

## Running the Evaluation

```bash
python evaluation/eval_runner.py
```

Runs the full LangGraph pipeline against all 6 mock patients and compares output to `data/golden_dataset.json`. Reports:

| Metric | What it checks | Target |
|---|---|---|
| **Critical Recall** | Are all CRITICAL-severity patients correctly identified? | **100%** (exits code 1 if missed) |
| **Conflict F1** | Micro-averaged precision / recall over detected drug pairs vs. ground truth | Maximise |
| **Normalisation Accuracy** | Does patient-006 "Brufen" resolve to "Ibuprofen" before interaction check? | 100% |
| **Average Latency** | Mean wall-clock seconds per patient end-to-end | Informational |
| **Audit Completeness** | All 6 patients have SQLite audit entries after their run | Pass / Fail |

Results are printed to the terminal and saved to `evaluation/results/eval_report.txt`.

---

## Tests

```bash
# Unit tests only (no live services needed)
pytest tests/test_profile_builder.py tests/test_interaction_auditor.py tests/test_report_generator.py -v

# Integration tests (requires Redis + ChromaDB; LLM is mocked)
pytest tests/test_integration.py -v -m integration
```

53 unit tests cover: brand normalisation, allergy detection, rule engine, LLM mock (MODERATE/NONE/failure paths), XML report parsing, fallback behaviour, and the full LangGraph pipeline on `patient_001`.

---

## Demo Walkthrough

### Step 1 — Scan a High-Risk Patient

POST to `/api/patient/scan` with the contents of `data/mock_patients/patient_002.json` (Priya Krishnamurthy — Warfarin + Aspirin) via the FastAPI Swagger UI at `http://localhost:8080/docs` or the React UI at `http://localhost:3000`.

**What to look for:** `overall_severity: "CRITICAL"`, a conflict entry for `Warfarin + Aspirin` (rule IR-001, major bleeding risk), and three populated reports (patient / coordinator / physician). The WebSocket trace in the UI shows each agent node completing in real time.

### Step 2 — Test Brand-Name Normalisation

POST `/api/patient/scan` with `data/mock_patients/patient_006.json` (Meena Pillai — prescribed "Brufen" by one doctor and Lisinopril by another).

**What to look for:** the `medications` array shows `drug_name: "Brufen"` with `generic_name: "Ibuprofen"` and `is_normalised: true`. The conflict `Lisinopril + Ibuprofen` (IR-003) is detected even though the prescription used the brand name — without normalisation this interaction would be silently missed.

### Step 3 — Run the Evaluation Suite

```bash
python evaluation/eval_runner.py
```

**What to look for:** Critical Recall = **100%**, Normalisation Accuracy = **100%**, conflict F1 score, and all 6 patients with audit entries in the results table.

---

## Commenting Convention

Three structured comment types are used throughout the codebase:

- `# NOTE:` — a non-obvious invariant or constraint a reader must know (e.g. `# NOTE: Redis stores JSON strings — deserialise on every read`)
- `# WHY:` — a design decision that would otherwise look arbitrary (e.g. `# WHY: CRITICAL severity bypasses human checkpoint — patient safety rule`)
- `# [REFS: file.py > function]` — a cross-file dependency callout linking the current code to the implementation it relies on

---

## Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over raw function calls | Explicit typed state, conditional routing, and `interrupt_before` HITL without boilerplate async plumbing |
| Separate rule engine + LLM | Deterministic rules catch known critical interactions with zero latency and no token cost; LLM handles novel or ambiguous pairs the rules don't cover |
| XML-delimited three-report output | Single LLM call for three audiences; regex extraction is more robust than JSON parsing for long prose; fallback templates guarantee non-empty output |
| Brand → generic normalisation before matching | Drug pair matching would silently miss interactions if one prescription used a brand name; the 60-entry synonym map covers the most common trade names |
| SQLite audit log (separate from Redis) | Compliance trail must survive Redis eviction and be queryable by severity; SQLite gives durable indexed storage with no extra service dependency |

---

## Project Structure

```
polypharmacy-agent/
│
├── agents/
│   ├── profile_builder.py       # Agent 1 — FHIR parse, brand normalise, Redis write
│   ├── interaction_auditor.py   # Agent 2 — allergy + rule + RAG + LLM pipeline
│   └── report_generator.py      # Agent 3 — 3-audience reports via XML-tagged LLM
│
├── graph/
│   └── safety_graph.py          # LangGraph StateGraph, nodes, edges, public run_*_flow()
│
├── memory/
│   ├── patient_store.py         # Redis wrapper — medications, conflicts, allergies, audit
│   └── knowledge_store.py       # ChromaDB wrapper — ingest + semantic query
│
├── tools/
│   ├── fhir_parser.py           # Drug dataclass, FHIR-lite parser, normalise_drug()
│   ├── rule_engine.py           # Deterministic pair check against interaction_rules.json
│   └── report_formatter.py      # Prompt builders + fallback templates for Agent 3
│
├── audit/
│   └── audit_logger.py          # SQLite audit log — every node event with severity
│
├── api/
│   ├── main.py                  # FastAPI: POST /patient/scan, POST /doctor/check,
│   │                            #          GET /patient/{id}/profile, WS /ws/{id}
│   └── models.py                # Pydantic request and response models
│
├── ui/
│   ├── index.html               # React 18 + Tailwind CDN shell
│   └── App.jsx                  # Single-file SPA: Dashboard, Patient Portal,
│                                # Doctor Station, Health Card, Audit Timeline
│
├── data/
│   ├── mock_patients/           # patient_001–006.json (FHIR-lite, covers all test cases)
│   ├── knowledge/
│   │   ├── interaction_rules.json   # 15 curated drug-interaction rules (IR-001–IR-015)
│   │   └── fda_interactions.txt     # 16 pharmacovigilance paragraphs for RAG
│   ├── drug_synonyms.json       # 60 brand → generic mappings
│   └── golden_dataset.json      # Ground-truth severities and conflict pairs for eval
│
├── evaluation/
│   └── eval_runner.py           # End-to-end eval: F1, critical recall, normalisation
│
├── tests/
│   ├── conftest.py              # Shared fixtures: FakeStore, make_drug, mock_redis
│   ├── test_profile_builder.py  # 15 unit tests (mock Redis)
│   ├── test_interaction_auditor.py  # 21 unit tests (mock ChromaDB + LLM)
│   ├── test_report_generator.py     # 17 unit tests (mock LLM)
│   └── test_integration.py          # 8 integration tests (real Redis + ChromaDB)
│
├── .env.example
├── requirements.txt
├── pytest.ini
└── setup_check.py               # Pre-flight: Redis ping, ChromaDB, embeddings
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `gemini` \| `groq` \| `cerebras` \| `ollama` |
| `UPSTASH_REDIS_URL` | Yes* | Full `rediss://` URL from Upstash console |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_SSL` | Yes* | Alternative to `UPSTASH_REDIS_URL` for split config |
| `CHROMA_PERSIST_DIR` | Yes | Local path for ChromaDB storage (default: `./data/chromadb`) |
| `LLM_MAX_RETRIES` | No | Tenacity retry count (default: `3`) |
| `LOG_LEVEL` | No | `INFO` \| `DEBUG` \| `WARNING` |

\* Supply either `UPSTASH_REDIS_URL` **or** the four split variables — not both.
