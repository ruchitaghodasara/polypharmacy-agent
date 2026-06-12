# Polypharmacy Safety Agent

AI-powered multi-agent system that detects dangerous drug interactions for patients consulting multiple specialist doctors simultaneously.

---

## The Problem

When a patient sees a cardiologist, a rheumatologist, and an endocrinologist independently, each doctor prescribes without visibility into the others' medications. This information asymmetry at the point of prescribing is a leading cause of preventable adverse drug events — polypharmacy affects an estimated 40% of patients over 65 and contributes to over 125,000 deaths annually in the US alone.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend (terminal 1)
uvicorn api.main:app --reload

# 3. Open the frontend (terminal 2 — or open ui/index.html directly in a browser)
npm start
```

Copy `.env.example` to `.env` and fill in your keys before starting. See [Environment Variables](#environment-variables) below.

> **Setup check:** run `python setup_check.py` first to verify Redis, ChromaDB, Anthropic API, and sentence-transformers are all reachable.

---

## Architecture

```
                   ┌─────────────────────────────────────────────────┐
                   │              LangGraph StateGraph                │
                   │                                                  │
  FHIR Patient ──► │  ┌─────────────────┐                            │
  JSON / New Rx    │  │ ProfileBuilder   │ writes Redis               │
                   │  │    Agent         │──────────────────────────► │
                   │  └────────┬────────┘                            │
                   │           │ medications[]                        │
                   │  ┌────────▼────────┐   ChromaDB RAG             │
                   │  │ Interaction      │──────────────────────────► │
                   │  │ Auditor Agent    │   + Claude JSON check      │
                   │  └────────┬────────┘                            │
                   │           │ conflicts[]  severity                │
                   │      ┌────▼─────┐  MODERATE                     │
                   │      │ Human    │◄── interrupt (HITL)            │
                   │      │Checkpoint│                                │
                   │      └────┬─────┘                               │
                   │           │ approve / reject                     │
                   │  ┌────────▼────────┐                            │
                   │  │ Report Generator │ 3-audience XML reports     │
                   │  │    Agent         │                            │
                   │  └─────────────────┘                            │
                   └─────────────────────────────────────────────────┘
                           │                     │
                     FastAPI REST          SQLite Audit Log
                     + WebSocket           (every node event)
```

### Agents

| Agent | Responsibility |
|---|---|
| **PatientProfileBuilderAgent** | Sole writer to patient memory — parses FHIR-lite JSON, normalises brand drug names to generics (60-entry synonym map), and persists the canonical medication list to Redis. |
| **DrugInteractionAuditorAgent** | Five-step pipeline: allergy check → deterministic rule engine (15 curated rules) → ChromaDB RAG retrieval → Claude semantic classification for uncovered pairs → conflict merge with CRITICAL-wins deduplication. |
| **ConflictReportGeneratorAgent** | Generates three audience-tailored reports (patient / care coordinator / physician) in a single Claude call using XML-delimited output; falls back to pre-templated reports on any parse failure so reports are never empty. |

### Orchestration

The three agents are composed into a **LangGraph `StateGraph`** with typed state (`AgentState`), conditional routing edges, and `interrupt_before` on the human-checkpoint node for MODERATE-severity cases. A `MemorySaver` checkpointer enables graph resumption after human review.

Routing logic:
- **CRITICAL** → `immediate_alert_node` → END  
- **MODERATE** → `human_checkpoint_node` (suspends) → `report_generator_node` → END  
- **NONE** → `safe_confirm_node` → END  
- **Doctor mode / NONE** → `confirm_and_persist_node` → END

### Memory Layer

| Store | Technology | What is kept |
|---|---|---|
| Patient medications, conflicts, allergies | **Redis** (Upstash-compatible) | Per-patient JSON blobs; keys `patient:{id}:medications` etc. |
| Pharmacovigilance literature | **ChromaDB** (local persistent) | 16 FDA interaction paragraphs chunked and embedded with `all-MiniLM-L6-v2`; queried by drug-pair at audit time |
| Compliance audit trail | **SQLite** | Every node execution, human decision, and alert dispatch with severity label and state snapshot |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph 0.1 (`StateGraph`, `MemorySaver`, `interrupt_before`) |
| **LLM** | Anthropic Claude (`claude-sonnet-4-6`) via `anthropic` SDK; tenacity retry (3×, exponential 2–16 s) |
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

# Integration tests (requires Redis + ChromaDB; Claude is mocked)
pytest tests/test_integration.py -v -m integration
```

53 unit tests cover: brand normalisation, allergy detection, rule engine, Claude mock (MODERATE/NONE/failure paths), XML report parsing, fallback behaviour, and the full LangGraph pipeline on `patient_001`.

---

## Project Structure

```
polypharmacy-agent/
│
├── agents/
│   ├── profile_builder.py       # Agent 1 — FHIR parse, brand normalise, Redis write
│   ├── interaction_auditor.py   # Agent 2 — allergy + rule + RAG + Claude pipeline
│   └── report_generator.py      # Agent 3 — 3-audience reports via XML-tagged Claude
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
│   └── main.py                  # FastAPI: POST /patient/scan, POST /doctor/check,
│                                #           GET /patient/{id}/profile, WS /ws/{id}
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
│   ├── conftest.py
│   ├── test_profile_builder.py  # 15 unit tests (mock Redis)
│   ├── test_interaction_auditor.py  # 21 unit tests (mock ChromaDB + Claude)
│   ├── test_report_generator.py     # 17 unit tests (mock Claude)
│   └── test_integration.py          # 8 integration tests (real Redis + ChromaDB)
│
├── .env.example
├── requirements.txt
├── pytest.ini
└── setup_check.py               # Pre-flight: Redis ping, ChromaDB, Anthropic, embeddings
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `UPSTASH_REDIS_URL` | Yes* | Full `rediss://` URL from Upstash console |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_SSL` | Yes* | Alternative to `UPSTASH_REDIS_URL` for split config |
| `CHROMA_PERSIST_DIR` | Yes | Local path for ChromaDB storage (default: `./data/chromadb`) |
| `CLAUDE_MODEL` | No | Model ID (default: `claude-sonnet-4-6`) |
| `LLM_MAX_RETRIES` | No | Tenacity retry count (default: `3`) |
| `LOG_LEVEL` | No | `INFO` \| `DEBUG` \| `WARNING` |

\* Supply either `UPSTASH_REDIS_URL` **or** the four split variables — not both.

---

## Demo Walkthrough

Follow these three steps to demonstrate the system end-to-end in an interview:

### Step 1 — Scan a High-Risk Patient

Open `http://localhost:8000/docs` (FastAPI Swagger UI) or use the React UI at `http://localhost:3000`.

POST to `/api/patient/scan` with the contents of `data/mock_patients/patient_002.json` (Priya Krishnamurthy — Warfarin + Aspirin).

**What to look for:** `overall_severity: "CRITICAL"`, a conflict entry for `Warfarin + Aspirin` (rule IR-001, major bleeding risk), and three populated reports (patient / coordinator / physician). The WebSocket trace in the UI shows each agent node completing in real time.

### Step 2 — Test Brand-Name Normalisation

POST `/api/patient/scan` with `data/mock_patients/patient_006.json` (Meena Pillai — prescribed "Brufen" by one doctor and Lisinopril by another).

**What to look for:** the `medications` array in the response shows `drug_name: "Brufen"` with `generic_name: "Ibuprofen"` and `is_normalised: true`. The conflict `Lisinopril + Ibuprofen` (IR-003) is detected even though the prescription used the brand name. Without normalisation this interaction would be silently missed.

### Step 3 — Run the Evaluation Suite

```bash
python evaluation/eval_runner.py
```

**What to look for:** the terminal report showing Critical Recall = **100%** (patient-002 CRITICAL correctly identified), Normalisation Accuracy = **100%** (patient-006 Brufen resolved), conflict F1 score, and all 6 patients with audit entries. The report is also saved to `evaluation/results/eval_report.txt`.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over raw function calls | Explicit typed state, conditional routing, and `interrupt_before` HITL without boilerplate async plumbing |
| Separate rule engine + Claude | Deterministic rules catch known critical interactions with zero latency and no token cost; Claude handles novel or ambiguous pairs the rules don't cover |
| XML-delimited three-report output | Single LLM call for three audiences; regex extraction is more robust than JSON parsing for long prose; fallback templates guarantee non-empty output |
| Brand → generic normalisation before matching | Drug pair matching would silently miss interactions if one prescription used a brand name; the 60-entry synonym map covers the most common trade names |
| SQLite audit log (separate from Redis) | Compliance trail must survive Redis eviction and be queryable by severity; SQLite gives durable indexed storage with no extra service dependency |
