# API: Polypharmacy Safety Agent — FastAPI Backend
# Routes: POST /api/patient/scan, POST /api/doctor/check, GET /api/patient/{id}/profile, GET /api/patient/{id}/audit
# WebSocket: /api/ws/{patient_id} — streams agent progress events

"""
FastAPI backend for the Polypharmacy Safety Agent.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from audit.audit_logger import AuditLogger
from graph.safety_graph import run_patient_flow, run_doctor_flow
from memory.knowledge_store import KnowledgeStore
from memory.patient_store import PatientStore
from api.models import (
    PatientScanRequest,
    DoctorCheckRequest,
    ScanResult,
    CheckResult,
    MedicationProfile,
    AuditTrailResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("polypharmacy.api")

# ── Shared singletons ─────────────────────────────────────────────────────────

_store: PatientStore | None = None
_knowledge_store: KnowledgeStore | None = None
_audit: AuditLogger | None = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _knowledge_store, _audit
    log.info("Polypharmacy Safety Agent API starting up…")

    try:
        _store = PatientStore()
        _store._r.ping()
        log.info("✓ Redis connection OK")
    except Exception as exc:
        log.error(f"✗ Redis connection FAILED: {exc}")
        _store = None

    try:
        _knowledge_store = KnowledgeStore()
        _knowledge_store.init()
        log.info(f"✓ ChromaDB OK ({_knowledge_store.document_count} chunks)")
    except Exception as exc:
        log.error(f"✗ ChromaDB FAILED: {exc}")
        _knowledge_store = None

    try:
        _audit = AuditLogger()
        log.info("✓ AuditLogger (SQLite) OK")
    except Exception as exc:
        log.error(f"✗ AuditLogger FAILED: {exc}")
        _audit = None

    _llm_ok = False
    try:
        from llm_config import get_llm
        from langchain_core.messages import HumanMessage
        llm = get_llm()
        llm.invoke([HumanMessage(content="ping")])
        _llm_ok = True
        log.info(f"✓ LLM ready: {os.getenv('LLM_PROVIDER', 'gemini')}")
    except Exception as exc:
        log.error(f"✗ LLM FAILED: {exc}")

    if _store and _knowledge_store and _audit and _llm_ok:
        log.info("✓ READY — all systems operational")
    else:
        log.warning("⚠ DEGRADED — one or more backing services (Redis / ChromaDB / LLM) failed to initialise")

    yield

    log.info("Polypharmacy Safety Agent API shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Polypharmacy Safety Agent API",
    description="Multi-agent drug interaction detection for patients seeing multiple doctors.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(patient_id, []).append(ws)

    def disconnect(self, patient_id: str, ws: WebSocket) -> None:
        sockets = self._connections.get(patient_id, [])
        if ws in sockets:
            sockets.remove(ws)
        if not sockets:
            self._connections.pop(patient_id, None)

    async def emit(self, patient_id: str, event: dict) -> None:
        """Broadcast *event* to all sockets subscribed to *patient_id*."""
        sockets = self._connections.get(patient_id, [])
        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(patient_id, ws)


_ws_manager = _ConnectionManager()


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _emit_progress(patient_id: str, agent: str, message: str) -> None:
    await _ws_manager.emit(
        patient_id,
        {"event": "agent_complete", "agent": agent, "timestamp": _ts(), "message": message},
    )


# ── Dependency helpers ────────────────────────────────────────────────────────

def _require_store() -> PatientStore:
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is not available. Check REDIS_HOST / UPSTASH_REDIS_URL in .env.",
        )
    return _store


def _require_audit() -> AuditLogger:
    if _audit is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AuditLogger (SQLite) is not available.",
        )
    return _audit


def _action_required(severity: str, conflicts: list[dict]) -> str:
    if severity == "CRITICAL":
        pairs = ", ".join(
            f"{c.get('drug_a','?')} + {c.get('drug_b','?')}"
            for c in conflicts if c.get("severity") == "CRITICAL"
        )
        return f"URGENT: Do not prescribe until critical interactions reviewed — {pairs}"
    if severity == "MODERATE":
        return "Review interactions with patient's other prescribers before dispensing."
    return "No action required. Safe to prescribe."


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Liveness probe — returns service status."""
    redis_ok = False
    if _store:
        try:
            _store._r.ping()
            redis_ok = True
        except Exception:
            pass
    chroma_ok = bool(_knowledge_store and _knowledge_store.document_count > 0)
    return {
        "status":    "ok" if (redis_ok and chroma_ok) else "degraded",
        "redis":     redis_ok,
        "chromadb":  chroma_ok,
        "timestamp": _ts(),
    }


# --- Patient routes -----------------------------------------------------------

@app.post(
    "/api/patient/scan",
    response_model=ScanResult,
    status_code=status.HTTP_200_OK,
    tags=["Patient"],
    summary="Run full polypharmacy safety scan for a patient",
)
async def patient_scan(req: PatientScanRequest) -> ScanResult:
    """
    Load all prescriptions, normalise brand names, check interactions via
    rule engine + LLM, and return tiered reports for patient / coordinator
    / physician.
    """
    store        = _require_store()
    audit_logger = _require_audit()
    t0           = time.monotonic()

    patient_json: dict = {
        "id":          req.patient_id,
        "name":        [{"given": [req.patient_name or req.patient_id], "family": ""}],
        "medications": req.prescriptions,
        "allergies":   [],
    }

    await _emit_progress(req.patient_id, "profile_builder", "Loading medication profile…")

    try:
        # [REFS: graph/safety_graph.py > run_patient_flow]
        final_state = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_patient_flow(
                patient_id=req.patient_id,
                patient_json=patient_json,
                patient_name=req.patient_name,
                thread_id=f"scan-{req.patient_id}-{uuid.uuid4().hex[:8]}",
            ),
        )
    except Exception as exc:
        log.exception(f"patient_scan failed for {req.patient_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        )

    await _emit_progress(
        req.patient_id, "interaction_auditor",
        f"Interaction check complete — severity: {final_state.get('overall_severity', 'NONE')}",
    )
    await _emit_progress(req.patient_id, "report_generator", "Reports generated.")

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    trail      = audit_logger.get_audit_trail(req.patient_id)

    return ScanResult(
        patient_id=req.patient_id,
        overall_severity=final_state.get("overall_severity", "NONE"),
        conflicts=final_state.get("conflicts", []),
        reports=final_state.get("reports", {}),
        processing_time_ms=elapsed_ms,
        audit_trail=trail,
    )


@app.get(
    "/api/patient/{patient_id}/profile",
    response_model=MedicationProfile,
    tags=["Patient"],
    summary="Fetch current medication list from Redis",
)
async def get_patient_profile(patient_id: str) -> MedicationProfile:
    store = _require_store()

    # [REFS: memory/patient_store.py > get_medications]
    medications = store.get_medications(patient_id)
    if not medications:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No medication profile found for patient '{patient_id}'.",
        )

    med_dicts = [
        {
            "drug_name":          d.drug_name,
            "generic_name":       d.generic_name,
            "dose":               d.dose,
            "frequency":          d.frequency,
            "prescribing_doctor": d.prescribing_doctor,
            "condition":          d.condition,
            "prescription_date":  d.prescription_date,
            "active_status":      d.active_status,
            "is_normalised":      d.is_normalised,
        }
        for d in medications
    ]

    return MedicationProfile(
        patient_id=patient_id,
        medications=med_dicts,
        medication_count=len(med_dicts),
    )


@app.get(
    "/api/patient/{patient_id}/audit",
    response_model=AuditTrailResponse,
    tags=["Patient"],
    summary="Fetch audit trail from SQLite",
)
async def get_patient_audit(patient_id: str) -> AuditTrailResponse:
    audit_logger = _require_audit()
    store        = _require_store()

    if not store.get_medications(patient_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for patient '{patient_id}'.",
        )

    # [REFS: audit/audit_logger.py > get_audit_trail]
    entries = audit_logger.get_audit_trail(patient_id)
    return AuditTrailResponse(
        patient_id=patient_id,
        entries=entries,
        entry_count=len(entries),
    )


# --- Doctor routes ------------------------------------------------------------

@app.post(
    "/api/doctor/check",
    response_model=CheckResult,
    status_code=status.HTTP_200_OK,
    tags=["Doctor"],
    summary="Check a new prescription against a patient's existing medications",
)
async def doctor_check(req: DoctorCheckRequest) -> CheckResult:
    """
    Append a new drug to an existing patient profile, run the safety pipeline,
    and return a prescribing-decision summary for the physician.
    Returns 404 if the patient has no base profile yet.
    """
    store = _require_store()
    t0    = time.monotonic()

    # [REFS: memory/patient_store.py > get_medications]
    existing = store.get_medications(req.patient_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No medication profile found for patient '{req.patient_id}'. "
                "Use POST /api/patient/scan to create a base profile first."
            ),
        )

    new_drug = {
        "drug_name":          req.drug_name,
        "dose":               req.dose,
        "frequency":          req.frequency,
        "prescribing_doctor": req.prescribing_doctor,
        "condition":          req.condition,
        "prescription_date":  req.prescription_date or datetime.now(timezone.utc).date().isoformat(),
    }

    await _emit_progress(req.patient_id, "profile_builder", f"Adding {req.drug_name} to profile…")

    try:
        # [REFS: graph/safety_graph.py > run_doctor_flow]
        final_state = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_doctor_flow(
                patient_id=req.patient_id,
                new_drug=new_drug,
                thread_id=f"doctor-{req.patient_id}-{uuid.uuid4().hex[:8]}",
            ),
        )
    except Exception as exc:
        log.exception(f"doctor_check failed for {req.patient_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        )

    await _emit_progress(
        req.patient_id, "interaction_auditor",
        f"Safety check complete — severity: {final_state.get('overall_severity', 'NONE')}",
    )

    severity  = final_state.get("overall_severity", "NONE")
    conflicts = final_state.get("conflicts", [])
    reports   = final_state.get("reports", {})

    return CheckResult(
        patient_id=req.patient_id,
        safe_to_prescribe=severity != "CRITICAL",
        severity=severity,
        conflicts=conflicts,
        physician_report=reports.get("physician", ""),
        action_required=_action_required(severity, conflicts),
    )


# --- WebSocket ----------------------------------------------------------------

# NOTE: WebSocket sends JSON — client parses event.type to update UI.
@app.websocket("/api/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str) -> None:
    """
    Real-time agent progress stream.

    Connect before calling POST /api/patient/scan or /api/doctor/check.
    Events:  {"event": "agent_complete", "agent": "<node>", "timestamp": "…", "message": "…"}
    Ping:    send {"event": "ping"} — server replies {"event": "pong", "timestamp": "…"}
    """
    await _ws_manager.connect(patient_id, websocket)
    log.info(f"WebSocket connected: patient_id={patient_id}")

    try:
        await websocket.send_json({
            "event":      "connected",
            "patient_id": patient_id,
            "timestamp":  _ts(),
            "message":    "Connected to Polypharmacy Safety Agent progress stream.",
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    msg = json.loads(data)
                    if msg.get("event") == "ping":
                        await websocket.send_json({"event": "pong", "timestamp": _ts()})
                except (json.JSONDecodeError, AttributeError):
                    pass
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "keepalive", "timestamp": _ts()})

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: patient_id={patient_id}")
    except Exception as exc:
        log.warning(f"WebSocket error for {patient_id}: {exc}")
    finally:
        _ws_manager.disconnect(patient_id, websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8080)),
        reload=True,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
