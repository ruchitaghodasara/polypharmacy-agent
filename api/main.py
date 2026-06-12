"""
FastAPI backend for the Polypharmacy Safety Agent.

Endpoints
─────────
POST /api/patient/scan          Run full patient safety pipeline
POST /api/doctor/check          Run doctor-mode drug check
GET  /api/patient/{id}/profile  Fetch current medication list from Redis
GET  /api/patient/{id}/audit    Fetch SQLite audit trail
WS   /api/ws/{id}               Real-time agent progress events

Startup
───────
Verifies Redis connectivity and ChromaDB initialisation on boot.
Logs READY or FAILED to stdout.

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
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Deferred project imports (after dotenv so env vars are set)
from audit.audit_logger import AuditLogger
from graph.safety_graph import run_patient_flow, run_doctor_flow
from memory.knowledge_store import KnowledgeStore
from memory.patient_store import PatientStore

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("polypharmacy.api")

# ── Startup / shutdown ────────────────────────────────────────────────────────

_store: PatientStore | None = None
_knowledge_store: KnowledgeStore | None = None
_audit: AuditLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _knowledge_store, _audit
    log.info("Polypharmacy Safety Agent API starting up…")

    # Redis
    try:
        _store = PatientStore()
        _store._r.ping()
        log.info("✓ Redis connection OK")
    except Exception as exc:
        log.error(f"✗ Redis connection FAILED: {exc}")
        _store = None

    # ChromaDB + knowledge ingestion
    try:
        _knowledge_store = KnowledgeStore()
        _knowledge_store.init()
        log.info(f"✓ ChromaDB OK ({_knowledge_store.document_count} chunks)")
    except Exception as exc:
        log.error(f"✗ ChromaDB FAILED: {exc}")
        _knowledge_store = None

    # SQLite audit logger
    try:
        _audit = AuditLogger()
        log.info("✓ AuditLogger (SQLite) OK")
    except Exception as exc:
        log.error(f"✗ AuditLogger FAILED: {exc}")
        _audit = None

    if _store and _knowledge_store and _audit:
        log.info("✓ READY — all systems operational")
    else:
        log.warning("⚠ DEGRADED — one or more backing services failed to initialise")

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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self) -> None:
        # patient_id → list of active sockets
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(patient_id, []).append(ws)
        log.debug(f"WS connected: patient_id={patient_id}")

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
        {
            "event": "agent_complete",
            "agent": agent,
            "timestamp": _ts(),
            "message": message,
        },
    )


# ── Pydantic models ───────────────────────────────────────────────────────────

class PatientScanRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, description="Unique patient identifier")
    prescriptions: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="List of prescription dicts in FHIR-lite format"
    )
    patient_name: Optional[str] = Field(None, description="Display name for reports")

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "patient-001",
            "patient_name": "Arjun Sharma",
            "prescriptions": [
                {
                    "drug_name": "Lisinopril", "dose": "10mg", "frequency": "once daily",
                    "prescribing_doctor": "Dr. Anjali Mehta", "condition": "Hypertension",
                    "prescription_date": "2023-05-20", "active_status": True
                }
            ]
        }
    }}


class DoctorCheckRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    drug_name: str = Field(..., min_length=1)
    dose: str = Field(..., min_length=1)
    frequency: str = Field(default="once daily")
    prescribing_doctor: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    prescription_date: Optional[str] = Field(None, description="ISO-8601 date, defaults to today")

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "patient-001",
            "drug_name": "Aspirin",
            "dose": "75mg",
            "frequency": "once daily",
            "prescribing_doctor": "Dr. Cardiac Specialist",
            "condition": "Cardiovascular prevention",
            "prescription_date": "2024-06-01"
        }
    }}


class ScanResult(BaseModel):
    patient_id: str
    overall_severity: str
    conflicts: List[Dict[str, Any]]
    reports: Dict[str, Any]
    processing_time_ms: int
    audit_trail: List[Dict[str, Any]]


class CheckResult(BaseModel):
    patient_id: str
    safe_to_prescribe: bool
    severity: str
    conflicts: List[Dict[str, Any]]
    physician_report: str
    action_required: str


class MedicationProfile(BaseModel):
    patient_id: str
    medications: List[Dict[str, Any]]
    medication_count: int


class AuditTrailResponse(BaseModel):
    patient_id: str
    entries: List[Dict[str, Any]]
    entry_count: int


# ── Helper: assert dependencies are up ────────────────────────────────────────

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
            for c in conflicts
            if c.get("severity") == "CRITICAL"
        )
        return f"URGENT: Do not prescribe until critical interactions reviewed — {pairs}"
    if severity == "MODERATE":
        return "Review interactions with patient's other prescribers before dispensing."
    return "No action required. Safe to prescribe."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Liveness probe — returns service status."""
    redis_ok = False
    chroma_ok = False

    if _store:
        try:
            _store._r.ping()
            redis_ok = True
        except Exception:
            pass

    chroma_ok = bool(_knowledge_store and _knowledge_store.document_count > 0)

    return {
        "status": "ok" if (redis_ok and chroma_ok) else "degraded",
        "redis": redis_ok,
        "chromadb": chroma_ok,
        "timestamp": _ts(),
    }


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
    rule engine + Claude, and return tiered reports for patient / coordinator
    / physician.
    """
    store = _require_store()
    audit_logger = _require_audit()
    t0 = time.monotonic()

    # Build a minimal FHIR-lite patient dict from the request
    patient_json: dict = {
        "id": req.patient_id,
        "name": [{"given": [req.patient_name or req.patient_id], "family": ""}],
        "medications": req.prescriptions,
        "allergies": [],
    }

    await _emit_progress(req.patient_id, "profile_builder", "Loading medication profile…")

    try:
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
        req.patient_id,
        "interaction_auditor",
        f"Interaction check complete — severity: {final_state.get('overall_severity', 'NONE')}",
    )
    await _emit_progress(
        req.patient_id,
        "report_generator",
        "Reports generated.",
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    trail = audit_logger.get_audit_trail(req.patient_id)

    return ScanResult(
        patient_id=req.patient_id,
        overall_severity=final_state.get("overall_severity", "NONE"),
        conflicts=final_state.get("conflicts", []),
        reports=final_state.get("reports", {}),
        processing_time_ms=elapsed_ms,
        audit_trail=trail,
    )


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
    t0 = time.monotonic()

    # Verify patient profile exists before running the full pipeline
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
        "drug_name": req.drug_name,
        "dose": req.dose,
        "frequency": req.frequency,
        "prescribing_doctor": req.prescribing_doctor,
        "condition": req.condition,
        "prescription_date": req.prescription_date or datetime.now(timezone.utc).date().isoformat(),
    }

    await _emit_progress(req.patient_id, "profile_builder", f"Adding {req.drug_name} to profile…")

    try:
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
        req.patient_id,
        "interaction_auditor",
        f"Safety check complete — severity: {final_state.get('overall_severity', 'NONE')}",
    )

    severity = final_state.get("overall_severity", "NONE")
    conflicts = final_state.get("conflicts", [])
    reports = final_state.get("reports", {})

    return CheckResult(
        patient_id=req.patient_id,
        safe_to_prescribe=severity not in ("CRITICAL",),
        severity=severity,
        conflicts=conflicts,
        physician_report=reports.get("physician", ""),
        action_required=_action_required(severity, conflicts),
    )


@app.get(
    "/api/patient/{patient_id}/profile",
    response_model=MedicationProfile,
    tags=["Patient"],
    summary="Fetch current medication list from Redis",
)
async def get_patient_profile(patient_id: str) -> MedicationProfile:
    store = _require_store()

    medications = store.get_medications(patient_id)
    if not medications:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No medication profile found for patient '{patient_id}'.",
        )

    med_dicts = [
        {
            "drug_name": d.drug_name,
            "generic_name": d.generic_name,
            "dose": d.dose,
            "frequency": d.frequency,
            "prescribing_doctor": d.prescribing_doctor,
            "condition": d.condition,
            "prescription_date": d.prescription_date,
            "active_status": d.active_status,
            "is_normalised": d.is_normalised,
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
    store = _require_store()

    # Check patient exists
    if not store.get_medications(patient_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for patient '{patient_id}'.",
        )

    entries = audit_logger.get_audit_trail(patient_id)
    return AuditTrailResponse(
        patient_id=patient_id,
        entries=entries,
        entry_count=len(entries),
    )


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/api/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str) -> None:
    """
    Real-time agent progress stream.

    Connect before calling POST /api/patient/scan or /api/doctor/check.
    Events have the shape:
        {"event": "agent_complete", "agent": "<node>", "timestamp": "…", "message": "…"}

    The server also echoes back any JSON message you send (ping/pong support).
    The connection closes automatically when the pipeline finishes or on error.
    """
    await _ws_manager.connect(patient_id, websocket)
    log.info(f"WebSocket connected: patient_id={patient_id}")

    try:
        await websocket.send_json(
            {
                "event": "connected",
                "patient_id": patient_id,
                "timestamp": _ts(),
                "message": "Connected to Polypharmacy Safety Agent progress stream.",
            }
        )

        # Keep connection alive; handle client pings
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
                # Send keepalive
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
