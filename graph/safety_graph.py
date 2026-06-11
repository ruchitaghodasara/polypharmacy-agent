"""
LangGraph StateGraph for the Polypharmacy Safety Agent.

Flow overview
─────────────
Patient mode:
  profile_builder_node
      │
      ▼
  interaction_auditor_node
      │
      ├─ CRITICAL  ──► immediate_alert_node ──► END
      ├─ MODERATE  ──► human_checkpoint_node (interrupt) ──► report_generator_node ──► END
      └─ NONE      ──► safe_confirm_node ──► END

Doctor mode:
  profile_builder_node
      │
      ├─ INCOMPLETE_PROFILE ──► END  (early exit, no profile yet)
      ▼
  interaction_auditor_node
      │
      ├─ CRITICAL  ──► immediate_alert_node ──► END
      ├─ MODERATE  ──► human_checkpoint_node (interrupt) ──► confirm_and_persist_node ──► END
      └─ NONE      ──► confirm_and_persist_node ──► END

Public API:
    run_patient_flow(patient_id, patient_json)  → final state dict
    run_doctor_flow(patient_id, new_drug)        → final state dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import anthropic

from agents.profile_builder import run_agent as _profile_builder_run
from agents.interaction_auditor import run_agent as _auditor_run
from agents.report_generator import run_agent as _report_gen_run
from memory.patient_store import PatientStore
from memory.knowledge_store import KnowledgeStore
from audit.audit_logger import AuditLogger

# ── Shared singletons (lazy-initialised) ─────────────────────────────────────

_store: PatientStore | None = None
_knowledge_store: KnowledgeStore | None = None
_anthropic_client: anthropic.Anthropic | None = None
_audit_logger: AuditLogger | None = None


def _get_store() -> PatientStore:
    global _store
    if _store is None:
        _store = PatientStore()
    return _store


def _get_knowledge_store() -> KnowledgeStore:
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore()
        _knowledge_store.init()
    return _knowledge_store


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return _anthropic_client


def _get_audit() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# ── AgentState ────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Identity & routing
    patient_id: str
    patient_name: str
    mode: str                    # "patient" | "doctor"

    # Profile builder outputs
    medications: List[dict]      # serialised Drug dicts
    skip_audit: bool
    incomplete_profile: bool     # True when doctor mode finds no base profile

    # Interaction auditor outputs
    conflicts: List[dict]
    overall_severity: str        # "CRITICAL" | "MODERATE" | "NONE"

    # Human checkpoint
    human_decision: str          # "approve" | "reject" | "pending"

    # Report generator outputs
    reports: dict                # {patient, coordinator, physician, fallback_used}

    # Cross-cutting
    audit_entries: List[dict]    # accumulated log entries for this run
    status: str
    error: Optional[str]

    # Inputs (not mutated by nodes)
    patient_json: dict           # raw FHIR patient dict (patient mode)
    new_drug: dict               # new prescription dict (doctor mode)


# ── Node implementations ──────────────────────────────────────────────────────

def profile_builder_node(state: AgentState) -> AgentState:
    audit = _get_audit()
    store = _get_store()

    result = _profile_builder_run(state, store=store)

    audit.log_event(
        patient_id=state.get("patient_id", ""),
        node="profile_builder_node",
        severity="INFO",
        action=f"profile_built status={result.get('status')} drugs={len(result.get('medications', []))}",
        state={"status": result.get("status"), "drug_count": len(result.get("medications", []))},
    )

    updates: AgentState = {
        "medications": result.get("medications", []),
        "skip_audit": result.get("skip_audit", False),
        "status": result.get("status", "ERROR"),
        "error": result.get("error"),
    }

    # Doctor mode: profile not found
    if result.get("status") == "INCOMPLETE_PROFILE":
        updates["incomplete_profile"] = True
        updates["overall_severity"] = "NONE"
        updates["conflicts"] = []
        updates["reports"] = {
            "patient": (
                "We could not find an existing medication record for you. "
                "Please ask your primary care provider to set up your medication profile first."
            ),
            "coordinator": "No patient profile found. Base profile must be created before doctor-mode updates.",
            "physician": "Patient profile not found in system. Base medication record required.",
            "fallback_used": True,
        }

    return updates


def interaction_auditor_node(state: AgentState) -> AgentState:
    audit = _get_audit()

    result = _auditor_run(
        state,
        store=_get_store(),
        knowledge_store=_get_knowledge_store(),
        anthropic_client=_get_client(),
    )

    severity = result.get("overall_severity", "NONE")
    n_conflicts = len(result.get("conflicts", []))

    audit.log_event(
        patient_id=state.get("patient_id", ""),
        node="interaction_auditor_node",
        severity=severity,
        action=f"audit_complete conflicts={n_conflicts} severity={severity}",
        state={
            "overall_severity": severity,
            "conflict_count": n_conflicts,
            "status": result.get("status"),
        },
    )

    return {
        "conflicts": result.get("conflicts", []),
        "overall_severity": severity,
        "status": result.get("status", "OK"),
        "error": result.get("error"),
    }


def human_checkpoint_node(state: AgentState) -> AgentState:
    """
    Interrupt point for MODERATE severity cases.

    In a live deployment LangGraph suspends here via interrupt_before and waits
    for an external event (web UI button, Slack approval, API call) to resume
    with human_decision set to "approve" or "reject".

    In batch/test mode this node auto-approves if human_decision is unset.
    """
    audit = _get_audit()
    decision = state.get("human_decision", "approve")   # default: auto-approve

    audit.log_event(
        patient_id=state.get("patient_id", ""),
        node="human_checkpoint_node",
        severity="INFO",
        action=f"human_decision={decision}",
        state={"decision": decision, "overall_severity": state.get("overall_severity")},
    )

    return {"human_decision": decision}


def report_generator_node(state: AgentState) -> AgentState:
    audit = _get_audit()

    result = _report_gen_run(
        state,
        store=_get_store(),
        anthropic_client=_get_client(),
    )

    audit.log_event(
        patient_id=state.get("patient_id", ""),
        node="report_generator_node",
        severity="INFO",
        action=f"reports_generated status={result.get('status')} fallback={result.get('reports', {}).get('fallback_used')}",
        state={"status": result.get("status"), "fallback_used": result.get("reports", {}).get("fallback_used")},
    )

    return {
        "reports": result.get("reports", {}),
        "status": result.get("status", "OK"),
        "error": result.get("error"),
    }


def immediate_alert_node(state: AgentState) -> AgentState:
    """
    CRITICAL path: generate reports then emit an immediate alert event.
    In production this would also trigger SMS/pager/EHR alert.
    """
    audit = _get_audit()
    patient_id = state.get("patient_id", "")
    conflicts = state.get("conflicts", [])

    # Generate reports for the alert
    report_result = _report_gen_run(
        state,
        store=_get_store(),
        anthropic_client=_get_client(),
    )

    critical_drugs = [
        f"{c.get('drug_a','?')} + {c.get('drug_b','?')}"
        for c in conflicts
        if c.get("severity") == "CRITICAL"
    ]

    audit.log_event(
        patient_id=patient_id,
        node="immediate_alert_node",
        severity="CRITICAL",
        action=f"CRITICAL_ALERT dispatched — {len(critical_drugs)} critical pair(s): {'; '.join(critical_drugs)}",
        state={
            "critical_pairs": critical_drugs,
            "conflict_count": len(conflicts),
            "report_status": report_result.get("status"),
        },
    )

    # Persist alert flag to Redis audit
    _get_store()._audit(
        patient_id,
        "graph:immediate_alert",
        {"critical_pairs": critical_drugs, "overall_severity": "CRITICAL"},
    )

    reports = report_result.get("reports", {})
    reports["alert_dispatched"] = True

    return {
        "reports": reports,
        "status": "CRITICAL_ALERT",
        "error": None,
    }


def safe_confirm_node(state: AgentState) -> AgentState:
    """NONE severity path: generate safe-confirmation reports."""
    audit = _get_audit()

    report_result = _report_gen_run(
        state,
        store=_get_store(),
        anthropic_client=_get_client(),
    )

    audit.log_event(
        patient_id=state.get("patient_id", ""),
        node="safe_confirm_node",
        severity="NONE",
        action="safe_confirmation_generated",
        state={"status": report_result.get("status")},
    )

    return {
        "reports": report_result.get("reports", {}),
        "status": "SAFE",
        "error": None,
    }


def confirm_and_persist_node(state: AgentState) -> AgentState:
    """
    Doctor mode only: remove pending_confirmation flag from the new drug
    and write the finalised medication list back to Redis.
    """
    audit = _get_audit()
    store = _get_store()
    patient_id = state.get("patient_id", "")

    medications = store.get_medications(patient_id)
    persisted_drug: str | None = None

    updated: list = []
    for drug in medications:
        d = {
            "drug_name": drug.drug_name,
            "generic_name": drug.generic_name,
            "dose": drug.dose,
            "frequency": drug.frequency,
            "prescribing_doctor": drug.prescribing_doctor,
            "condition": drug.condition,
            "prescription_date": drug.prescription_date,
            "active_status": drug.active_status,
            "is_normalised": drug.is_normalised,
        }
        updated.append(d)

    # Re-save via store (pending_confirmation is not a Drug field; the Drug
    # dataclass never carried it — it only existed on the pending_drug dict
    # in the profile_builder output, so saving the Drug objects here
    # automatically drops it).
    from memory.patient_store import _dict_to_drug
    drug_objects = [_dict_to_drug(d) for d in updated]
    store.save_medications(patient_id, drug_objects)

    # Generate reports (no-conflict or moderate confirmation)
    report_result = _report_gen_run(
        state,
        store=store,
        anthropic_client=_get_client(),
    )

    store._audit(
        patient_id,
        "graph:confirm_and_persist",
        {"total_medications": len(drug_objects), "human_decision": state.get("human_decision", "auto")},
    )

    audit.log_event(
        patient_id=patient_id,
        node="confirm_and_persist_node",
        severity="INFO",
        action=f"new_drug_persisted total_medications={len(drug_objects)}",
        state={"total_medications": len(drug_objects)},
    )

    return {
        "reports": report_result.get("reports", {}),
        "status": "PERSISTED",
        "error": None,
    }


# ── Conditional edge router ───────────────────────────────────────────────────

def _route_after_profile(state: AgentState) -> str:
    """Route after profile_builder_node."""
    if state.get("incomplete_profile"):
        return END
    if state.get("skip_audit"):
        mode = state.get("mode", "patient")
        return "safe_confirm_node" if mode == "patient" else "confirm_and_persist_node"
    return "interaction_auditor_node"


def _route_after_audit(state: AgentState) -> str:
    """Route after interaction_auditor_node based on overall_severity."""
    severity = state.get("overall_severity", "NONE")
    mode = state.get("mode", "patient")
    human_decision = state.get("human_decision", "pending")

    if severity == "CRITICAL":
        return "immediate_alert_node"

    if severity == "MODERATE":
        # If human already decided (resume after interrupt), skip checkpoint
        if human_decision == "reject":
            # Rejected: still generate safe report but don't persist
            return "safe_confirm_node"
        return "human_checkpoint_node"

    # NONE
    if mode == "doctor":
        return "confirm_and_persist_node"
    return "safe_confirm_node"


def _route_after_checkpoint(state: AgentState) -> str:
    """Route after human_checkpoint_node."""
    decision = state.get("human_decision", "approve")
    mode = state.get("mode", "patient")

    if decision == "reject":
        return "safe_confirm_node"

    # approved
    if mode == "doctor":
        return "confirm_and_persist_node"
    return "report_generator_node"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the safety StateGraph.

    Parameters
    ----------
    checkpointer : optional LangGraph checkpointer for persistence / resumption.
                   Defaults to an in-memory MemorySaver.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("profile_builder_node",      profile_builder_node)
    graph.add_node("interaction_auditor_node",  interaction_auditor_node)
    graph.add_node("human_checkpoint_node",     human_checkpoint_node)
    graph.add_node("report_generator_node",     report_generator_node)
    graph.add_node("immediate_alert_node",      immediate_alert_node)
    graph.add_node("safe_confirm_node",         safe_confirm_node)
    graph.add_node("confirm_and_persist_node",  confirm_and_persist_node)

    # Entry point
    graph.set_entry_point("profile_builder_node")

    # Edges
    graph.add_conditional_edges(
        "profile_builder_node",
        _route_after_profile,
        {
            "interaction_auditor_node": "interaction_auditor_node",
            "safe_confirm_node": "safe_confirm_node",
            "confirm_and_persist_node": "confirm_and_persist_node",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "interaction_auditor_node",
        _route_after_audit,
        {
            "immediate_alert_node":     "immediate_alert_node",
            "human_checkpoint_node":    "human_checkpoint_node",
            "safe_confirm_node":        "safe_confirm_node",
            "confirm_and_persist_node": "confirm_and_persist_node",
        },
    )

    # human_checkpoint_node is configured with interrupt_before in compile()
    graph.add_conditional_edges(
        "human_checkpoint_node",
        _route_after_checkpoint,
        {
            "report_generator_node":    "report_generator_node",
            "confirm_and_persist_node": "confirm_and_persist_node",
            "safe_confirm_node":        "safe_confirm_node",
        },
    )

    graph.add_edge("report_generator_node",    END)
    graph.add_edge("immediate_alert_node",     END)
    graph.add_edge("safe_confirm_node",        END)
    graph.add_edge("confirm_and_persist_node", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_checkpoint_node"],
    )
    return compiled


# ── Public flow functions ─────────────────────────────────────────────────────

def run_patient_flow(
    patient_id: str,
    patient_json: dict,
    patient_name: str | None = None,
    thread_id: str | None = None,
    human_decision: str = "approve",
) -> dict:
    """
    Run the full patient-mode safety pipeline.

    Parameters
    ----------
    patient_id    : str   patient identifier matching data/mock_patients/*.json "id"
    patient_json  : dict  full FHIR-lite patient dict
    patient_name  : str   display name for reports (auto-extracted from FHIR if None)
    thread_id     : str   LangGraph thread ID for checkpoint resumption
    human_decision: str   pre-set human decision ("approve") for batch/test mode
    """
    if patient_name is None:
        name = patient_json.get("name", [{}])[0]
        patient_name = f"{name.get('given', [''])[0]} {name.get('family', '')}".strip()

    if thread_id is None:
        thread_id = f"patient-flow-{patient_id}"

    compiled = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "mode": "patient",
        "patient_json": patient_json,
        "human_decision": human_decision,
    }

    final_state = compiled.invoke(initial_state, config=config)
    return final_state


def run_doctor_flow(
    patient_id: str,
    new_drug: dict,
    patient_name: str = "",
    thread_id: str | None = None,
    human_decision: str = "approve",
) -> dict:
    """
    Run the doctor-mode flow to append a new prescription.

    Parameters
    ----------
    patient_id    : str   existing patient identifier
    new_drug      : dict  {drug_name, dose, frequency, prescribing_doctor,
                           condition, prescription_date}
    patient_name  : str   display name for reports
    thread_id     : str   LangGraph thread ID
    human_decision: str   pre-set decision for batch/test mode
    """
    if thread_id is None:
        thread_id = f"doctor-flow-{patient_id}"

    compiled = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "patient_id": patient_id,
        "patient_name": patient_name or f"Patient {patient_id}",
        "mode": "doctor",
        "new_drug": new_drug,
        "human_decision": human_decision,
    }

    final_state = compiled.invoke(initial_state, config=config)
    return final_state


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    store = _get_store()
    try:
        store._r.ping()
        print("\n[PASS] Redis connection")
    except Exception as exc:
        print(f"\n[FAIL] Redis — {exc}")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    audit = _get_audit()

    # ── Test 1: patient flow with patient_001 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1: patient flow — patient_001 (Arjun, 4 drugs)")
    print("=" * 60)

    p1_file = project_root / "data" / "mock_patients" / "patient_001.json"
    with open(p1_file) as fh:
        p1_json = json.load(fh)

    patient_id = p1_json["id"]
    store.delete_patient(patient_id)
    audit.clear_patient(patient_id)

    final = run_patient_flow(
        patient_id=patient_id,
        patient_json=p1_json,
        thread_id=f"test-{patient_id}",
    )

    print(f"  status           : {final.get('status')}")
    print(f"  overall_severity : {final.get('overall_severity')}")
    print(f"  conflicts        : {len(final.get('conflicts', []))}")
    print(f"  fallback_used    : {final.get('reports', {}).get('fallback_used')}")
    print(f"  patient report   : {str(final.get('reports', {}).get('patient', ''))[:120]}…")

    trail = audit.get_audit_trail(patient_id)
    print(f"\n  Audit trail ({len(trail)} entries):")
    for e in trail:
        print(f"    [{e['severity']:8s}] {e['node_name']} — {e['action']}")

    assert final.get("overall_severity") in ("CRITICAL", "MODERATE", "NONE")
    assert "reports" in final
    assert all(k in final["reports"] for k in ("patient", "coordinator", "physician"))
    print("\n  [PASS]\n")

    # ── Test 2: patient flow with patient_005 (single drug, SKIPPED) ──────────
    print("=" * 60)
    print("TEST 2: patient flow — patient_005 (Kumar, 1 drug, expect SAFE)")
    print("=" * 60)

    p5_file = project_root / "data" / "mock_patients" / "patient_005.json"
    with open(p5_file) as fh:
        p5_json = json.load(fh)

    patient_id_5 = p5_json["id"]
    store.delete_patient(patient_id_5)

    final_5 = run_patient_flow(
        patient_id=patient_id_5,
        patient_json=p5_json,
        thread_id=f"test-{patient_id_5}",
    )

    print(f"  status           : {final_5.get('status')}")
    print(f"  overall_severity : {final_5.get('overall_severity')}")
    print(f"  patient report   : {str(final_5.get('reports', {}).get('patient', ''))[:120]}…")
    assert final_5.get("status") in ("SAFE", "OK", "SKIPPED", "FALLBACK")
    print("  [PASS]\n")

    # ── Test 3: doctor flow — append drug to patient_001 ──────────────────────
    print("=" * 60)
    print("TEST 3: doctor flow — append Aspirin to patient_001")
    print("=" * 60)

    # patient_001 was loaded in Test 1, profile exists in Redis
    final_doc = run_doctor_flow(
        patient_id=patient_id,
        new_drug={
            "drug_name": "Aspirin",
            "dose": "75mg",
            "frequency": "once daily",
            "prescribing_doctor": "Dr. Cardiac Specialist",
            "condition": "Cardiovascular prevention",
            "prescription_date": "2024-06-01",
        },
        patient_name="Arjun Sharma",
        thread_id=f"test-doctor-{patient_id}",
    )

    print(f"  status           : {final_doc.get('status')}")
    print(f"  overall_severity : {final_doc.get('overall_severity')}")
    medications_now = store.get_medications(patient_id)
    print(f"  medications now  : {len(medications_now)} in Redis")
    assert final_doc.get("status") in ("PERSISTED", "CRITICAL_ALERT", "SAFE", "FALLBACK", "OK")
    print("  [PASS]\n")

    # Cleanup
    store.delete_patient(patient_id)
    store.delete_patient(patient_id_5)
    audit.clear_patient(patient_id)
    print("All safety graph tests passed.")
