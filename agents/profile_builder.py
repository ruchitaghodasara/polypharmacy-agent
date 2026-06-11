"""
Agent 1 — PatientProfileBuilderAgent

Sole writer to patient memory.  All other agents read; only this agent writes
to patient:{id}:medications and patient:{id}:audit in Redis.

LangGraph integration
---------------------
Pass this agent's run_agent function as a node::

    from agents.profile_builder import run_agent as build_profile
    graph.add_node("profile_builder", build_profile)

State contract
--------------
Input keys consumed:
    mode            str   "patient" | "doctor"  (required)
    patient_id      str   (required)
    patient_json    dict  FHIR patient dict       (patient mode only)
    new_drug        dict  {drug_name, dose, frequency, prescribing_doctor,
                           condition, prescription_date}   (doctor mode only)

Output keys always present:
    patient_id      str
    mode            str
    medications     List[dict]   serialised Drug objects
    status          str   "OK" | "INCOMPLETE_PROFILE" | "ERROR"
    skip_audit      bool  True when < 2 active medications after processing
    error           str | None

Output keys added in doctor mode:
    pending_drug    dict | None  the appended drug dict (if profile existed)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Project imports ────────────────────────────────────────────────────────────
from tools.fhir_parser import Drug, parse_fhir_patient, normalise_drug
from memory.patient_store import PatientStore, _drug_to_dict

# Path to synonym file — resolved relative to this file so it works regardless
# of the working directory the process is started from.
_SYNONYMS_PATH = Path(__file__).parent.parent / "data" / "drug_synonyms.json"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_count(drugs: list[Drug]) -> int:
    return sum(1 for d in drugs if d.active_status)


def _drugs_to_dicts(drugs: list[Drug]) -> list[dict]:
    return [_drug_to_dict(d) for d in drugs]


def _make_ok(patient_id: str, mode: str, drugs: list[Drug]) -> dict:
    return {
        "patient_id": patient_id,
        "mode": mode,
        "medications": _drugs_to_dicts(drugs),
        "status": "OK",
        "skip_audit": _active_count(drugs) < 2,
        "error": None,
    }


def _make_error(patient_id: str, mode: str, message: str) -> dict:
    return {
        "patient_id": patient_id,
        "mode": mode,
        "medications": [],
        "status": "ERROR",
        "skip_audit": True,
        "error": message,
    }


# ── Patient mode ───────────────────────────────────────────────────────────────

def _run_patient_mode(state: dict, store: PatientStore) -> dict:
    """
    Bulk-load: parse all medications from a FHIR patient dict, normalise brand
    names, and persist to Redis.
    """
    patient_id: str = state["patient_id"]
    patient_json: dict = state.get("patient_json", {})

    if not patient_json:
        return _make_error(patient_id, "patient", "patient_json is missing or empty")

    # Parse + normalise
    try:
        drugs = parse_fhir_patient(patient_json, synonyms_path=_SYNONYMS_PATH)
    except Exception as exc:
        return _make_error(patient_id, "patient", f"FHIR parse failed: {exc}")

    normalised = [d for d in drugs if d.is_normalised]

    # Write medications
    store.save_medications(patient_id, drugs)

    # Write allergies if present in the FHIR dict
    allergies = patient_json.get("allergies", [])
    if allergies:
        store.save_allergies(patient_id, allergies)

    # Detailed audit entry
    store._audit(
        patient_id,
        "profile_builder:patient_mode",
        {
            "ts": _now_iso(),
            "drug_count": len(drugs),
            "active_count": _active_count(drugs),
            "normalised": [{"from": d.drug_name, "to": d.generic_name} for d in normalised],
            "allergy_count": len(allergies),
        },
    )

    result = _make_ok(patient_id, "patient", drugs)

    # Annotate skip reason for downstream nodes
    if result["skip_audit"]:
        result["skip_reason"] = (
            f"Only {_active_count(drugs)} active medication(s) — interaction check skipped."
        )

    return result


# ── Doctor mode ────────────────────────────────────────────────────────────────

def _run_doctor_mode(state: dict, store: PatientStore) -> dict:
    """
    Single-drug append: validate the patient profile exists, then append the
    new prescription with pending_confirmation=True.
    """
    patient_id: str = state["patient_id"]
    new_drug_input: dict = state.get("new_drug", {})

    if not new_drug_input:
        return _make_error(patient_id, "doctor", "new_drug is missing or empty")

    # Guard: profile must already exist
    existing_drugs = store.get_medications(patient_id)
    if not existing_drugs:
        store._audit(
            patient_id,
            "profile_builder:doctor_mode:no_profile",
            {"ts": _now_iso(), "attempted_drug": new_drug_input.get("drug_name")},
        )
        return {
            "patient_id": patient_id,
            "mode": "doctor",
            "medications": [],
            "status": "INCOMPLETE_PROFILE",
            "skip_audit": True,
            "pending_drug": None,
            "error": (
                f"No existing profile for patient '{patient_id}'. "
                "Run in patient mode first to build the base profile."
            ),
        }

    # Normalise the incoming drug name
    raw_name: str = new_drug_input.get("drug_name", "")
    generic_name = normalise_drug(raw_name, _SYNONYMS_PATH)
    is_normalised = generic_name.lower() != raw_name.lower()

    new_drug = Drug(
        drug_name=raw_name,
        generic_name=generic_name,
        dose=new_drug_input.get("dose", ""),
        frequency=new_drug_input.get("frequency", ""),
        prescribing_doctor=new_drug_input.get("prescribing_doctor", ""),
        condition=new_drug_input.get("condition", ""),
        prescription_date=new_drug_input.get("prescription_date", _now_iso()[:10]),
        active_status=True,
        is_normalised=is_normalised,
    )

    updated_drugs = existing_drugs + [new_drug]
    store.save_medications(patient_id, updated_drugs)

    # Capture the pending drug as a dict with the extra confirmation flag
    pending_drug_dict = _drug_to_dict(new_drug)
    pending_drug_dict["pending_confirmation"] = True

    store._audit(
        patient_id,
        "profile_builder:doctor_mode:drug_appended",
        {
            "ts": _now_iso(),
            "drug_name": raw_name,
            "generic_name": generic_name,
            "is_normalised": is_normalised,
            "prescribing_doctor": new_drug.prescribing_doctor,
            "pending_confirmation": True,
            "total_medications": len(updated_drugs),
        },
    )

    result = _make_ok(patient_id, "doctor", updated_drugs)
    result["pending_drug"] = pending_drug_dict
    return result


# ── LangGraph node ─────────────────────────────────────────────────────────────

def run_agent(state: dict, store: PatientStore | None = None) -> dict:
    """
    LangGraph-compatible node function.

    *store* is injectable for testing; defaults to a PatientStore built from
    environment variables when not supplied.
    """
    if store is None:
        store = PatientStore()

    mode = state.get("mode", "")
    patient_id = state.get("patient_id", "")

    if not patient_id:
        return _make_error("", mode, "patient_id is required in state")

    if mode == "patient":
        return _run_patient_mode(state, store)
    elif mode == "doctor":
        return _run_doctor_mode(state, store)
    else:
        return _make_error(patient_id, mode, f"Unknown mode '{mode}'. Must be 'patient' or 'doctor'.")


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    project_root = Path(__file__).parent.parent
    patient_file = project_root / "data" / "mock_patients" / "patient_001.json"

    with open(patient_file, "r", encoding="utf-8") as fh:
        patient_json = json.load(fh)

    patient_id = patient_json["id"]

    # ── Connect store (fail fast if Redis is down) ─────────────────────────────
    store = PatientStore()
    try:
        store._r.ping()
        print("\n[PASS] Redis connection\n")
    except Exception as exc:
        print(f"\n[FAIL] Redis connection — {exc}")
        sys.exit(1)

    # Clean slate for repeatable test runs
    store.delete_patient(patient_id)

    # ────────────────────────────────────────────────────────────────────────────
    # Test 1: patient mode — bulk load patient_001
    # ────────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: patient mode — bulk load patient_001.json")
    print("=" * 60)

    state_in = {
        "mode": "patient",
        "patient_id": patient_id,
        "patient_json": patient_json,
    }
    result = run_agent(state_in, store=store)

    print(f"  status       : {result['status']}")
    print(f"  skip_audit   : {result['skip_audit']}")
    print(f"  medications  : {len(result['medications'])} loaded")
    for med in result["medications"]:
        normalised_tag = f"  ← normalised from '{med['drug_name']}'" if med["is_normalised"] else ""
        print(f"    • {med['generic_name']} {med['dose']}{normalised_tag}")
    assert result["status"] == "OK", f"Expected OK, got {result['status']}"
    assert len(result["medications"]) == 4
    assert result["skip_audit"] is False
    print("  [PASS]\n")

    # ────────────────────────────────────────────────────────────────────────────
    # Test 2: doctor mode — append a new prescription to patient_001
    # ────────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2: doctor mode — append Nurofen (brand) for patient_001")
    print("=" * 60)

    state_in2 = {
        "mode": "doctor",
        "patient_id": patient_id,
        "new_drug": {
            "drug_name": "Nurofen",          # brand name — must normalise to Ibuprofen
            "dose": "200mg",
            "frequency": "as needed",
            "prescribing_doctor": "Dr. Testdoctor",
            "condition": "Headache",
            "prescription_date": "2024-03-01",
        },
    }
    result2 = run_agent(state_in2, store=store)

    print(f"  status           : {result2['status']}")
    print(f"  total medications: {len(result2['medications'])}")
    pending = result2.get("pending_drug", {})
    print(f"  pending_drug     : {pending.get('generic_name')} (normalised={pending.get('is_normalised')})")
    print(f"  pending_confirm  : {pending.get('pending_confirmation')}")
    assert result2["status"] == "OK"
    assert pending["generic_name"] == "Ibuprofen"
    assert pending["is_normalised"] is True
    assert pending["pending_confirmation"] is True
    assert len(result2["medications"]) == 5   # 4 original + 1 new
    print("  [PASS]\n")

    # ────────────────────────────────────────────────────────────────────────────
    # Test 3: doctor mode — no existing profile (expect INCOMPLETE_PROFILE)
    # ────────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 3: doctor mode — unknown patient (no profile)")
    print("=" * 60)

    state_in3 = {
        "mode": "doctor",
        "patient_id": "patient-UNKNOWN",
        "new_drug": {
            "drug_name": "Aspirin",
            "dose": "75mg",
            "frequency": "once daily",
            "prescribing_doctor": "Dr. Nobody",
            "condition": "Prevention",
            "prescription_date": "2024-03-01",
        },
    }
    result3 = run_agent(state_in3, store=store)

    print(f"  status     : {result3['status']}")
    print(f"  skip_audit : {result3['skip_audit']}")
    print(f"  error      : {result3['error']}")
    assert result3["status"] == "INCOMPLETE_PROFILE"
    assert result3["skip_audit"] is True
    print("  [PASS]\n")

    # ────────────────────────────────────────────────────────────────────────────
    # Test 4: audit log integrity
    # ────────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 4: audit log for patient_001")
    print("=" * 60)

    audit = store.get_audit_log(patient_id)
    for entry in audit:
        print(f"  [{entry['event']}]  {entry.get('detail', {})}")
    assert len(audit) >= 2, "Expected at least 2 audit entries"
    print(f"\n  [PASS] {len(audit)} audit entries written\n")

    # Cleanup
    store.delete_patient(patient_id)
    print("All PatientProfileBuilderAgent tests passed.")
