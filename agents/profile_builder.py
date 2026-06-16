# Agent: PatientProfileBuilderAgent
# Role:  Parse a FHIR patient bundle or a single new drug and persist to Redis.
# Input: mode, patient_id, patient_json (patient mode) | new_drug (doctor mode)
# Output: patient_id, mode, medications, status, skip_audit, error, pending_drug

"""
Agent 1 — PatientProfileBuilderAgent

Sole writer to patient memory.  All other agents read; only this agent writes
to patient:{id}:medications and patient:{id}:audit in Redis.

LangGraph integration::

    from agents.profile_builder import run_agent as build_profile
    graph.add_node("profile_builder", build_profile)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.fhir_parser import Drug, parse_fhir_patient, normalise_drug
from memory.patient_store import PatientStore, _drug_to_dict

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_RETRIES       = 3
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_NONE     = "NONE"

KEY_PATIENT_ID  = "patient_id"
KEY_MODE        = "mode"
KEY_MEDICATIONS = "medications"
KEY_STATUS      = "status"
KEY_SKIP_AUDIT  = "skip_audit"
KEY_ERROR       = "error"
KEY_PATIENT_JSON = "patient_json"
KEY_NEW_DRUG    = "new_drug"
KEY_PENDING_DRUG = "pending_drug"

_SYNONYMS_PATH = Path(__file__).parent.parent / "data" / "drug_synonyms.json"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _active_count(drugs: list[Drug]) -> int:
    """Count active medications in *drugs*."""
    return sum(1 for d in drugs if d.active_status)


def _drugs_to_dicts(drugs: list[Drug]) -> list[dict]:
    """Serialise a list of Drug objects to dicts."""
    return [_drug_to_dict(d) for d in drugs]


def _make_ok(patient_id: str, mode: str, drugs: list[Drug]) -> dict:
    """Build a successful output state dict."""
    return {
        KEY_PATIENT_ID:  patient_id,
        KEY_MODE:        mode,
        KEY_MEDICATIONS: _drugs_to_dicts(drugs),
        KEY_STATUS:      "OK",
        KEY_SKIP_AUDIT:  _active_count(drugs) < 2,
        KEY_ERROR:       None,
    }


def _make_error(patient_id: str, mode: str, message: str) -> dict:
    """Build an error output state dict."""
    return {
        KEY_PATIENT_ID:  patient_id,
        KEY_MODE:        mode,
        KEY_MEDICATIONS: [],
        KEY_STATUS:      "ERROR",
        KEY_SKIP_AUDIT:  True,
        KEY_ERROR:       message,
    }


# ── Patient mode ───────────────────────────────────────────────────────────────

def _run_patient_mode(state: dict, store: PatientStore) -> dict:
    """Bulk-load FHIR medications, normalise brand names, persist to Redis."""
    patient_id: str  = state[KEY_PATIENT_ID]
    patient_json: dict = state.get(KEY_PATIENT_JSON, {})

    if not patient_json:
        return _make_error(patient_id, "patient", "patient_json is missing or empty")

    try:
        drugs = parse_fhir_patient(patient_json, synonyms_path=_SYNONYMS_PATH)
    except Exception as exc:
        return _make_error(patient_id, "patient", f"FHIR parse failed: {exc}")

    normalised = [d for d in drugs if d.is_normalised]

    # [REFS: memory/patient_store.py > save_medications]
    store.save_medications(patient_id, drugs)

    allergies = patient_json.get("allergies", [])
    if allergies:
        # [REFS: memory/patient_store.py > save_allergies]
        store.save_allergies(patient_id, allergies)

    store._audit(
        patient_id,
        "profile_builder:patient_mode",
        {
            "ts":            _now_iso(),
            "drug_count":    len(drugs),
            "active_count":  _active_count(drugs),
            "normalised":    [{"from": d.drug_name, "to": d.generic_name} for d in normalised],
            "allergy_count": len(allergies),
        },
    )

    result = _make_ok(patient_id, "patient", drugs)

    if result[KEY_SKIP_AUDIT]:
        result["skip_reason"] = (
            f"Only {_active_count(drugs)} active medication(s) — interaction check skipped."
        )

    return result


# ── Doctor mode ────────────────────────────────────────────────────────────────

def _run_doctor_mode(state: dict, store: PatientStore) -> dict:
    """Append a single new prescription to an existing patient profile."""
    patient_id: str      = state[KEY_PATIENT_ID]
    new_drug_input: dict = state.get(KEY_NEW_DRUG, {})

    if not new_drug_input:
        return _make_error(patient_id, "doctor", "new_drug is missing or empty")

    # [REFS: memory/patient_store.py > get_medications]
    existing_drugs = store.get_medications(patient_id)
    if not existing_drugs:
        store._audit(
            patient_id,
            "profile_builder:doctor_mode:no_profile",
            {"ts": _now_iso(), "attempted_drug": new_drug_input.get("drug_name")},
        )
        return {
            KEY_PATIENT_ID:   patient_id,
            KEY_MODE:         "doctor",
            KEY_MEDICATIONS:  [],
            KEY_STATUS:       "INCOMPLETE_PROFILE",
            KEY_SKIP_AUDIT:   True,
            KEY_PENDING_DRUG: None,
            KEY_ERROR: (
                f"No existing profile for patient '{patient_id}'. "
                "Run in patient mode first to build the base profile."
            ),
        }

    raw_name: str = new_drug_input.get("drug_name", "")
    generic_name  = normalise_drug(raw_name, _SYNONYMS_PATH)
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
    # [REFS: memory/patient_store.py > save_medications]
    store.save_medications(patient_id, updated_drugs)

    pending_drug_dict = _drug_to_dict(new_drug)
    pending_drug_dict["pending_confirmation"] = True

    store._audit(
        patient_id,
        "profile_builder:doctor_mode:drug_appended",
        {
            "ts":                 _now_iso(),
            "drug_name":          raw_name,
            "generic_name":       generic_name,
            "is_normalised":      is_normalised,
            "prescribing_doctor": new_drug.prescribing_doctor,
            "pending_confirmation": True,
            "total_medications":  len(updated_drugs),
        },
    )

    result = _make_ok(patient_id, "doctor", updated_drugs)
    result[KEY_PENDING_DRUG] = pending_drug_dict
    return result


# ── LangGraph node ─────────────────────────────────────────────────────────────

def run_agent(state: dict, store: PatientStore | None = None) -> dict:
    """Dispatch to patient or doctor mode; inject *store* for testing."""
    if store is None:
        store = PatientStore()

    mode       = state.get(KEY_MODE, "")
    patient_id = state.get(KEY_PATIENT_ID, "")

    if not patient_id:
        return _make_error("", mode, "patient_id is required in state")

    if mode == "patient":
        return _run_patient_mode(state, store)
    if mode == "doctor":
        return _run_doctor_mode(state, store)
    return _make_error(patient_id, mode, f"Unknown mode '{mode}'. Must be 'patient' or 'doctor'.")


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    project_root = Path(__file__).parent.parent
    patient_file = project_root / "data" / "mock_patients" / "patient_001.json"

    with open(patient_file, encoding="utf-8") as fh:
        patient_json = json.load(fh)

    patient_id = patient_json["id"]

    store = PatientStore()
    try:
        store._r.ping()
        print("\n[PASS] Redis connection\n")
    except Exception as exc:
        print(f"\n[FAIL] Redis connection — {exc}")
        sys.exit(1)

    store.delete_patient(patient_id)

    # ── Test 1: patient mode ───────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: patient mode — bulk load patient_001.json")
    print("=" * 60)

    result = run_agent(
        {KEY_MODE: "patient", KEY_PATIENT_ID: patient_id, KEY_PATIENT_JSON: patient_json},
        store=store,
    )

    print(f"  status       : {result[KEY_STATUS]}")
    print(f"  skip_audit   : {result[KEY_SKIP_AUDIT]}")
    print(f"  medications  : {len(result[KEY_MEDICATIONS])} loaded")
    for med in result[KEY_MEDICATIONS]:
        tag = f"  ← normalised from '{med['drug_name']}'" if med["is_normalised"] else ""
        print(f"    • {med['generic_name']} {med['dose']}{tag}")
    assert result[KEY_STATUS] == "OK"
    assert len(result[KEY_MEDICATIONS]) == 4
    assert result[KEY_SKIP_AUDIT] is False
    print("  [PASS]\n")

    # ── Test 2: doctor mode ────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2: doctor mode — append Nurofen (brand) for patient_001")
    print("=" * 60)

    result2 = run_agent(
        {
            KEY_MODE:       "doctor",
            KEY_PATIENT_ID: patient_id,
            KEY_NEW_DRUG: {
                "drug_name": "Nurofen",
                "dose": "200mg",
                "frequency": "as needed",
                "prescribing_doctor": "Dr. Testdoctor",
                "condition": "Headache",
                "prescription_date": "2024-03-01",
            },
        },
        store=store,
    )

    pending = result2.get(KEY_PENDING_DRUG, {})
    print(f"  status           : {result2[KEY_STATUS]}")
    print(f"  total medications: {len(result2[KEY_MEDICATIONS])}")
    print(f"  pending_drug     : {pending.get('generic_name')} (normalised={pending.get('is_normalised')})")
    print(f"  pending_confirm  : {pending.get('pending_confirmation')}")
    assert result2[KEY_STATUS] == "OK"
    assert pending["generic_name"] == "Ibuprofen"
    assert pending["is_normalised"] is True
    assert pending["pending_confirmation"] is True
    assert len(result2[KEY_MEDICATIONS]) == 5
    print("  [PASS]\n")

    # ── Test 3: doctor mode — no existing profile ──────────────────────────────
    print("=" * 60)
    print("TEST 3: doctor mode — unknown patient (no profile)")
    print("=" * 60)

    result3 = run_agent(
        {
            KEY_MODE:       "doctor",
            KEY_PATIENT_ID: "patient-UNKNOWN",
            KEY_NEW_DRUG: {
                "drug_name": "Aspirin", "dose": "75mg", "frequency": "once daily",
                "prescribing_doctor": "Dr. Nobody", "condition": "Prevention",
                "prescription_date": "2024-03-01",
            },
        },
        store=store,
    )

    print(f"  status     : {result3[KEY_STATUS]}")
    print(f"  skip_audit : {result3[KEY_SKIP_AUDIT]}")
    print(f"  error      : {result3[KEY_ERROR]}")
    assert result3[KEY_STATUS] == "INCOMPLETE_PROFILE"
    assert result3[KEY_SKIP_AUDIT] is True
    print("  [PASS]\n")

    # ── Test 4: audit log integrity ────────────────────────────────────────────
    print("=" * 60)
    print("TEST 4: audit log for patient_001")
    print("=" * 60)

    audit = store.get_audit_log(patient_id)
    for entry in audit:
        print(f"  [{entry['event']}]  {entry.get('detail', {})}")
    assert len(audit) >= 2
    print(f"\n  [PASS] {len(audit)} audit entries written\n")

    store.delete_patient(patient_id)
    print("All PatientProfileBuilderAgent tests passed.")
