# Tests: agents/profile_builder.py — patient mode, doctor mode, edge cases
# Coverage: brand normalisation, Redis persistence, skip_audit logic, mode routing

"""
Unit tests for agents/profile_builder.py.

All Redis I/O is mocked via a FakeStore — no live Redis required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FakeStore, make_drug
from memory.patient_store import _drug_to_dict, _dict_to_drug

# ── Fixtures ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_P001 = _PROJECT_ROOT / "data" / "mock_patients" / "patient_001.json"
_P006 = _PROJECT_ROOT / "data" / "mock_patients" / "patient_006.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


# ── Import agent after fixtures so patching is applied cleanly ────────────────

from agents.profile_builder import run_agent  # noqa: E402


# ── Patient mode tests ────────────────────────────────────────────────────────

class TestPatientMode:
    def test_patient_mode_loads_four_drugs_and_returns_ok(self, store):
        """Bulk-load patient_001 — 4 drugs, status OK, skip_audit False."""
        patient_json = _load(_P001)
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": patient_json,
        }

        result = run_agent(state, store=store)

        assert result["status"] == "OK"
        assert result["skip_audit"] is False
        assert len(result["medications"]) == 4
        assert result["error"] is None

    def test_patient_mode_persists_medications_to_store(self, store):
        """Medications are saved to the fake store after a successful run."""
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": _load(_P001),
        }
        run_agent(state, store=store)
        saved = store.get_medications("patient-001")
        assert len(saved) == 4

    def test_patient_mode_writes_allergies_to_store(self, store):
        """Allergies list from FHIR dict is written to the store."""
        patient_json = _load(_P001)
        patient_json["allergies"] = [{"substance": "Penicillin", "reaction": "rash"}]
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": patient_json,
        }
        run_agent(state, store=store)
        allergies = store.get_allergies("patient-001")
        assert len(allergies) == 1
        assert allergies[0]["substance"] == "Penicillin"

    def test_single_active_drug_sets_skip_audit_true(self, store):
        """Patient with one active drug has skip_audit=True — no pair to check."""
        patient_json = _load(_P001)
        for med in patient_json["medications"][1:]:
            med["active_status"] = False

        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": patient_json,
        }
        result = run_agent(state, store=store)
        assert result["skip_audit"] is True

    def test_empty_patient_json_returns_error_status(self, store):
        """Empty patient_json returns ERROR status and skip_audit=True."""
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": {},
        }
        result = run_agent(state, store=store)
        assert result["status"] == "ERROR"
        assert result["skip_audit"] is True

    def test_patient006_brufen_normalised_to_ibuprofen(self, store):
        """patient_006 uses 'Brufen' which must be normalised to 'Ibuprofen'."""
        patient_json = _load(_P006)
        state = {
            "mode": "patient",
            "patient_id": "patient-006",
            "patient_json": patient_json,
        }
        result = run_agent(state, store=store)

        assert result["status"] == "OK"
        generics = [m["generic_name"] for m in result["medications"]]
        assert "Ibuprofen" in generics, f"Expected Ibuprofen in {generics}"

        normalised = [m for m in result["medications"] if m["is_normalised"]]
        assert len(normalised) >= 1
        assert any(m["drug_name"] == "Brufen" and m["generic_name"] == "Ibuprofen"
                   for m in result["medications"])

    def test_patient_mode_appends_at_least_one_audit_entry(self, store):
        """At least one audit entry is appended to the store after a patient-mode run."""
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": _load(_P001),
        }
        run_agent(state, store=store)
        entries = store.get_audit_log("patient-001")
        assert len(entries) >= 1


# ── Doctor mode tests ─────────────────────────────────────────────────────────

class TestDoctorMode:
    def _seed_profile(self, store: FakeStore, patient_id: str = "patient-001") -> None:
        """Pre-populate two active drugs so doctor mode has a base profile."""
        store.save_medications(patient_id, [
            make_drug("Lisinopril"),
            make_drug("Metformin"),
        ])

    def test_doctor_mode_appends_new_drug_to_existing_profile(self, store):
        """Doctor mode appends a new drug so the medication count increases by one."""
        self._seed_profile(store)
        state = {
            "mode": "doctor",
            "patient_id": "patient-001",
            "new_drug": {
                "drug_name": "Aspirin",
                "dose": "75mg",
                "frequency": "once daily",
                "prescribing_doctor": "Dr. Heart",
                "condition": "CVD prevention",
                "prescription_date": "2024-06-01",
            },
        }
        result = run_agent(state, store=store)

        assert result["status"] == "OK"
        assert len(result["medications"]) == 3

    def test_doctor_mode_sets_pending_confirmation_on_new_drug(self, store):
        """New drug dict carries pending_confirmation=True until the doctor confirms."""
        self._seed_profile(store)
        state = {
            "mode": "doctor",
            "patient_id": "patient-001",
            "new_drug": {
                "drug_name": "Aspirin",
                "dose": "75mg",
                "frequency": "once daily",
                "prescribing_doctor": "Dr. Heart",
                "condition": "CVD prevention",
                "prescription_date": "2024-06-01",
            },
        }
        result = run_agent(state, store=store)
        pending = result.get("pending_drug", {})
        assert pending.get("pending_confirmation") is True

    def test_doctor_mode_normalises_brand_name_to_generic(self, store):
        """Brand name Nurofen is normalised to Ibuprofen in doctor mode."""
        self._seed_profile(store)
        state = {
            "mode": "doctor",
            "patient_id": "patient-001",
            "new_drug": {
                "drug_name": "Nurofen",
                "dose": "200mg",
                "frequency": "as needed",
                "prescribing_doctor": "Dr. Test",
                "condition": "Pain",
                "prescription_date": "2024-01-01",
            },
        }
        result = run_agent(state, store=store)
        pending = result.get("pending_drug", {})
        assert pending["generic_name"] == "Ibuprofen"
        assert pending["is_normalised"] is True

    def test_doctor_mode_unknown_patient_returns_incomplete_profile(self, store):
        """No existing profile → status INCOMPLETE_PROFILE, skip_audit True, empty medications."""
        state = {
            "mode": "doctor",
            "patient_id": "patient-UNKNOWN",
            "new_drug": {
                "drug_name": "Aspirin",
                "dose": "75mg",
                "frequency": "once daily",
                "prescribing_doctor": "Dr. Nobody",
                "condition": "Prevention",
                "prescription_date": "2024-01-01",
            },
        }
        result = run_agent(state, store=store)
        assert result["status"] == "INCOMPLETE_PROFILE"
        assert result["skip_audit"] is True
        assert result["medications"] == []

    def test_doctor_mode_two_drug_profile_has_skip_audit_false(self, store):
        """After appending to a 2-drug profile the total is ≥2 active, so skip_audit=False."""
        self._seed_profile(store)
        state = {
            "mode": "doctor",
            "patient_id": "patient-001",
            "new_drug": {
                "drug_name": "Aspirin",
                "dose": "75mg",
                "frequency": "once daily",
                "prescribing_doctor": "Dr. Heart",
                "condition": "CVD prevention",
                "prescription_date": "2024-06-01",
            },
        }
        result = run_agent(state, store=store)
        assert result["skip_audit"] is False


# ── Edge-case tests ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unknown_mode_returns_error_status(self, store):
        """An unrecognised mode string returns ERROR without raising."""
        result = run_agent(
            {"mode": "radiologist", "patient_id": "p-999"},
            store=store,
        )
        assert result["status"] == "ERROR"
        assert "mode" in result["error"].lower() or "unknown" in result["error"].lower()

    def test_missing_patient_id_returns_error(self, store):
        """State dict with no patient_id returns ERROR status."""
        result = run_agent({"mode": "patient", "patient_json": {}}, store=store)
        assert result["status"] == "ERROR"

    def test_all_inactive_drugs_sets_skip_audit_true(self, store):
        """Profile with all inactive drugs has skip_audit=True — nothing to check."""
        patient_json = _load(_P001)
        for med in patient_json["medications"]:
            med["active_status"] = False
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": patient_json,
        }
        result = run_agent(state, store=store)
        assert result["skip_audit"] is True
