"""
Unit tests for agents/profile_builder.py.

All Redis I/O is mocked via a fake PatientStore — no live Redis required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.fhir_parser import Drug
from memory.patient_store import _drug_to_dict, _dict_to_drug

# ── Fixtures ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_P001 = _PROJECT_ROOT / "data" / "mock_patients" / "patient_001.json"
_P006 = _PROJECT_ROOT / "data" / "mock_patients" / "patient_006.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _make_drug(name: str, generic: str | None = None, active: bool = True) -> Drug:
    return Drug(
        drug_name=name,
        generic_name=generic or name,
        dose="10mg",
        frequency="once daily",
        prescribing_doctor="Dr. Test",
        condition="Test condition",
        prescription_date="2024-01-01",
        active_status=active,
        is_normalised=(generic is not None and generic.lower() != name.lower()),
    )


class FakeStore:
    """Minimal in-memory substitute for PatientStore."""

    def __init__(self):
        self._meds: dict[str, list[Drug]] = {}
        self._allergies: dict[str, list] = {}
        self._audit_log: dict[str, list] = {}

    def save_medications(self, patient_id: str, drugs: list[Drug]) -> None:
        self._meds[patient_id] = list(drugs)

    def get_medications(self, patient_id: str) -> list[Drug]:
        return list(self._meds.get(patient_id, []))

    def save_allergies(self, patient_id: str, allergies: list) -> None:
        self._allergies[patient_id] = allergies

    def get_allergies(self, patient_id: str) -> list:
        return self._allergies.get(patient_id, [])

    def _audit(self, patient_id: str, event: str, detail: Any = None) -> None:
        self._audit_log.setdefault(patient_id, []).append(
            {"event": event, "detail": detail}
        )

    def get_audit_log(self, patient_id: str) -> list[dict]:
        return self._audit_log.get(patient_id, [])

    def delete_patient(self, patient_id: str) -> None:
        for store in (self._meds, self._allergies, self._audit_log):
            store.pop(patient_id, None)


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


# ── Import agent after fixtures so patching is applied cleanly ────────────────

from agents.profile_builder import run_agent  # noqa: E402


# ── Patient mode tests ────────────────────────────────────────────────────────

class TestPatientMode:
    def test_patient_mode_ok(self, store):
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

    def test_patient_mode_persists_to_store(self, store):
        """Medications are saved to the fake store."""
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": _load(_P001),
        }
        run_agent(state, store=store)
        saved = store.get_medications("patient-001")
        assert len(saved) == 4

    def test_patient_mode_allergies_saved(self, store):
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

    def test_skip_audit_single_active_drug(self, store):
        """Patient with one active drug has skip_audit=True."""
        patient_json = _load(_P001)
        # Deactivate all but first drug
        for med in patient_json["medications"][1:]:
            med["active_status"] = False

        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": patient_json,
        }
        result = run_agent(state, store=store)
        assert result["skip_audit"] is True

    def test_missing_patient_json(self, store):
        """Empty patient_json returns ERROR status."""
        state = {
            "mode": "patient",
            "patient_id": "patient-001",
            "patient_json": {},
        }
        result = run_agent(state, store=store)
        assert result["status"] == "ERROR"
        assert result["skip_audit"] is True

    def test_brand_normalisation_patient006(self, store):
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

    def test_audit_entry_written(self, store):
        """At least one audit entry is appended after patient-mode run."""
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
            _make_drug("Lisinopril"),
            _make_drug("Metformin"),
        ])

    def test_doctor_mode_appends_drug(self, store):
        """Doctor mode appends a new drug to an existing profile."""
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
        assert len(result["medications"]) == 3  # 2 original + 1 new

    def test_doctor_mode_pending_confirmation(self, store):
        """New drug dict carries pending_confirmation=True."""
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

    def test_doctor_mode_normalises_brand_name(self, store):
        """Brand name (Nurofen → Ibuprofen) is normalised in doctor mode."""
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

    def test_doctor_mode_incomplete_profile(self, store):
        """No existing profile → status INCOMPLETE_PROFILE, skip_audit True."""
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

    def test_doctor_mode_skip_audit_after_append(self, store):
        """After appending a drug to a 2-drug profile, skip_audit is False (≥2 active)."""
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
    def test_unknown_mode_returns_error(self, store):
        result = run_agent(
            {"mode": "radiologist", "patient_id": "p-999"},
            store=store,
        )
        assert result["status"] == "ERROR"
        assert "mode" in result["error"].lower() or "unknown" in result["error"].lower()

    def test_missing_patient_id(self, store):
        result = run_agent({"mode": "patient", "patient_json": {}}, store=store)
        assert result["status"] == "ERROR"

    def test_all_inactive_drugs_skip_audit(self, store):
        """Profile with all inactive drugs has skip_audit=True."""
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
