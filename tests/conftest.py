"""
Shared pytest configuration and fixtures for the Polypharmacy Safety Agent test suite.

Inserts the project root on sys.path so every test file can import
project modules without installing the package.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.fhir_parser import Drug


# ── Shared test helpers ───────────────────────────────────────────────────────

def make_drug(name: str, generic: str | None = None, active: bool = True) -> Drug:
    """Build a minimal Drug dataclass for use in tests."""
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
    """In-memory substitute for PatientStore — no Redis required."""

    def __init__(self, allergies: list | None = None):
        self._meds: dict[str, list[Drug]] = {}
        self._allergies: dict[str, list] = {} if allergies is None else {"__default__": allergies}
        self._conflicts: dict[str, list] = {}
        self._audit_log: dict[str, list] = {}

    def save_medications(self, patient_id: str, drugs: list[Drug]) -> None:
        self._meds[patient_id] = list(drugs)

    def get_medications(self, patient_id: str) -> list[Drug]:
        return list(self._meds.get(patient_id, []))

    def save_allergies(self, patient_id: str, allergies: list) -> None:
        self._allergies[patient_id] = allergies

    def get_allergies(self, patient_id: str) -> list:
        # WHY: tests that initialise with a global allergy list use __default__ key
        return self._allergies.get(patient_id, self._allergies.get("__default__", []))

    def save_conflicts(self, patient_id: str, conflicts: list[dict]) -> None:
        self._conflicts[patient_id] = conflicts

    def get_conflicts(self, patient_id: str) -> list[dict]:
        return self._conflicts.get(patient_id, [])

    def _audit(self, patient_id: str, event: str, detail: Any = None) -> None:
        self._audit_log.setdefault(patient_id, []).append(
            {"event": event, "detail": detail}
        )

    def get_audit_log(self, patient_id: str) -> list[dict]:
        return self._audit_log.get(patient_id, [])

    def delete_patient(self, patient_id: str) -> None:
        for store in (self._meds, self._allergies, self._conflicts, self._audit_log):
            store.pop(patient_id, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis() -> FakeStore:
    """Fresh in-memory store — equivalent to an empty Redis instance."""
    return FakeStore()


@pytest.fixture
def sample_drug_profile() -> list[Drug]:
    """Two-drug base profile used across multiple test modules."""
    return [make_drug("Lisinopril"), make_drug("Metformin")]
