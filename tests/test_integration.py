"""
Integration test — full graph run on patient_001.

Uses REAL Redis and REAL ChromaDB (no Claude mock).
Claude API is mocked to avoid cost/latency in CI and to keep the test
deterministic, but every other component (Redis, ChromaDB, rule engine,
FHIR parser, LangGraph) runs with its production implementation.

Mark: pytest -m integration
Run with live services configured in .env.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Skip if Redis unavailable ─────────────────────────────────────────────────

def _redis_available() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
        import redis
        url = os.environ.get("REDIS_URL") or os.environ.get("UPSTASH_REDIS_URL")
        if url:
            r = redis.from_url(url, socket_connect_timeout=3, decode_responses=True)
        else:
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_connect_timeout=3,
            )
        r.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.integration

requires_redis = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available — set REDIS_URL or REDIS_HOST in .env",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def patient_001() -> dict:
    p = _PROJECT_ROOT / "data" / "mock_patients" / "patient_001.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def patient_store():
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    from memory.patient_store import PatientStore
    return PatientStore()


@pytest.fixture(scope="module")
def knowledge_store():
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    from memory.knowledge_store import KnowledgeStore
    ks = KnowledgeStore()
    ks.init()
    return ks


@pytest.fixture(scope="module")
def audit_logger():
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    from audit.audit_logger import AuditLogger
    return AuditLogger()


def _mock_claude_xml() -> str:
    return (
        "<patient_report>Your medications have been reviewed. "
        "Please follow up with your doctors.</patient_report>\n"
        "<coordinator_report>Medication review complete. "
        "Interactions identified — see physician report.</coordinator_report>\n"
        "<physician_report>Clinical review: interactions detected. "
        "Monitor patient closely.</physician_report>"
    )


def _make_mock_client() -> MagicMock:
    """Mock anthropic client returning valid XML for report generator
    and valid JSON for interaction auditor."""
    call_count = [0]

    def _create(**kwargs):
        call_count[0] += 1
        content = MagicMock()
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        # Auditor calls use max_tokens=512 and expect JSON
        if kwargs.get("max_tokens", 0) <= 512:
            content.text = json.dumps({"severity": "NONE", "mechanism": "No interaction."})
        else:
            content.text = _mock_claude_xml()
        msg = MagicMock()
        msg.content = [content]
        return msg

    client = MagicMock()
    client.messages.create.side_effect = _create
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────

@requires_redis
class TestFullGraphPatient001:
    """End-to-end integration tests using real Redis + ChromaDB."""

    def test_pipeline_completes_without_error(
        self, patient_001, patient_store, knowledge_store, audit_logger
    ):
        """run_patient_flow returns a final state with no error."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"

        # Clean state
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        sg._anthropic_client = mock_client

        try:
            from graph.safety_graph import run_patient_flow
            final = run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                patient_name="Arjun Sharma",
                thread_id=thread_id,
                human_decision="approve",
            )
        finally:
            sg._anthropic_client = orig_client
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

        assert final.get("error") is None, f"Pipeline error: {final.get('error')}"

    def test_medications_persisted_in_redis(
        self, patient_001, patient_store, knowledge_store, audit_logger
    ):
        """After running, patient's medications are saved in Redis."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        sg._anthropic_client = mock_client

        try:
            from graph.safety_graph import run_patient_flow
            run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                thread_id=thread_id,
                human_decision="approve",
            )
            meds = patient_store.get_medications(patient_id)
            assert len(meds) == 4
        finally:
            sg._anthropic_client = orig_client
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

    def test_conflicts_detected_for_patient_001(
        self, patient_001, patient_store, knowledge_store, audit_logger
    ):
        """patient_001 has known MODERATE interactions — at least one must be detected."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        sg._anthropic_client = mock_client

        try:
            from graph.safety_graph import run_patient_flow
            final = run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                thread_id=thread_id,
                human_decision="approve",
            )
            assert len(final.get("conflicts", [])) >= 1
            assert final.get("overall_severity") in ("MODERATE", "CRITICAL")
        finally:
            sg._anthropic_client = orig_client
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

    def test_reports_all_three_keys_present(
        self, patient_001, patient_store, knowledge_store, audit_logger
    ):
        """Final state must contain patient, coordinator, physician reports."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        sg._anthropic_client = mock_client

        try:
            from graph.safety_graph import run_patient_flow
            final = run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                thread_id=thread_id,
                human_decision="approve",
            )
            reports = final.get("reports", {})
            for key in ("patient", "coordinator", "physician"):
                assert key in reports, f"Missing report key: {key}"
                assert len(reports[key]) > 10, f"{key} report is too short"
        finally:
            sg._anthropic_client = orig_client
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

    def test_audit_trail_written_to_sqlite(
        self, patient_001, patient_store, knowledge_store, audit_logger
    ):
        """SQLite audit log must have entries for patient_001 after pipeline run."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        orig_audit = sg._audit_logger
        sg._anthropic_client = mock_client
        sg._audit_logger = audit_logger

        try:
            from graph.safety_graph import run_patient_flow
            run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                thread_id=thread_id,
                human_decision="approve",
            )
            trail = audit_logger.get_audit_trail(patient_id)
            assert len(trail) >= 2, f"Expected ≥2 audit entries, got {len(trail)}"
        finally:
            sg._anthropic_client = orig_client
            sg._audit_logger = orig_audit
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

    def test_severity_state_is_valid_value(
        self, patient_001, patient_store, audit_logger
    ):
        """overall_severity must be one of CRITICAL, MODERATE, NONE."""
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")

        patient_id = patient_001["id"]
        thread_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        patient_store.delete_patient(patient_id)
        audit_logger.clear_patient(patient_id)

        mock_client = _make_mock_client()

        import graph.safety_graph as sg
        orig_client = sg._anthropic_client
        sg._anthropic_client = mock_client

        try:
            from graph.safety_graph import run_patient_flow
            final = run_patient_flow(
                patient_id=patient_id,
                patient_json=patient_001,
                thread_id=thread_id,
                human_decision="approve",
            )
            assert final.get("overall_severity") in ("CRITICAL", "MODERATE", "NONE")
        finally:
            sg._anthropic_client = orig_client
            patient_store.delete_patient(patient_id)
            audit_logger.clear_patient(patient_id)

    def test_knowledge_store_queried(self, knowledge_store):
        """Sanity-check: knowledge store returns chunks for a drug pair query."""
        chunks = knowledge_store.query("Warfarin Aspirin interaction", n_results=3)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_rule_engine_catches_lisinopril_ibuprofen(self):
        """Rule engine alone detects the Lisinopril+Ibuprofen MODERATE interaction."""
        from tools.fhir_parser import Drug
        from tools.rule_engine import check_pairs_all_severity

        drugs = [
            Drug("Lisinopril", "Lisinopril", "10mg", "od", "Dr A", "HTN", "2024-01-01", True, False),
            Drug("Ibuprofen", "Ibuprofen", "400mg", "tid", "Dr B", "Pain", "2024-01-01", True, False),
        ]
        conflicts = check_pairs_all_severity(drugs)
        pairs = [
            frozenset({c["matched_drug_a"].lower(), c["matched_drug_b"].lower()})
            for c in conflicts
        ]
        assert frozenset({"lisinopril", "ibuprofen"}) in pairs
