"""
Unit tests for agents/interaction_auditor.py.

All external dependencies (Redis, ChromaDB, Claude) are replaced with
lightweight fakes or unittest.mock objects — no live services required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.fhir_parser import Drug
from memory.patient_store import _drug_to_dict

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_drug(
    name: str,
    generic: str | None = None,
    active: bool = True,
) -> Drug:
    return Drug(
        drug_name=name,
        generic_name=generic or name,
        dose="10mg",
        frequency="once daily",
        prescribing_doctor="Dr. Test",
        condition="Test",
        prescription_date="2024-01-01",
        active_status=active,
        is_normalised=False,
    )


def _drugs_as_dicts(drugs: list[Drug]) -> list[dict]:
    return [_drug_to_dict(d) for d in drugs]


# ── Fake store ────────────────────────────────────────────────────────────────

class FakeStore:
    def __init__(self, allergies: list | None = None):
        self._allergies = allergies or []
        self._saved_conflicts: list[dict] = []

    def get_allergies(self, patient_id: str) -> list:
        return self._allergies

    def save_conflicts(self, patient_id: str, conflicts: list[dict]) -> None:
        self._saved_conflicts = conflicts

    def get_conflicts(self, patient_id: str) -> list[dict]:
        return self._saved_conflicts


# ── Fake knowledge store ──────────────────────────────────────────────────────

class FakeKnowledgeStore:
    def __init__(self, chunks: list[str] | None = None):
        self._chunks = chunks or ["No specific interaction data found."]

    def query(self, text: str, n_results: int = 3) -> list[str]:
        return self._chunks[:n_results]


# ── Mock Claude client ────────────────────────────────────────────────────────

def _make_claude_mock(severity: str = "NONE", mechanism: str = "No interaction.") -> MagicMock:
    """Return a mock anthropic.Anthropic client whose messages.create() returns fixed JSON."""
    response_text = json.dumps({"severity": severity, "mechanism": mechanism})
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


from agents.interaction_auditor import run_agent, _check_allergies, _merge_conflicts, _overall_severity  # noqa: E402


# ── Allergy check ─────────────────────────────────────────────────────────────

class TestAllergyCheck:
    def test_detects_allergy_match(self):
        drugs = [_make_drug("Penicillin")]
        allergies = [{"substance": "Penicillin", "reaction": "anaphylaxis"}]
        conflicts = _check_allergies(drugs, allergies)
        assert len(conflicts) == 1
        assert conflicts[0]["severity"] == "CRITICAL"
        assert conflicts[0]["conflict_type"] == "ALLERGY"

    def test_no_match_returns_empty(self):
        drugs = [_make_drug("Metformin")]
        allergies = [{"substance": "Penicillin", "reaction": "rash"}]
        conflicts = _check_allergies(drugs, allergies)
        assert conflicts == []

    def test_case_insensitive_match(self):
        drugs = [_make_drug("ASPIRIN")]
        allergies = ["aspirin"]
        conflicts = _check_allergies(drugs, allergies)
        assert len(conflicts) == 1

    def test_inactive_drug_ignored(self):
        drugs = [_make_drug("Penicillin", active=False)]
        allergies = [{"substance": "Penicillin", "reaction": "rash"}]
        conflicts = _check_allergies(drugs, allergies)
        assert conflicts == []

    def test_string_allergy_format(self):
        drugs = [_make_drug("Ibuprofen")]
        allergies = ["ibuprofen"]
        conflicts = _check_allergies(drugs, allergies)
        assert len(conflicts) == 1


# ── Rule layer ────────────────────────────────────────────────────────────────

class TestRuleLayer:
    def test_warfarin_aspirin_detected_as_critical(self):
        """Warfarin + Aspirin is in interaction_rules.json as CRITICAL (IR-001)."""
        drugs = [_make_drug("Warfarin"), _make_drug("Aspirin")]
        state = {
            "patient_id": "test-001",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        store = FakeStore()
        ks = FakeKnowledgeStore()
        client = _make_claude_mock("NONE")

        result = run_agent(state, store=store, knowledge_store=ks, anthropic_client=client)

        assert result["overall_severity"] == "CRITICAL"
        critical = [c for c in result["conflicts"] if c["severity"] == "CRITICAL"]
        assert len(critical) >= 1
        pair_names = {c["matched_drug_a"].lower() for c in critical} | \
                     {c["matched_drug_b"].lower() for c in critical}
        assert "warfarin" in pair_names
        assert "aspirin" in pair_names

    def test_lisinopril_ibuprofen_moderate(self):
        """Lisinopril + Ibuprofen is MODERATE (IR-003) via rule engine."""
        drugs = [_make_drug("Lisinopril"), _make_drug("Ibuprofen")]
        state = {
            "patient_id": "test-002",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=_make_claude_mock("NONE"),
        )
        pairs = {
            frozenset({c["matched_drug_a"].lower(), c["matched_drug_b"].lower()})
            for c in result["conflicts"]
        }
        assert frozenset({"lisinopril", "ibuprofen"}) in pairs

    def test_skip_audit_returns_empty(self):
        """skip_audit=True bypasses all checks and returns status=SKIPPED."""
        drugs = [_make_drug("Warfarin"), _make_drug("Aspirin")]
        state = {
            "patient_id": "test-003",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": True,
        }
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=_make_claude_mock("NONE"),
        )
        assert result["status"] == "SKIPPED"
        assert result["conflicts"] == []
        assert result["overall_severity"] == "NONE"

    def test_no_medications_returns_error(self):
        state = {"patient_id": "test-004", "medications": [], "skip_audit": False}
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=_make_claude_mock("NONE"),
        )
        assert result["status"] == "ERROR"


# ── RAG / Claude layer ────────────────────────────────────────────────────────

class TestClaudeLayer:
    def test_claude_moderate_added_for_uncovered_pair(self):
        """Claude flags a pair that the rule engine doesn't cover."""
        # Two drugs with no known rule between them
        drugs = [_make_drug("DrugX"), _make_drug("DrugY")]
        state = {
            "patient_id": "test-005",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        client = _make_claude_mock("MODERATE", "Additive toxicity risk.")
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=client,
        )
        assert result["overall_severity"] == "MODERATE"
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["source"] == "claude"

    def test_claude_none_adds_no_conflict(self):
        drugs = [_make_drug("DrugA"), _make_drug("DrugB")]
        state = {
            "patient_id": "test-006",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        client = _make_claude_mock("NONE")
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=client,
        )
        assert result["conflicts"] == []
        assert result["overall_severity"] == "NONE"

    def test_claude_api_failure_treated_as_none(self):
        """If Claude raises, the pair is treated as NONE (no false positive)."""
        drugs = [_make_drug("DrugC"), _make_drug("DrugD")]
        state = {
            "patient_id": "test-007",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = RuntimeError("API timeout")

        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=bad_client,
        )
        # Should not raise; conflicts from Claude layer are zero
        claude_conflicts = [c for c in result["conflicts"] if c.get("source") == "claude"]
        assert claude_conflicts == []

    def test_claude_json_parse_failure_treated_as_none(self):
        """Malformed Claude response is ignored (NONE) rather than raising."""
        drugs = [_make_drug("DrugE"), _make_drug("DrugF")]
        state = {
            "patient_id": "test-008",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        bad_content = MagicMock()
        bad_content.text = "not valid JSON at all"
        bad_msg = MagicMock()
        bad_msg.content = [bad_content]
        bad_client = MagicMock()
        bad_client.messages.create.return_value = bad_msg

        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=bad_client,
        )
        claude_conflicts = [c for c in result["conflicts"] if c.get("source") == "claude"]
        assert claude_conflicts == []

    def test_covered_pair_not_sent_to_claude(self):
        """Pairs already flagged by rule engine must not be re-checked by Claude."""
        drugs = [_make_drug("Warfarin"), _make_drug("Aspirin")]
        state = {
            "patient_id": "test-009",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        client = _make_claude_mock("MODERATE", "Should not be returned.")
        result = run_agent(
            state,
            store=FakeStore(),
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=client,
        )
        # All conflicts should be from rules, none from claude for this pair
        for c in result["conflicts"]:
            if frozenset({c.get("matched_drug_a","").lower(), c.get("matched_drug_b","").lower()}) \
                    == frozenset({"warfarin", "aspirin"}):
                assert c.get("source") != "claude"


# ── Merge & dedup ─────────────────────────────────────────────────────────────

class TestMergeConflicts:
    def test_critical_wins_over_moderate_same_pair(self):
        critical = [{"drug_a": "Warfarin", "drug_b": "Aspirin",
                     "matched_drug_a": "Warfarin", "matched_drug_b": "Aspirin",
                     "severity": "CRITICAL", "source": "rules"}]
        moderate = [{"drug_a": "warfarin", "drug_b": "aspirin",
                     "matched_drug_a": "warfarin", "matched_drug_b": "aspirin",
                     "severity": "MODERATE", "source": "claude"}]
        merged = _merge_conflicts([], critical, moderate)
        assert len(merged) == 1
        assert merged[0]["severity"] == "CRITICAL"

    def test_deduplication_different_orderings(self):
        c1 = {"drug_a": "Warfarin", "drug_b": "Aspirin",
              "matched_drug_a": "Warfarin", "matched_drug_b": "Aspirin",
              "severity": "MODERATE", "source": "rules"}
        c2 = {"drug_a": "Aspirin", "drug_b": "Warfarin",
              "matched_drug_a": "Aspirin", "matched_drug_b": "Warfarin",
              "severity": "MODERATE", "source": "claude"}
        merged = _merge_conflicts([], [c1], [c2])
        assert len(merged) == 1

    def test_different_pairs_both_kept(self):
        c1 = {"drug_a": "A", "drug_b": "B", "matched_drug_a": "A",
              "matched_drug_b": "B", "severity": "MODERATE", "source": "rules"}
        c2 = {"drug_a": "C", "drug_b": "D", "matched_drug_a": "C",
              "matched_drug_b": "D", "severity": "MODERATE", "source": "rules"}
        merged = _merge_conflicts([], [c1, c2], [])
        assert len(merged) == 2


# ── Overall severity ──────────────────────────────────────────────────────────

class TestOverallSeverity:
    def test_critical_dominates(self):
        conflicts = [{"severity": "CRITICAL"}, {"severity": "MODERATE"}]
        assert _overall_severity(conflicts) == "CRITICAL"

    def test_moderate_when_no_critical(self):
        conflicts = [{"severity": "MODERATE"}, {"severity": "NONE"}]
        assert _overall_severity(conflicts) == "MODERATE"

    def test_none_when_empty(self):
        assert _overall_severity([]) == "NONE"


# ── Allergy + rule layer integration ─────────────────────────────────────────

class TestAllergyAndRuleIntegration:
    def test_allergy_conflict_is_critical_overall(self):
        """An allergy hit overrides any MODERATE rule result → CRITICAL."""
        drugs = [_make_drug("Lisinopril"), _make_drug("Ibuprofen")]
        state = {
            "patient_id": "allergy-test",
            "medications": _drugs_as_dicts(drugs),
            "skip_audit": False,
        }
        store = FakeStore(allergies=[{"substance": "Ibuprofen", "reaction": "anaphylaxis"}])
        result = run_agent(
            state,
            store=store,
            knowledge_store=FakeKnowledgeStore(),
            anthropic_client=_make_claude_mock("NONE"),
        )
        assert result["overall_severity"] == "CRITICAL"
        allergy_conflicts = [c for c in result["conflicts"] if c.get("conflict_type") == "ALLERGY"]
        assert len(allergy_conflicts) >= 1
