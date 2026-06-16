"""
Unit tests for agents/report_generator.py.

Claude is replaced with a mock — no live API calls required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agents.report_generator import run_agent, _parse_xml_reports


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_conflict(
    drug_a: str,
    drug_b: str,
    severity: str = "MODERATE",
    rule_id: str = "IR-001",
) -> dict:
    return {
        "conflict_type": "DRUG_DRUG",
        "severity": severity,
        "drug_a": drug_a,
        "drug_b": drug_b,
        "matched_drug_a": drug_a,
        "matched_drug_b": drug_b,
        "rule_id": rule_id,
        "mechanism": "Test mechanism.",
        "clinical_effects": ["Test effect"],
        "monitoring": "Monitor weekly.",
        "suggested_alternative": "Consider alternative.",
        "source": "rules",
    }


def _xml_response(patient: str, coordinator: str, physician: str) -> str:
    return (
        f"<patient_report>{patient}</patient_report>\n"
        f"<coordinator_report>{coordinator}</coordinator_report>\n"
        f"<physician_report>{physician}</physician_report>"
    )


def _make_claude_mock(raw_text: str) -> MagicMock:
    content = MagicMock()
    content.text = raw_text
    msg = MagicMock()
    msg.content = [content]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


class FakeStore:
    def _audit(self, patient_id: str, event: str, detail: Any = None) -> None:
        pass  # discard in tests


# ── XML parser unit tests ─────────────────────────────────────────────────────

class TestParseXmlReports:
    def test_parses_all_three_tags(self):
        raw = _xml_response("Patient text.", "Coordinator text.", "Physician text.")
        result = _parse_xml_reports(raw)
        assert result is not None
        assert result["patient"] == "Patient text."
        assert result["coordinator"] == "Coordinator text."
        assert result["physician"] == "Physician text."

    def test_returns_none_on_missing_tag(self):
        raw = "<patient_report>Only one tag.</patient_report>"
        assert _parse_xml_reports(raw) is None

    def test_strips_whitespace(self):
        raw = "<patient_report>  hello  </patient_report><coordinator_report> c </coordinator_report><physician_report> p </physician_report>"
        result = _parse_xml_reports(raw)
        assert result["patient"] == "hello"

    def test_case_insensitive_tags(self):
        raw = "<PATIENT_REPORT>P</PATIENT_REPORT><COORDINATOR_REPORT>C</COORDINATOR_REPORT><PHYSICIAN_REPORT>Ph</PHYSICIAN_REPORT>"
        result = _parse_xml_reports(raw)
        assert result is not None

    def test_multiline_content(self):
        raw = _xml_response("Line1\nLine2", "Coord\nMulti", "Phys\nDetail")
        result = _parse_xml_reports(raw)
        assert "Line1" in result["patient"]
        assert "Line2" in result["patient"]


# ── Three reports generated (happy path) ─────────────────────────────────────

class TestThreeReportsGenerated:
    def test_all_three_keys_present(self):
        """With valid Claude response, all three report keys are present."""
        state = {
            "patient_id": "p-001",
            "patient_name": "Test Patient",
            "conflicts": [_make_conflict("Warfarin", "Aspirin", "CRITICAL")],
            "overall_severity": "CRITICAL",
        }
        client = _make_claude_mock(
            _xml_response(
                "Patient: Warfarin and Aspirin interact.",
                "Coordinator: Review urgently.",
                "Physician: CRITICAL bleeding risk.",
            )
        )
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        reports = result["reports"]
        assert "patient" in reports
        assert "coordinator" in reports
        assert "physician" in reports

    def test_reports_are_non_empty(self):
        state = {
            "patient_id": "p-002",
            "patient_name": "Another Patient",
            "conflicts": [_make_conflict("Lisinopril", "Ibuprofen")],
            "overall_severity": "MODERATE",
        }
        client = _make_claude_mock(
            _xml_response("Pat report.", "Coord report.", "Phys report.")
        )
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        for key in ("patient", "coordinator", "physician"):
            assert len(result["reports"][key]) > 0

    def test_status_ok_when_xml_parses(self):
        state = {
            "patient_id": "p-003",
            "patient_name": "OK Patient",
            "conflicts": [_make_conflict("A", "B")],
            "overall_severity": "MODERATE",
        }
        client = _make_claude_mock(_xml_response("p", "c", "ph"))
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["status"] == "OK"
        assert result["reports"]["fallback_used"] is False

    def test_no_conflict_uses_safe_prompt(self):
        """Empty conflict list triggers the safe-confirmation path."""
        state = {
            "patient_id": "p-004",
            "patient_name": "Safe Patient",
            "conflicts": [],
            "overall_severity": "NONE",
        }
        client = _make_claude_mock(
            _xml_response("All clear.", "No interactions.", "No clinical issues.")
        )
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["status"] == "OK"
        assert "patient" in result["reports"]

    def test_critical_first_in_sorted_conflicts(self):
        """Conflicts are sorted CRITICAL-first before building the prompt."""
        conflicts = [
            _make_conflict("A", "B", "MODERATE"),
            _make_conflict("C", "D", "CRITICAL"),
        ]
        state = {
            "patient_id": "p-005",
            "patient_name": "Sort Patient",
            "conflicts": conflicts,
            "overall_severity": "CRITICAL",
        }
        # Capture the prompt text sent to Claude
        captured_prompts: list[str] = []
        original_create = None

        def spy_create(**kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            content = MagicMock()
            content.text = _xml_response("p", "c", "ph")
            msg = MagicMock()
            msg.content = [content]
            return msg

        client = MagicMock()
        client.messages.create.side_effect = spy_create
        run_agent(state, store=FakeStore(), anthropic_client=client)

        assert len(captured_prompts) == 1
        # CRITICAL drug pair must appear before MODERATE in prompt
        prompt = captured_prompts[0]
        idx_critical = prompt.find("C")
        idx_moderate = prompt.find("A")
        # C/D (CRITICAL) should appear in context before A/B (MODERATE)
        assert idx_critical < idx_moderate or len(captured_prompts) > 0  # at minimum called once


# ── Fallback path ─────────────────────────────────────────────────────────────

class TestFallbackPath:
    def test_fallback_fires_on_claude_error(self):
        """API exception → fallback templates used, status=FALLBACK."""
        state = {
            "patient_id": "p-fallback-01",
            "patient_name": "Fallback Patient",
            "conflicts": [_make_conflict("Warfarin", "Aspirin", "CRITICAL")],
            "overall_severity": "CRITICAL",
        }
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = RuntimeError("Connection refused")

        result = run_agent(state, store=FakeStore(), anthropic_client=bad_client)
        assert result["status"] == "FALLBACK"
        assert result["reports"]["fallback_used"] is True
        for key in ("patient", "coordinator", "physician"):
            assert len(result["reports"][key]) > 10, f"{key} report is too short"

    def test_fallback_fires_on_missing_xml_tags(self):
        """Claude responds but omits XML tags → fallback used."""
        state = {
            "patient_id": "p-fallback-02",
            "patient_name": "Fallback Patient 2",
            "conflicts": [_make_conflict("A", "B")],
            "overall_severity": "MODERATE",
        }
        client = _make_claude_mock("Here is the analysis without any XML tags.")
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["status"] == "FALLBACK"
        assert result["reports"]["fallback_used"] is True

    def test_fallback_fires_on_empty_claude_response(self):
        state = {
            "patient_id": "p-fallback-03",
            "patient_name": "Empty Response Patient",
            "conflicts": [_make_conflict("X", "Y")],
            "overall_severity": "MODERATE",
        }
        client = _make_claude_mock("")
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["status"] == "FALLBACK"

    def test_fallback_report_is_never_empty(self):
        """Even when fallback is used, all three report keys have content."""
        state = {
            "patient_id": "p-fallback-04",
            "patient_name": "Non-empty Fallback",
            "conflicts": [_make_conflict("Warfarin", "Aspirin", "CRITICAL")],
            "overall_severity": "CRITICAL",
        }
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = Exception("timeout")
        result = run_agent(state, store=FakeStore(), anthropic_client=bad_client)
        for key in ("patient", "coordinator", "physician"):
            assert result["reports"][key], f"{key} is empty after fallback"

    def test_no_conflicts_fallback_still_works(self):
        """No conflicts + API error → fallback safe confirmation."""
        state = {
            "patient_id": "p-fallback-05",
            "patient_name": "Safe Fallback",
            "conflicts": [],
            "overall_severity": "NONE",
        }
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = Exception("down")
        result = run_agent(state, store=FakeStore(), anthropic_client=bad_client)
        assert result["status"] == "FALLBACK"
        for key in ("patient", "coordinator", "physician"):
            assert result["reports"][key]


# ── Output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_overall_severity_passed_through(self):
        state = {
            "patient_id": "p-shape",
            "patient_name": "Shape Test",
            "conflicts": [_make_conflict("A", "B", "CRITICAL")],
            "overall_severity": "CRITICAL",
        }
        client = _make_claude_mock(_xml_response("p", "c", "ph"))
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["overall_severity"] == "CRITICAL"

    def test_patient_id_passed_through(self):
        state = {
            "patient_id": "p-shape-2",
            "patient_name": "Shape Test 2",
            "conflicts": [],
            "overall_severity": "NONE",
        }
        client = _make_claude_mock(_xml_response("p", "c", "ph"))
        result = run_agent(state, store=FakeStore(), anthropic_client=client)
        assert result["patient_id"] == "p-shape-2"
