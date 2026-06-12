# Agent: ConflictReportGeneratorAgent
# Role:  Generate patient, coordinator, and physician reports from detected conflicts.
# Input: patient_id, conflicts, overall_severity, patient_name
# Output: patient_id, reports, overall_severity, status, error

"""
Agent 3 — ConflictReportGeneratorAgent

Generates three tailored reports in a single LLM call, delimited by XML tags.
Falls back to pre-templated reports on any parse or API failure.

LangGraph integration::

    from agents.report_generator import run_agent as generate_reports
    graph.add_node("report_generator", generate_reports)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_config import get_llm
from langchain_core.messages import HumanMessage

from tools.report_formatter import (
    build_patient_prompt,
    build_coordinator_prompt,
    build_physician_prompt,
    build_fallback_report,
)
from memory.patient_store import PatientStore

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_RETRIES       = 3
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_NONE     = "NONE"

KEY_PATIENT_ID       = "patient_id"
KEY_CONFLICTS        = "conflicts"
KEY_OVERALL_SEVERITY = "overall_severity"
KEY_STATUS           = "status"
KEY_ERROR            = "error"
KEY_REPORTS          = "reports"
KEY_PATIENT_NAME     = "patient_name"

# ── XML tag patterns ───────────────────────────────────────────────────────────

_TAG_RE = {
    "patient": re.compile(
        r"<patient_report>(.*?)</patient_report>", re.DOTALL | re.IGNORECASE
    ),
    "coordinator": re.compile(
        r"<coordinator_report>(.*?)</coordinator_report>", re.DOTALL | re.IGNORECASE
    ),
    "physician": re.compile(
        r"<physician_report>(.*?)</physician_report>", re.DOTALL | re.IGNORECASE
    ),
}

# ── Prompts ────────────────────────────────────────────────────────────────────

REPORT_COMBINED_WRAPPER = """\
You must produce THREE separate reports for the same set of drug interaction findings.
Each report is for a different audience. Write all three in a single response.

Wrap each report in the exact XML tags shown — no extra text outside the tags:

<patient_report>
[TASK FOR PATIENT REPORT]
{patient_section}
</patient_report>

<coordinator_report>
[TASK FOR COORDINATOR REPORT]
{coordinator_section}
</coordinator_report>

<physician_report>
[TASK FOR PHYSICIAN REPORT]
{physician_section}
</physician_report>

Important:
- Write actual report content inside each tag pair (not the task instructions above).
- Do not include any text, preamble, or explanation outside the three XML tag pairs.
- Each report should be self-contained — the reader sees only their own report.
"""

REPORT_SAFE_CONFIRMATION = """\
You must produce THREE separate safety confirmation messages for patient: {patient_name}.
No drug interactions were detected in their current medication list.

Wrap each message in the exact XML tags — no extra text outside the tags:

<patient_report>
Write a warm, reassuring message (max 100 words) for the patient confirming their
medicines are safe together. Encourage them to keep all their doctors informed
whenever they start a new medicine.
</patient_report>

<coordinator_report>
Write a brief coordinator note (max 80 words) confirming no interactions were found,
listing the review as complete, and recommending the next scheduled medication review date.
</coordinator_report>

<physician_report>
Write a clinical note (max 120 words) confirming no clinically significant drug-drug
or drug-allergy interactions were identified in the automated screening of the active
medication list for {patient_name}. Include a recommendation for the next review interval.
</physician_report>

Do not include any text outside the three XML tag pairs.
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

# Retries 3x with exponential backoff on API rate limits
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True,
)
def _call_llm(prompt: str) -> str:
    """Call the configured LLM and return the raw text response."""
    # [REFS: llm_config.py > get_llm]
    # NOTE: LangChain .invoke() returns AIMessage — use .content for string.
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_combined_prompt(conflicts: list[dict], patient_name: str) -> str:
    """Build the three-audience combined prompt for conflict cases."""
    return REPORT_COMBINED_WRAPPER.format(
        patient_section=build_patient_prompt(conflicts),
        coordinator_section=build_coordinator_prompt(conflicts, patient_name),
        physician_section=build_physician_prompt(conflicts, patient_name),
    )


def _build_safe_prompt(patient_name: str) -> str:
    """Build the safe-confirmation prompt for no-conflict cases."""
    return REPORT_SAFE_CONFIRMATION.format(patient_name=patient_name)


# ── XML parser ─────────────────────────────────────────────────────────────────

def _parse_xml_reports(raw: str) -> dict | None:
    """Extract three reports from XML-delimited LLM response; None if any tag missing."""
    results: dict[str, str] = {}
    for role, pattern in _TAG_RE.items():
        match = pattern.search(raw)
        if not match:
            return None
        results[role] = match.group(1).strip()
    return results


# ── LangGraph node ─────────────────────────────────────────────────────────────

def run_agent(
    state: dict,
    store: PatientStore | None = None,
    anthropic_client: Any = None,  # kept for backward-compat; unused
) -> dict:
    """Generate patient, coordinator, and physician reports; never returns empty reports."""
    patient_id: str       = state.get(KEY_PATIENT_ID, "")
    conflicts: list[dict] = state.get(KEY_CONFLICTS, [])
    overall_severity: str = state.get(KEY_OVERALL_SEVERITY, SEVERITY_NONE)
    patient_name: str     = state.get(KEY_PATIENT_NAME, f"Patient {patient_id}")

    if store is None:
        store = PatientStore()

    rank = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1}
    sorted_conflicts = sorted(conflicts, key=lambda c: rank.get(c.get("severity", SEVERITY_MODERATE), 1))

    prompt = (
        _build_safe_prompt(patient_name)
        if not sorted_conflicts
        else _build_combined_prompt(sorted_conflicts, patient_name)
    )

    # [REFS: llm_config.py > get_llm]
    raw_response: str    = ""
    llm_error: str | None = None
    try:
        raw_response = _call_llm(prompt)
    except Exception as exc:
        llm_error = str(exc)

    reports: dict | None = None
    if raw_response:
        reports = _parse_xml_reports(raw_response)

    if reports is None:
        reason = llm_error or ("empty response" if not raw_response else "XML tags missing or malformed")

        store._audit(
            patient_id,
            "report_generator:fallback_used",
            {"reason": reason, "raw_preview": raw_response[:200] if raw_response else ""},
        )
        reports      = build_fallback_report(sorted_conflicts)
        status       = "FALLBACK"
        error        = reason
    else:
        reports["fallback_used"] = False
        status = "OK"
        error  = None

    store._audit(
        patient_id,
        "report_generator:complete",
        {
            KEY_STATUS:           status,
            KEY_OVERALL_SEVERITY: overall_severity,
            "conflict_count":     len(sorted_conflicts),
            "fallback_used":      (status == "FALLBACK"),
        },
    )

    return {
        KEY_PATIENT_ID:       patient_id,
        KEY_REPORTS:          reports,
        KEY_OVERALL_SEVERITY: overall_severity,
        KEY_STATUS:           status,
        KEY_ERROR:            error,
    }


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    from agents.profile_builder import run_agent as build_profile
    from agents.interaction_auditor import run_agent as audit_interactions
    from memory.knowledge_store import KnowledgeStore

    project_root = Path(__file__).parent.parent
    store        = PatientStore()

    try:
        store._r.ping()
        print("\n[PASS] Redis connection")
    except Exception as exc:
        print(f"\n[FAIL] Redis — {exc}")
        sys.exit(1)

    knowledge_store = KnowledgeStore()
    knowledge_store.init()
    print(f"[PASS] ChromaDB ({knowledge_store.document_count} chunks)\n")

    # ── Test A: patient_001 — conflicts expected ───────────────────────────────
    print("=" * 60)
    print("TEST A: patient_001 — Arjun Sharma (conflicts expected)")
    print("=" * 60)

    p1_file = project_root / "data" / "mock_patients" / "patient_001.json"
    with open(p1_file, encoding="utf-8") as fh:
        p1_json = json.load(fh)

    patient_id   = p1_json["id"]
    patient_name = f"{p1_json['name'][0]['given'][0]} {p1_json['name'][0]['family']}"
    store.delete_patient(patient_id)

    profile = build_profile(
        {"mode": "patient", KEY_PATIENT_ID: patient_id, "patient_json": p1_json},
        store=store,
    )
    audit = audit_interactions(profile, store=store, knowledge_store=knowledge_store)
    audit[KEY_PATIENT_NAME] = patient_name

    result = run_agent(audit, store=store)

    print(f"  status           : {result[KEY_STATUS]}")
    print(f"  overall_severity : {result[KEY_OVERALL_SEVERITY]}")
    print(f"  fallback_used    : {result[KEY_REPORTS].get('fallback_used')}")
    print(f"\n--- Patient report (first 300 chars) ---")
    print(result[KEY_REPORTS]["patient"][:300])
    print(f"\n--- Coordinator report (first 300 chars) ---")
    print(result[KEY_REPORTS]["coordinator"][:300])
    print(f"\n--- Physician report (first 300 chars) ---")
    print(result[KEY_REPORTS]["physician"][:300])

    assert result[KEY_STATUS] in ("OK", "FALLBACK")
    assert all(k in result[KEY_REPORTS] for k in ("patient", "coordinator", "physician"))
    assert all(len(result[KEY_REPORTS][k]) > 20 for k in ("patient", "coordinator", "physician"))
    print("\n  [PASS]\n")

    # ── Test B: patient_005 — no conflicts ─────────────────────────────────────
    print("=" * 60)
    print("TEST B: patient_005 — Kumar Nair (no conflicts)")
    print("=" * 60)

    p5_file = project_root / "data" / "mock_patients" / "patient_005.json"
    with open(p5_file, encoding="utf-8") as fh:
        p5_json = json.load(fh)

    patient_id_5   = p5_json["id"]
    patient_name_5 = f"{p5_json['name'][0]['given'][0]} {p5_json['name'][0]['family']}"
    store.delete_patient(patient_id_5)

    profile_5 = build_profile(
        {"mode": "patient", KEY_PATIENT_ID: patient_id_5, "patient_json": p5_json},
        store=store,
    )
    audit_5 = audit_interactions(profile_5, store=store, knowledge_store=knowledge_store)
    audit_5[KEY_PATIENT_NAME] = patient_name_5

    result_5 = run_agent(audit_5, store=store)

    print(f"  status           : {result_5[KEY_STATUS]}")
    print(f"  overall_severity : {result_5[KEY_OVERALL_SEVERITY]}")
    print(f"\n--- Patient report ---")
    print(result_5[KEY_REPORTS]["patient"])

    assert result_5[KEY_STATUS] in ("OK", "FALLBACK", "SKIPPED")
    assert all(k in result_5[KEY_REPORTS] for k in ("patient", "coordinator", "physician"))
    assert all(len(result_5[KEY_REPORTS][k]) > 10 for k in ("patient", "coordinator", "physician"))
    print("\n  [PASS]\n")

    store.delete_patient(patient_id)
    store.delete_patient(patient_id_5)
    print("All ConflictReportGeneratorAgent tests passed.")
