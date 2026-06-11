"""
Agent 3 — ConflictReportGeneratorAgent

Takes the merged conflict list from the interaction auditor and generates
three tailored reports in a single Claude call:
  - patient     : plain-language, action-oriented
  - coordinator : structured action plan for care coordinators / pharmacists
  - physician   : full clinical detail with mechanisms and alternatives

A combined prompt wraps all three instructions and asks Claude to delimit
each report with XML tags. The response is parsed with regex; on any failure
the pre-templated fallback from report_formatter is used instead.

LangGraph integration:
    from agents.report_generator import run_agent as generate_reports
    graph.add_node("report_generator", generate_reports)

State contract
--------------
Input (from interaction_auditor output or equivalent):
    patient_id        str
    conflicts         List[dict]
    overall_severity  str    "CRITICAL" | "MODERATE" | "NONE"
    patient_name      str    optional — used in coordinator and physician reports

Output (always present — never empty):
    patient_id        str
    reports           dict   {patient, coordinator, physician, fallback_used}
    overall_severity  str    passed through unchanged
    status            str    "OK" | "FALLBACK" | "SKIPPED" | "ERROR"
    error             str | None
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.report_formatter import (
    build_patient_prompt,
    build_coordinator_prompt,
    build_physician_prompt,
    build_fallback_report,
)
from memory.patient_store import PatientStore

_CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# ── XML tag regex ──────────────────────────────────────────────────────────────

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


# ── Tenacity-wrapped Claude call ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True,
)
def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Combined prompt builder ───────────────────────────────────────────────────

def _build_combined_prompt(
    conflicts: list[dict],
    patient_name: str,
) -> str:
    patient_section = build_patient_prompt(conflicts)
    coordinator_section = build_coordinator_prompt(conflicts, patient_name)
    physician_section = build_physician_prompt(conflicts, patient_name)

    return f"""You must produce THREE separate reports for the same set of drug interaction findings.
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


# ── XML parser ────────────────────────────────────────────────────────────────

def _parse_xml_reports(raw: str) -> dict | None:
    """
    Extract the three reports from Claude's XML-delimited response.
    Returns None if any tag is missing.
    """
    results: dict[str, str] = {}
    for role, pattern in _TAG_RE.items():
        match = pattern.search(raw)
        if not match:
            return None
        results[role] = match.group(1).strip()
    return results


# ── No-conflict safe reports ──────────────────────────────────────────────────

def _build_safe_confirmation_prompt(patient_name: str) -> str:
    return f"""You must produce THREE separate safety confirmation messages for patient: {patient_name}.
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


# ── LangGraph node ─────────────────────────────────────────────────────────────

def run_agent(
    state: dict,
    store: PatientStore | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> dict:
    """
    LangGraph-compatible node function.

    Produces reports dict: {patient, coordinator, physician, fallback_used}.
    Never returns an empty report — fallback templates are used on any failure.
    """
    patient_id: str = state.get("patient_id", "")
    conflicts: list[dict] = state.get("conflicts", [])
    overall_severity: str = state.get("overall_severity", "NONE")
    patient_name: str = state.get("patient_name", f"Patient {patient_id}")

    if store is None:
        store = PatientStore()
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    # ── Sort conflicts: CRITICAL first ─────────────────────────────────────────
    rank = {"CRITICAL": 0, "MODERATE": 1}
    sorted_conflicts = sorted(
        conflicts, key=lambda c: rank.get(c.get("severity", "MODERATE"), 1)
    )

    # ── Build prompt (no-conflict path uses a different prompt) ────────────────
    if not sorted_conflicts:
        prompt = _build_safe_confirmation_prompt(patient_name)
    else:
        prompt = _build_combined_prompt(sorted_conflicts, patient_name)

    # ── Single Claude call ─────────────────────────────────────────────────────
    raw_response: str = ""
    claude_error: str | None = None
    try:
        raw_response = _call_claude(anthropic_client, prompt)
    except Exception as exc:
        claude_error = str(exc)

    # ── Parse XML tags ─────────────────────────────────────────────────────────
    reports: dict | None = None
    if raw_response:
        reports = _parse_xml_reports(raw_response)

    fallback_used = False
    if reports is None:
        # Log parse failure to audit
        if claude_error or not raw_response:
            reason = claude_error or "empty response"
        else:
            reason = "XML tags missing or malformed in Claude response"

        store._audit(
            patient_id,
            "report_generator:fallback_used",
            {"reason": reason, "raw_preview": raw_response[:200] if raw_response else ""},
        )
        reports = build_fallback_report(sorted_conflicts)
        fallback_used = True
        status = "FALLBACK"
        error = reason
    else:
        reports["fallback_used"] = False
        fallback_used = False
        status = "OK"
        error = None

    store._audit(
        patient_id,
        "report_generator:complete",
        {
            "status": status,
            "overall_severity": overall_severity,
            "conflict_count": len(sorted_conflicts),
            "fallback_used": fallback_used,
        },
    )

    return {
        "patient_id": patient_id,
        "reports": reports,
        "overall_severity": overall_severity,
        "status": status,
        "error": error,
    }


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    from agents.profile_builder import run_agent as build_profile
    from agents.interaction_auditor import run_agent as audit_interactions
    from memory.knowledge_store import KnowledgeStore

    project_root = Path(__file__).parent.parent
    store = PatientStore()

    try:
        store._r.ping()
        print("\n[PASS] Redis connection")
    except Exception as exc:
        print(f"\n[FAIL] Redis — {exc}")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    knowledge_store = KnowledgeStore()
    knowledge_store.init()
    print(f"[PASS] ChromaDB ({knowledge_store.document_count} chunks)\n")

    # ── Test A: patient_001 — expect conflict reports ──────────────────────────
    print("=" * 60)
    print("TEST A: patient_001 — Arjun Sharma (conflicts expected)")
    print("=" * 60)

    p1_file = project_root / "data" / "mock_patients" / "patient_001.json"
    with open(p1_file) as fh:
        p1_json = json.load(fh)

    patient_id = p1_json["id"]
    patient_name = f"{p1_json['name'][0]['given'][0]} {p1_json['name'][0]['family']}"
    store.delete_patient(patient_id)

    profile = build_profile(
        {"mode": "patient", "patient_id": patient_id, "patient_json": p1_json},
        store=store,
    )
    audit = audit_interactions(
        profile,
        store=store,
        knowledge_store=knowledge_store,
        anthropic_client=client,
    )
    audit["patient_name"] = patient_name

    result = run_agent(audit, store=store, anthropic_client=client)

    print(f"  status           : {result['status']}")
    print(f"  overall_severity : {result['overall_severity']}")
    print(f"  fallback_used    : {result['reports'].get('fallback_used')}")
    print(f"\n--- Patient report (first 300 chars) ---")
    print(result["reports"]["patient"][:300])
    print(f"\n--- Coordinator report (first 300 chars) ---")
    print(result["reports"]["coordinator"][:300])
    print(f"\n--- Physician report (first 300 chars) ---")
    print(result["reports"]["physician"][:300])

    assert result["status"] in ("OK", "FALLBACK")
    assert all(k in result["reports"] for k in ("patient", "coordinator", "physician"))
    assert all(len(result["reports"][k]) > 20 for k in ("patient", "coordinator", "physician"))
    print("\n  [PASS]\n")

    # ── Test B: patient_005 — no conflicts, safe confirmation ─────────────────
    print("=" * 60)
    print("TEST B: patient_005 — Kumar Nair (no conflicts)")
    print("=" * 60)

    p5_file = project_root / "data" / "mock_patients" / "patient_005.json"
    with open(p5_file) as fh:
        p5_json = json.load(fh)

    patient_id_5 = p5_json["id"]
    patient_name_5 = f"{p5_json['name'][0]['given'][0]} {p5_json['name'][0]['family']}"
    store.delete_patient(patient_id_5)

    profile_5 = build_profile(
        {"mode": "patient", "patient_id": patient_id_5, "patient_json": p5_json},
        store=store,
    )
    audit_5 = audit_interactions(
        profile_5,
        store=store,
        knowledge_store=knowledge_store,
        anthropic_client=client,
    )
    audit_5["patient_name"] = patient_name_5

    result_5 = run_agent(audit_5, store=store, anthropic_client=client)

    print(f"  status           : {result_5['status']}")
    print(f"  overall_severity : {result_5['overall_severity']}")
    print(f"\n--- Patient report ---")
    print(result_5["reports"]["patient"])

    assert result_5["status"] in ("OK", "FALLBACK", "SKIPPED")
    assert all(k in result_5["reports"] for k in ("patient", "coordinator", "physician"))
    assert all(len(result_5["reports"][k]) > 10 for k in ("patient", "coordinator", "physician"))
    print("\n  [PASS]\n")

    # Cleanup
    store.delete_patient(patient_id)
    store.delete_patient(patient_id_5)
    print("All ConflictReportGeneratorAgent tests passed.")
