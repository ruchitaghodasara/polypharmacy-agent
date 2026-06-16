"""
Prompt and fallback-report builders for the ConflictReportGeneratorAgent.

Three audience tiers:
  patient      — plain language, action-oriented, no clinical jargon
  coordinator  — structured action list, which doctors to notify, priority
  physician    — full clinical detail, mechanisms, evidence-based alternatives
"""

from __future__ import annotations

from typing import List

# ── Internal helpers ──────────────────────────────────────────────────────────

_SEVERITY_EMOJI = {"CRITICAL": "🚨", "MODERATE": "⚠️"}
_SEVERITY_LABEL = {"CRITICAL": "URGENT", "MODERATE": "IMPORTANT"}


def _sorted_conflicts(conflicts: List[dict]) -> List[dict]:
    rank = {"CRITICAL": 0, "MODERATE": 1}
    return sorted(conflicts, key=lambda c: rank.get(c.get("severity", "MODERATE"), 1))


def _conflict_lines_patient(conflicts: List[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(_sorted_conflicts(conflicts), 1):
        sev = c.get("severity", "MODERATE")
        label = _SEVERITY_LABEL.get(sev, sev)
        drug_a = c.get("drug_a", c.get("matched_drug_a", "Drug A"))
        drug_b = c.get("drug_b", c.get("matched_drug_b", "Drug B"))
        alt = c.get("suggested_alternative", "")
        conflict_type = c.get("conflict_type", "DRUG_DRUG")

        if conflict_type == "ALLERGY":
            lines.append(
                f"{i}. [{label}] You have a documented allergy to {drug_b}. "
                f"The medicine {drug_a} may be related to it. "
                "Please tell your doctor immediately before taking this medicine."
            )
        else:
            action = (
                f"Tell your doctor about this before taking both medicines together."
            )
            if alt:
                action += f" There may be a safer option — ask about: {alt.split('.')[0]}."
            lines.append(
                f"{i}. [{label}] {drug_a} and {drug_b} can interact. {action}"
            )
    return "\n".join(lines)


def _conflict_lines_coordinator(conflicts: List[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(_sorted_conflicts(conflicts), 1):
        sev = c.get("severity", "MODERATE")
        drug_a = c.get("drug_a", c.get("matched_drug_a", "Drug A"))
        drug_b = c.get("drug_b", c.get("matched_drug_b", "Drug B"))
        source = c.get("source", "unknown")
        rule_id = c.get("rule_id") or "semantic"
        lines.append(
            f"{i}. [{sev}] {drug_a} ↔ {drug_b}  (rule: {rule_id}, source: {source})"
        )
    return "\n".join(lines)


def _conflict_lines_physician(conflicts: List[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(_sorted_conflicts(conflicts), 1):
        sev = c.get("severity", "MODERATE")
        drug_a = c.get("drug_a", c.get("matched_drug_a", "Drug A"))
        drug_b = c.get("drug_b", c.get("matched_drug_b", "Drug B"))
        mechanism = c.get("mechanism", "Mechanism not specified.")
        effects = ", ".join(c.get("clinical_effects", [])) or "See mechanism."
        monitoring = c.get("monitoring", "")
        alt = c.get("suggested_alternative", "")
        rule_id = c.get("rule_id") or "semantic-LLM"
        lines.append(
            f"{i}. [{sev}] {drug_a} + {drug_b}  [{rule_id}]\n"
            f"   Mechanism    : {mechanism}\n"
            f"   Effects      : {effects}\n"
            + (f"   Monitoring   : {monitoring}\n" if monitoring else "")
            + (f"   Alternative  : {alt}\n" if alt else "")
        )
    return "\n".join(lines)


# ── Public prompt builders ────────────────────────────────────────────────────

def build_patient_prompt(conflicts: List[dict]) -> str:
    """
    Plain-language, action-oriented prompt for Claude to write a patient-facing report.
    No medical jargon. Focus on what the patient should *do* right now.
    """
    n_critical = sum(1 for c in conflicts if c.get("severity") == "CRITICAL")
    n_moderate = sum(1 for c in conflicts if c.get("severity") == "MODERATE")
    conflict_summary = _conflict_lines_patient(conflicts)

    return f"""You are writing a medication safety message for a patient.
The patient takes multiple medicines prescribed by different doctors who may not know about each other's prescriptions.

Detected issues ({n_critical} urgent, {n_moderate} important):
{conflict_summary}

Write a clear, caring, non-alarming patient safety message. Rules:
- Use simple everyday language — no Latin, no medical abbreviations
- Start with a brief reassurance that this check is routine and they are not in danger right now
- List each issue with a specific action ("Before your next dose, call Dr X" or "At your next appointment, mention Y to your doctor")
- End with a single sentence: what to do if they feel unwell before their next appointment
- Maximum 250 words
- Do NOT include headers, bullet symbols, or markdown — plain paragraphs only
"""


def build_coordinator_prompt(conflicts: List[dict], patient_name: str) -> str:
    """
    Structured action list for a care coordinator or pharmacist.
    Which prescribers to contact, in what priority order, and why.
    """
    sorted_c = _sorted_conflicts(conflicts)
    n_critical = sum(1 for c in sorted_c if c.get("severity") == "CRITICAL")
    conflict_lines = _conflict_lines_coordinator(sorted_c)

    # Extract unique prescribers from conflict drugs
    prescribers: set[str] = set()
    for c in sorted_c:
        # We don't store prescribers in conflict dicts; coordinator prompt notes this
        pass

    return f"""You are writing an action plan for a care coordinator managing patient: {patient_name}.

Detected drug interactions requiring coordination:
{conflict_lines}

Write a structured coordinator action plan. Rules:
- Open with a one-line priority statement (e.g. "1 CRITICAL interaction requires same-day action")
- List actions in strict priority order (CRITICAL first)
- For each issue: state which prescribing specialty is most likely responsible for each drug,
  what specific information to communicate, and the recommended timeframe (same-day / within 48h / next appointment)
- Add a "Documentation" section: what to log in the patient record
- Add a "Follow-up" section: when to re-check this patient's medication list
- Use professional clinical language appropriate for a pharmacist or nurse coordinator
- Maximum 350 words
"""


def build_physician_prompt(conflicts: List[dict], patient_name: str) -> str:
    """
    Full clinical detail for the responsible physician(s).
    Drug names, mechanisms, evidence-based alternatives, urgency classification.
    """
    conflict_lines = _conflict_lines_physician(conflicts)
    n_critical = sum(1 for c in conflicts if c.get("severity") == "CRITICAL")

    return f"""You are writing a clinical drug interaction alert for the physician(s) caring for patient: {patient_name}.

Interaction details:
{conflict_lines}

Write a physician-level clinical alert. Rules:
- Open with a clinical summary: total interactions, highest severity, immediate risk assessment
- For each interaction provide:
    * Full pharmacological mechanism
    * Expected clinical manifestation and time course
    * Quantified risk where evidence exists (e.g. "3-fold increase in bleeding risk")
    * Evidence-based management options in order of preference
    * Specific monitoring parameters and frequency
- Close with a "Recommended Actions" section ranked by urgency:
    IMMEDIATE (same day): ...
    SHORT-TERM (within 1 week): ...
    ONGOING: ...
- Use standard clinical terminology (INN drug names, SI units)
- {"⚠️ At least one CRITICAL interaction is present — flag for urgent physician review." if n_critical else "No critical interactions; monitoring approach is appropriate."}
- Maximum 500 words
"""


# ── Fallback report (used when Claude fails) ──────────────────────────────────

def build_fallback_report(conflicts: List[dict]) -> dict:
    """
    Pre-templated reports used when the Claude API call fails or cannot be parsed.
    Always returns all three roles so the caller never gets an empty report.
    """
    sorted_c = _sorted_conflicts(conflicts)
    n_critical = sum(1 for c in sorted_c if c.get("severity") == "CRITICAL")
    n_moderate = sum(1 for c in sorted_c if c.get("severity") == "MODERATE")

    # ── Patient fallback ──
    if not sorted_c:
        patient_text = (
            "Good news — our review of your current medications found no safety concerns. "
            "Continue taking your medicines as prescribed. If you start any new medicine, "
            "vitamin, or supplement, please let all your doctors know."
        )
    else:
        issue_list = "\n".join(
            f"- {c.get('drug_a', '?')} and {c.get('drug_b', '?')}: "
            f"please mention this combination to your doctor."
            for c in sorted_c
        )
        patient_text = (
            f"Our medication safety review found {len(sorted_c)} item(s) that your doctor "
            f"should know about ({n_critical} urgent, {n_moderate} important). "
            f"Please do not change your medicines on your own — speak to your doctor first.\n\n"
            f"{issue_list}\n\n"
            "If you feel unwell at any time, contact your doctor or go to your nearest clinic immediately."
        )

    # ── Coordinator fallback ──
    if not sorted_c:
        coord_text = (
            "MEDICATION REVIEW COMPLETE — NO INTERACTIONS DETECTED\n"
            "No action required. Schedule routine next review per standard protocol."
        )
    else:
        coord_lines = "\n".join(
            f"[{c.get('severity', '?')}] {c.get('drug_a', '?')} + {c.get('drug_b', '?')} "
            f"— notify prescribing physician(s); rule: {c.get('rule_id') or 'semantic'}"
            for c in sorted_c
        )
        coord_text = (
            f"MEDICATION INTERACTION ALERT — {n_critical} CRITICAL, {n_moderate} MODERATE\n\n"
            f"Actions required:\n{coord_lines}\n\n"
            "Timeframe: CRITICAL items require same-day physician contact. "
            "MODERATE items require contact within 48 hours. "
            "Document all communications in the patient record."
        )

    # ── Physician fallback ──
    if not sorted_c:
        phys_text = (
            "CLINICAL MEDICATION REVIEW — NO INTERACTIONS IDENTIFIED\n"
            "Automated interaction screening found no clinically significant drug-drug or "
            "drug-allergy interactions in the current active medication list. "
            "Routine monitoring continues per clinical protocol."
        )
    else:
        phys_lines = "\n".join(
            f"[{c.get('severity', '?')}] {c.get('drug_a', '?')} + {c.get('drug_b', '?')}\n"
            f"  Mechanism: {c.get('mechanism', 'Not available')}\n"
            f"  Alternative: {c.get('suggested_alternative', 'Consult specialist')}"
            for c in sorted_c
        )
        phys_text = (
            f"CLINICAL DRUG INTERACTION ALERT\n"
            f"Summary: {len(sorted_c)} interaction(s) detected — {n_critical} CRITICAL, {n_moderate} MODERATE.\n\n"
            f"{phys_lines}\n\n"
            "ACTION REQUIRED: Review the above interactions and adjust therapy as clinically appropriate. "
            "CRITICAL interactions require same-day review."
        )

    return {
        "patient": patient_text,
        "coordinator": coord_text,
        "physician": phys_text,
        "fallback_used": True,
    }
