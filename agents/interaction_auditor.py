# Agent: DrugInteractionAuditorAgent
# Role:  Detect drug-drug and drug-allergy interactions via rules + LLM.
# Input: patient_id, medications, skip_audit
# Output: patient_id, conflicts, overall_severity, status, error

"""
Agent 2 — DrugInteractionAuditorAgent

Pipeline (5 steps):
  1. Allergy check   — any active drug matching a known allergy → CRITICAL
  2. Rule engine     — deterministic check of all pairs via interaction_rules.json
  3. LLM check       — unflagged pairs queried through ChromaDB then LLM
  4. Merge           — deduplicate; CRITICAL wins over MODERATE for same pair
  5. Persist         — write consolidated conflicts to Redis

LangGraph integration::

    from agents.interaction_auditor import run_agent as audit_interactions
    graph.add_node("interaction_auditor", audit_interactions)
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_config import get_llm
from langchain_core.messages import HumanMessage

from tools.fhir_parser import Drug
from tools.rule_engine import check_pairs, check_pairs_all_severity
from memory.patient_store import PatientStore, _dict_to_drug
from memory.knowledge_store import KnowledgeStore

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_RETRIES       = 3
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_NONE     = "NONE"

KEY_PATIENT_ID       = "patient_id"
KEY_MEDICATIONS      = "medications"
KEY_CONFLICTS        = "conflicts"
KEY_OVERALL_SEVERITY = "overall_severity"
KEY_STATUS           = "status"
KEY_ERROR            = "error"
KEY_SKIP_AUDIT       = "skip_audit"
KEY_MODE             = "mode"

_SYNONYMS_PATH = Path(__file__).parent.parent / "data" / "drug_synonyms.json"

# ── Prompt ─────────────────────────────────────────────────────────────────────

AUDIT_PROMPT = """\
You are a clinical pharmacist reviewing a potential drug interaction.

Drug A: {drug_a}
Drug B: {drug_b}

Relevant pharmacovigilance literature:
{context}

Task: Assess whether concurrent use of {drug_a} and {drug_b} poses a clinically significant interaction risk.

Respond ONLY with a JSON object in this exact format (no markdown, no explanation outside the JSON):
{{
  "severity": "MODERATE" or "NONE",
  "mechanism": "one sentence plain-English explanation, or null if NONE"
}}

Rules:
- Use "MODERATE" only if there is a real clinical interaction requiring monitoring or dose adjustment.
- Use "NONE" if the combination is generally safe.
- Do not use "CRITICAL" — critical interactions are handled by a separate rule engine.
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


# ── Step 1: Allergy check ──────────────────────────────────────────────────────

def _check_allergies(drugs: list[Drug], allergies: list[Any]) -> list[dict]:
    """Return CRITICAL conflicts for any active drug matching a known allergy."""
    conflicts: list[dict] = []
    allergy_substances: list[str] = []

    for a in allergies:
        if isinstance(a, dict):
            allergy_substances.append(a.get("substance", "").lower())
        elif isinstance(a, str):
            allergy_substances.append(a.lower())

    for drug in drugs:
        if not drug.active_status:
            continue
        drug_names = {drug.drug_name.lower(), drug.generic_name.lower()}
        for substance in allergy_substances:
            if substance and any(substance in name or name in substance for name in drug_names):
                reaction = ""
                for a in allergies:
                    if isinstance(a, dict) and a.get("substance", "").lower() == substance:
                        reaction = a.get("reaction", "")
                conflicts.append(
                    {
                        "conflict_type":         "ALLERGY",
                        "severity":              SEVERITY_CRITICAL,
                        "drug_a":                drug.generic_name,
                        "drug_b":                substance,
                        "rule_id":               "ALLERGY_CHECK",
                        "mechanism": (
                            f"Patient has a documented allergy to {substance}"
                            + (f" (reaction: {reaction})" if reaction else "")
                            + f". {drug.generic_name} matches or cross-reacts with this allergen."
                        ),
                        "clinical_effects":      [reaction] if reaction else ["allergic reaction"],
                        "monitoring":            "Do not administer. Consult prescribing physician immediately.",
                        "suggested_alternative": "Use a structurally unrelated drug. Review full allergy history.",
                        "source":                "allergy_check",
                    }
                )
    return conflicts


# ── Step 3: LLM semantic check ─────────────────────────────────────────────────

def _llm_check_unflagged_pairs(
    drugs: list[Drug],
    flagged_pairs: set[frozenset],
    knowledge_store: KnowledgeStore,
) -> list[dict]:
    """Query ChromaDB + LLM for pairs not covered by the rule engine."""
    active    = [d for d in drugs if d.active_status]
    conflicts: list[dict] = []

    for drug_a, drug_b in combinations(active, 2):
        pair_key = frozenset({drug_a.generic_name.lower(), drug_b.generic_name.lower()})
        if pair_key in flagged_pairs:
            continue

        query = f"{drug_a.generic_name} {drug_b.generic_name} interaction"
        try:
            # [REFS: memory/knowledge_store.py > query]
            chunks = knowledge_store.query(query, n_results=3)
        except Exception:
            chunks = []

        context = "\n\n".join(chunks) if chunks else "No specific literature found."
        prompt  = AUDIT_PROMPT.format(
            drug_a=drug_a.generic_name,
            drug_b=drug_b.generic_name,
            context=context,
        )

        try:
            # [REFS: llm_config.py > get_llm]
            raw       = _call_llm(prompt)
            raw       = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result    = json.loads(raw)
            severity  = result.get("severity", SEVERITY_NONE).upper()
            mechanism = result.get("mechanism") or ""
        except Exception as exc:
            severity  = SEVERITY_NONE
            mechanism = f"[LLM check failed: {exc}]"

        if severity == SEVERITY_MODERATE:
            conflicts.append(
                {
                    "conflict_type":         "DRUG_DRUG",
                    "severity":              SEVERITY_MODERATE,
                    "drug_a":                drug_a.generic_name,
                    "drug_b":                drug_b.generic_name,
                    "matched_drug_a":        drug_a.generic_name,
                    "matched_drug_b":        drug_b.generic_name,
                    "rule_id":               None,
                    "mechanism":             mechanism,
                    "clinical_effects":      [],
                    "monitoring":            "Monitor clinically; review with prescribing physician.",
                    "suggested_alternative": "",
                    "source":                "llm",
                }
            )

    return conflicts


# ── Step 4: Merge & deduplicate ────────────────────────────────────────────────

def _merge_conflicts(
    allergy_conflicts: list[dict],
    rule_conflicts: list[dict],
    llm_conflicts: list[dict],
) -> list[dict]:
    """Combine all sources; CRITICAL wins over MODERATE for the same drug pair."""
    all_conflicts = allergy_conflicts + rule_conflicts + llm_conflicts

    best: dict[frozenset, dict] = {}
    severity_rank = {SEVERITY_CRITICAL: 2, SEVERITY_MODERATE: 1, SEVERITY_NONE: 0}

    for c in all_conflicts:
        pair_key = frozenset(
            {
                c.get("matched_drug_a", c["drug_a"]).lower(),
                c.get("matched_drug_b", c["drug_b"]).lower(),
            }
        )
        existing = best.get(pair_key)
        if existing is None or (
            severity_rank.get(c["severity"], 0) > severity_rank.get(existing["severity"], 0)
        ):
            best[pair_key] = c

    return list(best.values())


# ── Step 5: Overall severity ───────────────────────────────────────────────────

def _overall_severity(conflicts: list[dict]) -> str:
    """Return the highest severity present across all conflicts."""
    severities = {c["severity"] for c in conflicts}
    if SEVERITY_CRITICAL in severities:
        return SEVERITY_CRITICAL
    if SEVERITY_MODERATE in severities:
        return SEVERITY_MODERATE
    return SEVERITY_NONE


# ── LangGraph node ─────────────────────────────────────────────────────────────

def run_agent(
    state: dict,
    store: PatientStore | None = None,
    knowledge_store: KnowledgeStore | None = None,
    anthropic_client: Any = None,  # kept for backward-compat; unused
) -> dict:
    """Run the full 5-step interaction audit pipeline."""
    patient_id: str = state.get(KEY_PATIENT_ID, "")

    if state.get(KEY_SKIP_AUDIT):
        return {
            KEY_PATIENT_ID:       patient_id,
            KEY_CONFLICTS:        [],
            KEY_OVERALL_SEVERITY: SEVERITY_NONE,
            KEY_STATUS:           "SKIPPED",
            KEY_ERROR:            None,
        }

    medication_dicts: list[dict] = state.get(KEY_MEDICATIONS, [])
    if not medication_dicts:
        return {
            KEY_PATIENT_ID:       patient_id,
            KEY_CONFLICTS:        [],
            KEY_OVERALL_SEVERITY: SEVERITY_NONE,
            KEY_STATUS:           "ERROR",
            KEY_ERROR:            "No medications in state — run profile_builder first.",
        }

    if store is None:
        store = PatientStore()
    if knowledge_store is None:
        knowledge_store = KnowledgeStore()
        knowledge_store.init()

    drugs: list[Drug] = [_dict_to_drug(d) for d in medication_dicts]

    # Step 1 — allergy check
    # [REFS: memory/patient_store.py > get_allergies]
    allergies         = store.get_allergies(patient_id)
    allergy_conflicts = _check_allergies(drugs, allergies)

    # Step 2 — rule engine
    all_rule_conflicts        = check_pairs_all_severity(drugs)
    rule_conflicts_for_output = [
        c for c in all_rule_conflicts
        if c["severity"] in (SEVERITY_CRITICAL, SEVERITY_MODERATE)
    ]
    flagged_pairs: set[frozenset] = {
        frozenset({c["matched_drug_a"].lower(), c["matched_drug_b"].lower()})
        for c in all_rule_conflicts
    }

    # Step 3 — LLM check for uncovered pairs
    llm_conflicts = _llm_check_unflagged_pairs(drugs, flagged_pairs, knowledge_store)

    # Step 4 — merge
    merged = _merge_conflicts(allergy_conflicts, rule_conflicts_for_output, llm_conflicts)

    # Step 5 — persist
    # [REFS: memory/patient_store.py > save_conflicts]
    store.save_conflicts(patient_id, merged)

    # WHY: CRITICAL severity bypasses human checkpoint — patient safety rule.
    return {
        KEY_PATIENT_ID:       patient_id,
        KEY_CONFLICTS:        merged,
        KEY_OVERALL_SEVERITY: _overall_severity(merged),
        KEY_STATUS:           "OK",
        KEY_ERROR:            None,
    }


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    from agents.profile_builder import run_agent as build_profile

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
    print(f"[PASS] ChromaDB knowledge store ({knowledge_store.document_count} chunks)\n")

    # ── Test A: patient_001 ────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST A: patient_001 — Arjun Sharma (4 drugs, expect MODERATE)")
    print("=" * 60)

    p1_file = project_root / "data" / "mock_patients" / "patient_001.json"
    with open(p1_file, encoding="utf-8") as fh:
        p1_json = json.load(fh)

    patient_id_1 = p1_json["id"]
    store.delete_patient(patient_id_1)

    profile_state = build_profile(
        {KEY_MODE: "patient", KEY_PATIENT_ID: patient_id_1, "patient_json": p1_json},
        store=store,
    )
    assert profile_state[KEY_STATUS] == "OK", f"Profile build failed: {profile_state}"

    audit_state = run_agent(profile_state, store=store, knowledge_store=knowledge_store)

    print(f"  overall_severity : {audit_state[KEY_OVERALL_SEVERITY]}")
    print(f"  conflicts found  : {len(audit_state[KEY_CONFLICTS])}")
    for c in audit_state[KEY_CONFLICTS]:
        src = c.get("rule_id") or c.get("source", "?")
        print(f"    [{c['severity']:8s}] {c['drug_a']} + {c['drug_b']}  [{src}]")
    assert audit_state[KEY_STATUS] == "OK"
    assert audit_state[KEY_OVERALL_SEVERITY] in (SEVERITY_MODERATE, SEVERITY_CRITICAL)
    assert len(audit_state[KEY_CONFLICTS]) >= 1
    print("  [PASS]\n")

    # ── Test B: patient_005 ────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST B: patient_005 — Kumar Nair (1 drug, expect SKIPPED/NONE)")
    print("=" * 60)

    p5_file = project_root / "data" / "mock_patients" / "patient_005.json"
    with open(p5_file, encoding="utf-8") as fh:
        p5_json = json.load(fh)

    patient_id_5 = p5_json["id"]
    store.delete_patient(patient_id_5)

    profile_state_5 = build_profile(
        {KEY_MODE: "patient", KEY_PATIENT_ID: patient_id_5, "patient_json": p5_json},
        store=store,
    )
    assert profile_state_5[KEY_STATUS] == "OK"
    assert profile_state_5[KEY_SKIP_AUDIT] is True

    audit_state_5 = run_agent(profile_state_5, store=store, knowledge_store=knowledge_store)

    print(f"  status           : {audit_state_5[KEY_STATUS]}")
    print(f"  overall_severity : {audit_state_5[KEY_OVERALL_SEVERITY]}")
    print(f"  conflicts found  : {len(audit_state_5[KEY_CONFLICTS])}")
    assert audit_state_5[KEY_STATUS] == "SKIPPED"
    assert audit_state_5[KEY_OVERALL_SEVERITY] == SEVERITY_NONE
    assert len(audit_state_5[KEY_CONFLICTS]) == 0
    print("  [PASS]\n")

    store.delete_patient(patient_id_1)
    store.delete_patient(patient_id_5)
    print("All DrugInteractionAuditorAgent tests passed.")
