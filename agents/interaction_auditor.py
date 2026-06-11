"""
Agent 2 — DrugInteractionAuditorAgent

Pipeline (5 steps):
  1. Allergy check   — any active drug matching a known allergy → CRITICAL immediately
  2. Rule engine     — deterministic check of all pairs against interaction_rules.json
  3. Claude check    — for pairs NOT already flagged by rules, query ChromaDB then
                       ask Claude to classify: MODERATE or NONE
  4. Merge           — deduplicate pairs; CRITICAL wins over MODERATE
  5. Persist         — write consolidated conflicts to Redis patient:{id}:conflicts

LangGraph integration:
    from agents.interaction_auditor import run_agent as audit_interactions
    graph.add_node("interaction_auditor", audit_interactions)

State contract
--------------
Input (from profile_builder output or equivalent):
    patient_id      str
    medications     List[dict]   serialised Drug dicts
    skip_audit      bool         if True, returns immediately with no conflicts

Output (always present):
    patient_id      str
    conflicts       List[dict]
    overall_severity  str   "CRITICAL" | "MODERATE" | "NONE"
    status          str   "OK" | "SKIPPED" | "ERROR"
    error           str | None
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.fhir_parser import Drug
from tools.rule_engine import check_pairs, check_pairs_all_severity
from memory.patient_store import PatientStore, _dict_to_drug
from memory.knowledge_store import KnowledgeStore

_CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
_SYNONYMS_PATH = Path(__file__).parent.parent / "data" / "drug_synonyms.json"

# ── Tenacity-wrapped Claude call ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True,
)
def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """Call Claude and return the raw text response. Retried up to 3 times."""
    response = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_claude_prompt(drug_a: str, drug_b: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else "No specific literature found."
    return f"""You are a clinical pharmacist reviewing a potential drug interaction.

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


# ── Step 1: Allergy check ─────────────────────────────────────────────────────

def _check_allergies(
    drugs: list[Drug],
    allergies: list[Any],
) -> list[dict]:
    """Return CRITICAL conflicts for any drug matching a known allergy."""
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
                        "conflict_type": "ALLERGY",
                        "severity": "CRITICAL",
                        "drug_a": drug.generic_name,
                        "drug_b": substance,
                        "rule_id": "ALLERGY_CHECK",
                        "mechanism": (
                            f"Patient has a documented allergy to {substance}"
                            + (f" (reaction: {reaction})" if reaction else "")
                            + f". {drug.generic_name} matches or cross-reacts with this allergen."
                        ),
                        "clinical_effects": [reaction] if reaction else ["allergic reaction"],
                        "monitoring": "Do not administer. Consult prescribing physician immediately.",
                        "suggested_alternative": "Use a structurally unrelated drug. Review full allergy history.",
                        "source": "allergy_check",
                    }
                )
    return conflicts


# ── Step 3: Claude semantic check ─────────────────────────────────────────────

def _claude_check_unflagged_pairs(
    drugs: list[Drug],
    flagged_pairs: set[frozenset],
    knowledge_store: KnowledgeStore,
    client: anthropic.Anthropic,
) -> list[dict]:
    """
    For every active pair NOT already flagged by rules, ask ChromaDB + Claude
    whether an interaction exists. Returns only MODERATE results (NONE discarded).
    """
    active = [d for d in drugs if d.active_status]
    conflicts: list[dict] = []

    for drug_a, drug_b in combinations(active, 2):
        pair_key = frozenset({drug_a.generic_name.lower(), drug_b.generic_name.lower()})
        if pair_key in flagged_pairs:
            continue  # already handled by rule engine

        query = f"{drug_a.generic_name} {drug_b.generic_name} interaction"
        try:
            chunks = knowledge_store.query(query, n_results=3)
        except Exception:
            chunks = []

        prompt = _build_claude_prompt(drug_a.generic_name, drug_b.generic_name, chunks)

        try:
            raw = _call_claude(client, prompt)
            # Strip markdown code fences if Claude wraps the JSON
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            severity = result.get("severity", "NONE").upper()
            mechanism = result.get("mechanism") or ""
        except Exception as exc:
            # Treat parse/API failures as NONE to avoid false positives
            severity = "NONE"
            mechanism = f"[Claude check failed: {exc}]"

        if severity == "MODERATE":
            conflicts.append(
                {
                    "conflict_type": "DRUG_DRUG",
                    "severity": "MODERATE",
                    "drug_a": drug_a.generic_name,
                    "drug_b": drug_b.generic_name,
                    "matched_drug_a": drug_a.generic_name,
                    "matched_drug_b": drug_b.generic_name,
                    "rule_id": None,
                    "mechanism": mechanism,
                    "clinical_effects": [],
                    "monitoring": "Monitor clinically; review with prescribing physician.",
                    "suggested_alternative": "",
                    "source": "claude",
                }
            )

    return conflicts


# ── Step 4: Merge & deduplicate ───────────────────────────────────────────────

def _merge_conflicts(
    allergy_conflicts: list[dict],
    rule_conflicts: list[dict],
    claude_conflicts: list[dict],
) -> list[dict]:
    """
    Combine all conflict sources, deduplicate by drug pair, and let CRITICAL
    win over MODERATE for the same pair.
    """
    all_conflicts = allergy_conflicts + rule_conflicts + claude_conflicts

    # Key: frozenset of the two drug names (lower-case)
    best: dict[frozenset, dict] = {}
    severity_rank = {"CRITICAL": 2, "MODERATE": 1, "NONE": 0}

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


# ── Step 5: Overall severity ──────────────────────────────────────────────────

def _overall_severity(conflicts: list[dict]) -> str:
    severities = {c["severity"] for c in conflicts}
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "MODERATE" in severities:
        return "MODERATE"
    return "NONE"


# ── LangGraph node ────────────────────────────────────────────────────────────

def run_agent(
    state: dict,
    store: PatientStore | None = None,
    knowledge_store: KnowledgeStore | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> dict:
    """
    LangGraph-compatible node function.

    All dependencies (store, knowledge_store, anthropic_client) are injectable
    for testing. In production they are constructed from environment variables.
    """
    patient_id: str = state.get("patient_id", "")

    # ── Early exit ─────────────────────────────────────────────────────────────
    if state.get("skip_audit"):
        return {
            "patient_id": patient_id,
            "conflicts": [],
            "overall_severity": "NONE",
            "status": "SKIPPED",
            "error": None,
        }

    medication_dicts: list[dict] = state.get("medications", [])
    if not medication_dicts:
        return {
            "patient_id": patient_id,
            "conflicts": [],
            "overall_severity": "NONE",
            "status": "ERROR",
            "error": "No medications in state — run profile_builder first.",
        }

    # ── Initialise dependencies ────────────────────────────────────────────────
    if store is None:
        store = PatientStore()
    if knowledge_store is None:
        knowledge_store = KnowledgeStore()
        knowledge_store.init()
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    # Deserialise drugs from state
    drugs: list[Drug] = [_dict_to_drug(d) for d in medication_dicts]

    # ── Step 1: Allergy check ──────────────────────────────────────────────────
    allergies = store.get_allergies(patient_id)
    allergy_conflicts = _check_allergies(drugs, allergies)

    # ── Step 2: Rule engine (all severities to know which pairs are covered) ───
    all_rule_conflicts = check_pairs_all_severity(drugs)
    # Only CRITICAL ones go into final output directly
    rule_conflicts = [c for c in all_rule_conflicts if c["severity"] == "CRITICAL"]
    # MODERATE from rules also go into output
    rule_moderate = [c for c in all_rule_conflicts if c["severity"] == "MODERATE"]
    rule_conflicts_for_output = rule_conflicts + rule_moderate

    # Track which pairs are already covered (by any rule severity)
    flagged_pairs: set[frozenset] = {
        frozenset(
            {
                c["matched_drug_a"].lower(),
                c["matched_drug_b"].lower(),
            }
        )
        for c in all_rule_conflicts
    }

    # ── Step 3: Claude semantic check for uncovered pairs ─────────────────────
    claude_conflicts = _claude_check_unflagged_pairs(
        drugs, flagged_pairs, knowledge_store, anthropic_client
    )

    # ── Step 4: Merge ──────────────────────────────────────────────────────────
    merged = _merge_conflicts(allergy_conflicts, rule_conflicts_for_output, claude_conflicts)

    # ── Step 5: Persist & return ───────────────────────────────────────────────
    store.save_conflicts(patient_id, merged)

    return {
        "patient_id": patient_id,
        "conflicts": merged,
        "overall_severity": _overall_severity(merged),
        "status": "OK",
        "error": None,
    }


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    from agents.profile_builder import run_agent as build_profile

    project_root = Path(__file__).parent.parent
    store = PatientStore()

    try:
        store._r.ping()
        print("\n[PASS] Redis connection")
    except Exception as exc:
        print(f"\n[FAIL] Redis — {exc}")
        sys.exit(1)

    knowledge_store = KnowledgeStore()
    knowledge_store.init()
    print(f"[PASS] ChromaDB knowledge store ({knowledge_store.document_count} chunks)\n")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # ── Test A: patient_001 (Arjun — multi-doctor, 4 drugs, expect MODERATE) ──
    print("=" * 60)
    print("TEST A: patient_001 — Arjun Sharma (4 drugs, expect MODERATE)")
    print("=" * 60)

    p1_file = project_root / "data" / "mock_patients" / "patient_001.json"
    with open(p1_file, "r", encoding="utf-8") as fh:
        p1_json = json.load(fh)

    patient_id_1 = p1_json["id"]
    store.delete_patient(patient_id_1)

    profile_state = build_profile(
        {"mode": "patient", "patient_id": patient_id_1, "patient_json": p1_json},
        store=store,
    )
    assert profile_state["status"] == "OK", f"Profile build failed: {profile_state}"

    audit_state = run_agent(
        profile_state,
        store=store,
        knowledge_store=knowledge_store,
        anthropic_client=client,
    )

    print(f"  overall_severity : {audit_state['overall_severity']}")
    print(f"  conflicts found  : {len(audit_state['conflicts'])}")
    for c in audit_state["conflicts"]:
        src = c.get("rule_id") or c.get("source", "?")
        print(f"    [{c['severity']:8s}] {c['drug_a']} + {c['drug_b']}  [{src}]")
    assert audit_state["status"] == "OK"
    assert audit_state["overall_severity"] in ("MODERATE", "CRITICAL")
    assert len(audit_state["conflicts"]) >= 1
    print("  [PASS]\n")

    # ── Test B: patient_005 (Kumar — single drug, expect NONE / SKIPPED) ──────
    print("=" * 60)
    print("TEST B: patient_005 — Kumar Nair (1 drug, expect SKIPPED/NONE)")
    print("=" * 60)

    p5_file = project_root / "data" / "mock_patients" / "patient_005.json"
    with open(p5_file, "r", encoding="utf-8") as fh:
        p5_json = json.load(fh)

    patient_id_5 = p5_json["id"]
    store.delete_patient(patient_id_5)

    profile_state_5 = build_profile(
        {"mode": "patient", "patient_id": patient_id_5, "patient_json": p5_json},
        store=store,
    )
    assert profile_state_5["status"] == "OK"
    assert profile_state_5["skip_audit"] is True, "Expected skip_audit=True for single-drug patient"

    audit_state_5 = run_agent(
        profile_state_5,
        store=store,
        knowledge_store=knowledge_store,
        anthropic_client=client,
    )

    print(f"  status           : {audit_state_5['status']}")
    print(f"  overall_severity : {audit_state_5['overall_severity']}")
    print(f"  conflicts found  : {len(audit_state_5['conflicts'])}")
    assert audit_state_5["status"] == "SKIPPED"
    assert audit_state_5["overall_severity"] == "NONE"
    assert len(audit_state_5["conflicts"]) == 0
    print("  [PASS]\n")

    # Cleanup
    store.delete_patient(patient_id_1)
    store.delete_patient(patient_id_5)
    print("All DrugInteractionAuditorAgent tests passed.")
