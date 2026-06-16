"""
Deterministic rule engine for drug-drug interaction checking.

Loads data/knowledge/interaction_rules.json and checks all pairwise
combinations of a patient's active medications against the rule set.
Only CRITICAL rules are returned; MODERATE pairs are left for the
Claude-backed semantic check in the interaction auditor.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import List

from tools.fhir_parser import Drug

_DEFAULT_RULES_PATH = (
    Path(__file__).parent.parent / "data" / "knowledge" / "interaction_rules.json"
)

# Module-level cache: rules_path → list of rule dicts
_RULES_CACHE: dict[str, list[dict]] = {}


def _load_rules(rules_path: str | Path = _DEFAULT_RULES_PATH) -> list[dict]:
    key = str(rules_path)
    if key not in _RULES_CACHE:
        with open(rules_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _RULES_CACHE[key] = data["interactions"]
    return _RULES_CACHE[key]


def _names(drug: Drug) -> set[str]:
    """Return lower-case lookup set: both the raw name and the generic name."""
    return {drug.drug_name.lower(), drug.generic_name.lower()}


def check_pairs(
    drug_list: List[Drug],
    rules_path: str | Path = _DEFAULT_RULES_PATH,
) -> List[dict]:
    """
    Generate all pairwise combinations of *drug_list* (active medications only)
    and return every CRITICAL match found in the rule set.

    Each result dict contains:
        conflict_type       str   always "DRUG_DRUG"
        severity            str   always "CRITICAL" (only CRITICAL rules returned)
        drug_a              str   generic name from the rule
        drug_b              str   generic name from the rule
        matched_drug_a      str   name as stored on the patient record
        matched_drug_b      str   name as stored on the patient record
        rule_id             str   e.g. "IR-001"
        mechanism           str   plain-English explanation
        clinical_effects    list
        monitoring          str
        suggested_alternative str
        source              str   always "rules"
    """
    rules = _load_rules(rules_path)
    active = [d for d in drug_list if d.active_status]
    conflicts: list[dict] = []

    for drug_a, drug_b in combinations(active, 2):
        names_a = _names(drug_a)
        names_b = _names(drug_b)

        for rule in rules:
            if rule.get("severity") != "CRITICAL":
                continue

            ra = rule["drug_a"].lower()
            rb = rule["drug_b"].lower()

            # Match in either order
            if (ra in names_a and rb in names_b) or (ra in names_b and rb in names_a):
                conflicts.append(
                    {
                        "conflict_type": "DRUG_DRUG",
                        "severity": "CRITICAL",
                        "drug_a": rule["drug_a"],
                        "drug_b": rule["drug_b"],
                        "matched_drug_a": drug_a.generic_name,
                        "matched_drug_b": drug_b.generic_name,
                        "rule_id": rule["id"],
                        "mechanism": rule["mechanism"],
                        "clinical_effects": rule.get("clinical_effects", []),
                        "monitoring": rule.get("monitoring", ""),
                        "suggested_alternative": rule.get("suggested_alternative", ""),
                        "source": "rules",
                    }
                )

    return conflicts


def check_pairs_all_severity(
    drug_list: List[Drug],
    rules_path: str | Path = _DEFAULT_RULES_PATH,
) -> List[dict]:
    """
    Same as check_pairs() but returns ALL severity levels (CRITICAL + MODERATE).
    Used internally by the auditor to identify which pairs are already covered
    by deterministic rules so they don't get re-checked by Claude.
    """
    rules = _load_rules(rules_path)
    active = [d for d in drug_list if d.active_status]
    conflicts: list[dict] = []

    for drug_a, drug_b in combinations(active, 2):
        names_a = _names(drug_a)
        names_b = _names(drug_b)

        for rule in rules:
            ra = rule["drug_a"].lower()
            rb = rule["drug_b"].lower()

            if (ra in names_a and rb in names_b) or (ra in names_b and rb in names_a):
                conflicts.append(
                    {
                        "conflict_type": "DRUG_DRUG",
                        "severity": rule["severity"],
                        "drug_a": rule["drug_a"],
                        "drug_b": rule["drug_b"],
                        "matched_drug_a": drug_a.generic_name,
                        "matched_drug_b": drug_b.generic_name,
                        "rule_id": rule["id"],
                        "mechanism": rule["mechanism"],
                        "clinical_effects": rule.get("clinical_effects", []),
                        "monitoring": rule.get("monitoring", ""),
                        "suggested_alternative": rule.get("suggested_alternative", ""),
                        "source": "rules",
                    }
                )

    return conflicts


if __name__ == "__main__":
    import json
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    patient_file = project_root / "data" / "mock_patients" / "patient_002.json"

    with open(patient_file, "r", encoding="utf-8") as fh:
        patient_json = json.load(fh)

    from tools.fhir_parser import parse_fhir_patient

    drugs = parse_fhir_patient(patient_json)
    print(f"\nChecking {len(drugs)} drugs for patient-002 (Warfarin + Aspirin + Atorvastatin + Metformin)\n")

    critical = check_pairs(drugs)
    print(f"CRITICAL conflicts found: {len(critical)}")
    for c in critical:
        print(f"  [{c['rule_id']}] {c['drug_a']} + {c['drug_b']}: {c['mechanism'][:80]}…")

    all_conflicts = check_pairs_all_severity(drugs)
    print(f"\nAll-severity conflicts found: {len(all_conflicts)}")
    for c in all_conflicts:
        print(f"  [{c['severity']:8s}] {c['drug_a']} + {c['drug_b']}")
