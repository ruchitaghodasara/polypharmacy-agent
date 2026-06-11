"""
FHIR-lite patient parser.

Reads the project's mock patient JSON format (data/mock_patients/*.json) and
produces Drug dataclass instances with brand-name normalisation applied.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Drug:
    drug_name: str                   # name as written on the prescription
    generic_name: str                # resolved generic (may equal drug_name)
    dose: str
    frequency: str
    prescribing_doctor: str
    condition: str
    prescription_date: str           # ISO-8601 string, e.g. "2023-09-05"
    active_status: bool
    is_normalised: bool              # True when generic_name != drug_name

    def __str__(self) -> str:
        normalised_tag = f" [normalised from '{self.drug_name}']" if self.is_normalised else ""
        status = "ACTIVE" if self.active_status else "INACTIVE"
        return (
            f"{self.generic_name}{normalised_tag} {self.dose} {self.frequency} "
            f"| {status} | Dr: {self.prescribing_doctor} | Condition: {self.condition} "
            f"| Prescribed: {self.prescription_date}"
        )


# ── Synonym loader (cached at module level) ───────────────────────────────────

_SYNONYM_CACHE: dict[str, dict[str, str]] = {}


def _load_synonyms(synonyms_path: str | Path) -> dict[str, str]:
    path = str(synonyms_path)
    if path not in _SYNONYM_CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Store as lower-case keys for case-insensitive lookup
        _SYNONYM_CACHE[path] = {k.lower(): v for k, v in data["synonyms"].items()}
    return _SYNONYM_CACHE[path]


# ── Public helpers ─────────────────────────────────────────────────────────────

def normalise_drug(drug_name: str, synonyms_path: str | Path) -> str:
    """Return the generic name for *drug_name*, or *drug_name* unchanged if not found."""
    synonyms = _load_synonyms(synonyms_path)
    return synonyms.get(drug_name.lower(), drug_name)


def parse_fhir_patient(
    json_dict: dict,
    synonyms_path: Optional[str | Path] = None,
) -> List[Drug]:
    """
    Parse a mock FHIR patient dict and return one Drug per active medication.

    *synonyms_path* defaults to <project_root>/data/drug_synonyms.json resolved
    relative to this file's location if not supplied.
    """
    if synonyms_path is None:
        synonyms_path = Path(__file__).parent.parent / "data" / "drug_synonyms.json"

    drugs: List[Drug] = []
    for med in json_dict.get("medications", []):
        raw_name: str = med["drug_name"]
        generic = normalise_drug(raw_name, synonyms_path)
        drugs.append(
            Drug(
                drug_name=raw_name,
                generic_name=generic,
                dose=med.get("dose", ""),
                frequency=med.get("frequency", ""),
                prescribing_doctor=med.get("prescribing_doctor", ""),
                condition=med.get("condition", ""),
                prescription_date=med.get("prescription_date", ""),
                active_status=bool(med.get("active_status", True)),
                is_normalised=generic.lower() != raw_name.lower(),
            )
        )
    return drugs


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    patient_file = project_root / "data" / "mock_patients" / "patient_001.json"

    if not patient_file.exists():
        print(f"ERROR: {patient_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(patient_file, "r", encoding="utf-8") as fh:
        patient_json = json.load(fh)

    name = patient_json["name"][0]
    full_name = f"{name['given'][0]} {name['family']}"
    print(f"\nPatient: {full_name}  (id={patient_json['id']})\n")
    print("-" * 70)

    drugs = parse_fhir_patient(patient_json)
    for i, drug in enumerate(drugs, 1):
        print(f"  [{i}] {drug}")

    print(f"\nTotal medications parsed: {len(drugs)}")
    normalised = [d for d in drugs if d.is_normalised]
    if normalised:
        print(f"Brand names normalised : {[d.drug_name for d in normalised]}")
    else:
        print("Brand names normalised : none")
