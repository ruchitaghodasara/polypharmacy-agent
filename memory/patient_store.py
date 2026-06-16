"""
Redis-backed patient store.

Keys:
  patient:{id}:medications   JSON array of Drug dicts
  patient:{id}:conflicts     JSON array of conflict dicts
  patient:{id}:allergies     JSON array of allergy strings / dicts
  patient:{id}:audit         Redis List — append-only event log

# NOTE: Redis stores JSON strings — deserialise on every read.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, List

import redis

from tools.fhir_parser import Drug

# ── Constants ──────────────────────────────────────────────────────────────────

_KEY_TEMPLATE        = "patient:{id}:{suffix}"
_SUFFIX_MEDICATIONS  = "medications"
_SUFFIX_CONFLICTS    = "conflicts"
_SUFFIX_ALLERGIES    = "allergies"
_SUFFIX_AUDIT        = "audit"

# WHY: TTL set to 86400 (24h) — conflicts re-audited daily on profile change.
TTL_CONFLICTS_SECONDS = 86_400

_ALL_SUFFIXES = (
    _SUFFIX_MEDICATIONS,
    _SUFFIX_CONFLICTS,
    _SUFFIX_ALLERGIES,
    _SUFFIX_AUDIT,
)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _drug_to_dict(drug: Drug) -> dict:
    """Convert a Drug dataclass to a JSON-serialisable dict."""
    return {
        "drug_name":          drug.drug_name,
        "generic_name":       drug.generic_name,
        "dose":               drug.dose,
        "frequency":          drug.frequency,
        "prescribing_doctor": drug.prescribing_doctor,
        "condition":          drug.condition,
        "prescription_date":  drug.prescription_date,
        "active_status":      drug.active_status,
        "is_normalised":      drug.is_normalised,
    }


def _dict_to_drug(d: dict) -> Drug:
    """Reconstruct a Drug dataclass from a stored dict."""
    return Drug(
        drug_name=d["drug_name"],
        generic_name=d["generic_name"],
        dose=d["dose"],
        frequency=d["frequency"],
        prescribing_doctor=d["prescribing_doctor"],
        condition=d["condition"],
        prescription_date=d["prescription_date"],
        active_status=bool(d["active_status"]),
        is_normalised=bool(d["is_normalised"]),
    )


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── PatientStore ───────────────────────────────────────────────────────────────

class PatientStore:
    """
    Thin Redis wrapper for per-patient medication, conflict, and allergy data.

    Pass *redis_url* explicitly, or set REDIS_URL / UPSTASH_REDIS_URL in the
    environment.  Falls back to REDIS_HOST / REDIS_PORT / REDIS_PASSWORD /
    REDIS_SSL when no URL is present.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url or os.environ.get("REDIS_URL") or os.environ.get("UPSTASH_REDIS_URL")
        if url:
            self._r = redis.from_url(url, decode_responses=True, socket_connect_timeout=5)
        else:
            self._r = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD") or None,
                ssl=os.environ.get("REDIS_SSL", "false").lower() == "true",
                decode_responses=True,
                socket_connect_timeout=5,
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _key(patient_id: str, suffix: str) -> str:
        """Build a namespaced Redis key."""
        return f"patient:{patient_id}:{suffix}"

    def _save_to_redis(self, key: str, data: Any) -> None:
        """Serialise *data* to JSON and write to *key*."""
        self._r.set(key, json.dumps(data))

    def _audit(self, patient_id: str, event: str, detail: Any = None) -> None:
        """Append one audit event to the patient's Redis list."""
        entry = json.dumps({"ts": _now_iso(), "event": event, "detail": detail})
        self._r.rpush(self._key(patient_id, _SUFFIX_AUDIT), entry)

    # --- Write operations -----------------------------------------------------

    # [REFS: agents/profile_builder.py > run_agent]
    def save_medications(self, patient_id: str, drugs: List[Drug]) -> None:
        """Overwrite the stored medication list for *patient_id*."""
        self._save_to_redis(
            self._key(patient_id, _SUFFIX_MEDICATIONS),
            [_drug_to_dict(d) for d in drugs],
        )
        self._audit(patient_id, "medications_saved", {"count": len(drugs)})

    def save_conflicts(self, patient_id: str, conflicts: List[dict]) -> None:
        """Persist detected interaction conflicts for *patient_id*."""
        self._save_to_redis(self._key(patient_id, _SUFFIX_CONFLICTS), conflicts)
        self._audit(patient_id, "conflicts_saved", {"count": len(conflicts)})

    def save_allergies(self, patient_id: str, allergies: List[Any]) -> None:
        """Persist allergy records (strings or {substance, reaction} dicts)."""
        self._save_to_redis(self._key(patient_id, _SUFFIX_ALLERGIES), allergies)
        self._audit(patient_id, "allergies_saved", {"count": len(allergies)})

    # --- Read operations ------------------------------------------------------

    def get_medications(self, patient_id: str) -> List[Drug]:
        """Return the stored Drug list, or [] if none saved yet."""
        raw = self._r.get(self._key(patient_id, _SUFFIX_MEDICATIONS))
        return [_dict_to_drug(d) for d in json.loads(raw)] if raw else []

    def get_conflicts(self, patient_id: str) -> List[dict]:
        """Return stored conflicts, or [] if none saved yet."""
        raw = self._r.get(self._key(patient_id, _SUFFIX_CONFLICTS))
        return json.loads(raw) if raw else []

    def get_allergies(self, patient_id: str) -> List[Any]:
        """Return stored allergy records, or [] if none saved yet."""
        raw = self._r.get(self._key(patient_id, _SUFFIX_ALLERGIES))
        return json.loads(raw) if raw else []

    def get_audit_log(self, patient_id: str) -> List[dict]:
        """Return all audit entries in insertion order."""
        entries = self._r.lrange(self._key(patient_id, _SUFFIX_AUDIT), 0, -1)
        return [json.loads(e) for e in entries]

    # --- Delete / TTL operations ----------------------------------------------

    def delete_patient(self, patient_id: str) -> None:
        """Remove all Redis keys for *patient_id* (useful in tests)."""
        for suffix in _ALL_SUFFIXES:
            self._r.delete(self._key(patient_id, suffix))


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    from tools.fhir_parser import parse_fhir_patient

    project_root = Path(__file__).parent.parent
    patient_file = project_root / "data" / "mock_patients" / "patient_001.json"

    with open(patient_file, encoding="utf-8") as fh:
        patient_json = json.load(fh)

    patient_id = patient_json["id"]
    allergies  = patient_json.get("allergies", [])
    drugs      = parse_fhir_patient(patient_json)

    print(f"\n=== PatientStore smoke-test  (patient_id={patient_id}) ===\n")

    try:
        store = PatientStore()
        store._r.ping()
        print("[PASS] Redis connection")
    except Exception as exc:
        print(f"[FAIL] Redis connection — {exc}")
        sys.exit(1)

    store.delete_patient(patient_id)

    store.save_medications(patient_id, drugs)
    print(f"[PASS] save_medications  ({len(drugs)} drugs)")

    store.save_allergies(patient_id, allergies)
    print(f"[PASS] save_allergies    ({len(allergies)} entries)")

    mock_conflicts = [
        {"drug_a": "Lisinopril", "drug_b": "Ibuprofen", "severity": "MODERATE", "rule_id": "IR-003"}
    ]
    store.save_conflicts(patient_id, mock_conflicts)
    print(f"[PASS] save_conflicts    ({len(mock_conflicts)} conflict)")

    retrieved_drugs = store.get_medications(patient_id)
    assert len(retrieved_drugs) == len(drugs), "medication count mismatch"
    assert retrieved_drugs[0].generic_name == drugs[0].generic_name, "name mismatch"
    print(f"[PASS] get_medications   ({len(retrieved_drugs)} drugs returned)")

    retrieved_conflicts = store.get_conflicts(patient_id)
    assert len(retrieved_conflicts) == 1
    print(f"[PASS] get_conflicts     ({len(retrieved_conflicts)} conflict returned)")

    retrieved_allergies = store.get_allergies(patient_id)
    print(f"[PASS] get_allergies     ({len(retrieved_allergies)} entries returned)")

    audit = store.get_audit_log(patient_id)
    print(f"[PASS] audit_log         ({len(audit)} entries)")
    for entry in audit:
        print(f"       {entry['ts']}  {entry['event']}  {entry['detail']}")

    store.delete_patient(patient_id)
    print("\nAll PatientStore checks passed.")
