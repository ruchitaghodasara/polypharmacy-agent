"""
SQLite-backed audit logger for the Polypharmacy Safety Agent.

Every node execution, human decision, and alert dispatch is recorded here
for compliance, debugging, and downstream analytics.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, List

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "audit.db"

# --- Table setup --------------------------------------------------------------

AUDIT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     TEXT    NOT NULL,
    node_name      TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    severity       TEXT,
    action         TEXT    NOT NULL,
    state_snapshot TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_patient_id  ON audit_log (patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp   ON audit_log (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_severity    ON audit_log (severity);",
]

# --- Write operations ---------------------------------------------------------

_INSERT_SQL = """
INSERT INTO audit_log (patient_id, node_name, timestamp, severity, action, state_snapshot)
VALUES (?, ?, ?, ?, ?, ?);
"""

# --- Read operations ----------------------------------------------------------

_SELECT_PATIENT_SQL = """
SELECT id, patient_id, node_name, timestamp, severity, action, state_snapshot
FROM audit_log
WHERE patient_id = ?
ORDER BY timestamp ASC, id ASC;
"""

_SELECT_ALL_SQL = """
SELECT id, patient_id, node_name, timestamp, severity, action, state_snapshot
FROM audit_log
ORDER BY timestamp ASC, id ASC;
"""

_SELECT_CRITICAL_SQL = """
SELECT * FROM audit_log
WHERE severity = 'CRITICAL'
ORDER BY timestamp ASC;
"""

_DELETE_PATIENT_SQL = "DELETE FROM audit_log WHERE patient_id = ?;"


# ── Connection context manager ─────────────────────────────────────────────────

@contextmanager
def _connect(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_chars: int = 4096) -> str:
    """Serialise obj to JSON, truncating large state snapshots."""
    try:
        raw = json.dumps(obj, default=str)
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "…[truncated]"
        return raw
    except Exception:
        return "{}"


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a dict, deserialising state_snapshot JSON."""
    d = dict(row)
    if d.get("state_snapshot"):
        try:
            d["state_snapshot"] = json.loads(d["state_snapshot"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


# ── AuditLogger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thin SQLite wrapper for structured audit logging.

    Usage::

        logger = AuditLogger()
        logger.log_event(
            patient_id="patient-001",
            node="profile_builder_node",
            severity="INFO",
            action="medications_loaded",
            state={"drug_count": 4},
        )
        trail = logger.get_audit_trail("patient-001")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path or os.environ.get("AUDIT_DB_PATH", _DEFAULT_DB_PATH))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # --- Table setup ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create table and indexes if they don't exist."""
        with _connect(self._db_path) as conn:
            conn.execute(AUDIT_TABLE_SCHEMA)
            for idx_sql in _CREATE_INDEXES_SQL:
                conn.execute(idx_sql)

    # --- Write operations -----------------------------------------------------

    def log_event(
        self,
        patient_id: str,
        node: str,
        severity: str,
        action: str,
        state: Any = None,
    ) -> None:
        """Append one audit record to the database."""
        with _connect(self._db_path) as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    patient_id,
                    node,
                    _now_iso(),
                    severity.upper() if severity else "INFO",
                    action,
                    _safe_json(state) if state is not None else None,
                ),
            )

    # --- Read operations ------------------------------------------------------

    def get_audit_trail(self, patient_id: str) -> List[dict]:
        """Return all audit entries for *patient_id*, oldest first."""
        with _connect(self._db_path) as conn:
            rows = conn.execute(_SELECT_PATIENT_SQL, (patient_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_all_events(self) -> List[dict]:
        """Return every audit entry across all patients, oldest first."""
        with _connect(self._db_path) as conn:
            rows = conn.execute(_SELECT_ALL_SQL).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_critical_events(self) -> List[dict]:
        """Return all CRITICAL entries across all patients."""
        with _connect(self._db_path) as conn:
            rows = conn.execute(_SELECT_CRITICAL_SQL).fetchall()
        return [_row_to_dict(r) for r in rows]

    # --- Delete / TTL operations ----------------------------------------------

    def clear_patient(self, patient_id: str) -> int:
        """Delete all audit records for *patient_id*; returns row count deleted."""
        with _connect(self._db_path) as conn:
            cur = conn.execute(_DELETE_PATIENT_SQL, (patient_id,))
            return cur.rowcount


# ── Module-level convenience functions ────────────────────────────────────────

_DEFAULT_LOGGER: AuditLogger | None = None


def _get_logger() -> AuditLogger:
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = AuditLogger()
    return _DEFAULT_LOGGER


def log_event(patient_id: str, node: str, severity: str, action: str, state: Any = None) -> None:
    """Module-level shortcut — uses the process-singleton AuditLogger."""
    _get_logger().log_event(patient_id, node, severity, action, state)


def get_audit_trail(patient_id: str) -> List[dict]:
    """Module-level shortcut — returns audit trail for *patient_id*."""
    return _get_logger().get_audit_trail(patient_id)


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        test_db = tmp.name

    print(f"\n=== AuditLogger smoke-test  (db={test_db}) ===\n")
    logger = AuditLogger(db_path=test_db)

    PID = "patient-test-001"

    logger.log_event(PID, "profile_builder_node", "INFO", "medications_loaded", {"drug_count": 4})
    logger.log_event(PID, "interaction_auditor_node", "MODERATE", "interaction_detected",
                     {"drug_a": "Lisinopril", "drug_b": "Ibuprofen", "rule_id": "IR-003"})
    logger.log_event("patient-test-002", "interaction_auditor_node", "CRITICAL",
                     "interaction_detected", {"drug_a": "Warfarin", "drug_b": "Aspirin"})
    logger.log_event(PID, "report_generator_node", "INFO", "reports_generated",
                     {"status": "OK", "fallback_used": False})
    logger.log_event(PID, "human_checkpoint_node", "INFO", "human_approved", {"decision": "approve"})

    trail = logger.get_audit_trail(PID)
    print(f"[PASS] get_audit_trail returned {len(trail)} entries for {PID}")
    assert len(trail) == 4, f"Expected 4, got {len(trail)}"
    for entry in trail:
        print(f"       [{entry['severity']:8s}] {entry['node_name']} — {entry['action']}")

    assert isinstance(trail[0]["state_snapshot"], dict), "state_snapshot should be dict"
    assert trail[0]["state_snapshot"]["drug_count"] == 4
    print("[PASS] state_snapshot round-trip (JSON → dict)")

    crits = logger.get_critical_events()
    print(f"[PASS] get_critical_events returned {len(crits)} CRITICAL record(s)")
    assert len(crits) == 1
    assert crits[0]["patient_id"] == "patient-test-002"

    all_events = logger.get_all_events()
    print(f"[PASS] get_all_events returned {len(all_events)} total records")
    assert len(all_events) == 5

    deleted   = logger.clear_patient(PID)
    remaining = logger.get_audit_trail(PID)
    print(f"[PASS] clear_patient deleted {deleted} row(s); {len(remaining)} remaining")
    assert deleted == 4
    assert len(remaining) == 0

    log_event("patient-shortcut", "test_node", "NONE", "shortcut_test")
    result = get_audit_trail("patient-shortcut")
    print("[PASS] module-level log_event / get_audit_trail work")
    assert len(result) == 1

    Path(test_db).unlink(missing_ok=True)
    print("\nAll AuditLogger checks passed.")
