"""
Evaluation runner for the Polypharmacy Safety Agent.

Loads all 6 mock patients, runs the full LangGraph pipeline for each, and
compares outputs against golden_dataset.json to compute:

  - Critical recall    : fraction of CRITICAL patients correctly identified
  - Conflict F1        : precision/recall/F1 over detected drug-pair conflicts
  - Normalisation acc  : patient-006 Brufen → Ibuprofen must succeed
  - Average latency    : mean wall-clock seconds per patient
  - Audit completeness : all 6 patients have SQLite audit entries

Prints a formatted report to stdout and writes it to
evaluation/results/eval_report.txt.

Exit code 1 if critical recall < 100 %.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── Bootstrap path so imports resolve from project root ───────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from graph.safety_graph import run_patient_flow, _get_store, _get_audit  # noqa: E402
from audit.audit_logger import AuditLogger  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_PATIENTS_DIR = _PROJECT_ROOT / "data" / "mock_patients"
_GOLDEN_PATH  = _PROJECT_ROOT / "data" / "golden_dataset.json"
_RESULTS_DIR  = _PROJECT_ROOT / "evaluation" / "results"
_REPORT_PATH  = _RESULTS_DIR / "eval_report.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_golden() -> dict[str, dict]:
    """Return golden records keyed by patient_id."""
    with open(_GOLDEN_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return {p["patient_id"]: p for p in data["patients"]}


def _load_patient_files() -> list[dict]:
    """Return all patient FHIR dicts, sorted by filename."""
    files = sorted(_PATIENTS_DIR.glob("patient_*.json"))
    patients = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            patients.append(json.load(fh))
    return patients


def _normalise_pair(drug_a: str, drug_b: str) -> frozenset:
    return frozenset({drug_a.strip().lower(), drug_b.strip().lower()})


def _detected_pairs(conflicts: list[dict]) -> set[frozenset]:
    """Return the set of detected drug pairs as frozensets of lowercase names."""
    pairs = set()
    for c in conflicts:
        a = c.get("matched_drug_a") or c.get("drug_a", "")
        b = c.get("matched_drug_b") or c.get("drug_b", "")
        if a and b:
            pairs.add(_normalise_pair(a, b))
    return pairs


def _expected_pairs(golden: dict) -> set[frozenset]:
    return {
        _normalise_pair(c["drug_a"], c["drug_b"])
        for c in golden.get("expected_conflicts", [])
    }


def _severity_label(overall: str, golden: dict) -> str:
    expected = golden["expected_overall_severity"]
    match = "✓" if overall == expected else "✗"
    return f"{match} got={overall} expected={expected}"


# ── Per-patient evaluation ────────────────────────────────────────────────────

def evaluate_patient(
    patient_json: dict,
    golden: dict,
    audit_logger: AuditLogger,
) -> dict:
    """Run one patient through the pipeline and return an evaluation dict."""
    patient_id = patient_json["id"]
    name = patient_json.get("name", [{}])[0]
    patient_name = f"{name.get('given', [''])[0]} {name.get('family', '')}".strip()

    # Clean up prior state so the run is repeatable
    store = _get_store()
    store.delete_patient(patient_id)
    audit_logger.clear_patient(patient_id)

    t0 = time.perf_counter()
    try:
        final = run_patient_flow(
            patient_id=patient_id,
            patient_json=patient_json,
            patient_name=patient_name,
            thread_id=f"eval-{patient_id}-{int(t0)}",
            human_decision="approve",
        )
        error = None
    except Exception as exc:
        final = {}
        error = str(exc)
    latency = time.perf_counter() - t0

    # Compute conflict metrics
    detected    = _detected_pairs(final.get("conflicts", []))
    expected    = _expected_pairs(golden)
    tp          = len(detected & expected)
    fp          = len(detected - expected)
    fn          = len(expected - detected)

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 1.0   # no FP → perfect precision
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 1.0   # no expected conflicts → 1.0

    f1          = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    overall_sev = final.get("overall_severity", "NONE")
    expected_sev = golden["expected_overall_severity"]
    severity_correct = (overall_sev == expected_sev)

    # Normalisation check (patient-006 only)
    norm_ok: bool | None = None
    if golden.get("normalisation_required"):
        medications = final.get("medications", [])
        generic_names = {m.get("generic_name", "").lower() for m in medications}
        norm_ok = "ibuprofen" in generic_names

    # Audit completeness
    trail = audit_logger.get_audit_trail(patient_id)
    has_audit = len(trail) > 0

    return {
        "patient_id":       patient_id,
        "patient_name":     patient_name,
        "overall_severity": overall_sev,
        "expected_severity":expected_sev,
        "severity_correct": severity_correct,
        "detected_pairs":   detected,
        "expected_pairs":   expected,
        "tp": tp, "fp": fp, "fn": fn,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "latency_s":        latency,
        "norm_ok":          norm_ok,
        "has_audit":        has_audit,
        "audit_entries":    len(trail),
        "reports_present":  all(k in final.get("reports", {}) for k in ("patient", "coordinator", "physician")),
        "error":            error,
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def compute_aggregate(results: list[dict]) -> dict:
    # Critical recall
    critical_patients = [r for r in results if r["expected_severity"] == "CRITICAL"]
    critical_correct  = [r for r in critical_patients if r["severity_correct"]]
    critical_recall   = (
        len(critical_correct) / len(critical_patients)
        if critical_patients else 1.0
    )

    # Micro-averaged conflict F1
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1        = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    # Normalisation
    norm_results = [r for r in results if r["norm_ok"] is not None]
    norm_acc     = (
        sum(1 for r in norm_results if r["norm_ok"]) / len(norm_results)
        if norm_results else None
    )

    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    audit_complete = all(r["has_audit"] for r in results)

    severity_accuracy = sum(1 for r in results if r["severity_correct"]) / len(results) if results else 0.0

    return {
        "critical_recall":    critical_recall,
        "micro_precision":    micro_precision,
        "micro_recall":       micro_recall,
        "micro_f1":           micro_f1,
        "norm_accuracy":      norm_acc,
        "avg_latency_s":      avg_latency,
        "audit_complete":     audit_complete,
        "severity_accuracy":  severity_accuracy,
        "n_patients":         len(results),
        "n_errors":           sum(1 for r in results if r["error"]),
    }


# ── Report formatter ──────────────────────────────────────────────────────────

def format_report(results: list[dict], agg: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  POLYPHARMACY SAFETY AGENT — EVALUATION REPORT")
    lines.append(f"  Generated : {ts}")
    lines.append("=" * 72)
    lines.append("")

    # Per-patient table
    lines.append("PER-PATIENT RESULTS")
    lines.append("-" * 72)
    hdr = f"{'Patient':<16} {'Severity':>10} {'Match':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6} {'Lat(s)':>7} {'Audit':>6}"
    lines.append(hdr)
    lines.append("-" * 72)

    for r in results:
        match_sym = "✓" if r["severity_correct"] else "✗"
        norm_note = ""
        if r["norm_ok"] is True:
            norm_note = " [NORM✓]"
        elif r["norm_ok"] is False:
            norm_note = " [NORM✗]"
        audit_sym = "✓" if r["has_audit"] else "✗"
        err_note  = f" ERROR: {r['error'][:40]}" if r["error"] else ""

        row = (
            f"{r['patient_id']:<16} "
            f"{r['overall_severity']:>10} "
            f"{match_sym:>6} "
            f"{r['tp']:>4} "
            f"{r['fp']:>4} "
            f"{r['fn']:>4} "
            f"{r['f1']:>6.3f} "
            f"{r['latency_s']:>7.2f} "
            f"{audit_sym:>6}"
            f"{norm_note}{err_note}"
        )
        lines.append(row)

    lines.append("-" * 72)
    lines.append("")

    # Conflict pair detail
    lines.append("CONFLICT PAIR DETAIL")
    lines.append("-" * 72)
    for r in results:
        if r["expected_pairs"] or r["detected_pairs"]:
            lines.append(f"  {r['patient_id']} ({r['patient_name']})")
            for pair in sorted(r["expected_pairs"], key=str):
                hit = pair in r["detected_pairs"]
                sym = "TP" if hit else "FN"
                parts = sorted(pair)
                lines.append(f"    [{sym}] {parts[0]} + {parts[1]}")
            for pair in sorted(r["detected_pairs"] - r["expected_pairs"], key=str):
                parts = sorted(pair)
                lines.append(f"    [FP] {parts[0]} + {parts[1]}")
    lines.append("")

    # Aggregate metrics
    lines.append("AGGREGATE METRICS")
    lines.append("=" * 72)

    cr_pct   = agg["critical_recall"] * 100
    cr_pass  = "PASS ✓" if agg["critical_recall"] >= 1.0 else "FAIL ✗  ← EXIT CODE 1"
    lines.append(f"  Critical Recall          : {cr_pct:6.1f}%   [{cr_pass}]  (target: 100%)")

    lines.append(f"  Conflict Micro-F1        : {agg['micro_f1']*100:6.1f}%")
    lines.append(f"    Precision              : {agg['micro_precision']*100:6.1f}%")
    lines.append(f"    Recall                 : {agg['micro_recall']*100:6.1f}%")

    if agg["norm_accuracy"] is not None:
        na_pct  = agg["norm_accuracy"] * 100
        na_pass = "PASS ✓" if agg["norm_accuracy"] >= 1.0 else "FAIL ✗"
        lines.append(f"  Normalisation Accuracy   : {na_pct:6.1f}%   [{na_pass}]  (patient-006 Brufen→Ibuprofen)")

    lines.append(f"  Severity Accuracy        : {agg['severity_accuracy']*100:6.1f}%")
    lines.append(f"  Average Latency          : {agg['avg_latency_s']:6.2f} s / patient")

    audit_sym = "PASS ✓" if agg["audit_complete"] else "FAIL ✗"
    lines.append(f"  Audit Log Completeness   : {audit_sym}")
    lines.append(f"  Patients evaluated       : {agg['n_patients']}")
    lines.append(f"  Pipeline errors          : {agg['n_errors']}")
    lines.append("=" * 72)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("Loading golden dataset …")
    golden_by_id = _load_golden()

    print("Loading patient files …")
    patient_files = _load_patient_files()
    print(f"  {len(patient_files)} patients found\n")

    audit_logger = AuditLogger()

    results: list[dict] = []
    for pj in patient_files:
        pid = pj["id"]
        golden = golden_by_id.get(pid)
        if golden is None:
            print(f"  [SKIP] {pid} — no golden record")
            continue
        print(f"  Running {pid} …", end=" ", flush=True)
        r = evaluate_patient(pj, golden, audit_logger)
        results.append(r)
        sev_sym = "✓" if r["severity_correct"] else "✗"
        print(f"done  ({r['latency_s']:.1f}s)  severity {sev_sym}  F1={r['f1']:.2f}")

    print()
    agg = compute_aggregate(results)
    report = format_report(results, agg)

    print(report)

    # Save to file
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {_REPORT_PATH}")

    # Exit code 1 if critical recall < 100 %
    if agg["critical_recall"] < 1.0:
        print("\nERROR: Critical recall below 100% — exiting with code 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
