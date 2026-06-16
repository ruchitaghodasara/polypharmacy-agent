"""
Pydantic request and response models for the Polypharmacy Safety Agent API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class PatientScanRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, description="Unique patient identifier")
    prescriptions: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="List of prescription dicts in FHIR-lite format"
    )
    patient_name: Optional[str] = Field(None, description="Display name for reports")

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "patient-001",
            "patient_name": "Arjun Sharma",
            "prescriptions": [
                {
                    "drug_name": "Lisinopril", "dose": "10mg", "frequency": "once daily",
                    "prescribing_doctor": "Dr. Anjali Mehta", "condition": "Hypertension",
                    "prescription_date": "2023-05-20", "active_status": True,
                }
            ],
        }
    }}


class DoctorCheckRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    drug_name: str = Field(..., min_length=1)
    dose: str = Field(..., min_length=1)
    frequency: str = Field(default="once daily")
    prescribing_doctor: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    prescription_date: Optional[str] = Field(None, description="ISO-8601 date, defaults to today")

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "patient-001",
            "drug_name": "Aspirin",
            "dose": "75mg",
            "frequency": "once daily",
            "prescribing_doctor": "Dr. Cardiac Specialist",
            "condition": "Cardiovascular prevention",
            "prescription_date": "2024-06-01",
        }
    }}


# ── Response models ────────────────────────────────────────────────────────────

class ScanResult(BaseModel):
    patient_id: str
    overall_severity: str
    conflicts: List[Dict[str, Any]]
    reports: Dict[str, Any]
    processing_time_ms: int
    audit_trail: List[Dict[str, Any]]


class CheckResult(BaseModel):
    patient_id: str
    safe_to_prescribe: bool
    severity: str
    conflicts: List[Dict[str, Any]]
    physician_report: str
    action_required: str


class MedicationProfile(BaseModel):
    patient_id: str
    medications: List[Dict[str, Any]]
    medication_count: int


class AuditTrailResponse(BaseModel):
    patient_id: str
    entries: List[Dict[str, Any]]
    entry_count: int
