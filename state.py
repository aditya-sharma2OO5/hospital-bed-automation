"""
state.py — Shared state TypedDict for the Hospital Bed Request LangGraph.

Every node reads from and writes to this dict. Fields are all Optional so
any node can return only the keys it updates; LangGraph merges partials.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class HospitalState(TypedDict, total=False):
    # ── Patient details (provided at intake) ──────────────────────────────
    patient_name: str
    patient_age: Optional[float]          # None triggers validation error
    patient_mobile: str
    patient_email: str
    symptoms: str
    location: str                          # Delhi zone, e.g. "South Delhi"

    # ── Triage output (set by ai_triage_node + validate_triage_node) ──────
    urgency_level: str                     # "High" | "Medium" | "Low"
    bed_type_required: str                 # "ICU" | "General" | "Standard"

    # ── Hospital search (set by hospital_search_node) ──────────────────────
    hospital_search_results: List[Dict[str, Any]]

    # ── Availability check (set by check_availability_node) ───────────────
    selected_hospital: Optional[Dict[str, Any]]
    available_beds_count: int

    # ── Reservation (set by reservation_node) ─────────────────────────────
    reservation_id: Optional[str]          # "RES-YYYYMMDD-XXXXXX"

    # ── Notification (set by confirmation / no_beds_notification nodes) ────
    confirmation_sent: bool

    # ── Lifecycle metadata ─────────────────────────────────────────────────
    timestamp: str                         # ISO 8601 UTC, set at intake
    error_message: Optional[str]           # set on validation failure
