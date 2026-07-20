"""
nodes.py — LangGraph node functions. Each takes and returns a (partial)
HospitalState dict, per LangGraph conventions.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from state import HospitalState
import tools

logger = logging.getLogger("hospital_bed_agent.nodes")


def patient_intake_node(state: HospitalState) -> Dict[str, Any]:
    """Validate patient data and stamp the request timestamp."""
    is_valid, error = tools.validate_patient_input(state)
    if not is_valid:
        logger.error("Patient intake validation failed: %s", error)
        return {"error_message": error}

    return {
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def ai_triage_node(state: HospitalState) -> Dict[str, Any]:
    """Call Groq to classify symptoms. Falls back to defaults on failure."""
    if state.get("error_message"):
        return {}

    raw = tools.classify_symptoms_groq(state["symptoms"])
    return {"urgency_level": raw["urgency_level"], "bed_type_required": raw["bed_type_required"]}


def validate_triage_node(state: HospitalState) -> Dict[str, Any]:
    """Sanitize/normalize the raw triage output."""
    if state.get("error_message"):
        return {}

    sanitized = tools.sanitize_triage_response(
        {
            "urgency_level": state.get("urgency_level", ""),
            "bed_type_required": state.get("bed_type_required", ""),
        }
    )
    return sanitized


def hospital_search_node(state: HospitalState) -> Dict[str, Any]:
    """Query the hospital database by zone."""
    if state.get("error_message"):
        return {}

    results = tools.search_hospitals_by_zone(state["location"])
    if not results:
        logger.info("No hospitals found in zone %s", state["location"])
    return {"hospital_search_results": results}


def check_availability_node(state: HospitalState) -> Dict[str, Any]:
    """
    Select the best-matched hospital from the search results based on the
    bed type assigned by triage, then record its available bed count.

    Sorting priority:
      - bed_type_required == "ICU"                 -> sort by icu_beds DESC, then available_beds DESC
      - bed_type_required == "General" / "Standard" -> sort by general_beds DESC, then available_beds DESC

    Hospitals with zero total available beds are ranked last so the router
    can still trigger Path B (no-beds notification) when none have capacity.

    Routing itself happens via the conditional edge in graph.py, which reads
    `available_beds_count` from this output.
    """
    if state.get("error_message"):
        return {"available_beds_count": 0, "selected_hospital": None}

    results = state.get("hospital_search_results") or []
    if not results:
        return {"available_beds_count": 0, "selected_hospital": None}

    bed_type = (state.get("bed_type_required") or "General").strip()

    if bed_type == "ICU":
        sorted_results = sorted(
            results,
            key=lambda h: (
                int(h.get("available_beds", 0) or 0) > 0,  # any-beds hospitals first
                int(h.get("icu_beds", 0) or 0),             # most ICU beds
                int(h.get("available_beds", 0) or 0),       # tiebreak: most total beds
            ),
            reverse=True,
        )
    else:
        sorted_results = sorted(
            results,
            key=lambda h: (
                int(h.get("available_beds", 0) or 0) > 0,
                int(h.get("general_beds", 0) or 0),
                int(h.get("available_beds", 0) or 0),
            ),
            reverse=True,
        )

    best = sorted_results[0]
    beds = int(best.get("available_beds", 0) or 0)

    logger.info(
        "Selected hospital: %s (ICU beds: %s, General beds: %s) for bed_type=%s",
        best.get("hospital_name"), best.get("icu_beds"), best.get("general_beds"), bed_type,
    )

    return {"selected_hospital": best, "available_beds_count": beds}


def reservation_node(state: HospitalState) -> Dict[str, Any]:
    """Simulate a bed reservation: generate an ID and decrement the count."""
    hospital = state.get("selected_hospital") or {}
    reservation_id = tools.generate_reservation_id()

    if hospital:
        tools.decrement_available_beds(hospital.get("hospital_name", ""), state.get("location", ""))

    logger.info(
        "Reserved bed at %s for %s -> %s",
        hospital.get("hospital_name"), state.get("patient_name"), reservation_id,
    )
    return {"reservation_id": reservation_id}


def confirmation_node(state: HospitalState) -> Dict[str, Any]:
    """Send the confirmation email for a successful reservation."""
    hospital = state.get("selected_hospital") or {}
    sent = tools.send_confirmation_email(
        to=state["patient_email"],
        patient_name=state["patient_name"],
        patient_age=state.get("patient_age", 0),
        patient_mobile=state.get("patient_mobile", ""),
        hospital=hospital,
        urgency_level=state.get("urgency_level", "High"),
        bed_type=state.get("bed_type_required", "General"),
        reservation_id=state.get("reservation_id", ""),
        symptoms=state.get("symptoms", ""),
    )
    return {"confirmation_sent": sent}


def no_beds_notification_node(state: HospitalState) -> Dict[str, Any]:
    """Path B: notify the patient that no beds are currently available."""
    if state.get("error_message"):
        # Intake validation already failed (e.g. bad email) — nothing to
        # notify, and we don't have a trustworthy address to send to.
        return {"confirmation_sent": False, "reservation_id": None}

    results = state.get("hospital_search_results") or []

    request_id = f"REQ-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{state.get('patient_mobile', '')[-6:]}"

    sent = tools.send_no_beds_notification(
        to=state["patient_email"],
        patient_name=state["patient_name"],
        zone=state.get("location", ""),
        hospitals_in_zone=results,
        request_id=request_id,
    )
    return {"confirmation_sent": sent, "reservation_id": None}
