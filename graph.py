"""
graph.py — Assembles the LangGraph StateGraph for hospital bed requests.

Flow:
    patient_intake → ai_triage → validate_triage → hospital_search
                                                         → check_availability
                                                               │
                                          ┌────────────────────┴────────────────────┐
                                     beds > 0                               beds == 0 or error
                                          │                                          │
                                    reservation                         no_beds_notification
                                          │                                          │
                                    confirmation                                    END
                                          │
                                         END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from nodes import (
    ai_triage_node,
    check_availability_node,
    confirmation_node,
    hospital_search_node,
    no_beds_notification_node,
    patient_intake_node,
    reservation_node,
    validate_triage_node,
)
from state import HospitalState


def _route_on_availability(state: HospitalState) -> str:
    """
    Conditional edge: if beds > 0 (and no upstream error), take Path A
    (reserve + confirm). Otherwise take Path B (no-beds notification).
    """
    if state.get("error_message"):
        return "no_beds_notification"
    if (state.get("available_beds_count") or 0) > 0:
        return "reservation"
    return "no_beds_notification"


def build_graph() -> StateGraph:
    graph = StateGraph(HospitalState)

    # Register all nodes
    graph.add_node("patient_intake", patient_intake_node)
    graph.add_node("ai_triage", ai_triage_node)
    graph.add_node("validate_triage", validate_triage_node)
    graph.add_node("hospital_search", hospital_search_node)
    graph.add_node("check_availability", check_availability_node)
    graph.add_node("reservation", reservation_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("no_beds_notification", no_beds_notification_node)

    # Linear edges up to the routing fork
    graph.set_entry_point("patient_intake")
    graph.add_edge("patient_intake", "ai_triage")
    graph.add_edge("ai_triage", "validate_triage")
    graph.add_edge("validate_triage", "hospital_search")
    graph.add_edge("hospital_search", "check_availability")

    # Conditional branch: Path A vs Path B
    graph.add_conditional_edges(
        "check_availability",
        _route_on_availability,
        {
            "reservation": "reservation",
            "no_beds_notification": "no_beds_notification",
        },
    )

    # Path A tail
    graph.add_edge("reservation", "confirmation")
    graph.add_edge("confirmation", END)

    # Path B tail
    graph.add_edge("no_beds_notification", END)

    return graph.compile()


# Module-level compiled graph — import this in main.py and test_agent.py
hospital_bed_graph = build_graph()
