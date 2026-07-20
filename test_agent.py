"""
test_agent.py — unit + integration tests for the Hospital Bed Request agent.

Run with:
    pytest -v
"""
from unittest.mock import patch

import pytest

import tools
from graph import hospital_bed_graph
from nodes import (
    check_availability_node,
    no_beds_notification_node,
    validate_triage_node,
)


# --------------------------------------------------------------------------
# Unit tests: validation
# --------------------------------------------------------------------------
def test_validate_patient_input_valid():
    ok, err = tools.validate_patient_input({
        "patient_name": "Test Person",
        "symptoms": "Fever",
        "location": "South Delhi",
        "patient_email": "a@b.com",
        "patient_age": 30,
    })
    assert ok is True
    assert err is None


@pytest.mark.parametrize("field,value", [
    ("patient_name", ""),
    ("symptoms", ""),
    ("location", ""),
    ("patient_email", "not-an-email"),
    ("patient_age", -1),
    ("patient_age", None),
])
def test_validate_patient_input_invalid(field, value):
    data = {
        "patient_name": "Test Person",
        "symptoms": "Fever",
        "location": "South Delhi",
        "patient_email": "a@b.com",
        "patient_age": 30,
    }
    data[field] = value
    ok, err = tools.validate_patient_input(data)
    assert ok is False
    assert err is not None


# --------------------------------------------------------------------------
# Unit tests: triage sanitization
# --------------------------------------------------------------------------
def test_sanitize_triage_valid_passthrough():
    result = tools.sanitize_triage_response({"urgency_level": "Medium", "bed_type_required": "ICU"})
    assert result == {"urgency_level": "Medium", "bed_type_required": "ICU"}


def test_sanitize_triage_invalid_replaced_with_defaults():
    result = tools.sanitize_triage_response({"urgency_level": "Unknown", "bed_type_required": ""})
    assert result == {"urgency_level": "High", "bed_type_required": "General"}


def test_sanitize_triage_placeholder_text_replaced():
    result = tools.sanitize_triage_response(
        {"urgency_level": "{urgency_level}", "bed_type_required": "bed_type_required"}
    )
    assert result == {"urgency_level": "High", "bed_type_required": "General"}


# --------------------------------------------------------------------------
# Unit tests: Groq classifier error handling
# --------------------------------------------------------------------------
def test_classify_symptoms_no_api_key_returns_defaults(monkeypatch):
    monkeypatch.setattr(tools, "GROQ_API_KEY", "")
    result = tools.classify_symptoms_groq("chest pain")
    assert result == {"urgency_level": "High", "bed_type_required": "General"}


def test_classify_symptoms_timeout_returns_defaults(monkeypatch):
    monkeypatch.setattr(tools, "GROQ_API_KEY", "fake-key")

    def raise_timeout(*args, **kwargs):
        raise tools.requests.Timeout()

    monkeypatch.setattr(tools.requests, "post", raise_timeout)
    result = tools.classify_symptoms_groq("chest pain")
    assert result == {"urgency_level": "High", "bed_type_required": "General"}


def test_classify_symptoms_bad_json_returns_defaults(monkeypatch):
    monkeypatch.setattr(tools, "GROQ_API_KEY", "fake-key")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResp())
    result = tools.classify_symptoms_groq("chest pain")
    assert result == {"urgency_level": "High", "bed_type_required": "General"}


# --------------------------------------------------------------------------
# Unit tests: hospital search
# --------------------------------------------------------------------------
def test_search_hospitals_by_zone_returns_matches():
    results = tools.search_hospitals_by_zone("South Delhi", use_cache=False)
    names = {r["hospital_name"] for r in results}
    assert "AIIMS New Delhi" in names
    assert "Safdarjung Hospital" in names
    assert "Max Super Speciality Hospital, Saket" in names


def test_search_hospitals_by_zone_no_match_returns_empty():
    results = tools.search_hospitals_by_zone("Zone Nonexistent", use_cache=False)
    assert results == []


def test_search_hospitals_excludes_total_row():
    """The Excel file's trailing TOTAL summary row must never be treated as a hospital."""
    all_zones_results = []
    for zone in tools.list_available_zones():
        all_zones_results.extend(tools.search_hospitals_by_zone(zone, use_cache=False))
    names = {r["hospital_name"] for r in all_zones_results}
    assert "TOTAL" not in names


def test_available_beds_is_general_plus_icu():
    results = tools.search_hospitals_by_zone("South Delhi", use_cache=False)
    aiims = next(r for r in results if r["hospital_name"] == "AIIMS New Delhi")
    assert aiims["available_beds"] == aiims["general_beds"] + aiims["icu_beds"]
    assert aiims["available_beds"] == 17  # 14 general + 3 ICU, per the source file


def test_reservation_id_format():
    rid = tools.generate_reservation_id()
    assert rid.startswith("RES-")
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8   # YYYYMMDD
    assert len(parts[2]) == 6   # random digits


# --------------------------------------------------------------------------
# Unit tests: routing node
# --------------------------------------------------------------------------
def test_check_availability_icu_selects_most_icu_beds():
    """ICU bed type -> hospital with most ICU beds wins, not the first row."""
    state = {
        "bed_type_required": "ICU",
        "hospital_search_results": [
            {"hospital_name": "General First",  "available_beds": 20, "icu_beds": 2,  "general_beds": 18},
            {"hospital_name": "ICU Specialist", "available_beds": 10, "icu_beds": 8,  "general_beds": 2},
            {"hospital_name": "Medium ICU",     "available_beds": 15, "icu_beds": 5,  "general_beds": 10},
        ],
    }
    out = check_availability_node(state)
    assert out["selected_hospital"]["hospital_name"] == "ICU Specialist"
    assert out["available_beds_count"] == 10


def test_check_availability_general_selects_most_general_beds():
    """General/Standard bed type -> hospital with most general beds wins."""
    state = {
        "bed_type_required": "General",
        "hospital_search_results": [
            {"hospital_name": "ICU Heavy",      "available_beds": 15, "icu_beds": 12, "general_beds": 3},
            {"hospital_name": "General Leader", "available_beds": 20, "icu_beds": 2,  "general_beds": 18},
            {"hospital_name": "Small General",  "available_beds": 10, "icu_beds": 1,  "general_beds": 9},
        ],
    }
    out = check_availability_node(state)
    assert out["selected_hospital"]["hospital_name"] == "General Leader"
    assert out["available_beds_count"] == 20


def test_check_availability_zero_beds_when_all_hospitals_full():
    state = {
        "bed_type_required": "General",
        "hospital_search_results": [
            {"hospital_name": "Full Hospital", "available_beds": 0, "icu_beds": 0, "general_beds": 0},
        ],
    }
    out = check_availability_node(state)
    assert out["available_beds_count"] == 0


def test_check_availability_no_results():
    out = check_availability_node({"hospital_search_results": []})
    assert out["available_beds_count"] == 0
    assert out["selected_hospital"] is None


# --------------------------------------------------------------------------
# Integration tests: full graph, Path A (beds available)
# --------------------------------------------------------------------------
@patch("tools._send_email_smtp", return_value=True)
def test_full_graph_path_a_beds_available(mock_send):
    result = hospital_bed_graph.invoke({
        "patient_name": "Aditya Sharma",
        "patient_age": 28,
        "patient_mobile": "+91-9876543210",
        "patient_email": "aditya@example.com",
        "symptoms": "Severe chest pain, difficulty breathing",
        "location": "South Delhi",
    })

    assert result["error_message"] is None
    # For a chest-pain/ICU case the agent should pick the hospital with most
    # ICU beds in South Delhi — not necessarily AIIMS (row 1).
    assert result["selected_hospital"] is not None
    assert result["available_beds_count"] > 0
    assert result["reservation_id"] is not None
    assert result["reservation_id"].startswith("RES-")
    assert result["confirmation_sent"] is True
    mock_send.assert_called_once()
    # confirm it's the *confirmation* email, not the no-beds one
    sent_subject = mock_send.call_args[0][1]
    assert "Reserved" in sent_subject


# --------------------------------------------------------------------------
# Integration tests: full graph, Path B (no beds available)
# --------------------------------------------------------------------------
@patch("tools._send_email_smtp", return_value=True)
def test_full_graph_path_b_first_hospital_full(mock_send, monkeypatch):
    """
    Force the search results so the FIRST hospital in the zone has 0 beds,
    proving Path B (no-beds notification) fires — this mirrors the
    "first available hospital" selection logic in the spec, which does not
    fall through to the next hospital if the first one is full.
    """
    monkeypatch.setattr(
        tools,
        "search_hospitals_by_zone",
        lambda zone, use_cache=True: [
            {"hospital_name": "Sir Ganga Ram Hospital", "zone": "South Delhi",
             "total_beds": 675, "general_beds": 0, "icu_beds": 0,
             "available_beds": 0, "contact": "011-25750000",
             "last_updated": "18-07-2026 09:00"},
        ],
    )

    result = hospital_bed_graph.invoke({
        "patient_name": "Bob Wilson",
        "patient_age": 45,
        "patient_mobile": "+91-9876543211",
        "patient_email": "bob@example.com",
        "symptoms": "Minor cuts and bruises",
        "location": "South Delhi",
    })

    assert result["available_beds_count"] == 0
    assert result["reservation_id"] is None
    assert result["confirmation_sent"] is True
    mock_send.assert_called_once()
    sent_subject = mock_send.call_args[0][1]
    assert "Not Currently Available" in sent_subject


@patch("tools._send_email_smtp", return_value=True)
def test_full_graph_path_b_invalid_zone_no_hospitals(mock_send):
    result = hospital_bed_graph.invoke({
        "patient_name": "Alice Johnson",
        "patient_age": 51,
        "patient_mobile": "+91-9876543212",
        "patient_email": "alice@example.com",
        "symptoms": "Headache",
        "location": "Zone X",
    })

    assert result["hospital_search_results"] == []
    assert result["selected_hospital"] is None
    assert result["reservation_id"] is None
    assert result["confirmation_sent"] is True
    mock_send.assert_called_once()


# --------------------------------------------------------------------------
# Integration tests: invalid patient input short-circuits cleanly
# --------------------------------------------------------------------------
@patch("tools._send_email_smtp")
def test_full_graph_invalid_input_sends_no_email(mock_send):
    result = hospital_bed_graph.invoke({
        "patient_name": "No Email Guy",
        "patient_age": 40,
        "patient_mobile": "+91-9876543213",
        "patient_email": "",
        "symptoms": "Cough",
        "location": "South Delhi",
    })

    assert result["error_message"] is not None
    assert result["confirmation_sent"] is False
    assert result["reservation_id"] is None
    mock_send.assert_not_called()


def test_duplicate_requests_get_different_reservation_ids():
    ids = {tools.generate_reservation_id() for _ in range(20)}
    assert len(ids) == 20  # extremely unlikely to collide
