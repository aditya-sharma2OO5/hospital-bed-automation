"""
test_intake.py — Tests for patient_intake_source.py and check_new_patients.py

Run with: pytest test_intake.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

import patient_intake_source as intake


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_CSV_ROWS = [
    # header
    ["Timestamp", "Patient Name", "Age", "Symptoms", "Area in Delhi", "Contact Number", "Email Address"],
    # valid row
    ["19/07/2026 10:00:00", "Raj Kumar", "45", "High fever and difficulty breathing", "South Delhi", "9876543210", "raj@example.com"],
    # duplicate valid row (different timestamp)
    ["19/07/2026 11:00:00", "Priya Singh", "32", "Mild cough, fatigue", "Central Delhi", "8765432109", "priya@example.com"],
    # row with blank age (should parse, but age=None → validation catches it)
    ["19/07/2026 12:00:00", "No Age Person", "", "Headache", "West Delhi", "7654321098", "noage@example.com"],
    # row with non-numeric age
    ["19/07/2026 13:00:00", "Bad Age Person", "abc", "Cough", "West Delhi", "6543210987", "badage@example.com"],
    # row with too few columns (malformed — should be skipped)
    ["19/07/2026 14:00:00", "Short Row"],
    # row with blank name (should be skipped)
    ["19/07/2026 15:00:00", "", "25", "Fever", "South Delhi", "5432109876", "blank@example.com"],
]


def _make_csv(rows: List[List[str]]) -> str:
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


# ── Unit tests: _parse_csv_row ────────────────────────────────────────────────

def test_parse_valid_row():
    row = ["19/07/2026 10:00:00", "Raj Kumar", "45", "High fever", "South Delhi", "9876543210", "raj@example.com"]
    result = intake._parse_csv_row(1, row)
    assert result is not None
    assert result["patient_name"] == "Raj Kumar"
    assert result["patient_age"] == 45.0
    assert result["symptoms"] == "High fever"
    assert result["location"] == "South Delhi"
    assert result["patient_mobile"] == "9876543210"
    assert result["patient_email"] == "raj@example.com"
    assert result["_row_index"] == 1


def test_parse_row_blank_age_gives_none():
    row = ["ts", "Name", "", "Cough", "South Delhi", "1234567890", "a@b.com"]
    result = intake._parse_csv_row(2, row)
    assert result is not None
    assert result["patient_age"] is None


def test_parse_row_nonnumeric_age_gives_none():
    row = ["ts", "Name", "xyz", "Cough", "South Delhi", "1234567890", "a@b.com"]
    result = intake._parse_csv_row(3, row)
    assert result is not None
    assert result["patient_age"] is None


def test_parse_row_too_few_columns_returns_none():
    result = intake._parse_csv_row(4, ["ts", "short"])
    assert result is None


def test_parse_row_blank_name_returns_none():
    row = ["ts", "", "30", "Fever", "South Delhi", "1234567890", "a@b.com"]
    result = intake._parse_csv_row(5, row)
    assert result is None


# ── Unit tests: processed-row tracking ───────────────────────────────────────

def test_load_save_processed_rows(tmp_path, monkeypatch):
    pf = tmp_path / "processed.json"
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(pf))

    # Initially empty
    assert intake._load_processed_rows() == set()

    # Save some rows
    intake._save_processed_rows({1, 3, 7})
    loaded = intake._load_processed_rows()
    assert loaded == {1, 3, 7}


def test_load_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "nonexistent.json"))
    assert intake._load_processed_rows() == set()


def test_load_malformed_json_returns_empty(tmp_path, monkeypatch):
    pf = tmp_path / "bad.json"
    pf.write_text("not json at all")
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(pf))
    assert intake._load_processed_rows() == set()


# ── Unit tests: fetch_new_patients ────────────────────────────────────────────

def test_fetch_new_patients_parses_all_valid_rows(tmp_path, monkeypatch):
    """Valid rows are returned; malformed/blank rows are silently skipped."""
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))
    csv_text = _make_csv(SAMPLE_CSV_ROWS)
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: csv_text)

    patients = intake.fetch_new_patients()

    names = [p["patient_name"] for p in patients]
    assert "Raj Kumar" in names
    assert "Priya Singh" in names
    assert "No Age Person" in names    # parsed (age=None), validation will catch it later
    assert "Bad Age Person" in names   # same
    # Malformed/blank-name rows should NOT appear
    assert all(p["patient_name"] for p in patients)


def test_fetch_new_patients_deduplicates_exact_duplicates(tmp_path, monkeypatch):
    """Exact duplicate rows (same name+email+symptoms+location) appear only once."""
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))

    dup_rows = [
        ["Timestamp", "Patient Name", "Age", "Symptoms", "Area in Delhi", "Contact Number", "Email Address"],
        ["19/07/2026 10:00", "John Doe", "30", "Headache", "South Delhi", "9999999999", "john@example.com"],
        ["19/07/2026 10:05", "John Doe", "30", "Headache", "South Delhi", "9999999999", "john@example.com"],
        ["19/07/2026 10:10", "John Doe", "30", "Headache", "South Delhi", "9999999999", "john@example.com"],
        ["19/07/2026 10:15", "Jane Doe", "25", "Fever", "Central Delhi", "8888888888", "jane@example.com"],
    ]
    csv_text = _make_csv(dup_rows)
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: csv_text)

    patients = intake.fetch_new_patients()
    names = [p["patient_name"] for p in patients]

    assert names.count("John Doe") == 1   # deduplicated to one
    assert "Jane Doe" in names            # distinct patient still appears
    assert len(patients) == 2



def test_fetch_new_patients_skips_processed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))
    # Pre-mark row 1 (Raj Kumar) as processed
    intake._save_processed_rows({1})

    csv_text = _make_csv(SAMPLE_CSV_ROWS)
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: csv_text)

    patients = intake.fetch_new_patients()
    names = [p["patient_name"] for p in patients]
    assert "Raj Kumar" not in names
    assert "Priya Singh" in names


def test_fetch_new_patients_empty_sheet(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: "")
    assert intake.fetch_new_patients() == []


# ── Unit tests: mark_patients_processed ──────────────────────────────────────

def test_mark_patients_processed_saves_row_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))

    patients = [
        {"patient_name": "A", "_row_index": 2},
        {"patient_name": "B", "_row_index": 5},
    ]
    intake.mark_patients_processed(patients)
    assert intake._load_processed_rows() == {2, 5}


def test_mark_patients_processed_accumulates(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(tmp_path / "p.json"))

    intake.mark_patients_processed([{"_row_index": 1}])
    intake.mark_patients_processed([{"_row_index": 3}])
    assert intake._load_processed_rows() == {1, 3}


# ── Integration: check_new_patients runner ───────────────────────────────────

def test_check_new_patients_runs_end_to_end(tmp_path, monkeypatch, capsys):
    """
    Feed two valid rows through check_new_patients.run() with mocked CSV
    and mocked graph + email. Confirm summary output is printed and rows
    are marked processed.
    """
    import check_new_patients as cnp

    pf = tmp_path / "p.json"
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(pf))

    valid_csv = _make_csv([
        SAMPLE_CSV_ROWS[0],   # header
        SAMPLE_CSV_ROWS[1],   # Raj Kumar (valid)
        SAMPLE_CSV_ROWS[2],   # Priya Singh (valid)
    ])
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: valid_csv)

    # Mock the graph to return a successful Path A result
    mock_result = {
        "error_message": None,
        "reservation_id": "RES-20260719-123456",
        "selected_hospital": {"hospital_name": "AIIMS New Delhi"},
        "confirmation_sent": True,
    }
    monkeypatch.setattr(cnp.hospital_bed_graph, "invoke", lambda inp: mock_result)

    cnp.run()

    captured = capsys.readouterr()
    assert "Raj Kumar" in captured.out
    assert "Priya Singh" in captured.out
    assert "RES-20260719-123456" in captured.out

    # Both rows should now be marked processed
    processed = intake._load_processed_rows()
    assert 1 in processed
    assert 2 in processed


def test_check_new_patients_flags_validation_errors(tmp_path, monkeypatch, capsys):
    """Validation-error rows should appear in the manual-follow-up section."""
    import check_new_patients as cnp

    pf = tmp_path / "p.json"
    monkeypatch.setattr(intake, "PROCESSED_ROWS_FILE", str(pf))

    error_csv = _make_csv([
        SAMPLE_CSV_ROWS[0],   # header
        SAMPLE_CSV_ROWS[3],   # No Age Person — will trigger validation error
    ])
    monkeypatch.setattr(intake, "_fetch_csv_text", lambda *a, **k: error_csv)

    error_result = {
        "error_message": "Age must be a positive number.",
        "reservation_id": None,
        "confirmation_sent": False,
    }
    monkeypatch.setattr(cnp.hospital_bed_graph, "invoke", lambda inp: error_result)

    cnp.run()

    captured = capsys.readouterr()
    assert "manual" in captured.out.lower() or "follow" in captured.out.lower()
    assert "No Age Person" in captured.out
