"""
patient_intake_source.py — Fetches new patient submissions from the
public Google Sheet CSV export and returns them as dicts ready to feed
into the LangGraph hospital_bed_graph.

The Sheet is the one linked at:
  https://docs.google.com/spreadsheets/d/1jD70uIx-0MZQAvvAOqrahUG4Uajj1MnMiZlzUJj2V-Y

Expected column layout (Form Responses 1 tab):
  A – Timestamp
  B – Patient Name
  C – Age
  D – Symptoms
  E – Area in Delhi
  F – Contact Number
  G – Email Address

Configuration (via .env):
  SHEET_ID             – the long ID in the Sheet URL (set by default to the
                         linked sheet above)
  FORM_RESPONSES_GID   – the gid= query param of the "Form Responses 1" tab.
                         To find it: open the sheet, click the tab, look at the
                         URL — ?gid=XXXXXXX.  Default is "0" which is correct
                         if "Form Responses 1" is the *first* tab; change it in
                         .env if the URL shows a different number.
  PROCESSED_ROWS_FILE  – path to the JSON file used to track already-processed
                         row numbers so re-runs don't double-process.
                         Default: processed_rows.json next to this file.

No credentials are required — the sheet must be "Anyone with the link
can view" (link-viewable) for the CSV export to work.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger("hospital_bed_agent.intake")

# ── Config ────────────────────────────────────────────────────────────────────
SHEET_ID = os.getenv(
    "SHEET_ID",
    "1jD70uIx-0MZQAvvAOqrahUG4Uajj1MnMiZlzUJj2V-Y",  # the shared sheet
)
FORM_RESPONSES_GID = os.getenv("FORM_RESPONSES_GID", "0")
PROCESSED_ROWS_FILE = os.getenv(
    "PROCESSED_ROWS_FILE",
    str(Path(__file__).parent / "processed_rows.json"),
)

_CSV_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    "?format=csv&gid={gid}"
)

# How to find the correct GID if "0" doesn't work:
# 1. Open the spreadsheet in a browser.
# 2. Click the "Form Responses 1" tab at the bottom.
# 3. Look at the URL bar — it will end with something like  #gid=12345678
# 4. Set FORM_RESPONSES_GID=12345678 in your .env file.


# ── Processed-row tracking ───────────────────────────────────────────────────

def _load_processed_rows() -> Set[int]:
    """Load the set of already-processed CSV row indices (0-indexed, header = row 0)."""
    try:
        with open(PROCESSED_ROWS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_rows", []))
    except FileNotFoundError:
        return set()
    except (json.JSONDecodeError, KeyError):
        logger.warning("processed_rows.json is malformed; starting fresh.")
        return set()


def _save_processed_rows(rows: Set[int]) -> None:
    """Persist the set of processed row indices to disk."""
    with open(PROCESSED_ROWS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_rows": sorted(rows)}, f, indent=2)


# ── CSV fetch & parse ─────────────────────────────────────────────────────────

def _fetch_csv_text(sheet_id: str = SHEET_ID, gid: str = FORM_RESPONSES_GID) -> str:
    """Download the sheet as a CSV string. Raises requests.RequestException on failure."""
    url = _CSV_EXPORT_URL.format(sheet_id=sheet_id, gid=gid)
    logger.info("Fetching form responses from %s", url)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def _parse_csv_row(row_index: int, row: List[str]) -> Optional[Dict[str, Any]]:
    """
    Convert a raw CSV row (7 columns expected) into a HospitalState-compatible
    dict.  Returns None and logs a warning if the row is obviously malformed
    (too few columns, blank name, etc.).

    Column mapping:
      0 – Timestamp
      1 – Patient Name
      2 – Age
      3 – Symptoms
      4 – Area in Delhi  (→ location)
      5 – Contact Number (→ patient_mobile)
      6 – Email Address  (→ patient_email)
    """
    if len(row) < 7:
        logger.warning("Row %d has only %d columns (expected 7); skipping.", row_index, len(row))
        return None

    name = row[1].strip()
    if not name:
        logger.warning("Row %d has an empty Patient Name; skipping.", row_index)
        return None

    # Parse age robustly — blank or non-numeric becomes None so validation
    # catches it and routes it to the error path rather than crashing.
    raw_age = row[2].strip()
    try:
        age: Optional[float] = float(raw_age) if raw_age else None
    except ValueError:
        age = None

    return {
        "patient_name": name,
        "patient_age": age,
        "symptoms": row[3].strip(),
        "location": row[4].strip(),
        "patient_mobile": row[5].strip(),
        "patient_email": row[6].strip(),
        "_row_index": row_index,   # internal tracking key, stripped before graph invoke
        "_timestamp": row[0].strip(),
    }


def fetch_new_patients(
    sheet_id: str = SHEET_ID,
    gid: str = FORM_RESPONSES_GID,
) -> List[Dict[str, Any]]:
    """
    Fetch the Google Sheet CSV and return only rows that haven't been
    processed yet (tracked in processed_rows.json).

    Each returned dict has an extra '_row_index' key used by
    mark_patients_processed(); strip it before passing to the graph:
        patient = {k: v for k, v in patient.items() if not k.startswith('_')}

    Raises requests.RequestException if the sheet is unreachable.
    """
    csv_text = _fetch_csv_text(sheet_id, gid)
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    if not rows:
        logger.info("Sheet is empty.")
        return []

    processed = _load_processed_rows()
    new_patients: List[Dict[str, Any]] = []
    # Tracks (name, email, symptoms, location) fingerprints seen THIS run
    # so exact duplicate form submissions are skipped automatically.
    _seen_fingerprints: set = set()

    # Row 0 is the header; data rows start at index 1
    for i, row in enumerate(rows):
        if i == 0:
            continue  # skip header
        if i in processed:
            continue  # already handled in a previous run

        parsed = _parse_csv_row(i, row)
        if parsed is None:
            continue

        # Build a dedup fingerprint from the fields that uniquely identify a
        # real submission (case-insensitive, stripped).
        fingerprint = (
            parsed["patient_name"].lower(),
            parsed["patient_email"].lower(),
            parsed["symptoms"].lower(),
            parsed["location"].lower(),
        )

        if fingerprint in _seen_fingerprints:
            logger.warning(
                "Row %d is an exact duplicate of an earlier row (name=%s, email=%s); "
                "skipping and marking as processed to suppress future repeats.",
                i, parsed["patient_name"], parsed["patient_email"],
            )
            # Mark processed so the duplicate never surfaces again.
            processed.add(i)
            continue

        _seen_fingerprints.add(fingerprint)
        new_patients.append(parsed)

    # Persist the newly-identified duplicate rows so they don't reappear.
    _save_processed_rows(processed)

    logger.info(
        "Found %d new unique patient submission(s) (skipped %d already-processed row(s)).",
        len(new_patients),
        len(processed),
    )
    return new_patients


def mark_patients_processed(patients: List[Dict[str, Any]]) -> None:
    """
    Mark each patient dict (as returned by fetch_new_patients) as processed.
    Call this AFTER successfully running each patient through the graph.
    """
    processed = _load_processed_rows()
    for p in patients:
        row_idx = p.get("_row_index")
        if row_idx is not None:
            processed.add(row_idx)
    _save_processed_rows(processed)
