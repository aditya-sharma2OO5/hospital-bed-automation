"""
check_new_patients.py — On-demand runner: fetches new patient submissions
from the Google Sheet, runs each through the LangGraph hospital-bed agent,
and prints a summary.

Usage:
    python check_new_patients.py

Designed to be run manually whenever you want to process new form responses.
No scheduler or long-running process required.

Configure via .env:
    SHEET_ID            – Google Sheet ID (default: the shared demo sheet)
    FORM_RESPONSES_GID  – tab gid for "Form Responses 1" (default: "0")
    GROQ_API_KEY        – optional; falls back to default triage if absent
    SMTP_HOST / etc.    – optional; emails are logged if SMTP not configured
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── Configure logging before importing anything else ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hospital_bed_agent.check_new_patients")

# ── Imports (after load_dotenv so env vars are visible to all modules) ────────
import requests  # noqa: E402

from graph import hospital_bed_graph  # noqa: E402
from patient_intake_source import fetch_new_patients, mark_patients_processed  # noqa: E402


# ── Result tracking ───────────────────────────────────────────────────────────

def _strip_internal_keys(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Remove _row_index / _timestamp tracking keys before passing to the graph."""
    return {k: v for k, v in patient.items() if not k.startswith("_")}


def _summarise_result(result: Dict[str, Any], patient: Dict[str, Any]) -> str:
    """One-line human-readable summary of what happened for this patient."""
    name = patient.get("patient_name", "Unknown")
    if result.get("error_message"):
        return f"  [ERR] {name} -- validation error: {result['error_message']}"
    if result.get("reservation_id"):
        hospital = (result.get("selected_hospital") or {}).get("hospital_name", "Unknown hospital")
        rid = result["reservation_id"]
        sent = "email sent" if result.get("confirmation_sent") else "email NOT sent (check SMTP config)"
        return f"  [OK]  {name} -- reserved at {hospital} [{rid}] -- {sent}"
    # Path B: no beds
    sent = "no-beds email sent" if result.get("confirmation_sent") else "no-beds email NOT sent (check SMTP config)"
    return f"  [--]  {name} -- no beds available in {patient.get('location', '?')} -- {sent}"


def run() -> None:
    print("\n" + "=" * 70)
    print("  Hospital Bed Agent -- Check New Patients")
    print("=" * 70)

    # ── 1. Fetch new submissions ──────────────────────────────────────────
    try:
        new_patients = fetch_new_patients()
    except requests.RequestException as exc:
        logger.error("Could not fetch the Google Sheet: %s", exc)
        print(
            "\n⚠️  Could not reach the Google Sheet. Check your internet "
            "connection and that SHEET_ID / FORM_RESPONSES_GID are correct in .env.\n"
        )
        sys.exit(1)

    if not new_patients:
        print("\n[OK]  No new submissions to process.\n")
        return

    print(f"\n[>>] {len(new_patients)} new submission(s) to process.\n")

    # ── 2. Run each patient through the graph ─────────────────────────────
    processed_ok: List[Dict[str, Any]] = []   # patients marked processed
    validation_failures: List[Dict[str, Any]] = []  # need manual follow-up
    results_summary: List[str] = []

    for patient in new_patients:
        name = patient.get("patient_name", "Unknown")
        logger.info("Processing patient: %s (row %s)", name, patient.get("_row_index"))

        graph_input = _strip_internal_keys(patient)

        try:
            result = hospital_bed_graph.invoke(graph_input)
        except Exception as exc:  # noqa: BLE001
            logger.error("Graph raised an exception for %s: %s", name, exc, exc_info=True)
            results_summary.append(
                f"  [!!] {name} -- unexpected error (see logs): {exc}"
            )
            # Don't mark as processed — let the operator retry after fixing.
            continue

        summary_line = _summarise_result(result, patient)
        results_summary.append(summary_line)

        if result.get("error_message"):
            # Validation failure — operator needs to follow up manually.
            validation_failures.append({
                "row": patient.get("_row_index"),
                "timestamp": patient.get("_timestamp"),
                "patient_name": patient.get("patient_name"),
                "patient_mobile": patient.get("patient_mobile"),
                "patient_email": patient.get("patient_email"),
                "error": result["error_message"],
            })
            # Still mark as processed so we don't re-attempt every run.
            processed_ok.append(patient)
        else:
            processed_ok.append(patient)

    # ── 3. Persist processed rows ─────────────────────────────────────────
    if processed_ok:
        mark_patients_processed(processed_ok)

    # ── 4. Print run summary ──────────────────────────────────────────────
    print("\nRun summary:")
    for line in results_summary:
        print(line)

    # ── 5. Flag validation failures for manual follow-up ─────────────────
    if validation_failures:
        print(
            f"\n[WARN] {len(validation_failures)} submission(s) had invalid data and "
            f"could NOT be automatically processed.\n"
            f"Please follow up with these patients manually:\n"
        )
        for vf in validation_failures:
            print(
                f"  Row {vf['row']} | {vf['timestamp']}\n"
                f"    Name:   {vf['patient_name']}\n"
                f"    Mobile: {vf['patient_mobile']}\n"
                f"    Email:  {vf['patient_email']}\n"
                f"    Error:  {vf['error']}\n"
            )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()
