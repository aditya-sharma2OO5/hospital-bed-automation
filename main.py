"""
main.py — CLI entry point for running the Hospital Bed Request agent.

Usage:
    python main.py                # runs the built-in example scenarios
    python main.py --interactive  # prompts you for a single patient request
"""
from __future__ import annotations

import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from graph import hospital_bed_graph  # noqa: E402  (after load_dotenv)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hospital_bed_agent.main")


EXAMPLE_SCENARIOS = [
    {
        "name": "Critical case (South Delhi, multiple hospitals with beds)",
        "input": {
            "patient_name": "Aditya Sharma",
            "patient_age": 28,
            "patient_mobile": "+91-9876543210",
            "patient_email": "aditya@example.com",
            "symptoms": "Severe chest pain, difficulty breathing, loss of consciousness",
            "location": "South Delhi",
        },
    },
    {
        "name": "Medium case (Central Delhi, beds available)",
        "input": {
            "patient_name": "Jane Smith",
            "patient_age": 34,
            "patient_mobile": "+91-8765432109",
            "patient_email": "jane@example.com",
            "symptoms": "Mild fever, body ache, fatigue, slight cough",
            "location": "Central Delhi",
        },
    },
    {
        "name": "Multiple hospitals in zone (South Delhi: first match has beds)",
        "input": {
            "patient_name": "Diana Prince",
            "patient_age": 33,
            "patient_mobile": "+91-9876543214",
            "patient_email": "diana@example.com",
            "symptoms": "General checkup",
            "location": "South Delhi",
        },
    },
    {
        "name": "Invalid zone (no hospitals found)",
        "input": {
            "patient_name": "Alice Johnson",
            "patient_age": 51,
            "patient_mobile": "+91-9876543212",
            "patient_email": "alice@example.com",
            "symptoms": "Headache",
            "location": "Zone X",
        },
    },
    {
        "name": "Invalid input (missing email)",
        "input": {
            "patient_name": "No Email Guy",
            "patient_age": 40,
            "patient_mobile": "+91-9876543213",
            "patient_email": "",
            "symptoms": "Cough",
            "location": "South Delhi",
        },
    },
]


def run_scenario(name: str, patient_input: dict) -> None:
    print(f"\n{'=' * 70}\nSCENARIO: {name}\n{'=' * 70}")
    result = hospital_bed_graph.invoke(patient_input)
    print(json.dumps(result, indent=2, default=str))


def run_interactive() -> None:
    print("Enter patient request details:")
    patient_input = {
        "patient_name": input("Name: ").strip(),
        "patient_age": int(input("Age: ").strip() or 0),
        "patient_mobile": input("Mobile: ").strip(),
        "patient_email": input("Email: ").strip(),
        "symptoms": input("Symptoms: ").strip(),
        "location": input("Zone (e.g. South Delhi, Central Delhi, West Delhi): ").strip(),
    }
    run_scenario("Interactive request", patient_input)


def main() -> None:
    if "--interactive" in sys.argv:
        run_interactive()
        return

    for scenario in EXAMPLE_SCENARIOS:
        run_scenario(scenario["name"], scenario["input"])


if __name__ == "__main__":
    main()
