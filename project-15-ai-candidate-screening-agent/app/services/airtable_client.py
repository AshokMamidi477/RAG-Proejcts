"""airtable_client.py — Airtable REST API wrapper"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://api.airtable.com/v0"

PAT = os.getenv("AIRTABLE_PAT")
BASE_ID = os.getenv("AIRTABLE_BASE_ID")
TABLE_NAME = os.getenv(
    "AIRTABLE_TABLE_NAME",
    "Candidates"
)


def create_record(fields: dict) -> str:
    """Create a new Airtable record and return its record ID."""

    fields["Submitted At"] = datetime.now().isoformat()

    response = requests.post(
        f"{BASE_URL}/{BASE_ID}/{TABLE_NAME}",
        headers={
            "Authorization": f"Bearer {PAT}",
            "Content-Type": "application/json"
        },
        json={
            "fields": fields
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()["id"]