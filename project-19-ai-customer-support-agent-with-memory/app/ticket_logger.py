"""ticket_logger.py — Local JSON ticket store (Zendesk optional)"""
import json, os
from datetime import datetime

TICKET_DIR = "samples/tickets"


def log_ticket(session_id: str, user_message: str, conversation: list) -> str:
    os.makedirs(TICKET_DIR, exist_ok=True)
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    ticket = {
        "ticket_id":   ticket_id,
        "session_id":  session_id,
        "created_at":  datetime.now().isoformat(),
        "status":      "open",
        "user_message": user_message,
        "conversation": conversation,
    }
    path = os.path.join(TICKET_DIR, f"{ticket_id}.json")
    with open(path, "w") as f:
        json.dump(ticket, f, indent=2)
    return ticket_id
