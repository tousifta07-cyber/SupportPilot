
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKETS_FILE = (
    PROJECT_ROOT / "data" / "support_tickets.jsonl"
)

VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


def create_support_ticket(
    question: str,
    description: str = "",
    priority: str = "normal",
    assistant_answer: str = "",
) -> dict:
    """Save a support ticket locally and return its details."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A question is required.")

    if len(question) > 2000:
        raise ValueError("Question must not exceed 2000 characters.")

    if not isinstance(description, str):
        raise ValueError("Description must be text.")

    if len(description) > 5000:
        raise ValueError("Description must not exceed 5000 characters.")

    if not isinstance(assistant_answer, str):
        raise ValueError("Assistant answer must be text.")

    if len(assistant_answer) > 10000:
        raise ValueError("Assistant answer is too long.")

    priority = priority.strip().lower()

    if priority not in VALID_PRIORITIES:
        raise ValueError(
            "Priority must be low, normal, high, or urgent."
        )

    ticket = {
        "ticket_id": f"SP-{uuid4().hex[:10].upper()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "priority": priority,
        "question": question.strip(),
        "description": description.strip(),
        "assistant_answer": assistant_answer.strip(),
    }

    TICKETS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TICKETS_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(ticket, ensure_ascii=False) + "\n"
        )

    return ticket
