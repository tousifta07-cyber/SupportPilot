
import json
import re

import pytest
from fastapi.testclient import TestClient

from app import api
from app.services import ticket_service
from app.services.llm_service import validate_citations

client = TestClient(api.app)


# -----------------------------
# Citation validation tests
# -----------------------------

def test_valid_citation_is_accepted():
    answer = "Review the installation logs [swg21396515]."

    valid, message = validate_citations(
        answer,
        {"swg21396515", "nas8N1011459"},
    )

    assert valid is True
    assert "retrieved" in message.lower()


def test_unknown_citation_is_rejected():
    answer = "Review the logs [unknown_source]."

    valid, message = validate_citations(
        answer,
        {"swg21396515"},
    )

    assert valid is False
    assert "unknown_source" in message


def test_answer_without_citation_is_rejected():
    valid, message = validate_citations(
        "Review the installation logs.",
        {"swg21396515"},
    )

    assert valid is False
    assert "No source citations" in message


# -----------------------------
# Ticket service tests
# -----------------------------

def test_ticket_is_saved_to_jsonl(tmp_path, monkeypatch):
    ticket_file = tmp_path / "support_tickets.jsonl"

    monkeypatch.setattr(
        ticket_service,
        "TICKETS_FILE",
        ticket_file,
    )

    ticket = ticket_service.create_support_ticket(
        question="Installation Manager failed.",
        description="Please investigate the logs.",
        priority="high",
        assistant_answer="Collect the installation logs.",
    )

    assert re.fullmatch(r"SP-[A-F0-9]{10}", ticket["ticket_id"])
    assert ticket["status"] == "open"
    assert ticket["priority"] == "high"
    assert ticket_file.exists()

    with ticket_file.open("r", encoding="utf-8") as file:
        saved_ticket = json.loads(file.readline())

    assert saved_ticket["ticket_id"] == ticket["ticket_id"]
    assert saved_ticket["question"] == "Installation Manager failed."
    assert saved_ticket["priority"] == "high"


def test_ticket_rejects_empty_question(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ticket_service,
        "TICKETS_FILE",
        tmp_path / "support_tickets.jsonl",
    )

    with pytest.raises(ValueError, match="question is required"):
        ticket_service.create_support_ticket(
            question="   ",
        )


def test_ticket_rejects_invalid_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ticket_service,
        "TICKETS_FILE",
        tmp_path / "support_tickets.jsonl",
    )

    with pytest.raises(ValueError, match="Priority must be"):
        ticket_service.create_support_ticket(
            question="Installation failed.",
            priority="critical",
        )


# -----------------------------
# API tests
# -----------------------------

def test_ask_api_returns_answer_and_sources(monkeypatch):
    """Test the API through LangGraph without making a Gemini request."""

    passages = [
        {
            "source_id": "swg21396515",
            "chunk_index": 0,
            "text": "Collect IBM Installation Manager logs.",
            "distance": None,
            "rerank_score": 0.95,
            "retrieval_method": "hybrid_rrf_flashrank",
        }
    ]

    # Mock the new LangGraph workflow rather than the old direct calls.
    monkeypatch.setattr(
        api,
        "run_support_graph",
        lambda question: {
            "question": question,
            "passages": passages,
            "answer": (
                "Collect IBM Installation Manager logs "
                "[swg21396515]."
            ),
            "status": "answered",
        },
    )

    response = client.post(
        "/ask",
        json={
            "question": "How do I troubleshoot Installation Manager?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "How do I troubleshoot Installation Manager?"
    )
    assert "swg21396515" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "swg21396515"
    assert data["sources"][0]["chunk_index"] == 0
    assert data["sources"][0]["retrieval_method"] == "hybrid_rrf_flashrank"


def test_ask_api_rejects_blank_question():
    """A blank question must not trigger the workflow."""

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ticket_api_creates_ticket(tmp_path, monkeypatch):
    """Test ticket creation through the API using temporary storage."""

    ticket_file = tmp_path / "support_tickets.jsonl"

    monkeypatch.setattr(
        ticket_service,
        "TICKETS_FILE",
        ticket_file,
    )

    monkeypatch.setattr(
        api,
        "create_support_ticket",
        ticket_service.create_support_ticket,
    )

    response = client.post(
        "/tickets",
        json={
            "question": "Installation Manager failed.",
            "description": "Please investigate the installation logs.",
            "priority": "high",
            "assistant_answer": "Collect the installation logs.",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["ticket_id"].startswith("SP-")
    assert data["status"] == "open"
    assert data["priority"] == "high"
    assert ticket_file.exists()
