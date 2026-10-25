
import logging

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from app.services.ticket_service import create_support_ticket

from app.rag.retriever import (
    DATABASE_DIR,
    DEFAULT_COLLECTION,
)
from app.agents.support_graph import (
    AnswerGenerationFailure,
    RetrievalFailure,
    run_support_graph,
)


logger = logging.getLogger(__name__)

app = FastAPI(
    title="SupportPilot API",
    description=(
        "Technical support assistant using hybrid retrieval, "
        "reranking, and evidence-grounded answers."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourcePassage(BaseModel):
    source_id: str
    chunk_index: int
    text: str
    distance: float | None = None
    rerank_score: float | None = None
    retrieval_method: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourcePassage]

class TicketRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    description: str = Field(default="", max_length=5000)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    assistant_answer: str = Field(default="", max_length=10000)


class TicketResponse(BaseModel):
    ticket_id: str
    created_at: str
    status: str
    priority: str
    question: str
    description: str
    assistant_answer: str


@app.get("/")
def home():
    return {
        "message": "Welcome to SupportPilot API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Check whether the full-corpus ChromaDB collection is available."""
    try:
        client = chromadb.PersistentClient(
            path=str(DATABASE_DIR)
        )

        collection = client.get_collection(
            name=DEFAULT_COLLECTION
        )

        count = collection.count()

        if count == 0:
            raise HTTPException(
                status_code=503,
                detail="The SupportPilot collection is empty.",
            )

        return {
            "status": "ok",
            "collection": DEFAULT_COLLECTION,
            "indexed_chunks": count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("SupportPilot health check failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge index is unavailable. "
                "Check the server terminal for details."
            ),
        ) from exc



@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Run the LangGraph support workflow and return its evidence."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a question.",
        )

    # Step 1: Run retrieval, answer generation, and abstention routing.
    try:
        result = run_support_graph(question)

    except RetrievalFailure as exc:
        logger.exception("SupportPilot document retrieval failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Document retrieval failed. "
                "Check the server terminal for details."
            ),
        ) from exc

    except AnswerGenerationFailure as exc:
        logger.exception("Gemini answer generation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI answer service failed. "
                "Check the Gemini configuration and server terminal."
            ),
        ) from exc

    except Exception as exc:
        logger.exception("SupportPilot LangGraph workflow failed")
        raise HTTPException(
            status_code=500,
            detail="The support workflow failed. Check the server terminal.",
        ) from exc

    # Step 2: Convert retrieved passages to the existing API response model.
    passages = result.get("passages", [])

    sources = [
        SourcePassage(
            source_id=passage["source_id"],
            chunk_index=passage["chunk_index"],
            text=passage["text"][:600],
            distance=passage.get("distance"),
            rerank_score=passage.get("rerank_score"),
            retrieval_method=passage.get("retrieval_method"),
        )
        for passage in passages
    ]

    return AskResponse(
        question=question,
        answer=result["answer"],
        sources=sources,
    )

@app.post(
    "/tickets",
    response_model=TicketResponse,
    status_code=201,
)
def create_ticket(request: TicketRequest):
    """Create a support ticket for human follow-up."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="A question is required to create a ticket.",
        )

    try:
        ticket = create_support_ticket(
            question=question,
            description=request.description,
            priority=request.priority,
            assistant_answer=request.assistant_answer,
        )

        logger.info(
            "Created support ticket %s",
            ticket["ticket_id"],
        )

        return TicketResponse(**ticket)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Support-ticket creation failed")

        raise HTTPException(
            status_code=500,
            detail="Could not save the support ticket.",
        ) from exc