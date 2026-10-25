
import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.rag.retriever import retrieve_passages
from app.services.llm_service import (
    ABSTENTION_MESSAGE,
    generate_grounded_answer,
)

logger = logging.getLogger(__name__)


class SupportGraphState(TypedDict, total=False):
    question: str
    passages: list[dict[str, Any]]
    answer: str
    status: Literal["answered", "abstained"]


class RetrievalFailure(RuntimeError):
    """Raised when technical-document retrieval fails."""


class AnswerGenerationFailure(RuntimeError):
    """Raised when grounded answer generation fails."""


def retrieve_node(state: SupportGraphState) -> dict[str, Any]:
    """Retrieve candidate passages from the technical knowledge base."""
    try:
        passages = retrieve_passages(
            question=state["question"],
            top_k=5,
        )
    except Exception as exc:
        raise RetrievalFailure(
            "Technical-document retrieval failed."
        ) from exc

    return {"passages": passages}


def route_after_retrieval(
    state: SupportGraphState,
) -> Literal["generate", "abstain"]:
    """Choose a path based on whether retrieval returned evidence."""
    if state.get("passages"):
        return "generate"

    return "abstain"


def generate_node(state: SupportGraphState) -> dict[str, Any]:
    """Generate an answer grounded in the retrieved passages."""
    try:
        answer = generate_grounded_answer(
            question=state["question"],
            passages=state["passages"],
        )
    except Exception as exc:
        raise AnswerGenerationFailure(
            "Grounded answer generation failed."
        ) from exc

    status = (
        "abstained"
        if answer.strip() == ABSTENTION_MESSAGE.strip()
        else "answered"
    )

    return {
        "answer": answer,
        "status": status,
    }


def abstain_node(state: SupportGraphState) -> dict[str, Any]:
    """Return a safe response when retrieval finds no passages."""
    return {
        "answer": ABSTENTION_MESSAGE,
        "status": "abstained",
    }


def build_support_graph():
    """Build and compile the SupportPilot LangGraph workflow."""
    workflow = StateGraph(SupportGraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("abstain", abstain_node)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "abstain": "abstain",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("abstain", END)

    return workflow.compile()


support_graph = build_support_graph()


def run_support_graph(question: str) -> dict[str, Any]:
    """Run the complete support workflow for a question."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("A question is required.")

    return support_graph.invoke(
        {"question": cleaned_question}
    )
