
import pytest

from app.agents import support_graph as graph_module
from app.services.llm_service import ABSTENTION_MESSAGE


def test_graph_generates_answer_when_passages_are_found(monkeypatch):
    """The graph should generate an answer when evidence is found."""

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

    def fake_retrieve_passages(question, top_k):
        assert top_k == 5
        return passages

    def fake_generate_answer(question, passages):
        return "Collect the installation logs [swg21396515]."

    monkeypatch.setattr(
        graph_module,
        "retrieve_passages",
        fake_retrieve_passages,
    )
    monkeypatch.setattr(
        graph_module,
        "generate_grounded_answer",
        fake_generate_answer,
    )

    result = graph_module.run_support_graph(
        "How do I troubleshoot Installation Manager?"
    )

    assert result["status"] == "answered"
    assert result["answer"] == (
        "Collect the installation logs [swg21396515]."
    )
    assert len(result["passages"]) == 1
    assert result["passages"][0]["source_id"] == "swg21396515"


def test_graph_abstains_when_no_passages_are_found(monkeypatch):
    """The graph should abstain when retrieval finds no evidence."""

    monkeypatch.setattr(
        graph_module,
        "retrieve_passages",
        lambda question, top_k: [],
    )

    def unexpected_generation(question, passages):
        pytest.fail("Generation should not run without retrieved evidence.")

    monkeypatch.setattr(
        graph_module,
        "generate_grounded_answer",
        unexpected_generation,
    )

    result = graph_module.run_support_graph(
        "How do I troubleshoot an undocumented error?"
    )

    assert result["status"] == "abstained"
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["passages"] == []


def test_graph_rejects_blank_question():
    """The graph should reject blank input."""

    with pytest.raises(ValueError, match="question is required"):
        graph_module.run_support_graph("   ")
