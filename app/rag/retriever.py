import json
import re
import warnings
from functools import lru_cache
from pathlib import Path

import chromadb
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db_fixed"
DEFAULT_COLLECTION = "techqa_full_corpus_fixed"

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_corpus_fixed_chunks.jsonl"
)

RERANKER_CACHE_DIR = PROJECT_ROOT / "data" / "reranker_cache"

CANDIDATE_K = 20
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently with the evaluation experiment."""
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def get_chroma_client(database_path: str):
    """Reuse the ChromaDB client within this Python process."""
    return chromadb.PersistentClient(path=database_path)


@lru_cache(maxsize=1)
def get_collection(collection_name: str):
    """Reuse the requested collection within this process."""
    client = get_chroma_client(str(DATABASE_DIR))
    return client.get_collection(name=collection_name)


@lru_cache(maxsize=1)
def get_chunks_and_bm25(chunks_path: str):
    """Load the full-corpus chunks and build BM25 once per process."""
    path = Path(chunks_path)

    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks = []

    print("Building BM25 index for SupportPilot.")
    print("This happens on the first hybrid retrieval request.")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

            required = {"chunk_id", "source_id", "chunk_index", "text"}

            if not required.issubset(chunk):
                raise ValueError(
                    f"Missing required fields on line {line_number}."
                )

            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise ValueError(
                    f"Empty chunk text on line {line_number}."
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("The full-corpus chunks file is empty.")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    print(f"BM25 index ready: {len(chunks):,} chunks.")

    return chunks, bm25


@lru_cache(maxsize=1)
def get_reranker(cache_path: str):
    """Load and reuse the FlashRank reranker."""
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    return Ranker(
        cache_dir=cache_path,
        max_length=512,
    )


def retrieve_vector_only(
    question: str,
    top_k: int,
    collection_name: str,
) -> list[dict]:
    """Keep custom collection retrieval compatible with older callers."""
    collection = get_collection(collection_name)
    count = collection.count()

    if count == 0:
        raise ValueError("The selected collection is empty.")

    result = collection.query(
        query_texts=[question],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    passages = []

    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        passages.append({
            "source_id": metadata["source_id"],
            "chunk_index": metadata["chunk_index"],
            "text": text,
            "distance": float(distance),
            "retrieval_method": "vector",
        })

    return passages


def retrieve_passages(
    question: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict]:
    """Retrieve passages with hybrid search and FlashRank reranking."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Preserve vector-only behavior for explicitly selected legacy
    # collections. Hybrid retrieval is configured for the full corpus.
    if collection_name != DEFAULT_COLLECTION:
        return retrieve_vector_only(
            question.strip(),
            top_k,
            collection_name,
        )

    collection = get_collection(collection_name)
    collection_count = collection.count()

    if collection_count == 0:
        raise ValueError("The full-corpus collection is empty.")

    chunks, bm25 = get_chunks_and_bm25(str(CHUNKS_FILE))

    if collection_count != len(chunks):
        raise ValueError(
            "The ChromaDB collection and BM25 chunk file have "
            f"different sizes: {collection_count:,} vs {len(chunks):,}."
        )

    candidate_k = max(CANDIDATE_K, top_k)

    # 1. Vector retrieval
    vector_result = collection.query(
        query_texts=[question.strip()],
        n_results=min(candidate_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    candidates = {}
    fusion_scores = {}
    distances_by_id = {}

    vector_ids = vector_result["ids"][0]
    vector_docs = vector_result["documents"][0]
    vector_metas = vector_result["metadatas"][0]
    vector_distances = vector_result["distances"][0]

    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(
            vector_ids,
            vector_docs,
            vector_metas,
            vector_distances,
        ),
        start=1,
    ):
        candidates[chunk_id] = {
            "id": chunk_id,
            "text": text,
            "meta": {
                "source_id": metadata["source_id"],
                "chunk_index": metadata["chunk_index"],
            },
        }

        distances_by_id[chunk_id] = float(distance)

        fusion_scores[chunk_id] = (
            fusion_scores.get(chunk_id, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # 2. BM25 keyword retrieval
    bm25_scores = bm25.get_scores(tokenize(question))

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )

    bm25_indices = [
        index
        for index in bm25_indices
        if bm25_scores[index] > 0
    ][:candidate_k]

    for rank, chunk_index in enumerate(bm25_indices, start=1):
        chunk = chunks[chunk_index]
        chunk_id = chunk["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "id": chunk_id,
                "text": chunk["text"],
                "meta": {
                    "source_id": chunk["source_id"],
                    "chunk_index": chunk["chunk_index"],
                },
            }
            distances_by_id[chunk_id] = None

        fusion_scores[chunk_id] = (
            fusion_scores.get(chunk_id, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # 3. Reciprocal Rank Fusion
    fused_ids = sorted(
        fusion_scores,
        key=lambda chunk_id: fusion_scores[chunk_id],
        reverse=True,
    )

    fused_candidates = [
        candidates[chunk_id]
        for chunk_id in fused_ids
    ]

    # 4. FlashRank reranking
    ranker = get_reranker(str(RERANKER_CACHE_DIR))

    try:
        request = RerankRequest(
            query=question.strip(),
            passages=fused_candidates,
        )

        reranked = ranker.rerank(request)[:top_k]

    except Exception as exc:
        warnings.warn(
            "FlashRank failed; falling back to RRF ordering. "
            f"Reason: {exc}",
            RuntimeWarning,
        )

        reranked = [
            {
                **candidates[chunk_id],
                "score": None,
            }
            for chunk_id in fused_ids[:top_k]
        ]

    # 5. Return the same core fields used by the LLM service.
    passages = []

    for item in reranked:
        chunk_id = item["id"]
        metadata = item["meta"]

        passages.append({
            "source_id": metadata["source_id"],
            "chunk_index": metadata["chunk_index"],
            "text": item["text"],
            "distance": distances_by_id.get(chunk_id),
            "rerank_score": (
                float(item["score"])
                if item.get("score") is not None
                else None
            ),
            "retrieval_method": "hybrid_rrf_flashrank",
        })

    return passages