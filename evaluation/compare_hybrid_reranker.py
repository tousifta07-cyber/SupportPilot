
import json
import re
import time
from pathlib import Path

import chromadb
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_corpus_fixed_chunks.jsonl"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "techqa_dev_eval_50.jsonl"
)

DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db_fixed"
COLLECTION_NAME = "techqa_full_corpus_fixed"

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
CACHE_DIR = PROJECT_ROOT / "data" / "reranker_cache"

DETAILS_FILE = (
    RESULTS_DIR / "full_corpus_hybrid_reranker_details.jsonl"
)
SUMMARY_FILE = (
    RESULTS_DIR / "full_corpus_hybrid_reranker_summary.json"
)

CANDIDATE_K = 20
FINAL_K = 5
RRF_K = 60


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def gold_rank(results, gold_ids):
    for rank, item in enumerate(results, start=1):
        if item["source_id"] in gold_ids:
            return rank
    return None


def calculate_metrics(ranks):
    total = len(ranks)
    hits = sum(rank is not None for rank in ranks)

    mrr = sum(
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    ) / total if total else 0.0

    return {
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": mrr,
    }


def main():
    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(EVALUATION_FILE)

    answerable = [
        question for question in questions
        if question["answerable"]
    ]

    if not answerable:
        raise ValueError("No answerable questions found.")

    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    # Build BM25 once and reuse it for every question.
    print("Building BM25 keyword index...")

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() != len(chunks):
        raise ValueError(
            "The fixed-size chunk file and ChromaDB index "
            "have different numbers of chunks."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FlashRank reranker...")
    ranker = Ranker(
        cache_dir=str(CACHE_DIR),
        max_length=512,
    )

    print("\n" + "=" * 65)
    print("SUPPORTPILOT: HYBRID SEARCH + RERANKER")
    print("=" * 65)
    print("Answerable questions:", len(answerable))
    print("Vector candidates:", CANDIDATE_K)
    print("BM25 candidates:", CANDIDATE_K)
    print("Final results:", FINAL_K)

    rrf_ranks = []
    reranked_ranks = []
    candidate_hits = []

    retrieval_times = []
    rerank_times = []
    details = []

    for index, question in enumerate(answerable, start=1):
        query = question["question"]
        gold_ids = set(question["gold_source_ids"])

        # -------------------------------------------------
        # 1. Vector search
        # -------------------------------------------------
        search_start = time.perf_counter()

        vector_result = collection.query(
            query_texts=[query],
            n_results=min(CANDIDATE_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        candidates = {}
        fusion_scores = {}

        vector_ids = vector_result["ids"][0]
        vector_docs = vector_result["documents"][0]
        vector_metas = vector_result["metadatas"][0]
        vector_distances = vector_result["distances"][0]

        vector_results = []

        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                vector_ids,
                vector_docs,
                vector_metas,
                vector_distances,
            ),
            start=1,
        ):
            candidate = {
                "id": chunk_id,
                "text": text,
                "meta": {
                    "source_id": metadata["source_id"],
                    "chunk_index": metadata["chunk_index"],
                },
            }

            candidates[chunk_id] = candidate
            fusion_scores[chunk_id] = (
                fusion_scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

            vector_results.append({
                "chunk_id": chunk_id,
                "source_id": metadata["source_id"],
            })

        # -------------------------------------------------
        # 2. BM25 keyword search
        # -------------------------------------------------
        query_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )

        bm25_indices = [
            idx for idx in bm25_indices
            if bm25_scores[idx] > 0
        ][:CANDIDATE_K]

        bm25_results = []

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

            fusion_scores[chunk_id] = (
                fusion_scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

            bm25_results.append({
                "chunk_id": chunk_id,
                "source_id": chunk["source_id"],
            })

        retrieval_ms = (
            time.perf_counter() - search_start
        ) * 1000
        retrieval_times.append(retrieval_ms)

        # -------------------------------------------------
        # 3. RRF baseline ranking of the combined candidates
        # -------------------------------------------------
        fused_ids = sorted(
            fusion_scores,
            key=lambda chunk_id: fusion_scores[chunk_id],
            reverse=True,
        )

        hybrid_top5 = [
            {
                "chunk_id": chunk_id,
                "source_id": candidates[chunk_id]["meta"]["source_id"],
            }
            for chunk_id in fused_ids[:FINAL_K]
        ]

        rrf_rank = gold_rank(hybrid_top5, gold_ids)

        # Candidate recall: does the combined pool contain gold?
        candidate_source_ids = {
            item["meta"]["source_id"]
            for item in candidates.values()
        }

        candidate_hit = bool(candidate_source_ids & gold_ids)

        # -------------------------------------------------
        # 4. Rerank the combined, deduplicated candidate pool
        # -------------------------------------------------
        rerank_start = time.perf_counter()

        rerank_request = RerankRequest(
            query=query,
            passages=[
                candidates[chunk_id]
                for chunk_id in fused_ids
            ],
        )

        reranked = ranker.rerank(rerank_request)[:FINAL_K]

        rerank_ms = (
            time.perf_counter() - rerank_start
        ) * 1000
        rerank_times.append(rerank_ms)

        reranked_top5 = [
            {
                "chunk_id": item["id"],
                "source_id": item["meta"]["source_id"],
                "score": float(item["score"]),
            }
            for item in reranked
        ]

        reranked_rank = gold_rank(reranked_top5, gold_ids)

        rrf_ranks.append(rrf_rank)
        reranked_ranks.append(reranked_rank)
        candidate_hits.append(candidate_hit)

        details.append({
            "question_id": question["question_id"],
            "gold_source_ids": sorted(gold_ids),
            "unique_candidates": len(candidates),
            "candidate_hit": candidate_hit,
            "rrf_gold_rank": rrf_rank,
            "reranked_gold_rank": reranked_rank,
            "hybrid_top5": hybrid_top5,
            "reranked_top5": reranked_top5,
            "vector_top20": vector_results,
            "bm25_top20": bm25_results,
            "retrieval_latency_ms": retrieval_ms,
            "reranking_latency_ms": rerank_ms,
        })

        print(
            f"[{index:02}/{len(answerable)}] "
            f"{question['question_id']} | "
            f"Candidates: {len(candidates)} | "
            f"RRF: {rrf_rank or '-'} | "
            f"Reranked: {reranked_rank or '-'}"
        )

    rrf_metrics = calculate_metrics(rrf_ranks)
    rerank_metrics = calculate_metrics(reranked_ranks)

    summary = {
        "experiment": "hybrid_rrf_vs_reranker",
        "collection": COLLECTION_NAME,
        "answerable_questions": len(answerable),
        "vector_candidate_k": CANDIDATE_K,
        "bm25_candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "rrf_k": RRF_K,
        "union_candidate_hit_rate": (
            sum(candidate_hits) / len(candidate_hits)
        ),
        "hybrid_rrf": rrf_metrics,
        "hybrid_reranked": rerank_metrics,
        "mean_retrieval_latency_ms": (
            sum(retrieval_times) / len(retrieval_times)
        ),
        "mean_reranking_latency_ms": (
            sum(rerank_times) / len(rerank_times)
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with DETAILS_FILE.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 65)
    print("HYBRID + RERANKER RESULTS")
    print("=" * 65)

    print(
        f"Combined candidate Hit Rate: "
        f"{sum(candidate_hits)}/{len(answerable)} = "
        f"{sum(candidate_hits) / len(answerable) * 100:.2f}%"
    )

    print(
        f"Hybrid RRF Hit Rate@5: "
        f"{rrf_metrics['hits']}/{len(answerable)} = "
        f"{rrf_metrics['hit_rate'] * 100:.2f}%"
    )
    print(f"Hybrid RRF MRR@5: {rrf_metrics['mrr']:.4f}")

    print(
        f"\nHybrid + reranker Hit Rate@5: "
        f"{rerank_metrics['hits']}/{len(answerable)} = "
        f"{rerank_metrics['hit_rate'] * 100:.2f}%"
    )
    print(f"Hybrid + reranker MRR@5: {rerank_metrics['mrr']:.4f}")

    print(
        f"\nMean retrieval latency: "
        f"{summary['mean_retrieval_latency_ms']:.2f} ms"
    )
    print(
        f"Mean reranking latency: "
        f"{summary['mean_reranking_latency_ms']:.2f} ms"
    )

    print("\nSaved results:")
    print(SUMMARY_FILE)
    print(DETAILS_FILE)


if __name__ == "__main__":
    main()
