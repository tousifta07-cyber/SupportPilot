
import json
from pathlib import Path

import chromadb


# Find the SupportPilot project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


EVALUATION_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "techqa_dev_eval_50.jsonl"
)

DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db_fixed"
COLLECTION_NAME = "techqa_full_corpus_fixed"

TOP_K = 5

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DETAILS_FILE = (
    RESULTS_DIR / "full_corpus_vector_baseline_details.jsonl"
)

SUMMARY_FILE = (
    RESULTS_DIR / "full_corpus_vector_baseline_summary.json"
)

def load_evaluation_data():
    """Load the evaluation questions from JSONL."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    records = []

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("The evaluation dataset is empty.")

    return records


def main():
    records = load_evaluation_data()

    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError("The ChromaDB collection is empty.")

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    print("=" * 60)
    print("SUPPORTPILOT: FULL-CORPUS VECTOR RETRIEVAL BASELINE")
    print("=" * 60)
    print("Total evaluation questions:", len(records))
    print("Answerable questions:", len(answerable))
    print("Unanswerable questions:", len(unanswerable))
    print("Indexed chunks:", collection.count())
    print("Retrieval depth:", TOP_K)

    detailed_results = []
    hits = 0
    reciprocal_ranks = []

    for index, record in enumerate(records, start=1):
        query = record["question"]

        result = collection.query(
            query_texts=[query],
            n_results=min(TOP_K, collection.count()),
            include=["metadatas", "distances"],
        )

        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        retrieved = []

        for rank, (metadata, distance) in enumerate(
            zip(metadatas, distances),
            start=1,
        ):
            retrieved.append({
                "rank": rank,
                "source_id": metadata["source_id"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance),
            })

        gold_ids = set(record["gold_source_ids"])
        is_answerable = record["answerable"]

        rank_of_gold = None
        hit_at_5 = None
        reciprocal_rank = None

        if is_answerable:
            for item in retrieved:
                if item["source_id"] in gold_ids:
                    rank_of_gold = item["rank"]
                    break

            hit_at_5 = rank_of_gold is not None

            if hit_at_5:
                hits += 1
                reciprocal_rank = 1.0 / rank_of_gold

            reciprocal_ranks.append(
                reciprocal_rank if reciprocal_rank is not None else 0.0
            )

        detailed_results.append({
            "question_id": record["question_id"],
            "question": query,
            "answerable": is_answerable,
            "gold_source_ids": sorted(gold_ids),
            "retrieved": retrieved,
            "rank_of_gold": rank_of_gold,
            "hit_at_5": hit_at_5,
            "reciprocal_rank_at_5": reciprocal_rank,
        })

        if is_answerable:
            status = "HIT" if hit_at_5 else "MISS"
            rank_text = (
                str(rank_of_gold)
                if rank_of_gold is not None
                else "not in top 5"
            )
            print(
                f"[{index:02}] {record['question_id']} | "
                f"{status} | gold rank: {rank_text}"
            )
        else:
            print(
                f"[{index:02}] {record['question_id']} | "
                "UNANSWERABLE | excluded from retrieval hit rate"
            )

    total_answerable = len(answerable)

    hit_rate = hits / total_answerable if total_answerable else 0.0

    mrr_at_5 = (
        sum(reciprocal_ranks) / total_answerable
        if total_answerable
        else 0.0
    )

    summary = {
        "experiment": "full_corpus_vector_baseline",
        "chunk_size_words": 300,
        "overlap_words": 50,
        "collection": COLLECTION_NAME,
        "indexed_chunks": collection.count(),
        "total_questions": len(records),
        "answerable_questions": total_answerable,
        "unanswerable_questions": len(unanswerable),
        "top_k": TOP_K,
        "hits_at_5": hits,
        "hit_rate_at_5": hit_rate,
        "hit_rate_at_5_percent": round(hit_rate * 100, 2),
        "mrr_at_5": mrr_at_5,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with DETAILS_FILE.open("w", encoding="utf-8") as file:
        for item in detailed_results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Hit Rate@5: {hits}/{total_answerable} "
          f"= {hit_rate * 100:.2f}%")
    print(f"MRR@5: {mrr_at_5:.4f}")
    print("\nResults saved to:")
    print(DETAILS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
