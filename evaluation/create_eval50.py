
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "techqa"
    / "extracted"
)

DEV_FILE = DATA_DIR / "training_and_dev" / "dev_Q_A.json"
VALIDATION_FILE = DATA_DIR / "validation" / "validation_questions.json"
DOCUMENT_FILE = (
    DATA_DIR
    / "training_and_dev"
    / "training_dev_technotes.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "techqa" / "benchmark_50"

QUESTION_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "techqa_dev_eval_50.jsonl"
)

CORPUS_OUTPUT = OUTPUT_DIR / "technotes.json"

QUESTIONS_PER_CLASS = 25
RANDOM_SEED = 42


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def normalize(text):
    return " ".join(
        re.findall(r"[a-z0-9]+", (text or "").lower())
    )


def question_key(record):
    title = normalize(record.get("QUESTION_TITLE", ""))
    body = normalize(
        record.get("QUESTION_TEXT")
        or record.get("QUESTION_BODY")
        or ""
    )
    return title, body


def parse_offset(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main():
    print("Loading TechQA development questions...")
    development = load_json(DEV_FILE)

    validation_questions = load_json(VALIDATION_FILE)

    print("Loading the technical-document corpus...")
    documents = load_json(DOCUMENT_FILE)

    # Exclude questions already used in the original benchmark.
    previous_keys = {
        question_key(item)
        for item in validation_questions
    }

    available = [
        item for item in development
        if question_key(item) not in previous_keys
    ]

    answerable = [
        item for item in available
        if str(item.get("ANSWERABLE", "")).upper() == "Y"
        and item.get("DOCUMENT") in documents
    ]

    unanswerable = [
        item for item in available
        if str(item.get("ANSWERABLE", "")).upper() != "Y"
    ]

    if len(answerable) < QUESTIONS_PER_CLASS:
        raise ValueError(
            f"Only {len(answerable)} eligible answerable questions found."
        )

    if len(unanswerable) < QUESTIONS_PER_CLASS:
        raise ValueError(
            f"Only {len(unanswerable)} unanswerable questions found."
        )

    rng = random.Random(RANDOM_SEED)

    selected = (
        rng.sample(answerable, QUESTIONS_PER_CLASS)
        + rng.sample(unanswerable, QUESTIONS_PER_CLASS)
    )

    # Stable ordering makes the generated files easy to compare.
    selected.sort(key=lambda item: item["QUESTION_ID"])

    candidate_ids = set()

    for item in selected:
        ids = item.get("DOC_IDS", [])

        if isinstance(ids, list):
            candidate_ids.update(
                doc_id for doc_id in ids
                if isinstance(doc_id, str) and doc_id
            )

        # Always include the verified gold document.
        if str(item.get("ANSWERABLE", "")).upper() == "Y":
            candidate_ids.add(item["DOCUMENT"])

    found_ids = candidate_ids.intersection(documents.keys())
    missing_ids = candidate_ids - found_ids

    # Keep only documents available in the extracted IBM corpus.
    selected_documents = {
        doc_id: documents[doc_id]
        for doc_id in sorted(found_ids)
    }

    records = []

    for item in selected:
        is_answerable = (
            str(item.get("ANSWERABLE", "")).upper() == "Y"
        )

        gold_ids = (
            [item["DOCUMENT"]]
            if is_answerable
            else []
        )

        question_text = (
            item.get("QUESTION_TITLE", "")
            + "\n\n"
            + (
                item.get("QUESTION_TEXT")
                or item.get("QUESTION_BODY")
                or ""
            )
        ).strip()

        record = {
            "question_id": item["QUESTION_ID"],
            "question_title": item.get("QUESTION_TITLE", ""),
            "question": question_text,
            "expected_answer": (
                item.get("ANSWER") if is_answerable else None
            ),
            "answerable": is_answerable,
            "gold_source_ids": gold_ids,
            "gold_start_offset": (
                parse_offset(item.get("START_OFFSET"))
                if is_answerable
                else None
            ),
            "gold_end_offset": (
                parse_offset(item.get("END_OFFSET"))
                if is_answerable
                else None
            ),
            "candidate_doc_ids": [
                doc_id
                for doc_id in item.get("DOC_IDS", [])
                if doc_id in found_ids
            ],
            "benchmark_source": "TechQA development set",
        }

        records.append(record)

    # Verify critical properties before writing the files.
    assert len(records) == 50
    assert sum(item["answerable"] for item in records) == 25
    assert sum(not item["answerable"] for item in records) == 25
    assert not any(
        question_key(item) in previous_keys
        for item in selected
    )

    for record in records:
        for source_id in record["gold_source_ids"]:
            if source_id not in selected_documents:
                raise ValueError(
                    f"Gold document missing from corpus: {source_id}"
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUESTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with QUESTION_OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    with CORPUS_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            selected_documents,
            file,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("50-QUESTION BENCHMARK CREATED")
    print("=" * 65)
    print("Original validation questions excluded:", len(previous_keys))
    print("Questions selected:", len(records))
    print("Answerable:", sum(r["answerable"] for r in records))
    print("Unanswerable:", sum(not r["answerable"] for r in records))
    print("Candidate document IDs:", len(candidate_ids))
    print("Documents included in corpus:", len(selected_documents))
    print("Candidate IDs missing from corpus:", len(missing_ids))
    print("Random seed:", RANDOM_SEED)
    print("\nQuestion dataset:", QUESTION_OUTPUT)
    print("Document corpus:", CORPUS_OUTPUT)


if __name__ == "__main__":
    main()
