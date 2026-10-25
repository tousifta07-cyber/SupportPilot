
import argparse
import json
from pathlib import Path
from typing import Iterator

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "fixed_chunks.jsonl"
)
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db_fixed"
DEFAULT_COLLECTION_NAME = "techqa_fixed_baseline"

BATCH_SIZE = 100


def validate_chunks_file(chunks_file: Path) -> int:
    """Check the entire JSONL file before creating or resetting a collection."""
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Chunk file not found: {chunks_file}")

    count = 0
    required = {"chunk_id", "source_id", "chunk_index", "text"}

    with chunks_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from exc

            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(
                    f"Line {line_number} is missing required chunk fields."
                )

            if not isinstance(item["text"], str) or not item["text"].strip():
                raise ValueError(
                    f"Line {line_number} contains empty chunk text."
                )

            count += 1

    if count == 0:
        raise ValueError("The chunk file contains no records.")

    return count


def read_batches(
    chunks_file: Path,
    batch_size: int = BATCH_SIZE,
) -> Iterator[list[dict]]:
    """Read chunks in batches instead of loading the whole corpus into RAM."""
    batch = []

    with chunks_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main(
    chunks_file: Path,
    database_dir: Path,
    collection_name: str,
    reset_collection: bool = False,
) -> None:
    expected_count = validate_chunks_file(chunks_file)

    print(f"Chunk file: {chunks_file}")
    print(f"Chunks to index: {expected_count:,}")
    print(f"Database directory: {database_dir}")
    print(f"Target collection: {collection_name}")

    database_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(database_dir))

    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }

    if collection_name in existing_names:
        if not reset_collection:
            raise RuntimeError(
                f"Collection '{collection_name}' already exists. "
                "Nothing was overwritten. To rebuild this collection "
                "intentionally, rerun with --reset-collection."
            )

        client.delete_collection(name=collection_name)
        print("Removed only the requested collection.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print("\nCreating embeddings and indexing chunks.")
    print("Please keep this terminal open during indexing.")

    indexed = 0

    for batch in read_batches(chunks_file):
        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[
                {
                    "source_id": str(item["source_id"]),
                    "chunk_index": int(item["chunk_index"]),
                    "chunk_size_words": int(
                        item.get("chunk_size_words", len(item["text"].split()))
                    ),
                }
                for item in batch
            ],
        )

        indexed += len(batch)

        print(f"Indexed {indexed:,}/{expected_count:,} chunks.")

    actual_count = collection.count()

    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} chunks, but collection contains "
            f"{actual_count}. Investigate before using this collection."
        )

    print("\nFull-corpus indexing completed successfully!")
    print(f"Collection: {collection_name}")
    print(f"Stored chunks: {actual_count:,}")

    results = collection.query(
        query_texts=[
            "How can I troubleshoot an IBM software installation error?"
        ],
        n_results=min(3, actual_count),
        include=["documents", "metadatas", "distances"],
    )

    print("\nSample search results:")

    for rank, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"\nResult {rank}")
        print(f"Source: {metadata.get('source_id', 'unknown')}")
        print(f"Distance: {distance:.4f}")
        print(f"Text: {document[:250]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB index from a JSONL chunk file."
    )

    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=DEFAULT_CHUNKS_FILE,
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=DEFAULT_DATABASE_DIR,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and rebuild only the specified collection.",
    )

    args = parser.parse_args()

    main(
        chunks_file=args.chunks_file,
        database_dir=args.database_dir,
        collection_name=args.collection,
        reset_collection=args.reset_collection,
    )
