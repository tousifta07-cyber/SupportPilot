
import argparse
import html
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "techqa"
    / "extracted"
    / "validation"
    / "validation_technotes.json"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fixed_chunks.jsonl"
)

CHUNK_SIZE_WORDS = 300
OVERLAP_WORDS = 50


def extract_text(html_content: str) -> str:
    """Convert HTML content into clean, readable text."""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[str]:
    """Split text into fixed-size word chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def main(input_file: Path, output_file: Path) -> None:
    """Create chunks from the selected document corpus."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Document file not found: {input_file}"
        )

    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output files must be different.")

    print(f"Loading documents from: {input_file}")

    with input_file.open("r", encoding="utf-8-sig") as file:
        documents = json.load(file)

    if not isinstance(documents, dict):
        raise ValueError(
            "Expected the TechQA documents to be a JSON object."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    empty_documents = 0

    with output_file.open("w", encoding="utf-8") as output:
        for document_number, (source_id, document) in enumerate(
            documents.items(), start=1
        ):
            raw_html = document.get("content", "")

            if not isinstance(raw_html, str) or not raw_html.strip():
                empty_documents += 1
                continue

            text = extract_text(raw_html)

            if not text:
                empty_documents += 1
                continue

            chunks = split_into_chunks(text)

            for chunk_index, chunk_text in enumerate(chunks):
                record = {
                    "chunk_id": f"{source_id}_chunk_{chunk_index}",
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "chunk_size_words": len(chunk_text.split()),
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                total_chunks += 1

            if document_number % 5000 == 0:
                print(
                    f"Processed {document_number}/"
                    f"{len(documents)} documents..."
                )

    print("\nIngestion completed!")
    print(f"Documents loaded: {len(documents)}")
    print(f"Documents without usable text: {empty_documents}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Chunk size: {CHUNK_SIZE_WORDS} words")
    print(f"Overlap: {OVERLAP_WORDS} words")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create fixed-size chunks from TechQA documents."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the source TechQA JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path where the JSONL chunks should be saved.",
    )

    args = parser.parse_args()
    main(args.input, args.output)
