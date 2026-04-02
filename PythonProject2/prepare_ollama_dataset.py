import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if parts:
        return parts
    return [line.strip() for line in text.split("\n") if line.strip()]


def chunk_paragraphs(paragraphs: Iterable[str], chunk_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_length = 0

            start = 0
            step = max(1, chunk_chars - overlap_chars)
            while start < len(paragraph):
                piece = paragraph[start : start + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
                start += step
            continue

        projected = current_length + len(paragraph) + (2 if current else 0)
        if projected <= chunk_chars:
            current.append(paragraph)
            current_length = projected
            continue

        if current:
            chunks.append("\n\n".join(current).strip())

        if overlap_chars > 0 and chunks:
            tail = chunks[-1][-overlap_chars:].strip()
            current = [tail, paragraph] if tail else [paragraph]
        else:
            current = [paragraph]
        current_length = sum(len(item) for item in current) + max(0, len(current) - 1) * 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_outputs(
    input_csv: Path,
    output_dir: Path,
    chunk_chars: int,
    overlap_chars: int,
    max_docs: Optional[int],
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    corpus_path = output_dir / "pretrain_corpus.txt"

    documents: list[dict] = []
    chunks: list[dict] = []
    corpus_blocks: list[str] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for index, row in enumerate(reader):
            if max_docs is not None and index >= max_docs:
                break

            url = (row.get("url") or "").strip()
            title = normalize_text(row.get("title") or "")
            content = normalize_text(row.get("content") or "")

            if not title and not content:
                continue

            document_id = f"doc-{index + 1:05d}"
            full_text = f"{title}\n\n{content}".strip() if content else title
            paragraphs = split_into_paragraphs(full_text)
            chunk_texts = chunk_paragraphs(paragraphs, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

            documents.append(
                {
                    "document_id": document_id,
                    "url": url,
                    "title": title,
                    "text": full_text,
                    "chunk_count": len(chunk_texts),
                }
            )

            for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                chunks.append(
                    {
                        "id": f"{document_id}-chunk-{chunk_index:03d}",
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "url": url,
                        "title": title,
                        "text": chunk_text,
                    }
                )

            corpus_blocks.append(full_text)

    document_count = write_jsonl(documents_path, documents)
    chunk_count = write_jsonl(chunks_path, chunks)

    with corpus_path.open("w", encoding="utf-8") as handle:
        for block in corpus_blocks:
            handle.write(block)
            handle.write("\n\n<|endofdoc|>\n\n")

    return document_count, chunk_count


def main() -> None:
    raise_csv_field_limit()

    parser = argparse.ArgumentParser(
        description="Prepare Adilet CSV data for local Ollama workflows and downstream fine-tuning."
    )
    parser.add_argument(
        "--input",
        default="PythonProject2/adilet_parsed_texts_15k.csv",
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="PythonProject2/prepared_ollama",
        help="Directory where prepared files will be written.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1800,
        help="Approximate chunk size in characters for retrieval.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=250,
        help="Character overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for quick experiments.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise SystemExit(f"Input file not found: {input_csv}")

    documents, chunks = build_outputs(
        input_csv=input_csv,
        output_dir=output_dir,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        max_docs=args.max_docs,
    )

    print(f"Prepared {documents} documents and {chunks} chunks in {output_dir}")
    print(f"- {output_dir / 'documents.jsonl'}")
    print(f"- {output_dir / 'chunks.jsonl'}")
    print(f"- {output_dir / 'pretrain_corpus.txt'}")


if __name__ == "__main__":
    main()
