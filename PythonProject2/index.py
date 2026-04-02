import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from langchain_community.vectorstores import Chroma
from openai_embeddings import OpenAIEmbeddingsAdapter
from prepare_ollama_dataset import chunk_paragraphs, normalize_text, split_into_paragraphs


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_DIR = BASE_DIR / "db"
DEFAULT_CHUNKS_PATH = BASE_DIR / "prepared_ollama" / "chunks.jsonl"
DEFAULT_CSV_PATH = BASE_DIR / "adilet_scraper" / "output.csv"
LEGACY_CSV_PATH = BASE_DIR / "adilet_parsed_texts.csv"


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_chunks(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "metadata": {
                        "source_type": "chunk",
                        "chunk_id": payload.get("id", f"line-{line_number}"),
                        "document_id": payload.get("document_id", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "title": payload.get("title", ""),
                        "url": payload.get("url", ""),
                    },
                }
            )
    return rows


def load_csv(path: Path, limit: Optional[int] = None) -> list[dict]:
    raise_csv_field_limit()

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if limit is not None and len(rows) >= limit:
                break

            title = (row.get("title") or "").strip()
            raw_text = row.get("content")
            if raw_text is None:
                raw_text = row.get("text") or ""
            content = normalize_text(raw_text)
            full_text = "\n\n".join(part for part in [title, content] if part).strip()
            if not full_text:
                continue

            paragraphs = split_into_paragraphs(full_text)
            chunks = chunk_paragraphs(paragraphs, chunk_chars=1800, overlap_chars=250)
            url = (row.get("url") or "").strip()

            for chunk_index, chunk_text in enumerate(chunks, start=1):
                rows.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "source_type": "csv_chunk",
                            "row_id": index,
                            "chunk_index": chunk_index,
                            "title": title,
                            "url": url,
                        },
                    }
                )
    return rows


def load_documents(
    chunks_path: Path,
    csv_path: Path,
    limit: Optional[int] = None,
    source_preference: str = "csv",
) -> list[dict]:
    candidate_csv_paths = [csv_path]
    if csv_path != LEGACY_CSV_PATH:
        candidate_csv_paths.append(LEGACY_CSV_PATH)

    if source_preference == "csv" and csv_path.exists():
        return load_csv(csv_path, limit=limit)
    if source_preference == "chunks" and chunks_path.exists():
        return load_chunks(chunks_path, limit=limit)
    for candidate in candidate_csv_paths:
        if candidate.exists():
            return load_csv(candidate, limit=limit)
    if chunks_path.exists():
        return load_chunks(chunks_path, limit=limit)
    raise FileNotFoundError(
        f"Не найден источник данных. Ожидался {csv_path}, {LEGACY_CSV_PATH} или {chunks_path}."
    )


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_index(
    docs: list[dict],
    db_dir: Path,
    embedding_model: str,
    batch_size: int,
) -> None:
    texts = [doc["text"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]

    embeddings = OpenAIEmbeddingsAdapter(model=embedding_model)
    db_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = None
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        print(
            f"Индексация батча {batch_number}: "
            f"{start + 1}-{start + len(batch_texts)} из {len(texts)}"
        )
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                batch_texts,
                embeddings,
                metadatas=batch_metadatas,
                persist_directory=str(db_dir),
            )
        else:
            vectorstore.add_texts(batch_texts, metadatas=batch_metadatas)

    if vectorstore is None:
        raise RuntimeError("Нет документов для индексации.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация правовых текстов в Chroma для локального RAG-поиска."
    )
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="Каталог для Chroma DB.",
    )
    parser.add_argument(
        "--chunks-path",
        default=str(DEFAULT_CHUNKS_PATH),
        help="Путь к chunks.jsonl.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Запасной путь к CSV с текстами.",
    )
    parser.add_argument(
        "--source-preference",
        choices=["csv", "chunks"],
        default="csv",
        help="Какой источник предпочитать для индексации. По умолчанию CSV, чтобы избежать битой кодировки в старых chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Название embedding-модели в OpenAI API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение числа документов для быстрого теста.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча при записи в Chroma.",
    )
    args = parser.parse_args()

    docs = load_documents(
        chunks_path=Path(args.chunks_path),
        csv_path=Path(args.csv_path),
        limit=args.limit,
        source_preference=args.source_preference,
    )
    print(f"Загружено документов для индексации: {len(docs)}")

    build_index(
        docs=docs,
        db_dir=Path(args.db_dir),
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    print(f"Индексация завершена. База сохранена в: {Path(args.db_dir).resolve()}")


if __name__ == "__main__":
    main()
