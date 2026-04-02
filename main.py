import csv
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
PROJECT2_DIR = ROOT_DIR / "PythonProject2"
if str(PROJECT2_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT2_DIR))

from nli_judge import judge_norm_pair  # noqa: E402
from nli_utils import DEFAULT_NLI_MODEL, build_nli_pipeline, run_nli  # noqa: E402
from openai_config import OPENAI_LLM_MODEL, OPENAI_TIMEOUT_SEC, get_openai_client  # noqa: E402
from openai_embeddings import OpenAIEmbeddingsAdapter  # noqa: E402
from prepare_ollama_dataset import chunk_paragraphs, normalize_text, split_into_paragraphs  # noqa: E402

try:
    from langchain_chroma import Chroma  # noqa: E402
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore # noqa: E402


DB_DIR = PROJECT2_DIR / "db"
PRIMARY_CSV_PATH = PROJECT2_DIR / "adilet_scraper" / "output.csv"
LEGACY_CSV_PATH = PROJECT2_DIR / "adilet_parsed_texts.csv"
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", OPENAI_LLM_MODEL)
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
NLI_MODEL = os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)
MAX_DISTANCE = 1.1
DEFAULT_TOP_K = 3
MAX_SOURCE_TEXT_CHARS = 700
MAX_TOTAL_CONTEXT_CHARS = 2200
KEYWORD_CHUNK_CHARS = 1400
KEYWORD_OVERLAP_CHARS = 180
MAX_JUDGE_VECTOR_DISTANCE = 0.55
MIN_JUDGE_KEYWORD_SCORE = 3.0
NORM_HINT_PATTERNS = (
    r"\bзапрещ",
    r"\bразреш",
    r"\bобязан",
    r"\bвлеч[её]т",
    r"\bнаказыва",
    r"\bподлеж",
    r"\bне допуска",
    r"\bпредусмотр",
    r"\bзапрещает",
)
TITLE_QUERY_PATTERNS = (
    r"\bкодекс\b",
    r"\bзакон\b",
    r"\bправил[аы]\b",
    r"\bпостановлени[ея]\b",
    r"\bприказ\b",
    r"\bконституци[яи]\b",
)
SEARCH_STOPWORDS = {
    "что",
    "будет",
    "если",
    "когда",
    "какой",
    "какая",
    "какие",
    "как",
    "где",
    "куда",
    "кто",
    "это",
    "при",
    "для",
    "без",
    "над",
    "под",
    "или",
    "ли",
    "по",
    "на",
    "за",
    "от",
    "до",
    "из",
    "в",
    "во",
    "с",
    "со",
    "у",
    "о",
    "об",
    "про",
    "не",
    "я",
    "мы",
    "вы",
    "он",
    "она",
    "они",
    "мне",
    "тебе",
    "руль",
    "сесть",
}
QUERY_TOKEN_EXPANSIONS = {
    "пьяным": ["опьянения", "алкогольного", "нетрезвом"],
    "пьяный": ["опьянения", "алкогольного", "нетрезвом"],
    "пьяная": ["опьянения", "алкогольного", "нетрезвом"],
    "пьяное": ["опьянения", "алкогольного", "нетрезвом"],
    "пьяные": ["опьянения", "алкогольного", "нетрезвом"],
    "рулем": ["управлять", "транспортным", "средством"],
    "рулём": ["управлять", "транспортным", "средством"],
    "вождение": ["управлять", "транспортным", "средством"],
    "машиной": ["транспортным", "средством", "автомобилем"],
    "авто": ["автомобилем", "транспортным", "средством"],
}

app = FastAPI(
    title="ZanAi Backend",
    description="API for legal search, retrieval-augmented answers, and contradiction analysis.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Norm or legal query to analyze.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10, description="How many passages to inspect.")


class CompareRequest(BaseModel):
    text_a: str = Field(..., min_length=5, description="First norm to compare.")
    text_b: str = Field(..., min_length=5, description="Second norm to compare.")


class JudgePayload(BaseModel):
    label: str
    confidence: float
    routing: str
    requires_human_review: bool
    step_1_extract_A: str
    step_2_extract_B: str
    step_3_compare: str


class SourceItem(BaseModel):
    title: str
    url: str
    text: str
    distance: float
    baseline_label: str
    baseline_score: float
    contradiction: bool
    judge: Optional[JudgePayload]


class FindingItem(BaseModel):
    title: str
    url: str
    signal: str
    explanation: str


class NormStatus(BaseModel):
    label: Literal["likely_active", "amendment_detected", "stale_or_lost_force", "requires_review"]
    title: str
    summary: str


class AnalyzeResponse(BaseModel):
    status: Literal["ok"]
    query: str
    answer: str
    sources: list[SourceItem]
    judge_summary: str
    analysis_mode: Literal["norm_review", "title_search", "legal_question"]
    executive_summary: str
    primary_document_title: str
    norm_status: NormStatus
    related_documents: list[FindingItem]
    possible_conflicts: list[FindingItem]
    possible_duplicates: list[FindingItem]
    staleness_signals: list[FindingItem]
    total_sources: int
    retrieval_threshold: float


class CompareResponse(BaseModel):
    status: Literal["ok"]
    text_a: str
    text_b: str
    contradiction: bool
    baseline_label: str
    baseline_score: float
    judge: JudgePayload
    summary: str


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def get_csv_path() -> Path:
    if PRIMARY_CSV_PATH.exists():
        return PRIMARY_CSV_PATH
    return LEGACY_CSV_PATH


def get_db() -> Chroma:
    if not DB_DIR.exists():
        raise RuntimeError(f"Vector DB not found: {DB_DIR}")
    embeddings = OpenAIEmbeddingsAdapter(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )


def get_nli():
    if not hasattr(get_nli, "_instance"):
        get_nli._instance = build_nli_pipeline(NLI_MODEL)  # type: ignore[attr-defined]
    return get_nli._instance  # type: ignore[attr-defined]


def get_nli_result(a: str, b: str) -> dict:
    nli = get_nli()
    result = run_nli(nli, a, b)
    return {
        "label": result["label"],
        "score": result["score"],
        "is_contradiction": result["is_contradiction_label"] and result["score"] >= 0.5,
    }


def classify_analysis_mode(query: str) -> Literal["norm_review", "title_search", "legal_question"]:
    if query_looks_like_norm(query):
        return "norm_review"
    if query_looks_like_title_search(query):
        return "title_search"
    return "legal_question"


def tokenize_for_search(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    raw_tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁәіңғүұқөһІі0-9-]+", normalized)
    tokens: list[str] = []
    seen: set[str] = set()

    for token in raw_tokens:
        if len(token) < 3 or token in SEARCH_STOPWORDS:
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
        for expanded in QUERY_TOKEN_EXPANSIONS.get(token, []):
            if expanded not in seen:
                tokens.append(expanded)
                seen.add(expanded)

    return tokens


def get_keyword_chunks() -> list[dict]:
    if hasattr(get_keyword_chunks, "_cache"):
        return get_keyword_chunks._cache  # type: ignore[attr-defined]

    raise_csv_field_limit()
    chunks: list[dict] = []
    csv_path = get_csv_path()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_id, row in enumerate(reader, start=1):
            title = normalize_text(row.get("title") or "")
            content = normalize_text(row.get("content") or row.get("text") or "")
            url = (row.get("url") or "").strip()
            full_text = "\n\n".join(part for part in [title, content] if part).strip()
            if not full_text:
                continue

            paragraphs = split_into_paragraphs(full_text)
            for chunk_index, chunk_text in enumerate(
                chunk_paragraphs(
                    paragraphs,
                    chunk_chars=KEYWORD_CHUNK_CHARS,
                    overlap_chars=KEYWORD_OVERLAP_CHARS,
                ),
                start=1,
            ):
                chunks.append(
                    {
                        "row_id": row_id,
                        "chunk_index": chunk_index,
                        "title": title or "Найденная норма",
                        "url": url,
                        "text": chunk_text,
                        "normalized_title": title.lower(),
                        "normalized_text": chunk_text.lower(),
                    }
                )

    get_keyword_chunks._cache = chunks  # type: ignore[attr-defined]
    return chunks


def query_looks_like_norm(query: str) -> bool:
    normalized = normalize_text(query).lower()
    if len(normalized) < 40:
        return False
    if re.search(r"\bстать[яи]\b|\bпункт\b|\bчаст[ьи]\b", normalized):
        return True
    return any(re.search(pattern, normalized) for pattern in NORM_HINT_PATTERNS)


def query_looks_like_title_search(query: str) -> bool:
    normalized = normalize_text(query).lower().strip()
    if len(normalized) < 12:
        return False
    if not any(re.search(pattern, normalized) for pattern in TITLE_QUERY_PATTERNS):
        return False
    return not query_looks_like_norm(normalized)


def score_keyword_chunk(query: str, query_tokens: list[str], chunk: dict) -> float:
    title = chunk["normalized_title"]
    text = chunk["normalized_text"]
    full_query = normalize_text(query).lower()

    token_hits_in_title = sum(1 for token in query_tokens if token in title)
    token_hits_in_text = sum(1 for token in query_tokens if token in text)
    exact_phrase_bonus = 4.0 if full_query and (full_query in text or full_query in title) else 0.0
    exact_title_bonus = 10.0 if full_query and full_query == title.strip() else 0.0
    title_prefix_bonus = 6.0 if full_query and title.startswith(full_query) else 0.0
    amendment_penalty = 0.0

    if full_query and full_query in title and title.startswith("о внесении "):
        amendment_penalty = 7.0

    return (
        exact_phrase_bonus
        + exact_title_bonus
        + title_prefix_bonus
        + token_hits_in_title * 2.5
        + token_hits_in_text * 1.0
        - amendment_penalty
    )


def keyword_search(query: str, limit: int) -> list[dict]:
    query_tokens = tokenize_for_search(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for chunk in get_keyword_chunks():
        score = score_keyword_chunk(query, query_tokens, chunk)
        if score <= 0:
            continue
        scored.append((score, chunk))

    scored.sort(
        key=lambda item: (
            item[0],
            len(item[1]["text"]),
        ),
        reverse=True,
    )

    results: list[dict] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for score, chunk in scored:
        key = (chunk["title"], chunk["url"], chunk["chunk_index"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pseudo_distance = max(0.05, 0.95 - min(score, 6.0) * 0.12)
        results.append(
            {
                "title": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
                "distance": pseudo_distance,
                "retrieval_method": "keyword",
                "keyword_score": score,
            }
        )
        if len(results) >= limit:
            break

    return results


def vector_search(query: str, limit: int) -> list[dict]:
    db = get_db()
    retrieved = db.similarity_search_with_score(query, k=max(limit * 2, 6))
    results: list[dict] = []
    for doc, distance in retrieved:
        metadata = doc.metadata or {}
        results.append(
            {
                "title": str(metadata.get("title", "Найденная норма")),
                "url": str(metadata.get("url", "")),
                "text": doc.page_content,
                "distance": float(distance),
                "retrieval_method": "vector",
                "keyword_score": 0.0,
            }
        )
    return results


def hybrid_search(query: str, limit: int) -> list[dict]:
    keyword_results = keyword_search(query, limit=max(limit, 6))
    vector_results = vector_search(query, limit=max(limit, 6))

    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for item in keyword_results + vector_results:
        key = (
            item["title"],
            item["url"],
            normalize_text(item["text"][:220]).lower(),
        )
        if key in seen:
            continue
        seen.add(key)

        if item["retrieval_method"] == "vector" and item["distance"] > MAX_DISTANCE:
            continue

        merged.append(item)

    if query_looks_like_title_search(query):
        base_acts = [item for item in merged if not item["title"].lower().startswith("о внесении ")]
        amendment_acts = [item for item in merged if item["title"].lower().startswith("о внесении ")]
        if base_acts:
            merged = base_acts + amendment_acts

    merged.sort(
        key=lambda item: (
            1 if query_looks_like_title_search(query) and item["title"].lower().startswith("о внесении ") else 0,
            0 if item["retrieval_method"] == "keyword" else 1,
            item["distance"],
            -item["keyword_score"],
        )
    )
    return merged[:limit]


def is_strong_match(item: dict) -> bool:
    if item["retrieval_method"] == "keyword":
        return float(item["keyword_score"]) >= MIN_JUDGE_KEYWORD_SCORE
    return float(item["distance"]) <= MAX_JUDGE_VECTOR_DISTANCE


def looks_stale(source: SourceItem) -> bool:
    haystack = normalize_text(f"{source.title}\n{source.text}").lower()
    return "утратил силу" in haystack or "утратила силу" in haystack


def looks_like_amendment(source: SourceItem) -> bool:
    return source.title.strip().lower().startswith("о внесении ")


def build_related_documents(sources: list[SourceItem]) -> list[FindingItem]:
    items: list[FindingItem] = []
    for source in sources:
        signal = "related_act"
        explanation = "Найден связанный нормативный акт или релевантная редакция по теме запроса."
        if looks_like_amendment(source):
            signal = "amendment_act"
            explanation = "Найден акт о внесении изменений, который может описывать эволюцию базовой нормы."
        elif looks_stale(source):
            signal = "stale_act"
            explanation = "Найден акт с явным сигналом устаревания или утраты силы."

        items.append(
            FindingItem(
                title=source.title,
                url=source.url,
                signal=signal,
                explanation=explanation,
            )
        )
    return items


def build_conflict_findings(sources: list[SourceItem]) -> list[FindingItem]:
    findings: list[FindingItem] = []
    for source in sources:
        if not source.contradiction:
            continue
        explanation = "Найден потенциальный конфликт по результатам baseline NLI и judge."
        if source.judge and source.judge.step_3_compare:
            explanation = source.judge.step_3_compare
        findings.append(
            FindingItem(
                title=source.title,
                url=source.url,
                signal="contradiction",
                explanation=explanation,
            )
        )
    return findings


def build_duplicate_findings(sources: list[SourceItem]) -> list[FindingItem]:
    findings: list[FindingItem] = []
    seen_titles: set[str] = set()
    for source in sources:
        normalized_title = normalize_text(source.title).lower()
        if normalized_title in seen_titles:
            findings.append(
                FindingItem(
                    title=source.title,
                    url=source.url,
                    signal="duplicate_title",
                    explanation="В выдаче повторяется очень близкий по названию акт, возможен дубль или близкая редакция.",
                )
            )
            continue
        seen_titles.add(normalized_title)

        if looks_like_amendment(source):
            findings.append(
                FindingItem(
                    title=source.title,
                    url=source.url,
                    signal="version_overlap",
                    explanation="Документ выглядит как поправка к базовому акту и требует сравнения версий.",
                )
            )
    return findings


def build_staleness_findings(sources: list[SourceItem]) -> list[FindingItem]:
    findings: list[FindingItem] = []
    for source in sources:
        if not looks_stale(source):
            continue
        findings.append(
            FindingItem(
                title=source.title,
                url=source.url,
                signal="lost_force",
                explanation="В тексте найден сигнал 'утратил силу', значит норма может быть устаревшей.",
            )
        )
    return findings


def build_executive_summary(
    mode: Literal["norm_review", "title_search", "legal_question"],
    sources: list[SourceItem],
    conflicts: list[FindingItem],
    duplicates: list[FindingItem],
    staleness: list[FindingItem],
) -> str:
    if not sources:
        return "В базе не найдено достаточно близких актов для анализа."

    lead = f"Найдено связанных актов: {len(sources)}."
    if mode == "norm_review":
        lead = f"Проверка нормы завершена. Найдено связанных актов: {len(sources)}."
    elif mode == "title_search":
        lead = f"Поиск по названию акта завершен. Найдено связанных документов: {len(sources)}."

    risk_parts: list[str] = []
    if conflicts:
        risk_parts.append(f"Потенциальных противоречий: {len(conflicts)}")
    if duplicates:
        risk_parts.append(f"сигналов дублирования/версий: {len(duplicates)}")
    if staleness:
        risk_parts.append(f"сигналов устаревания: {len(staleness)}")

    if not risk_parts:
        return f"{lead} Явных сигналов конфликта, дублирования или устаревания среди верхних результатов не найдено."
    return f"{lead} " + ", ".join(risk_parts) + "."


def build_norm_status(
    sources: list[SourceItem],
    conflicts: list[FindingItem],
    duplicates: list[FindingItem],
    staleness: list[FindingItem],
) -> NormStatus:
    if not sources:
        return NormStatus(
            label="requires_review",
            title="Требует проверки",
            summary="Верхние результаты не дали достаточно надёжной базы для вывода по актуальности нормы.",
        )

    if staleness:
        return NormStatus(
            label="stale_or_lost_force",
            title="Есть риск устаревания",
            summary="Среди найденных актов есть явные сигналы утраты силы или устаревания.",
        )

    if conflicts:
        return NormStatus(
            label="requires_review",
            title="Требует проверки",
            summary="Найдены потенциальные противоречия, поэтому норму лучше отправить на ручную юридическую проверку.",
        )

    if duplicates:
        return NormStatus(
            label="amendment_detected",
            title="Есть поправки или близкие версии",
            summary="По найденным актам видно, что норма связана с поправками или несколькими близкими редакциями.",
        )

    return NormStatus(
        label="likely_active",
        title="Похоже на актуальную норму",
        summary="Верхние результаты не показывают явных конфликтов, устаревания или проблемных версий.",
    )


def generate_with_openai(prompt: str, model: str = LLM_MODEL, timeout: int = int(OPENAI_TIMEOUT_SEC)) -> str:
    client = get_openai_client()
    response = client.responses.create(
        model=model,
        input=prompt,
        timeout=timeout,
    )
    return (response.output_text or "").strip()


def generate_answer(query: str, sources: list[SourceItem]) -> str:
    context_blocks = []
    total_context_chars = 0
    for index, source in enumerate(sources, start=1):
        trimmed_text = source.text.strip()
        if len(trimmed_text) > MAX_SOURCE_TEXT_CHARS:
            trimmed_text = trimmed_text[:MAX_SOURCE_TEXT_CHARS].rstrip() + "..."

        block = (
            f"Источник {index}\n"
            f"Название: {source.title}\n"
            f"URL: {source.url or 'не указан'}\n"
            f"Текст: {trimmed_text}"
        )

        projected_total = total_context_chars + len(block)
        if projected_total > MAX_TOTAL_CONTEXT_CHARS and context_blocks:
            break

        context_blocks.append(
            block
        )
        total_context_chars += len(block)
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Ты — AI-система для анализа законодательной энтропии в нормативных актах Республики Казахстан.
Используя только предоставленный контекст, дай краткий аналитический вывод по запросу.
Сфокусируйся на четырех вещах: что это за норма или акт, какие есть связанные документы, есть ли признаки конфликта/дублирования/устаревания, что нужно проверить дальше.
Обязательно упоминай названия документов, на которые опираешься.
Ответ должен быть коротким: не больше 5 предложений.
Если в контексте нет точного ответа, честно скажи: "Я не нашел точной информации в загруженной базе."

КОНТЕКСТ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}
"""
    return generate_with_openai(prompt, timeout=60)


def build_judge_summary(sources: list[SourceItem], judge_enabled: bool) -> str:
    if not judge_enabled:
        return "Judge не запускался: запрос интерпретирован как поисковый, а не как сравнение двух норм."

    contradictions = [source for source in sources if source.contradiction]
    if not contradictions:
        return "Judge не обнаружил явных противоречий в найденных источниках."

    top = max(
        contradictions,
        key=lambda item: (
            item.judge.confidence if item.judge else 0.0,
            item.baseline_score,
        ),
    )
    if not top.judge:
        return "Judge отметил потенциальное противоречие, но explainability недоступен."

    return (
        f"Judge обнаружил противоречие в источнике «{top.title}». "
        f"Метка: {top.judge.label}, confidence: {top.judge.confidence:.2f}, routing: {top.judge.routing}. "
        f"Объяснение: {top.judge.step_3_compare}"
    )


def build_compare_summary(baseline: dict, judge: JudgePayload) -> str:
    contradiction = judge.label == "contradiction"
    verdict = "обнаружено противоречие" if contradiction else "явного противоречия не обнаружено"
    return (
        f"Прямое сравнение завершено: {verdict}. "
        f"Baseline NLI: {baseline['label']} ({baseline['score']:.2f}). "
        f"Judge: {judge.label}, confidence {judge.confidence:.2f}, routing {judge.routing}."
    )


def analyze_query(query: str, top_k: int) -> Tuple[str, list[SourceItem], str]:
    retrieved = hybrid_search(query, limit=top_k)
    if not retrieved:
        answer = "Я не нашел точной информации в загруженной базе. Попробуйте переформулировать запрос."
        return answer, [], "Judge не запускался, потому что retrieval не нашел достаточно близких источников."

    strong_matches = [item for item in retrieved if is_strong_match(item)]
    judge_enabled = query_looks_like_norm(query) and bool(strong_matches)
    sources: list[SourceItem] = []
    for item in retrieved:
        baseline = get_nli_result(query, item["text"])
        judge_payload = None
        contradiction = False

        if judge_enabled and is_strong_match(item) and baseline["is_contradiction"]:
            judge = judge_norm_pair(query, item["text"], model=LLM_MODEL)
            contradiction = judge.is_contradiction
            judge_payload = JudgePayload(
                label=judge.final_label,
                confidence=judge.confidence_score,
                routing=judge.routing,
                requires_human_review=judge.requires_human_review,
                step_1_extract_A=judge.step_1_extract_A,
                step_2_extract_B=judge.step_2_extract_B,
                step_3_compare=judge.step_3_compare,
            )

        sources.append(
            SourceItem(
                title=item["title"],
                url=item["url"],
                text=item["text"],
                distance=float(item["distance"]),
                baseline_label=baseline["label"],
                baseline_score=baseline["score"],
                contradiction=contradiction,
                judge=judge_payload,
            )
        )

    answer = generate_answer(query, sources)
    if query_looks_like_norm(query) and not strong_matches:
        judge_summary = (
            "Judge не запускался: в базе не найдено достаточно близкого нормативного совпадения "
            "для уверенного сравнения с запросом."
        )
    else:
        judge_summary = build_judge_summary(sources, judge_enabled=judge_enabled)
    return answer, sources, judge_summary


@app.get("/health")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "service": "zanai-backend",
        "db_ready": DB_DIR.exists(),
        "llm_model": LLM_MODEL,
        "nli_model": NLI_MODEL,
        "default_top_k": DEFAULT_TOP_K,
        "retrieval_threshold": MAX_DISTANCE,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_law(request: AnalyzeRequest) -> AnalyzeResponse:
    query = request.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        answer, sources, judge_summary = analyze_query(query, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    analysis_mode = classify_analysis_mode(query)
    related_documents = build_related_documents(sources)
    possible_conflicts = build_conflict_findings(sources)
    possible_duplicates = build_duplicate_findings(sources)
    staleness_signals = build_staleness_findings(sources)
    norm_status = build_norm_status(
        sources,
        possible_conflicts,
        possible_duplicates,
        staleness_signals,
    )
    executive_summary = build_executive_summary(
        analysis_mode,
        sources,
        possible_conflicts,
        possible_duplicates,
        staleness_signals,
    )
    primary_document_title = sources[0].title if sources else ""

    return AnalyzeResponse(
        status="ok",
        query=query,
        answer=answer,
        sources=sources,
        judge_summary=judge_summary,
        analysis_mode=analysis_mode,
        executive_summary=executive_summary,
        primary_document_title=primary_document_title,
        norm_status=norm_status,
        related_documents=related_documents,
        possible_conflicts=possible_conflicts,
        possible_duplicates=possible_duplicates,
        staleness_signals=staleness_signals,
        total_sources=len(sources),
        retrieval_threshold=MAX_DISTANCE,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_norms(request: CompareRequest) -> CompareResponse:
    text_a = request.text_a.strip()
    text_b = request.text_b.strip()
    if not text_a or not text_b:
        raise HTTPException(status_code=400, detail="Both text_a and text_b must not be empty.")

    try:
        baseline = get_nli_result(text_a, text_b)
        judge = judge_norm_pair(text_a, text_b, model=LLM_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    judge_payload = JudgePayload(
        label=judge.final_label,
        confidence=judge.confidence_score,
        routing=judge.routing,
        requires_human_review=judge.requires_human_review,
        step_1_extract_A=judge.step_1_extract_A,
        step_2_extract_B=judge.step_2_extract_B,
        step_3_compare=judge.step_3_compare,
    )
    contradiction = judge_payload.label == "contradiction"

    return CompareResponse(
        status="ok",
        text_a=text_a,
        text_b=text_b,
        contradiction=contradiction,
        baseline_label=baseline["label"],
        baseline_score=baseline["score"],
        judge=judge_payload,
        summary=build_compare_summary(baseline, judge_payload),
    )


# Run locally with:
# python -m uvicorn main:app --reload
