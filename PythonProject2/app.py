import os
from pathlib import Path

import requests
import streamlit as st
from langchain_community.vectorstores import Chroma
from nli_judge import judge_norm_pair
from nli_utils import DEFAULT_NLI_MODEL, build_nli_pipeline, run_nli

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover - fallback for older environments
    from langchain_community.embeddings import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3:8b")
EMBEDDING_MODEL = "nomic-embed-text"
NLI_MODEL = os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)


@st.cache_resource
def load_db() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )


@st.cache_resource
def load_nli():
    return build_nli_pipeline(NLI_MODEL)


def is_contradiction(a: str, b: str) -> bool:
    nli = load_nli()
    result = run_nli(nli, a, b)
    return result["is_contradiction_label"] and result["score"] > 0.7


def get_nli_result(a: str, b: str) -> dict:
    nli = load_nli()
    result = run_nli(nli, a, b)
    return {
        "label": result["label"],
        "score": result["score"],
    }


def explain_conflict(query: str, candidate: str) -> str:
    prompt = f"""Ты юридический помощник по законодательству Республики Казахстан.

Норма пользователя:
{query}

Найденная норма:
{candidate}

Если есть противоречие, объясни его в одном коротком предложении.
Если явного противоречия нет, так и напиши: "Явного противоречия не найдено."
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return (payload.get("response") or "").strip()


def find_conflicts(query: str, k: int) -> list[dict]:
    db = load_db()
    docs = db.similarity_search(query, k=k)

    results: list[dict] = []
    for doc in docs:
        if not is_contradiction(query, doc.page_content):
            continue
        judge = judge_norm_pair(query, doc.page_content, model=LLM_MODEL)
        if not judge.is_contradiction and judge.routing != "human_review":
            continue
        explanation = explain_conflict(query, doc.page_content)
        results.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "explanation": explanation,
                "judge": {
                    "label": judge.final_label,
                    "confidence": judge.confidence_score,
                    "requires_human_review": judge.requires_human_review,
                    "routing": judge.routing,
                    "step_1_extract_A": judge.step_1_extract_A,
                    "step_2_extract_B": judge.step_2_extract_B,
                    "step_3_compare": judge.step_3_compare,
                },
            }
        )
    return results


def explain_direct_conflict(a: str, b: str) -> dict:
    nli_result = get_nli_result(a, b)
    judge = judge_norm_pair(a, b, model=LLM_MODEL)
    contradiction = judge.is_contradiction
    explanation = explain_conflict(a, b)
    return {
        "contradiction": contradiction,
        "label": nli_result["label"],
        "score": nli_result["score"],
        "explanation": explanation,
        "judge_label": judge.final_label,
        "judge_confidence": judge.confidence_score,
        "requires_human_review": judge.requires_human_review,
        "routing": judge.routing,
        "step_1_extract_A": judge.step_1_extract_A,
        "step_2_extract_B": judge.step_2_extract_B,
        "step_3_compare": judge.step_3_compare,
    }


def render_result(item: dict, index: int) -> None:
    metadata = item.get("metadata") or {}
    title = metadata.get("title") or f"Норма {index}"
    url = metadata.get("url") or ""

    st.subheader(title)
    if url:
        st.caption(url)
    st.write(item["text"])
    judge = item.get("judge") or {}
    if judge:
        st.write(
            f"Judge label: `{judge.get('label', 'unknown')}` | "
            f"confidence: `{judge.get('confidence', 0.0):.2f}` | "
            f"routing: `{judge.get('routing', 'pass')}` | "
            f"human review: `{judge.get('requires_human_review', False)}`"
        )
        if judge.get("step_1_extract_A"):
            st.write(f"A: {judge['step_1_extract_A']}")
        if judge.get("step_2_extract_B"):
            st.write(f"B: {judge['step_2_extract_B']}")
        if judge.get("step_3_compare"):
            st.write(f"Compare: {judge['step_3_compare']}")
    st.success(item["explanation"])


st.set_page_config(page_title="Поиск правовых противоречий", layout="wide")
st.title("Поиск правовых противоречий")
st.write(
    "Приложение ищет похожие нормы в локальной базе и просит локальную LLM кратко "
    "объяснить возможное противоречие."
)

if not DB_DIR.exists():
    st.warning(
        f"База {DB_DIR} пока не создана. Сначала запустите: `python index.py --limit 500` "
        "или без `--limit` для полной индексации."
    )

query = st.text_area(
    "Введите норму",
    height=220,
    placeholder="Например: Срок подачи налоговой декларации для ИП составляет не позднее 31 марта...",
)
k = st.slider("Сколько похожих норм анализировать", min_value=1, max_value=10, value=3)

st.divider()
st.subheader("Прямая проверка двух норм")
test_a = st.text_area(
    "Норма A",
    height=120,
    value="Курение разрешено во всех общественных местах без ограничений.",
)
test_b = st.text_area(
    "Норма B",
    height=120,
    value="Курение в общественных местах полностью запрещено.",
)

if st.button("Проверить NLI", type="secondary"):
    if not test_a.strip() or not test_b.strip():
        st.error("Введите обе нормы для прямой проверки.")
    else:
        with st.spinner("Проверяю противоречие между двумя нормами..."):
            try:
                direct_result = explain_direct_conflict(test_a.strip(), test_b.strip())
                verdict = "Да" if direct_result["contradiction"] else "Нет"
                st.write(f"BART NLI label: `{direct_result['label']}`")
                st.write(f"BART NLI score: `{direct_result['score']:.4f}`")
                st.write(f"Judge verdict: `{verdict}`")
                st.write(f"Judge label: `{direct_result['judge_label']}`")
                st.write(f"Judge confidence: `{direct_result['judge_confidence']:.4f}`")
                st.write(f"Routing: `{direct_result['routing']}`")
                st.write(f"Human review: `{direct_result['requires_human_review']}`")
                st.write(f"A: {direct_result['step_1_extract_A']}")
                st.write(f"B: {direct_result['step_2_extract_B']}")
                st.write(f"Compare: {direct_result['step_3_compare']}")
                st.success(direct_result["explanation"])
            except Exception as exc:
                import traceback

                st.error(f"Ошибка: {str(exc)}")
                st.text(traceback.format_exc())

st.divider()

if st.button("Найти", type="primary"):
    if not query.strip():
        st.error("Введите текст нормы перед поиском.")
    elif not DB_DIR.exists():
        st.error("Сначала создайте базу командой `python index.py`.")
    else:
        with st.spinner("Ищу похожие нормы и формирую объяснения..."):
            try:
                results = find_conflicts(query.strip(), k=k)
            except Exception as exc:
                import traceback

                st.error(f"Ошибка: {str(exc)}")
                st.text(traceback.format_exc())
                results = []

        if results:
            st.divider()
            for index, item in enumerate(results, start=1):
                render_result(item, index=index)
        else:
            st.info("Ничего не найдено или ответ не удалось сформировать.")
