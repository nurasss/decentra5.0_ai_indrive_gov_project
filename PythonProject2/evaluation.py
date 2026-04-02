import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from langchain_community.vectorstores import Chroma
from nli_judge import judge_norm_pair
from nli_utils import DEFAULT_NLI_MODEL, build_nli_pipeline, run_nli

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover - fallback for older environments
    from langchain_community.embeddings import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "eval_dataset.jsonl"
DEFAULT_PIPELINE_DATASET = BASE_DIR / "pipeline_eval_dataset.jsonl"
DEFAULT_ERROR_EXPORT = BASE_DIR / "judge_errors.jsonl"
DEFAULT_BASELINE_ERROR_EXPORT = BASE_DIR / "baseline_errors.jsonl"
DEFAULT_HELDOUT_TEMPLATE = BASE_DIR / "heldout_eval_template.jsonl"
DEFAULT_DB_DIR = BASE_DIR / "db"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
NLI_MODEL = os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"case-{line_number:03d}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def build_nli():
    return build_nli_pipeline(NLI_MODEL)


def load_vector_db(db_dir: Path) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )


def get_nli_result(nli, a: str, b: str, threshold: float) -> dict:
    result = run_nli(nli, a, b)
    predicted = result["is_contradiction_label"] and result["score"] >= threshold
    return {
        "label": result["label"],
        "normalized_label": result["normalized_label"],
        "score": result["score"],
        "predicted_contradiction": predicted,
    }


def explain_conflict(a: str, b: str) -> str:
    prompt = f"""Ты юридический помощник по законодательству Республики Казахстан.

Норма А:
{a}

Норма Б:
{b}

Если между нормами есть противоречие, объясни его одним коротким предложением.
Если явного противоречия нет, ответь: "Явного противоречия не найдено."
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


def compute_metrics(rows: list[dict]) -> dict:
    tp = sum(1 for row in rows if row["actual"] and row["predicted"])
    tn = sum(1 for row in rows if not row["actual"] and not row["predicted"])
    fp = sum(1 for row in rows if not row["actual"] and row["predicted"])
    fn = sum(1 for row in rows if row["actual"] and not row["predicted"])
    total = len(rows)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": {
            "actual_contradiction_predicted_contradiction": tp,
            "actual_no_contradiction_predicted_no_contradiction": tn,
            "actual_no_contradiction_predicted_contradiction": fp,
            "actual_contradiction_predicted_no_contradiction": fn,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_sweep(dataset: list[dict], thresholds: list[float]) -> list[dict]:
    nli = build_nli()
    summary: list[dict] = []

    for threshold in thresholds:
        rows: list[dict] = []
        for item in dataset:
            nli_result = get_nli_result(nli, item["text_a"], item["text_b"], threshold=threshold)
            rows.append(
                {
                    "actual": bool(item["contradiction"]),
                    "predicted": bool(nli_result["predicted_contradiction"]),
                }
            )
        metrics = compute_metrics(rows)
        metrics["threshold"] = threshold
        summary.append(metrics)

    summary.sort(key=lambda row: (row["f1"], row["accuracy"], row["precision"], row["recall"]), reverse=True)
    return summary


def evaluate_nli(dataset: list[dict], threshold: float) -> Tuple[list[dict], dict]:
    nli = build_nli()
    results: list[dict] = []

    for row in dataset:
        started = time.perf_counter()
        nli_result = get_nli_result(nli, row["text_a"], row["text_b"], threshold=threshold)
        duration = time.perf_counter() - started
        results.append(
            {
                "id": row["id"],
                "title": row.get("title", row["id"]),
                "actual": bool(row["contradiction"]),
                "predicted": bool(nli_result["predicted_contradiction"]),
                "label": nli_result["label"],
                "normalized_label": nli_result["normalized_label"],
                "score": nli_result["score"],
                "latency_sec": duration,
            }
        )

    metrics = compute_metrics(results)
    metrics["avg_latency_sec"] = statistics.mean(row["latency_sec"] for row in results)
    return results, metrics


def evaluate_ollama(dataset: list[dict], predicted_positive_rows: list[dict]) -> dict:
    matched = {row["id"] for row in predicted_positive_rows}
    explanation_rows = [row for row in dataset if row["id"] in matched]

    if not explanation_rows:
        return {
            "count": 0,
            "success_rate": 0.0,
            "avg_latency_sec": 0.0,
        }

    latencies: list[float] = []
    success_count = 0
    for row in explanation_rows:
        started = time.perf_counter()
        try:
            answer = explain_conflict(row["text_a"], row["text_b"])
            if answer:
                success_count += 1
        except Exception:
            answer = ""
        latencies.append(time.perf_counter() - started)
        row["explanation"] = answer

    return {
        "count": len(explanation_rows),
        "success_rate": success_count / len(explanation_rows),
        "avg_latency_sec": statistics.mean(latencies) if latencies else 0.0,
    }


def evaluate_judge(dataset: list[dict]) -> Tuple[list[dict], dict]:
    results: list[dict] = []
    for row in dataset:
        started = time.perf_counter()
        decision = judge_norm_pair(row["text_a"], row["text_b"], model=LLM_MODEL)
        duration = time.perf_counter() - started
        results.append(
            {
                "id": row["id"],
                "title": row.get("title", row["id"]),
                "actual": bool(row["contradiction"]),
                "predicted": decision.is_contradiction,
                "label": decision.final_label,
                "score": decision.confidence_score,
                "routing": decision.routing,
                "review": decision.requires_human_review,
                "latency_sec": duration,
            }
        )
    metrics = compute_metrics(results)
    metrics["avg_latency_sec"] = statistics.mean(row["latency_sec"] for row in results)
    return results, metrics


def print_report(results: list[dict], metrics: dict, ollama_metrics: Optional[dict]) -> None:
    print("\n=== NLI METRICS ===")
    print(f"Total cases:   {metrics['total']}")
    print(f"Accuracy:      {metrics['accuracy']:.3f}")
    print(f"Precision:     {metrics['precision']:.3f}")
    print(f"Recall:        {metrics['recall']:.3f}")
    print(f"F1:            {metrics['f1']:.3f}")
    print(f"Avg latency:   {metrics['avg_latency_sec']:.3f}s")
    print(f"TP / TN / FP / FN: {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print("Confusion matrix:")
    print(
        f"  actual contradiction -> predicted contradiction: {metrics['confusion_matrix']['actual_contradiction_predicted_contradiction']}"
    )
    print(
        f"  actual contradiction -> predicted no contradiction: {metrics['confusion_matrix']['actual_contradiction_predicted_no_contradiction']}"
    )
    print(
        f"  actual no contradiction -> predicted contradiction: {metrics['confusion_matrix']['actual_no_contradiction_predicted_contradiction']}"
    )
    print(
        f"  actual no contradiction -> predicted no contradiction: {metrics['confusion_matrix']['actual_no_contradiction_predicted_no_contradiction']}"
    )

    print("\n=== CASES ===")
    for row in results:
        status = "OK" if row["actual"] == row["predicted"] else "MISS"
        actual = "contradiction" if row["actual"] else "no contradiction"
        predicted = "contradiction" if row["predicted"] else "no contradiction"
        print(
            f"[{status}] {row['id']} | actual={actual} | predicted={predicted} "
            f"| label={row['label']} | score={row['score']:.4f} | {row['latency_sec']:.3f}s"
        )

    if ollama_metrics is not None:
        print("\n=== OLLAMA EXPLANATION METRICS ===")
        print(f"Evaluated cases: {ollama_metrics['count']}")
        print(f"Success rate:    {ollama_metrics['success_rate']:.3f}")
        print(f"Avg latency:     {ollama_metrics['avg_latency_sec']:.3f}s")


def print_threshold_report(rows: list[dict]) -> None:
    print("\n=== THRESHOLD SWEEP ===")
    for row in rows:
        print(
            f"threshold={row['threshold']:.2f} | accuracy={row['accuracy']:.3f} "
            f"| precision={row['precision']:.3f} | recall={row['recall']:.3f} | f1={row['f1']:.3f}"
        )


def print_judge_report(results: list[dict], metrics: dict) -> None:
    print("\n=== LLM JUDGE METRICS ===")
    print(f"Total cases:   {metrics['total']}")
    print(f"Accuracy:      {metrics['accuracy']:.3f}")
    print(f"Precision:     {metrics['precision']:.3f}")
    print(f"Recall:        {metrics['recall']:.3f}")
    print(f"F1:            {metrics['f1']:.3f}")
    print(f"Avg latency:   {metrics['avg_latency_sec']:.3f}s")
    print(f"TP / TN / FP / FN: {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print("Confusion matrix:")
    print(
        f"  actual contradiction -> predicted contradiction: {metrics['confusion_matrix']['actual_contradiction_predicted_contradiction']}"
    )
    print(
        f"  actual contradiction -> predicted no contradiction: {metrics['confusion_matrix']['actual_contradiction_predicted_no_contradiction']}"
    )
    print(
        f"  actual no contradiction -> predicted contradiction: {metrics['confusion_matrix']['actual_no_contradiction_predicted_contradiction']}"
    )
    print(
        f"  actual no contradiction -> predicted no contradiction: {metrics['confusion_matrix']['actual_no_contradiction_predicted_no_contradiction']}"
    )
    for row in results:
        status = "OK" if row["actual"] == row["predicted"] else "MISS"
        print(
            f"[{status}] {row['id']} | label={row['label']} | score={row['score']:.4f} "
            f"| routing={row['routing']} | review={row['review']} | {row['latency_sec']:.3f}s"
        )


def export_errors(dataset: list[dict], results: list[dict], output_path: Path, model_name: str) -> int:
    by_id = {row["id"]: row for row in dataset}
    errors = [row for row in results if row["actual"] != row["predicted"]]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in errors:
            source = by_id[row["id"]]
            payload = {
                "id": row["id"],
                "title": source.get("title", row["id"]),
                "model": model_name,
                "actual_contradiction": row["actual"],
                "predicted_contradiction": row["predicted"],
                "label": row.get("label", ""),
                "score": row.get("score", 0.0),
                "text_a": source["text_a"],
                "text_b": source["text_b"],
            }
            if "routing" in row:
                payload["routing"] = row["routing"]
            if "review" in row:
                payload["requires_human_review"] = row["review"]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(errors)


def compare_models(baseline_metrics: dict, judge_metrics: dict) -> None:
    print("\n=== BASELINE VS JUDGE ===")
    print(
        f"Accuracy:  baseline={baseline_metrics['accuracy']:.3f} | judge={judge_metrics['accuracy']:.3f}"
    )
    print(
        f"Precision: baseline={baseline_metrics['precision']:.3f} | judge={judge_metrics['precision']:.3f}"
    )
    print(
        f"Recall:    baseline={baseline_metrics['recall']:.3f} | judge={judge_metrics['recall']:.3f}"
    )
    print(
        f"F1:        baseline={baseline_metrics['f1']:.3f} | judge={judge_metrics['f1']:.3f}"
    )
    print(
        f"FP:        baseline={baseline_metrics['fp']} | judge={judge_metrics['fp']}"
    )
    print(
        f"FN:        baseline={baseline_metrics['fn']} | judge={judge_metrics['fn']}"
    )


def write_heldout_template(path: Path) -> None:
    rows = [
        {
            "id": "heldout-001",
            "title": "Пример скрытого ограничения",
            "text_a": "Закон разрешает подать заявление в бумажном или электронном виде.",
            "text_b": "Подзаконный акт требует подачу только через портал электронного правительства.",
            "contradiction": True,
            "notes": "Добавить реальные пары из Adilet, не похожие на few-shot примеры.",
        },
        {
            "id": "heldout-002",
            "title": "Пример допустимой процедуры",
            "text_a": "Лицензия выдается в течение 5 рабочих дней.",
            "text_b": "Для получения лицензии заявитель подает заявление по форме из приложения.",
            "contradiction": False,
            "notes": "Нужны реальные нейтральные procedural cases.",
        },
        {
            "id": "heldout-003",
            "title": "Пример прямого конфликта по сроку",
            "text_a": "Налог должен быть уплачен в течение 10 дней.",
            "text_b": "Налог должен быть уплачен в течение 30 дней.",
            "contradiction": True,
            "notes": "Добавить реальные конфликтующие нормы с разными сроками.",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def candidate_matches_expectation(doc, row: dict) -> bool:
    metadata = doc.metadata or {}
    haystacks = [
        normalize_text(doc.page_content),
        normalize_text(str(metadata.get("title", ""))),
        normalize_text(str(metadata.get("url", ""))),
    ]
    checks: list[str] = []
    for key in ["expected_text_contains", "expected_title_contains", "expected_url_contains"]:
        value = row.get(key)
        if value:
            checks.append(normalize_text(str(value)))
    if not checks:
        return False
    return any(check in haystack for check in checks for haystack in haystacks)


def evaluate_pipeline(dataset: list[dict], db_dir: Path, threshold: float, k: int) -> Tuple[list[dict], dict]:
    db = load_vector_db(db_dir)
    nli = build_nli()
    results: list[dict] = []

    for row in dataset:
        started = time.perf_counter()
        docs = db.similarity_search(row["query"], k=k)
        retrieval_hit = next((doc for doc in docs if candidate_matches_expectation(doc, row)), None)
        baseline_predicted = False
        judge_predicted = False
        judge_routing = "not_run"
        matched_title = ""

        if retrieval_hit is not None:
            matched_title = str((retrieval_hit.metadata or {}).get("title", "")).strip()
            nli_result = get_nli_result(nli, row["query"], retrieval_hit.page_content, threshold=threshold)
            baseline_predicted = bool(nli_result["predicted_contradiction"])
            if baseline_predicted:
                judge = judge_norm_pair(row["query"], retrieval_hit.page_content, model=LLM_MODEL)
                judge_predicted = judge.is_contradiction
                judge_routing = judge.routing

        duration = time.perf_counter() - started
        final_predicted = judge_predicted
        end_to_end_success = (final_predicted == bool(row["contradiction"])) and (
            retrieval_hit is not None or not bool(row["contradiction"])
        )

        results.append(
            {
                "id": row["id"],
                "title": row.get("title", row["id"]),
                "actual": bool(row["contradiction"]),
                "predicted": final_predicted,
                "retrieval_hit": retrieval_hit is not None,
                "baseline_predicted": baseline_predicted,
                "judge_predicted": judge_predicted,
                "judge_routing": judge_routing,
                "matched_title": matched_title,
                "end_to_end_success": end_to_end_success,
                "latency_sec": duration,
            }
        )

    metrics = compute_metrics(results)
    metrics["avg_latency_sec"] = statistics.mean(row["latency_sec"] for row in results)
    metrics["retrieval_hit_rate"] = sum(1 for row in results if row["retrieval_hit"]) / len(results)
    metrics["baseline_gate_rate"] = sum(1 for row in results if row["baseline_predicted"]) / len(results)
    metrics["judge_positive_rate"] = sum(1 for row in results if row["judge_predicted"]) / len(results)
    return results, metrics


def print_pipeline_report(results: list[dict], metrics: dict) -> None:
    print("\n=== FULL PIPELINE METRICS ===")
    print(f"Total cases:        {metrics['total']}")
    print(f"Accuracy:           {metrics['accuracy']:.3f}")
    print(f"Precision:          {metrics['precision']:.3f}")
    print(f"Recall:             {metrics['recall']:.3f}")
    print(f"F1:                 {metrics['f1']:.3f}")
    print(f"Avg latency:        {metrics['avg_latency_sec']:.3f}s")
    print(f"Retrieval hit rate: {metrics['retrieval_hit_rate']:.3f}")
    print(f"Baseline gate rate: {metrics['baseline_gate_rate']:.3f}")
    print(f"Judge positive rate:{metrics['judge_positive_rate']:.3f}")
    print(f"TP / TN / FP / FN:  {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    for row in results:
        status = "OK" if row["actual"] == row["predicted"] else "MISS"
        print(
            f"[{status}] {row['id']} | retrieval_hit={row['retrieval_hit']} | "
            f"baseline={row['baseline_predicted']} | judge={row['judge_predicted']} | "
            f"routing={row['judge_routing']} | end_to_end_success={row['end_to_end_success']} | "
            f"matched_title={row['matched_title'] or '-'} | "
            f"{row['latency_sec']:.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLI and explanation quality for the legal conflict MVP.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to JSONL dataset with text_a, text_b, contradiction.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Score threshold for contradiction decision.",
    )
    parser.add_argument(
        "--with-ollama",
        action="store_true",
        help="Also measure whether Ollama returns explanations for predicted contradictions.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Evaluate several thresholds and print the metric summary.",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Also evaluate the structured Ollama judge with few-shot JSON prompting.",
    )
    parser.add_argument(
        "--export-errors",
        action="store_true",
        help="Export misclassified cases to JSONL for prompt iteration.",
    )
    parser.add_argument(
        "--write-heldout-template",
        action="store_true",
        help="Write a starter held-out dataset template for manual expansion.",
    )
    parser.add_argument(
        "--pipeline-dataset",
        default=str(DEFAULT_PIPELINE_DATASET),
        help="Path to retrieval+jump pipeline evaluation dataset.",
    )
    parser.add_argument(
        "--with-pipeline",
        action="store_true",
        help="Evaluate retrieval -> baseline -> judge pipeline using the vector DB.",
    )
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="Path to Chroma DB for retrieval pipeline evaluation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="How many documents to retrieve in pipeline evaluation.",
    )
    args = parser.parse_args()

    if args.write_heldout_template:
        write_heldout_template(DEFAULT_HELDOUT_TEMPLATE)
        print(f"Wrote held-out template to: {DEFAULT_HELDOUT_TEMPLATE}")
        if not any([args.with_ollama, args.sweep_thresholds, args.with_judge, args.export_errors, args.with_pipeline]):
            return

    dataset = load_dataset(Path(args.dataset))
    results, metrics = evaluate_nli(dataset, threshold=args.threshold)

    ollama_metrics = None
    if args.with_ollama:
        predicted_positive_rows = [row for row in results if row["predicted"]]
        ollama_metrics = evaluate_ollama(dataset, predicted_positive_rows)

    print_report(results, metrics, ollama_metrics)

    if args.sweep_thresholds:
        sweep = threshold_sweep(dataset, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])
        print_threshold_report(sweep)

    if args.with_judge:
        judge_results, judge_metrics = evaluate_judge(dataset)
        print_judge_report(judge_results, judge_metrics)
        compare_models(metrics, judge_metrics)

        if args.export_errors:
            judge_exported = export_errors(dataset, judge_results, DEFAULT_ERROR_EXPORT, model_name="llm_judge")
            print(f"\nExported {judge_exported} judge errors to: {DEFAULT_ERROR_EXPORT}")
    elif args.export_errors:
        exported = export_errors(dataset, results, DEFAULT_BASELINE_ERROR_EXPORT, model_name="baseline_nli")
        print(f"\nExported {exported} baseline errors to: {DEFAULT_BASELINE_ERROR_EXPORT}")

    if args.with_pipeline:
        pipeline_dataset = load_dataset(Path(args.pipeline_dataset))
        pipeline_results, pipeline_metrics = evaluate_pipeline(
            pipeline_dataset,
            db_dir=Path(args.db_dir),
            threshold=args.threshold,
            k=args.k,
        )
        print_pipeline_report(pipeline_results, pipeline_metrics)


if __name__ == "__main__":
    main()
