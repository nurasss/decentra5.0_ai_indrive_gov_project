import os
from typing import Any

from transformers import pipeline


DEFAULT_NLI_MODEL = os.getenv("NLI_MODEL", "cointegrated/rubert-base-cased-nli-threeway")
DEFAULT_NLI_TEXT_CHAR_LIMIT = int(os.getenv("NLI_TEXT_CHAR_LIMIT", "1500"))
DEFAULT_CONTRADICTION_LABELS = {
    label.strip().upper().replace("-", "_").replace(" ", "_")
    for label in os.getenv("NLI_CONTRADICTION_LABELS", "CONTRADICTION").split(",")
    if label.strip()
}


def build_nli_pipeline(model_name: str = DEFAULT_NLI_MODEL):
    return pipeline(
        "text-classification",
        model=model_name,
        device=-1,
    )


def get_nli_max_length(nli: Any) -> int:
    tokenizer = getattr(nli, "tokenizer", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit <= 8192:
        return tokenizer_limit

    model = getattr(nli, "model", None)
    config = getattr(model, "config", None)
    config_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and config_limit > 0:
        return config_limit

    return 512


def normalize_nli_label(label: Any) -> str:
    return str(label or "UNKNOWN").strip().upper().replace("-", "_").replace(" ", "_")


def contradiction_labels_for_pipeline(nli: Any) -> set[str]:
    aliases = set(DEFAULT_CONTRADICTION_LABELS)
    model = getattr(nli, "model", None)
    config = getattr(model, "config", None)
    id2label = getattr(config, "id2label", {}) or {}
    for raw_label in id2label.values():
        label = normalize_nli_label(raw_label)
        if "CONTRAD" in label:
            aliases.add(label)
    return aliases


def run_nli(
    nli: Any,
    text_a: str,
    text_b: str,
    *,
    char_limit: int = DEFAULT_NLI_TEXT_CHAR_LIMIT,
) -> dict[str, Any]:
    max_length = get_nli_max_length(nli)
    result = nli(
        {
            "text": text_a.strip()[:char_limit],
            "text_pair": text_b.strip()[:char_limit],
        },
        truncation=True,
        max_length=max_length,
    )

    if isinstance(result, list):
        item = result[0] if result else {"label": "UNKNOWN", "score": 0.0}
    elif isinstance(result, dict):
        item = result
    else:
        raise TypeError(f"Unexpected NLI response type: {type(result)!r}")

    label = str(item.get("label", "UNKNOWN")).strip()
    normalized_label = normalize_nli_label(label)
    return {
        "label": label,
        "normalized_label": normalized_label,
        "score": float(item.get("score", 0.0)),
        "is_contradiction_label": normalized_label in contradiction_labels_for_pipeline(nli),
    }
