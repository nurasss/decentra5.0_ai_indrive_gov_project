from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT2_DIR = ROOT_DIR / "PythonProject2"
if str(PROJECT2_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT2_DIR))

from nli_judge import JudgeDecision, extract_json_object


def test_extract_json_object_reads_embedded_payload() -> None:
    payload = extract_json_object('prefix {"final_label":"contradiction","confidence_score":0.9,"requires_human_review":false} suffix')
    assert payload["final_label"] == "contradiction"
    assert payload["confidence_score"] == 0.9


def test_judge_routing_rules_cover_critical_and_review_paths() -> None:
    critical = JudgeDecision("", "", "", "contradiction", 0.91, False, "{}")
    review = JudgeDecision("", "", "", "neutral", 0.5, False, "{}")
    passed = JudgeDecision("", "", "", "entailment", 0.9, False, "{}")

    assert critical.routing == "critical_conflict"
    assert review.routing == "human_review"
    assert passed.routing == "pass"
