from nli_judge import judge_norm_pair


def main() -> None:
    text_a = "Налогоплательщик имеет право предоставить декларацию в бумажном или электронном виде."
    text_b = "Декларация предоставляется исключительно через веб-портал электронного правительства."

    result = judge_norm_pair(text_a, text_b)

    print("=== DEMO JUDGE ===")
    print(f"label: {result.final_label}")
    print(f"confidence: {result.confidence_score:.2f}")
    print(f"requires_human_review: {result.requires_human_review}")
    print(f"routing: {result.routing}")
    print(f"step_1_extract_A: {result.step_1_extract_A}")
    print(f"step_2_extract_B: {result.step_2_extract_B}")
    print(f"step_3_compare: {result.step_3_compare}")


if __name__ == "__main__":
    main()
