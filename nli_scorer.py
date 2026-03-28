import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загружаем специализированную модель для поиска логических связей в русском языке
MODEL_NAME = 'cointegrated/rubert-base-cased-nli-threeway'

print("Загрузка NLI-модели (потребуется скачать около 700 МБ)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Модель готова к работе!\n")


def analyze_contradiction(text1, text2):
    """
    Анализирует два текста и возвращает вероятности противоречия, следствия и нейтральности.
    """
    # Токенизируем сразу два текста как пару (модель сама вставит разделитель [SEP] между ними)
    tokens = tokenizer(text1, text2, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**tokens)

    # Превращаем сырые логиты нейросети в понятные вероятности от 0 до 1 (в сумме 100%)
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)[0]

    # В этой конкретной модели классы распределены так:
    # 0 - entailment (следствие/совпадение)
    # 1 - contradiction (противоречие)
    # 2 - neutral (нейтральность)

    entailment_prob = probabilities[0].item()
    contradiction_prob = probabilities[1].item()
    neutral_prob = probabilities[2].item()

    return {
        'contradiction': contradiction_prob,
        'entailment': entailment_prob,
        'neutral': neutral_prob
    }


if __name__ == "__main__":
    # --- ТЕСТИРОВАНИЕ НА ЖИВЫХ ПРИМЕРАХ ---

    # Пример 1: Явное противоречие (то, что мы ищем)
    norma_a = "Срок подачи налоговой декларации для ИП составляет не позднее 31 марта года, следующего за отчетным."
    norma_b = "Индивидуальные предприниматели обязаны сдать налоговую декларацию строго до 15 февраля."

    # Пример 2: Следствие / Совпадение (дублирование норм)
    norma_c = "Запрещается управление транспортным средством в состоянии алкогольного опьянения."
    norma_d = "Водитель не имеет права садиться за руль автомобиля, если он находится в нетрезвом виде."

    # Пример 3: Нейтральность (вообще разные темы)
    norma_e = "Государственная пошлина за выдачу паспорта составляет 2 МРП."
    norma_f = "Штраф за превышение скорости уплачивается в течение 30 дней."

    pairs_to_test = [
        ("ТЕСТ 1 (Ожидаем противоречие)", norma_a, norma_b),
        ("ТЕСТ 2 (Ожидаем совпадение/дубль)", norma_c, norma_d),
        ("ТЕСТ 3 (Ожидаем нейтральность)", norma_e, norma_f)
    ]

    for test_name, text1, text2 in pairs_to_test:
        print(f"--- {test_name} ---")
        print(f"Текст А: {text1}")
        print(f"Текст Б: {text2}")

        results = analyze_contradiction(text1, text2)

        print("\nВЕРДИКТ ИИ:")
        print(f"Вероятность ПРОТИВОРЕЧИЯ: {results['contradiction'] * 100:.1f}%")
        print(f"Вероятность СОВПАДЕНИЯ:   {results['entailment'] * 100:.1f}%")
        print(f"Вероятность НЕЙТРАЛЬНОСТИ: {results['neutral'] * 100:.1f}%\n")
        print("-" * 50)