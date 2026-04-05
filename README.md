# ZanAi / LexMirror

> AI-система для поиска противоречий, дублирований и устаревших норм в нормативных правовых актах Республики Казахстан.

## Реальный пример: что нашла система

Запрос: *«Земельный налог, срок уплаты»*

| | Документ | Норма |
|---|---|---|
| **Норма A** | Налоговый кодекс РК, ст. 507 | «...уплачивается равными долями не позднее **25 февраля** и **25 августа** текущего года» |
| **Норма B** | Приказ МФ РК № 486 | «...вносится единовременно не позднее **31 марта** текущего налогового года» |

**Вердикт системы:** `critical_conflict` · уверенность 97% · требует проверки юриста

**Шаг 1:** Кодекс устанавливает два срока уплаты — февраль и август.  
**Шаг 2:** Приказ устанавливает один срок — март.  
**Шаг 3:** Требования несовместимы. Налогоплательщик не может одновременно выполнить оба акта.

## Метрики качества (Ablation Study)

Оценка на `eval_dataset.jsonl` — 50 размеченных пар из 12 категорий казахстанского законодательства:

| Слой | Accuracy | Precision | Recall | F1 | Latency |
|---|---|---|---|---|---|
| Baseline NLI (RuBERT только) | 0.78 | 0.74 | 0.82 | 0.78 | ~0.3s |
| + LLM Judge (GPT-4.1-mini) | **0.90** | **0.91** | **0.88** | **0.89** | ~2.1s |
| Full pipeline (RAG + NLI + Judge) | **0.88** | 0.89 | 0.87 | 0.88 | ~2.4s |

**Вывод:** LLM-судья улучшает F1 на **+11 п.п.** относительно baseline. Основной прирост — снижение ложных позитивов (Precision +17 п.п.).

Запустить самостоятельно:
```bash
cd PythonProject2
python3 evaluation.py                # baseline
python3 evaluation.py --with-judge  # + LLM judge
```

## Что умеет система

- Индексирует корпус НПА из Әділет (adilet.zan.kz) — 13 751 документов
- Делает семантический поиск через embedding-базу (Chroma + OpenAI)
- Прогоняет кандидатов через NLI-модель (`cointegrated/rubert-base-cased-nli-threeway`)
- При потенциальном конфликте вызывает LLM-судью (GPT-4.1-mini) с chain-of-thought рассуждением
- Возвращает explainability-объяснение с маршрутизацией: `critical_conflict` / `human_review` / `pass`
- Строит граф связей между актами: `related`, `references`, `amends`, `contradicts`, `obsolete`
- **Новый эндпоинт `/api/bulk-check`**: перекрёстная проверка 2–10 норм на противоречия и дублирования
- Два интерфейса: HTML/JS фронтенд + Streamlit MVP

## Структура проекта

```
.
├── main.py                      # FastAPI backend
├── index.html / chat.html       # HTML/JS фронтенд
├── Dockerfile                   # Docker-образ для backend
├── docker-compose.yml           # Запуск backend + nginx frontend
├── .env.example                 # Шаблон конфигурации
├── PythonProject2/
│   ├── app.py                   # Streamlit UI
│   ├── index.py                 # Индексация корпуса в Chroma
│   ├── nli_judge.py             # LLM-судья (OpenAI, chain-of-thought)
│   ├── nli_utils.py             # NLI pipeline (RuBERT)
│   ├── evaluation.py            # Оценка качества (Accuracy/F1/confusion matrix)
│   ├── eval_dataset.jsonl       # 50 размеченных NLI-пар для оценки
│   ├── pipeline_eval_dataset.jsonl  # 15 pipeline-тестов
│   ├── db/                      # Chroma vector database (предзаполнена)
│   └── adilet_scraper/          # Scrapy-спайдер для Әділет
│       └── output.csv           # 13 751 документов (~5 GB, не в git)
└── ARCHITECTURE.md              # Описание архитектуры и data flow
```

## Форматы входных данных

### `/api/analyze` — анализ нормы или запроса
```json
{
  "text": "Налог на добавленную стоимость уплачивается по ставке 12 процентов.",
  "top_k": 3
}
```
Принимает: текст нормы (от 5 символов), юридический вопрос или название документа.

### `/api/compare` — попарное сравнение двух норм
```json
{
  "text_a": "Лицензия выдаётся в течение 5 рабочих дней.",
  "text_b": "Лицензия оформляется в течение 15 рабочих дней."
}
```

### `/api/bulk-check` — перекрёстная проверка 2–10 норм
```json
{
  "norms": [
    "Минимальная заработная плата составляет 70 000 тенге.",
    "Минимальный размер оплаты труда устанавливается в 85 000 тенге.",
    "Работник имеет право на ежегодный отпуск не менее 24 дней."
  ]
}
```
Возвращает все пары с классификацией: `contradiction` / `duplicate` / `neutral`.

## Требования

- Python **3.11** или **3.12** (3.14 ломает часть ML-стека)
- OpenAI API ключ
- macOS: `xcode-select --install`

## Быстрый старт (без Docker)

### 1. Клонировать и создать окружение

```bash
git clone https://github.com/nurasss/decentra5.0_ai_indrive_gov_project.git
cd decentra5.0_ai_indrive_gov_project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r PythonProject2/requirements.txt
```

### 2. Настроить переменные окружения

```bash
cp .env.example .env
# Открыть .env и вставить OPENAI_API_KEY
```

### 3. Проверить или построить векторную базу

База `PythonProject2/db/` уже включена в репозиторий. Если нужно переиндексировать:

```bash
cd PythonProject2
python3 index.py --limit 500 --source-preference csv   # быстрый тест
python3 index.py --source-preference csv               # полная индексация
```

### 4. Запустить backend

```bash
# Из корня проекта:
uvicorn main:app --reload
# Проверка: http://127.0.0.1:8000/health
```

### 5. Открыть фронтенд

```bash
python3 -m http.server 5500
# Открыть: http://127.0.0.1:5500/index.html
```

### 6. (Опционально) Streamlit UI

```bash
cd PythonProject2
python3 -m streamlit run app.py
# Открыть: http://localhost:8501
```

## Быстрый старт (Docker)

```bash
cp .env.example .env
# Вставить OPENAI_API_KEY в .env
docker compose up --build
# API: http://localhost:8000
# Frontend: http://localhost:5500
```

## Оценка качества

```bash
cd PythonProject2

# Только NLI-модель (baseline):
python3 evaluation.py

# NLI + LLM-судья:
python3 evaluation.py --with-judge

# Полный pipeline (retrieval + NLI + judge):
python3 evaluation.py --with-pipeline
```

Датасет для оценки: `eval_dataset.jsonl` — **50 размеченных пар** из 12 категорий казахстанского законодательства (налоги, трудовое право, гражданское право, экология, антикоррупция и др.).

## Сценарии применения

### Сценарий 1: Юрист проверяет новый приказ
1. Открыть `http://127.0.0.1:5500/chat.html`
2. Вставить текст нормы из нового приказа в поле анализа
3. Система найдёт похожие нормы в базе и укажет потенциальные противоречия
4. Для каждого конфликта судья выдаст трёхшаговое объяснение

### Сценарий 2: Аналитик проверяет набор норм на дублирования
```bash
curl -X POST http://127.0.0.1:8000/api/bulk-check \
  -H "Content-Type: application/json" \
  -d '{
    "norms": [
      "Срок исковой давности составляет три года.",
      "Общий срок исковой давности — 3 года.",
      "Срок исковой давности по трудовым спорам — один год."
    ]
  }'
```

### Сценарий 3: Попарное сравнение двух редакций нормы
```bash
curl -X POST http://127.0.0.1:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text_a": "Лицензия выдаётся в течение 5 рабочих дней.",
    "text_b": "Лицензия оформляется в течение 15 рабочих дней."
  }'
```

## API-эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Статус сервиса |
| GET | `/api/stats` | Статистика корпуса: количество документов, покрытие, типы актов |
| POST | `/api/analyze` | Анализ нормы/запроса (RAG + NLI + judge) |
| POST | `/api/compare` | Попарное сравнение двух норм |
| POST | `/api/bulk-check` | Перекрёстная проверка 2–10 норм на противоречия и дубли |

Интерактивная документация API: `http://127.0.0.1:8000/docs`

## Источники данных

| Источник | Описание | Объём |
|---|---|---|
| [Әділет](https://adilet.zan.kz) | База НПА Республики Казахстан | 13 751 документов |
| `adilet_scraper/` | Scrapy-спайдер для сбора документов | — |
| `eval_dataset.jsonl` | Размеченные NLI-пары для оценки | 50 примеров |
| `pipeline_eval_dataset.jsonl` | Pipeline-тесты с ожидаемыми заголовками | 15 примеров |

Файл `output.csv` (~5 GB) не включён в git из-за размера. Чтобы воспроизвести:
```bash
cd PythonProject2/adilet_scraper
pip install scrapy
scrapy crawl adilet --urls-file ../adilet_document_links.csv
```

## Честные ограничения и что дальше

| Ограничение | Влияние | Что нужно для устранения |
|---|---|---|
| Eval dataset синтетический | Метрики ориентировочные | Разметка 200+ реальных пар юристом |
| Граф связей — top-K, не весь корпус | Пропускает транзитивные связи | Knowledge graph на Neo4j |
| Определение устаревших норм — по тексту «утратил силу» | ~15% miss rate | Семантическая классификация + дата |
| Нет структурного парсинга статей/частей | Поиск на уровне документа | Парсер номеров статей и пунктов |
| Requires OpenAI API key | Нет оффлайн-режима | Локальная LLM (Ollama + llama3) |

## Ограничения

- Требуется активный `OPENAI_API_KEY` (judge + embeddings)
- `eval_dataset.jsonl` содержит синтетические примеры; для production-оценки нужен датасет из реальных НПА
- Граф связей строится по top-K результатам, а не по всему корпусу
- Анализ устаревших норм — эвристический (по ключевым словам "утратил силу")
- Файл `output.csv` не включён в репозиторий из-за размера

## Известные проблемы

- Python 3.14 несовместим с частью ML-зависимостей — используйте 3.11/3.12
- Без `OPENAI_API_KEY` backend, индексация и judge не работают
- При первом запуске NLI-модель (~400 MB) загружается с HuggingFace Hub
