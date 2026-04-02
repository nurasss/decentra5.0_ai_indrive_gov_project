# LexMirror MVP

MVP для поиска и объяснения правовых противоречий в нормативных актах Республики Казахстан.

## Что уже работает

- Индексация правовых текстов в `Chroma`
- Поиск похожих норм по embedding-поиску
- Базовый `NLI`-фильтр на `facebook/bart-large-mnli`
- `LLM judge` через `Ollama` с `few-shot` JSON-промптом
- `Streamlit`-интерфейс для ручной проверки
- Локальная `evaluation` для метрик и сравнения `baseline` против `judge`

## Ключевые файлы

- `app.py` — веб-интерфейс MVP
- `index.py` — индексация корпуса в `Chroma`
- `nli_judge.py` — основной JSON judge через `Ollama`
- `evaluation.py` — метрики, confusion matrix, pipeline eval
- `prepare_ollama_dataset.py` — подготовка корпуса и чанков
- `pipeline_eval_dataset.jsonl` — стартовый набор для проверки полного пайплайна
- `eval_dataset.jsonl` — стартовый набор размеченных NLI-кейсов
- `ARCHITECTURE.md` — краткое описание архитектуры решения

## Быстрый запуск

### 1. Установить зависимости

```powershell
pip install -r ../requirements.txt
```

### 2. Настроить OpenAI API

```powershell
set OPENAI_API_KEY=ваш_ключ
set OPENAI_LLM_MODEL=gpt-4.1-mini
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Построить векторную базу

```powershell
python index.py --limit 500 --source-preference csv
```

Для полной индексации:

```powershell
python index.py --source-preference csv
```

### 4. Запустить интерфейс

```powershell
python -m streamlit run app.py
```

## Минимально работающий модуль

Самый простой демонстрационный запуск без UI:

```powershell
python demo_judge.py
```

Этот скрипт сравнивает две нормы и печатает структурированный результат judge.

## Оценка качества

Базовая оценка:

```powershell
python evaluation.py
```

Сравнение baseline и LLM judge:

```powershell
python evaluation.py --with-judge
```

Подбор порога:

```powershell
python evaluation.py --with-judge --sweep-thresholds
```

Экспорт ошибок:

```powershell
python evaluation.py --export-errors
python evaluation.py --with-judge --export-errors
```

Проверка полного пайплайна:

```powershell
python evaluation.py --with-pipeline
```

## Что можно показать на защите

- Архитектуру `retrieval -> baseline NLI -> LLM judge -> explanation`
- Метрики `baseline` и `judge`
- Прямую проверку двух норм в `Streamlit`
- Логику маршрутизации: `critical_conflict`, `human_review`, `pass`

## Ограничения текущего MVP

- Локальный стек сильно зависит от `Ollama`
- `Python 3.14` даёт предупреждения совместимости для части ML-стека
- `pipeline eval` пока требует расширения набора реальных retrieval-кейсов
- Текущая валидация judge всё ещё маленькая, её нужно расширять до `held-out` набора
- Baseline NLI теперь по умолчанию настраивается через `NLI_MODEL`; без переменной окружения используется `cointegrated/rubert-base-cased-nli-threeway`
