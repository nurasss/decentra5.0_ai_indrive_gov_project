# ZanAi / LexMirror MVP

Рабочий прототип для поиска связанных норм, выявления возможных противоречий и объяснения результата через локальный `RAG + NLI + LLM judge`.

## Что находится в проекте

- `index.html`, `styles.css`, `script.js` - фронтенд-демо интерфейс
- `main.py` - `FastAPI` backend для фронтенда
- `PythonProject2/app.py` - `Streamlit` MVP для ручной проверки
- `PythonProject2/index.py` - индексация корпуса в `Chroma`
- `PythonProject2/nli_judge.py` - judge-модуль через `Ollama`
- `PythonProject2/evaluation.py` - локальная оценка качества
- `PythonProject2/ARCHITECTURE.md` - краткая архитектура
- `PythonProject2/README.md` - README для Python-части MVP

## Статус артефактов

- Рабочий прототип: есть
- Реализованные ключевые требования: есть базовая реализация
- GitHub-репозиторий: remote `origin` настроен на `https://github.com/nurasss/decentra5.0_ai_indrive_gov_project.git`
- README: теперь есть в корне и внутри `PythonProject2`
- Demo-видео: в проекте не найдено

## Что умеет прототип

- Индексирует локальный корпус правовых текстов в `Chroma`
- Делает retrieval по embedding-поиску
- Прогоняет кандидатов через `facebook/bart-large-mnli`
- При потенциальном конфликте вызывает `llama3:8b` через `Ollama`
- Возвращает explainability и routing: `critical_conflict`, `human_review`, `pass`
- Даёт два UI:
  - `Streamlit` для демонстрации MVP
  - HTML/CSS/JS фронтенд + `FastAPI` backend

## Важные замечания по окружению

- Рекомендуемая версия Python: `3.11` или `3.12`
- `Python 3.14` может ломать часть ML-стека
- Для `macOS` нужны `Xcode Command Line Tools`, иначе `git`, `python3` и сборка зависимостей могут не запускаться
- Для полной работы нужен локально установленный `Ollama`

## Полный запуск с нуля

### 1. Установить системные инструменты

На `macOS`:

```bash
xcode-select --install
```

Проверьте, что команды доступны:

```bash
git --version
python3 --version
pip3 --version
```

### 2. Перейти в папку проекта

```bash
cd "/Users/nuras/Desktop/copilot agent web"
```

### 3. Создать виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Для Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

По умолчанию baseline NLI теперь берётся из переменной `NLI_MODEL` и, если она не задана,
использует русскоязычную модель `cointegrated/rubert-base-cased-nli-threeway`.

### 5. Настроить OpenAI API

Экспортируйте ключ и модели:

```bash
export OPENAI_API_KEY="ваш_ключ"
export OPENAI_LLM_MODEL="gpt-4.1-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

### 6. Построить или проверить векторную базу

Если папка `PythonProject2/db` уже существует, этот шаг можно пропустить.

По умолчанию индексация теперь берёт новый источник:

```text
PythonProject2/adilet_scraper/output.csv
```

Если его нет, используется старый запасной файл:

```text
PythonProject2/adilet_parsed_texts.csv
```

Быстрый тестовый индекс:

```bash
cd "/Users/nuras/Desktop/copilot agent web/PythonProject2"
python3 index.py --limit 500 --source-preference csv
```

Полная индексация:

```bash
cd "/Users/nuras/Desktop/copilot agent web/PythonProject2"
python3 index.py --source-preference csv
```

Если нужно явно указать файл:

```bash
python3 index.py --csv-path "/Users/nuras/Desktop/copilot agent web/PythonProject2/adilet_scraper/output.csv" --source-preference csv
```

### 7. Запустить backend для фронтенда

В новом терминале:

```bash
cd "/Users/nuras/Desktop/copilot agent web"
uvicorn main:app --reload
```

Проверка backend:

```bash
curl http://127.0.0.1:8000/health
```

### 8. Открыть HTML-фронтенд

Проще всего поднять локальный статический сервер в корне проекта:

```bash
cd "/Users/nuras/Desktop/copilot agent web"
python3 -m http.server 5500
```

После этого откройте в браузере:

```text
http://127.0.0.1:5500/index.html
```

Фронтенд отправляет запросы в `http://127.0.0.1:8000/api/analyze`, поэтому backend должен быть уже запущен.

### 9. Запустить Streamlit MVP

Альтернативный интерфейс:

```bash
cd "/Users/nuras/Desktop/copilot agent web/PythonProject2"
python3 -m streamlit run app.py
```

### 10. Запустить демонстрационный judge без UI

```bash
cd "/Users/nuras/Desktop/copilot agent web/PythonProject2"
python3 demo_judge.py
```

### 11. Запустить оценку качества

Базовая:

```bash
cd "/Users/nuras/Desktop/copilot agent web/PythonProject2"
python3 evaluation.py
```

С judge:

```bash
python3 evaluation.py --with-judge
```

Проверка полного пайплайна:

```bash
python3 evaluation.py --with-pipeline
```

## Быстрая схема запуска для защиты

1. Запустить `Ollama`
2. Запустить backend: `uvicorn main:app --reload`
3. Открыть `index.html` через локальный сервер
4. Показать сценарий вопроса пользователя
5. При необходимости параллельно открыть `Streamlit` интерфейс

## Известные проблемы проекта

- В корне проекта нет отдельного файла с demo-видео
- Корневой каталог сам не является git-репозиторием, git находится в `PythonProject2/.git`
- Без `OPENAI_API_KEY` backend, индексация, judge и evaluation не будут работать полноценно
- Без установленного `Xcode Command Line Tools` на `macOS` не работают базовые команды разработки
- Для полноценной демонстрации всё ещё нужен более крупный реальный eval-датасет, текущие JSONL-наборы подходят в основном для smoke-тестов

## Минимальная команда для старта

Если всё уже установлено:

```bash
source .venv/bin/activate
cd "/Users/nuras/Desktop/copilot agent web"
uvicorn main:app --reload
```

И в отдельном окне:

```bash
cd "/Users/nuras/Desktop/copilot agent web"
python3 -m http.server 5500
```
