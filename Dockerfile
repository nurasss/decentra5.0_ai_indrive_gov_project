FROM python:3.12-slim

WORKDIR /app

# System dependencies for torch and transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY PythonProject2/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pre-download the NLI model at build time so the container starts faster
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cointegrated/rubert-base-cased-nli-threeway')" || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
