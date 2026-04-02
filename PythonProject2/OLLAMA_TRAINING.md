# Local `gpt-oss:20b` workflow

## What to do with this CSV

`adilet_parsed_texts_15k.csv` is a strong legal corpus, but it is **not** an instruction-tuning dataset yet.

That means there are two realistic paths:

1. `RAG` first
   - chunk the documents
   - embed the chunks
   - retrieve relevant legal passages at inference time
   - send retrieved context to `gpt-oss:20b` in Ollama

2. `LoRA/SFT` later
   - first prepare or generate question/answer pairs
   - then fine-tune outside Ollama with a training framework
   - finally serve the adapted model through Ollama

`Ollama` itself is mainly for serving models, not for doing the full training loop.

## Files produced by the prep script

Run:

```powershell
python PythonProject2/prepare_ollama_dataset.py
```

This creates:

- `PythonProject2/prepared_ollama/documents.jsonl`
- `PythonProject2/prepared_ollama/chunks.jsonl`
- `PythonProject2/prepared_ollama/pretrain_corpus.txt`

## Recommended order

### 1. Prepare the corpus

```powershell
python PythonProject2/prepare_ollama_dataset.py --chunk-chars 1800 --overlap-chars 250
```

### 2. Start with RAG

Why this is the best first move:

- legal texts are large and factual
- RAG preserves source grounding
- you can cite `url` and `title`
- you avoid catastrophic forgetting from low-quality fine-tuning

Minimal retrieval record shape from `chunks.jsonl`:

```json
{
  "id": "doc-00001-chunk-001",
  "document_id": "doc-00001",
  "chunk_index": 1,
  "url": "https://adilet.zan.kz/...",
  "title": "О признании ...",
  "text": "..."
}
```

### 3. Fine-tune only after we have supervision

If you still want to adapt `gpt-oss:20b`, the practical way is:

1. generate synthetic legal instructions from the corpus
2. review a sample manually
3. train a LoRA adapter in a framework such as `LLaMA-Factory`, `Axolotl`, or `Unsloth`
4. merge or keep the adapter
5. package for Ollama with a `Modelfile`

## Example supervised record format

For later `SFT`, aim for JSONL like this:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Ты юридический помощник по законодательству Казахстана. Отвечай точно и со ссылкой на источник."
    },
    {
      "role": "user",
      "content": "Кратко объясни, о чем этот документ: ...текст..."
    },
    {
      "role": "assistant",
      "content": "Документ регулирует ... Источник: ... "
    }
  ]
}
```

## Important note about the current environment

In the current workspace, `ollama` was not available in `PATH`, so the actual local model run could not be verified from here.

When `ollama` is installed on your machine, we can do the next step:

- connect `gpt-oss:20b` to the chunked corpus
- build retrieval
- or prepare an actual LoRA training dataset
