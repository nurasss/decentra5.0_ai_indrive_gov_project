from typing import List

from langchain_core.embeddings import Embeddings

from openai_config import OPENAI_EMBEDDING_MODEL, get_openai_client


class OpenAIEmbeddingsAdapter(Embeddings):
    def __init__(self, model: str = OPENAI_EMBEDDING_MODEL):
        self.model = model
        self.client = get_openai_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
