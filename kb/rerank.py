from typing import Protocol

from .config import Settings


class Reranker(Protocol):
    enabled: bool

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return (index into documents, relevance score), best first."""
        ...


class CohereReranker:
    def __init__(self, settings: Settings):
        self.model = settings.rerank_model
        self.enabled = bool(settings.cohere_api_key)
        self._client = None
        if self.enabled:
            import cohere

            self._client = cohere.ClientV2(api_key=settings.cohere_api_key)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not self._client or not documents:
            return []
        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        return [(r.index, r.relevance_score) for r in response.results]
