import os
from typing import Any, Protocol

from .config import Settings


class EmbeddingError(RuntimeError):
    """Embedding provider unreachable, misconfigured or rejected the request."""


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _field(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


class LiteLLMEmbedder:
    """OpenAI (or any LiteLLM-supported) embedding model, batched."""

    def __init__(self, settings: Settings):
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = settings.embedding_batch_size
        self._kwargs: dict[str, Any] = {"num_retries": 3}
        if settings.openai_api_key:
            self._kwargs["api_key"] = settings.openai_api_key
        if settings.embedding_api_base:
            self._kwargs["api_base"] = settings.embedding_api_base
        if settings.embedding_send_dimensions:
            self._kwargs["dimensions"] = settings.embedding_dim

    def _check_credentials(self) -> None:
        # Bare model names (no "provider/" prefix) go to OpenAI in LiteLLM.
        uses_openai = "/" not in self.model or self.model.startswith("openai/")
        if (uses_openai and "api_key" not in self._kwargs and "api_base" not in self._kwargs
                and not os.environ.get("OPENAI_API_KEY")):
            raise EmbeddingError("OPENAI_API_KEY is not set. Add it to the .env file and restart the server.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        self._check_credentials()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = litellm.embedding(model=self.model, input=batch, **self._kwargs)
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed ({self.model}): {exc}") from exc
            data = sorted(response.data, key=lambda d: _field(d, "index"))
            vectors.extend(_field(d, "embedding") for d in data)
        for v in vectors:
            if len(v) != self.dim:
                raise EmbeddingError(
                    f"Embedding model returned {len(v)} dims, expected KB_EMBEDDING_DIM={self.dim}"
                )
        return vectors
