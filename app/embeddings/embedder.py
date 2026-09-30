"""Shared SentenceTransformer embedding resource."""

from __future__ import annotations

import threading

import app.config  # noqa: F401
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

_embedding_cache = {}
_embedding_lock = threading.Lock()


def get_shared_embedding_model(
    model_name=DEFAULT_EMBEDDING_MODEL,
):
    """
    Load one SentenceTransformer instance per model name.

    Repeated EmbeddingModel() objects will reuse the same
    underlying model instead of loading duplicate copies.
    """

    name = str(
        model_name
        or DEFAULT_EMBEDDING_MODEL
    ).strip()

    cached = _embedding_cache.get(
        name
    )

    if cached is not None:
        return cached

    with _embedding_lock:

        cached = _embedding_cache.get(
            name
        )

        if cached is not None:
            return cached

        print(
            f"Loading embedding model: {name}"
        )

        model = SentenceTransformer(
            name
        )

        _embedding_cache[name] = model

        print(
            "Embedding model loaded."
        )

        return model


class EmbeddingModel:

    def __init__(
        self,
        model_name=DEFAULT_EMBEDDING_MODEL,
        model=None,
    ):

        self.model_name = str(
            model_name
            or DEFAULT_EMBEDDING_MODEL
        ).strip()

        self.model = (
            model
            if model is not None
            else get_shared_embedding_model(
                self.model_name
            )
        )

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ):

        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(
        self,
        query,
    ):

        return self.model.encode(
            str(query or ""),
            normalize_embeddings=True,
            show_progress_bar=False,
        )