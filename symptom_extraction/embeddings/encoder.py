"""Embedding encoder wrapper for multilingual dense retrieval."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}

DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Lazy-load and cache the embedding model."""
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Encode a list of texts into normalized dense vectors."""
    model = get_encoder(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings)


def compute_cosine_similarity(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between a query vector and document vectors.

    Assumes all vectors are already L2-normalized.
    """
    return doc_vecs @ query_vec
