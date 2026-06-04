"""
embed.py
Local embeddings using BAAI/bge-m3 (best multilingual retrieval model).
No API needed — runs entirely on local machine.
"""

import os
import torch
from typing import List
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_BATCH_SIZE = 16
CACHE_DIR = os.environ.get("HF_HOME", "./model_cache")


class EmbeddingModel:
    """
    Local sentence embeddings using BAAI/bge-m3.
    Supports Arabic (MSA + Egyptian dialect), English, and 100+ languages.
    No query/passage prefixes needed (unlike e5 models).
    """

    def __init__(self):
        self._model = None
        print(f"⚙️  Embeddings: {EMBEDDING_MODEL_NAME} (local, device: {EMBEDDING_DEVICE})")

    def load(self):
        if self._model is not None:
            return
        print(f"📥 Loading {EMBEDDING_MODEL_NAME} from local cache...")
        self._model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
        )
        print(f"✅ Embedding model ready on {EMBEDDING_DEVICE}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        self.load()
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        return 1024  # bge-m3 output dimension


_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
