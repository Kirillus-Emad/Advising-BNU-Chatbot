"""
reranker.py
Reranker using Cohere Rerank API (rerank-multilingual-v3.0).
Supports Arabic + English — no local model download required.
"""

import os
from typing import List, Dict

import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANKER_MODEL = "rerank-multilingual-v3.0"


class Reranker:
    def __init__(self):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in .env file")
        self._client = cohere.Client(api_key=COHERE_API_KEY)
        print(f"⚙️  Reranker: {RERANKER_MODEL} via Cohere API")

    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Re-score each chunk against the query via Cohere and return top_k."""
        if not chunks:
            return chunks

        documents = [chunk["text"] for chunk in chunks]

        response = self._client.rerank(
            model=RERANKER_MODEL,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=False,
        )

        reranked = []
        for result in response.results:
            chunk = dict(chunks[result.index])
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        return reranked


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
