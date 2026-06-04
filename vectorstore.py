"""
vectorstore.py
ChromaDB local vector store management.
Handles indexing, persistence, and semantic retrieval.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from embed import get_embedding_model

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "bnu_university"
TOP_K_RESULTS = 7


class VectorStore:
    """
    Manages ChromaDB collection for BNU university knowledge base.
    Supports Arabic and English semantic search.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedder = get_embedding_model()

    def _get_client(self):
        """Initialize ChromaDB client with persistence."""
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # use cosine similarity
            )
        return self._collection

    def is_populated(self) -> bool:
        """Check if the vector store already has documents."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return count > 0
        except Exception:
            return False

    def get_count(self) -> int:
        """Return number of documents in the store."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def add_documents(self, documents: List[Dict], batch_size: int = 50):
        """
        Add documents to the vector store.
        Documents: list of {'id', 'text', 'metadata'}
        """
        collection = self._get_collection()

        # Process in batches
        total = len(documents)
        print(f"\n🔢 Indexing {total} chunks into ChromaDB...")

        for i in range(0, total, batch_size):
            batch = documents[i: i + batch_size]

            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]

            # Generate embeddings
            embeddings = self._embedder.embed_documents(texts)

            # Upsert to ChromaDB (update if exists)
            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(f"   ✅ Indexed batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        print(f"✅ Vector store populated: {collection.count()} total chunks")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        section_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantic search over the vector store.
        Returns list of {'text', 'metadata', 'score'} dicts.
        """
        collection = self._get_collection()

        count = collection.count()
        if count == 0:
            return []

        # Embed the query
        query_embedding = self._embedder.embed_query(query)

        # Build optional metadata filter
        where_filter = None
        if section_filter:
            where_filter = {"section": {"$eq": section_filter}}

        # n_results must be <= collection size
        n = min(top_k, count)

        # Query ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Retry without filter if filter caused error (e.g. no matching section)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

        # Format results
        # BUG8 FIX: ChromaDB cosine distance is in range [0, 2] when using "cosine" space.
        # With normalized embeddings: distance=0 means identical, distance=2 means opposite.
        # similarity = 1 - (distance / 2)  → range [0, 1]
        # We use a threshold of 0.5 (moderate relevance minimum).
        formatted = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Normalize to [0, 1] similarity
                score = 1.0 - (dist / 2.0)
                if score > 0.40:  # minimum relevance threshold
                    formatted.append({
                        "text": doc,
                        "metadata": meta,
                        "score": round(score, 4),
                    })

        # Sort by score descending
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted

    def build_bm25(self, documents: List[Dict]) -> None:
        """Build BM25 sparse index alongside ChromaDB for hybrid retrieval."""
        try:
            from bm25_index import get_bm25_index
            get_bm25_index().build(documents)
        except Exception as e:
            print(f"⚠️  BM25 index not built: {e}")

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        section_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion.
        Falls back to dense-only if BM25 index not available.

        Retrieves top_k*2 candidates from each method, fuses, returns top_k.
        """
        try:
            from bm25_index import get_bm25_index, reciprocal_rank_fusion
            bm25_idx = get_bm25_index()
            if not bm25_idx.is_loaded():
                return self.search(query, top_k=top_k, section_filter=section_filter)
            dense = self.search(query, top_k=top_k * 2, section_filter=section_filter)
            sparse = bm25_idx.search(query, top_k=top_k * 2)
            return reciprocal_rank_fusion(dense, sparse, top_k=top_k)
        except Exception:
            return self.search(query, top_k=top_k, section_filter=section_filter)

    def hybrid_rerank_search(
        self,
        query: str,
        top_k: int = 8,
        section_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid retrieval followed by cross-encoder reranking.
        Fetches top_k*3 candidates so the reranker has enough to work with,
        then re-scores each (query, chunk) pair and returns the best top_k.
        Falls back to plain hybrid search if the reranker is unavailable.
        """
        candidates = self.hybrid_search(query, top_k=top_k * 3, section_filter=section_filter)
        if not candidates:
            return []
        try:
            from reranker import get_reranker
            return get_reranker().rerank(query, candidates, top_k=top_k)
        except Exception as e:
            print(f"⚠️  Reranker unavailable, falling back to hybrid search: {e}")
            return candidates[:top_k]

    def reset(self):
        """Clear all documents from the vector store."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            print(f"🗑️  Collection '{self.collection_name}' cleared.")
        except Exception as e:
            print(f"⚠️  Could not clear collection: {e}")


# Singleton
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
