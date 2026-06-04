"""
bm25_index.py — BM25 sparse retrieval for hybrid search.

BM25 complements dense vector search for:
  - Exact Arabic keyword matching  (رسوم، قبول، هندسة)
  - Fee amounts and percentages   (75 ألف، 97.80%)
  - Abbreviations                 (IGCSE، SAT، STEM)
  - Faculty / program names

Hyperparameters (tuned for Arabic university Q&A):
  k1 = 1.5   — term-frequency saturation; Arabic docs repeat key nouns often
  b  = 0.75  — length normalization; standard value works well here
  RRF k = 60 — dampens outlier ranks (de-facto standard)
  dense_weight = 0.6  — semantic query paraphrase matching
  bm25_weight  = 0.4  — exact-keyword / number matching
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

BM25_INDEX_FILE = "./bm25_index.pkl"

_STOP_AR = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'التي', 'الذي',
    'وفي', 'وعلى', 'ومن', 'أو', 'لا', 'ما', 'كان', 'يكون', 'هو', 'هي',
    'به', 'لها', 'له', 'لهم', 'بها', 'بهم', 'كما', 'حيث', 'إذ', 'عند',
    'كل', 'بعض', 'غير', 'حتى', 'إن', 'أن', 'عبر', 'وهو', 'وهي',
}
_STOP_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
    'could', 'should', 'may', 'might', 'shall',
}

# Normalize alef variants before tokenizing (same as loader normalization)
_ALEF_NORM = str.maketrans('أإآٱ', 'اااا')


def _strip_article(token: str) -> str:
    """Strip Arabic definite article 'ال' so 'الهندسة' matches 'هندسة'."""
    if token.startswith('ال') and len(token) > 3:
        return token[2:]
    return token


def _tokenize(text: str) -> List[str]:
    """Tokenize for BM25 — handles Arabic + English, keeps % for fee/score queries."""
    text = text.translate(_ALEF_NORM)
    text = re.sub(r'[^\w\s%]', ' ', text, flags=re.UNICODE)
    tokens = text.lower().split()
    tokens = [_strip_article(t) for t in tokens]
    return [
        t for t in tokens
        if len(t) >= 2 and t not in _STOP_AR and t not in _STOP_EN
    ]


class BM25Index:
    K1 = 1.5
    B  = 0.75

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Dict] = []
        self._index_file = Path(BM25_INDEX_FILE)

    def build(self, documents: List[Dict]) -> None:
        if not BM25_AVAILABLE:
            print("⚠️  rank_bm25 not installed — BM25 disabled")
            return
        self._documents = documents
        corpus = [_tokenize(doc['text']) for doc in documents]
        self._bm25 = BM25Okapi(corpus, k1=self.K1, b=self.B)
        self._save()
        print(f"✅ BM25 index built: {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self._bm25 or not self._documents:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self._documents), key=lambda x: x[0], reverse=True)
        return [
            {**doc, 'bm25_score': float(score)}
            for score, doc in ranked[:top_k]
            if score > 0
        ]

    def is_loaded(self) -> bool:
        return self._bm25 is not None

    def _save(self) -> None:
        try:
            with open(self._index_file, 'wb') as f:
                pickle.dump({'bm25': self._bm25, 'docs': self._documents}, f)
        except Exception:
            pass

    def load(self) -> bool:
        if not BM25_AVAILABLE or not self._index_file.exists():
            return False
        try:
            with open(self._index_file, 'rb') as f:
                data = pickle.load(f)
            self._bm25 = data['bm25']
            self._documents = data['docs']
            return True
        except Exception:
            return False


def reciprocal_rank_fusion(
    dense: List[Dict],
    bm25: List[Dict],
    top_k: int = 8,
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict]:
    """
    RRF: score(d) = Σ weight_i / (rank_i(d) + rrf_k)
    rrf_k=60 dampens rank outliers — de-facto standard value.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for rank, doc in enumerate(dense):
        key = doc.get('id', doc['text'][:60])
        scores[key] = scores.get(key, 0.0) + dense_weight / (rank + 1 + rrf_k)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25):
        key = doc.get('id', doc['text'][:60])
        scores[key] = scores.get(key, 0.0) + bm25_weight / (rank + 1 + rrf_k)
        if key not in doc_map:
            doc_map[key] = doc

    ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [doc_map[k] for k in ranked_keys]


_instance: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    global _instance
    if _instance is None:
        _instance = BM25Index()
        _instance.load()
    return _instance
