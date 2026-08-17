"""BM25 sparse retrieval with protected codes and identifier tokens."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from .models import ChildChunk, RetrievalHit

try:  # Optional accelerator; the deterministic fallback keeps tests portable.
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    BM25Okapi = None

try:
    import jieba
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    jieba = None


PROTECTED_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z]{2,}(?:[_-][A-Z0-9]+)+|[A-Z]{2,}\d{3,}|[A-Za-z]+_\w+|\d{3,})(?![A-Za-z0-9_])"
)
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Tokenize Chinese text while keeping codes such as AUTH_4011 intact."""
    text = str(text or "")
    protected = [match.group(0) for match in PROTECTED_TOKEN_RE.finditer(text)]
    masked = PROTECTED_TOKEN_RE.sub(" ", text)
    tokens: List[str] = []
    if jieba is not None:
        tokens.extend(str(token).strip().lower() for token in jieba.cut(masked, cut_all=False) if str(token).strip())
    else:
        for run in CJK_RE.findall(masked):
            tokens.extend(list(run))
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        tokens.extend(token.lower() for token in ASCII_TOKEN_RE.findall(masked))
    tokens.extend(token.lower() for token in protected)
    return tokens


class BM25Retriever:
    def __init__(self, children: Sequence[ChildChunk], top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        self.children = list(children)
        self.by_id = {child.child_id: child for child in self.children}
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(child.content) for child in self.children]
        self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b) if BM25Okapi and self.children else None
        self._avgdl = sum(len(row) for row in self._tokenized) / len(self._tokenized) if self._tokenized else 0.0
        self._idf = self._build_idf()

    def search(self, query: str, top_k: Optional[int] = None, query_source: str = "original") -> List[RetrievalHit]:
        return self.search_many([query], top_k=top_k, query_sources=[query_source])[0]

    def search_many(self, queries: Sequence[str], top_k: Optional[int] = None, query_sources: Optional[Sequence[str]] = None) -> List[List[RetrievalHit]]:
        top_k = top_k or self.top_k
        query_sources = list(query_sources or ["original"] * len(queries))
        return [
            self._search_one(query, top_k, query_sources[index])
            for index, query in enumerate(queries)
        ]

    def _search_one(self, query: str, top_k: int, query_source: str) -> List[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.children:
            return []
        if self._bm25 is not None:
            scores = [float(value) for value in self._bm25.get_scores(query_tokens)]
        else:
            scores = [self._fallback_score(query_tokens, tokens) for tokens in self._tokenized]
        protected_query_tokens = {
            token for token in query_tokens
            if PROTECTED_TOKEN_RE.fullmatch(token.upper()) or token.isdigit()
        }
        if protected_query_tokens:
            # rank_bm25 may assign an IDF of zero when a protected token
            # appears in exactly half of a tiny corpus. Exact code matches
            # must still be visible and rank above unrelated text.
            for index, document_tokens in enumerate(self._tokenized):
                exact_matches = protected_query_tokens.intersection(document_tokens)
                scores[index] += float(len(exact_matches))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(
                child_id=self.children[index].child_id,
                parent_id=self.children[index].parent_id,
                content=self.children[index].content,
                source=self.children[index].source,
                metadata={**self.children[index].metadata, "bm25_score": score},
                retriever="bm25",
                rank=rank + 1,
                score=score,
                query_source=query_source,
            )
            for rank, (index, score) in enumerate(ranked[:top_k])
            if score > 0
        ]

    def _build_idf(self):
        document_frequency = Counter(token for row in self._tokenized for token in set(row))
        count = len(self._tokenized)
        return {token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5)) for token, frequency in document_frequency.items()}

    def _fallback_score(self, query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        if not document_tokens:
            return 0.0
        frequencies = Counter(document_tokens)
        score = 0.0
        doc_len = len(document_tokens)
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            idf = self._idf.get(token, 0.0)
            denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1.0))
            score += idf * frequency * (self.k1 + 1) / denominator
        return score
