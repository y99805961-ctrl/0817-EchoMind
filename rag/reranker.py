"""Lazy local Cross-Encoder reranker with a safe RRF fallback."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional, Sequence

from .models import FusedHit, RerankedHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto", scorer: Optional[Callable] = None):
        self.model_name = model_name
        self.requested_device = device
        self.scorer = scorer
        self.local_files_only = str(os.getenv("RAG_LOCAL_FILES_ONLY", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self._model = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        self.cold_start_ms = 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None or self.scorer is not None

    @property
    def device(self) -> Optional[str]:
        return self._device

    def _ensure_model(self):
        if self.scorer is not None:
            return self.scorer
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            started = time.perf_counter()
            try:
                import torch
                from sentence_transformers import CrossEncoder
                device = self._select_device(torch)
                self._model = CrossEncoder(self.model_name, device=device, local_files_only=self.local_files_only)
                self._device = device
                self.cold_start_ms += (time.perf_counter() - started) * 1000
                return self._model
            except Exception as exc:
                self.cold_start_ms += (time.perf_counter() - started) * 1000
                raise RuntimeError(f"Unable to load reranker {self.model_name!r}: {exc}") from exc

    def _select_device(self, torch) -> str:
        requested = (self.requested_device or "auto").lower()
        if requested in {"auto", ""}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RERANKER_DEVICE=cuda but CUDA is unavailable")
        return requested

    def score(self, query: str, hits: Sequence[FusedHit]) -> List[float]:
        model = self._ensure_model()
        pairs = [(query, hit.content) for hit in hits]
        if self.scorer is not None:
            values = model(query, list(hits))
        else:
            values = model.predict(pairs, show_progress_bar=False)
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [float(value[0] if isinstance(value, (list, tuple)) else value) for value in values]

    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> tuple[List[RerankedHit], str]:
        limited = list(hits)
        if not limited:
            return [], "disabled"
        try:
            scores = self.score(query, limited)
            ranked = sorted(zip(limited, scores), key=lambda item: item[1], reverse=True)
            return [
                RerankedHit(
                    child_id=hit.child_id,
                    parent_id=hit.parent_id,
                    content=hit.content,
                    source=hit.source,
                    metadata=dict(hit.metadata),
                    reranker_score=float(score),
                    reranker_rank=index + 1,
                    rrf_score=hit.rrf_score,
                    matched_queries=list(hit.matched_queries),
                )
                for index, (hit, score) in enumerate(ranked[:top_k])
            ], "ok"
        except Exception as exc:
            logger.warning("Cross-Encoder reranker failed; retaining RRF order: %s", exc)
            return [
                RerankedHit(
                    child_id=hit.child_id,
                    parent_id=hit.parent_id,
                    content=hit.content,
                    source=hit.source,
                    metadata=dict(hit.metadata),
                    reranker_score=hit.rrf_score,
                    reranker_rank=index + 1,
                    rrf_score=hit.rrf_score,
                    matched_queries=list(hit.matched_queries),
                )
                for index, hit in enumerate(limited[:top_k])
            ], "failed"
