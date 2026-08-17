"""Shared, lazy-loaded BGE-M3 embedding service.

The service deliberately does not import or load a model at module import time.
Intent recognition and future retrieval code can share the same singleton while
tests inject a small fake encoder and never download model weights.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the local embedding runtime cannot be loaded or used."""


class EmbeddingService:
    """Lazy BGE-M3 encoder with shared configuration and safe device fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
        self.requested_device = (device or os.getenv("BGE_DEVICE", "auto")).lower()
        self.use_fp16 = _as_bool(
            use_fp16 if use_fp16 is not None else os.getenv("BGE_USE_FP16", "true")
        )
        self.batch_size = max(
            1,
            int(batch_size or os.getenv("BGE_BATCH_SIZE", "16")),
        )
        self._model = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        return self._device

    def encode_query(self, text: str) -> List[float]:
        """Encode one query using the same normalized embedding path as batches."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch, retrying once on GPU OOM with a CPU model."""
        values = [str(text) for text in texts]
        if not values:
            return []
        model = self._ensure_model()
        try:
            return self._encode_with_model(model, values)
        except RuntimeError as exc:
            if self._device != "cuda" or not _looks_like_oom(exc):
                raise EmbeddingServiceError(
                    f"BGE-M3 encoding failed on {self._device}: {exc}"
                ) from exc
            logger.warning("BGE-M3 GPU OOM; falling back to CPU inference")
            self._fallback_to_cpu()
            try:
                return self._encode_with_model(self._ensure_model(), values)
            except Exception as cpu_exc:
                raise EmbeddingServiceError(
                    f"BGE-M3 CPU fallback failed after GPU OOM: {cpu_exc}"
                ) from cpu_exc
        except Exception as exc:
            raise EmbeddingServiceError(f"BGE-M3 encoding failed: {exc}") from exc

    async def aencode_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.encode_query, text)

    async def aencode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.encode_batch, texts)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingServiceError(
                    "BGE-M3 requires sentence-transformers and torch; "
                    "install the Intent v2 requirements before enabling local embeddings"
                ) from exc

            device = self._select_device(torch)
            try:
                model = SentenceTransformer(self.model_name, device=device)
                if self.use_fp16 and device == "cuda":
                    model.half()
            except Exception as exc:
                raise EmbeddingServiceError(
                    f"Unable to load embedding model {self.model_name!r} on {device}: {exc}"
                ) from exc
            self._model = model
            self._device = device
            logger.info(
                "Loaded shared embedding model %s on %s (fp16=%s)",
                self.model_name,
                device,
                self.use_fp16 and device == "cuda",
            )
            return model

    def _select_device(self, torch) -> str:
        cuda_available = bool(torch.cuda.is_available())
        if self.requested_device in {"auto", ""}:
            return "cuda" if cuda_available else "cpu"
        if self.requested_device == "cuda" and not cuda_available:
            raise EmbeddingServiceError("BGE_DEVICE=cuda but CUDA is unavailable")
        return self.requested_device

    def _encode_with_model(self, model, texts: Sequence[str]) -> List[List[float]]:
        encoded = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [[float(value) for value in row] for row in rows]

    def _fallback_to_cpu(self) -> None:
        with self._lock:
            self._model = None
            self._device = "cpu"
            self.requested_device = "cpu"
            self.use_fp16 = False


_shared_service: Optional[EmbeddingService] = None
_shared_lock = threading.Lock()


def get_embedding_service(**kwargs) -> EmbeddingService:
    """Return the process-wide shared service, creating it without loading weights."""
    global _shared_service
    if _shared_service is None:
        with _shared_lock:
            if _shared_service is None:
                _shared_service = EmbeddingService(**kwargs)
    return _shared_service


def reset_embedding_service() -> None:
    """Reset the singleton for isolated tests or an explicit application reload."""
    global _shared_service
    with _shared_lock:
        _shared_service = None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text
