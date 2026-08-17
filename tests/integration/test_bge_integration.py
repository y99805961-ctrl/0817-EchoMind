import os

import pytest

from core.embedding_service import EmbeddingService
from core.intent_recognizer import _cosine


@pytest.mark.integration
def test_bge_refund_query_is_closer_to_refund_than_crash():
    if os.getenv("RUN_BGE_INTEGRATION") != "1":
        pytest.skip("set RUN_BGE_INTEGRATION=1 to download/use BAAI/bge-m3")
    service = EmbeddingService()
    query = service.encode_query("钱什么时候退回来")
    refund = service.encode_query("退款申请处理进度")
    crash = service.encode_query("应用运行时突然闪退")
    assert _cosine(query, refund) > _cosine(query, crash)
