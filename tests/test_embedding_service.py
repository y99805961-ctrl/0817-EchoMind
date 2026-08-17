from core.embedding_service import EmbeddingService, get_embedding_service, reset_embedding_service


def test_embedding_service_is_lazy_and_shared():
    reset_embedding_service()
    first = get_embedding_service(model_name="test-model", device="cpu")
    second = get_embedding_service(model_name="ignored")
    assert first is second
    assert first.model_name == "test-model"
    assert first.loaded is False
    reset_embedding_service()


def test_embedding_service_accepts_batch_configuration():
    service = EmbeddingService(model_name="test", device="cpu", use_fp16=True, batch_size=4)
    assert service.batch_size == 4
    assert service.use_fp16 is True
    assert service.device is None
