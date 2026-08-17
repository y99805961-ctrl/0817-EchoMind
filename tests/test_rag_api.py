from api.main import app, _should_use_knowledge


def test_rag_and_knowledge_v2_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert "/rag/debug" in paths
    assert "/knowledge/rebuild" in paths
    assert "/knowledge/stats" in paths


def test_chat_gate_matches_intent_v2_boundary():
    assert not _should_use_knowledge("你好", "greeting")
    assert _should_use_knowledge("退款多久到账", "refund")
    assert not _should_use_knowledge("我要转人工", "human_handoff")
