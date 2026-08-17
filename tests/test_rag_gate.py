from rag.gate import should_use_knowledge


def test_gate_keeps_intent_v2_skip_and_business_rules():
    assert not should_use_knowledge("你好", "greeting")
    assert not should_use_knowledge("我要转人工", "human_handoff")
    assert should_use_knowledge("退款多久到账", "refund")
    assert should_use_knowledge("投诉退款到账问题", "complaint")
    assert not should_use_knowledge("服务态度不好", "complaint")
