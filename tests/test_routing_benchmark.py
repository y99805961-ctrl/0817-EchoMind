import asyncio

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer


def test_existing_orchestrator_routes_compound_technical_billing_case():
    async def run():
        recognizer = IntentRecognizer(mode="rules_only")
        intent = await recognizer.recognize("登录报401，而且重复扣了我299元")
        orchestrator = AgentOrchestrator(api_key="unit-test-placeholder")
        req = Request(
            message="登录报401，而且重复扣了我299元",
            user_id="test",
            conv_id="test",
            intent=intent.intent,
            intent_group=intent.intent_group,
            urgency=intent.urgency,
            entities=intent.entities,
            intent_confidence=intent.confidence,
        )
        return orchestrator._route_decision(req)

    decision = asyncio.run(run())
    assert decision.primary_agent.value == "technical"
    assert [agent.value for agent in decision.supporting_agents] == ["billing"]
    assert decision.domain_scores["technical"] > decision.domain_scores["billing"]
    assert "top_candidates" in decision.reason


def test_explicit_technical_failure_can_be_primary_over_semantic_billing():
    orchestrator = AgentOrchestrator(api_key="unit-test-placeholder")
    req = Request(
        message="页面报500，而且我今天必须拿到退款结果，挺急的",
        user_id="test",
        conv_id="compound",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        urgency=None,
        intent_confidence=0.9,
        top_candidates=[{"intent": "refund", "score": 0.8}, {"intent": "technical_crash", "score": 0.7}],
        rule_signals=[{"intent": "refund", "rule": "refund"}],
    )
    decision = orchestrator._route_decision(req)
    assert decision.primary_agent == AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
