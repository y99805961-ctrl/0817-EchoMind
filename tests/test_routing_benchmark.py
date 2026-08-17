import asyncio

from agents.agent_orchestrator import AgentOrchestrator, Request
from core.intent_recognizer import IntentRecognizer


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
