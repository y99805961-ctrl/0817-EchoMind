import asyncio
from types import SimpleNamespace

from evaluation.rag_generation_eval import AnthropicRAGJudge


def _response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_judge_parses_fenced_json_and_validates_all_scores():
    assert AnthropicRAGJudge._parse_scores("```json\n{\"correctness\": 1, \"completeness\": 0.5, \"relevance\": 0.8, \"faithfulness\": 0.7, \"abstention_accuracy\": 0}\n```") == {
        "correctness": 1.0,
        "completeness": 0.5,
        "relevance": 0.8,
        "faithfulness": 0.7,
        "abstention_accuracy": 0.0,
    }


def test_judge_retries_transient_error_and_records_retry():
    class Client:
        def __init__(self):
            self.calls = 0
            self.messages = self

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return _response('{"correctness":1,"completeness":1,"relevance":1,"faithfulness":1,"abstention_accuracy":1}')

    client = Client()
    judge = AnthropicRAGJudge(client, "test", max_retries=1, retry_base_seconds=0)
    outcome = asyncio.run(judge({"query": "q"}, "context", "answer"))
    assert outcome["judge_status"] == "ok"
    assert outcome["judge_retried"] is True
    assert outcome["judge_retry_count"] == 1
    assert judge.stats["successes"] == 1


def test_judge_increases_budget_after_max_token_truncation():
    class Client:
        def __init__(self):
            self.calls = 0
            self.tokens = []
            self.messages = self

        async def create(self, **kwargs):
            self.calls += 1
            self.tokens.append(kwargs["max_tokens"])
            if self.calls == 1:
                return SimpleNamespace(content=[SimpleNamespace(type="thinking", text=None)], stop_reason="max_tokens")
            return _response('{"correctness":1,"completeness":1,"relevance":1,"faithfulness":1,"abstention_accuracy":1}')

    client = Client()
    judge = AnthropicRAGJudge(client, "test", max_retries=1, retry_base_seconds=0, max_tokens=2048, max_tokens_cap=4096)
    outcome = asyncio.run(judge({"query": "q"}, "context", "answer"))
    assert outcome["judge_status"] == "ok"
    assert client.tokens == [2048, 4096]


def test_judge_failure_is_unavailable_not_zero_score():
    class Client:
        def __init__(self):
            self.messages = self

        async def create(self, **kwargs):
            return _response("not json")

    judge = AnthropicRAGJudge(Client(), "test", max_retries=0)
    outcome = asyncio.run(judge({"query": "q"}, "context", "answer"))
    assert outcome["judge_status"] == "judge_failed"
    assert all(value is None for value in outcome["scores"].values())
    assert outcome["judge_error_type"] == "JudgeFormatError"
    assert outcome["judge_raw_output"] == "not json"
