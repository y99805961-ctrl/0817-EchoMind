from rag.context_builder import pack_context
from rag.models import ParentChunk, ParentSelection


def test_context_contains_source_and_section_but_not_retrieval_debug():
    parent = ParentChunk("doc", "doc#p", "Title", "Refund", ["Title", "Refund"], "refund.md", "Refund takes 3 days.")
    text = pack_context([ParentSelection(parent, 0.9, 1, ["doc#p#c01"])], max_chars=1000)
    assert "Source: refund.md" in text
    assert "Section: Refund" in text
    assert "rrf_score" not in text
    assert "reranker_raw_score" not in text
