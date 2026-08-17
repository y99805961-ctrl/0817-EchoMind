from rag.chunking import ParentChildChunker
from rag.models import ParentChunk


def test_children_stay_inside_one_parent_and_have_stable_ids():
    parent = ParentChunk("refund_return", "refund_return#timing", "退款", "到账", ["退款", "到账"], "02.md", "\n\n".join(["退款规则。" * 30] * 4))
    children = ParentChildChunker(child_size=100, overlap=20).split_parent(parent)
    assert len(children) > 1
    assert all(child.parent_id == parent.parent_id for child in children)
    assert [child.child_id for child in children] == [f"refund_return#timing#c{i:02d}" for i in range(1, len(children) + 1)]
    assert all(child.child_index == index for index, child in enumerate(children))


def test_overlap_is_never_taken_from_another_parent():
    first = ParentChunk("a", "a#one", "A", "One", ["A", "One"], "a.md", "第一段。" * 100)
    second = ParentChunk("a", "a#two", "A", "Two", ["A", "Two"], "a.md", "第二段。" * 100)
    chunker = ParentChildChunker(80, 10)
    first_text = "".join(child.content for child in chunker.split_parent(first))
    second_text = "".join(child.content for child in chunker.split_parent(second))
    assert "第二段" not in first_text
    assert "第一段" not in second_text
