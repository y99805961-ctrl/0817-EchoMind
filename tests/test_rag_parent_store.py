from rag.models import ParentChunk
from rag.parent_store import ParentStore


def test_parent_store_maps_stable_parent_id():
    parent = ParentChunk("doc", "doc#section", "Title", "Section", ["Title", "Section"], "doc.md", "content")
    store = ParentStore([parent])
    assert store.get("doc#section") is parent
    assert store.get("missing") is None
    assert len(store) == 1
