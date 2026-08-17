from pathlib import Path

from rag.config import load_rag_config
from rag.corpus import build_corpus


def test_rebuilding_same_sources_keeps_ids_stable():
    config = load_rag_config()
    parents1, children1, _ = build_corpus(config)
    parents2, children2, _ = build_corpus(config)
    assert [item.parent_id for item in parents1] == [item.parent_id for item in parents2]
    assert [item.child_id for item in children1] == [item.child_id for item in children2]
    assert not any("\\" in item.parent_id for item in parents1)
    assert not any(str(Path.cwd()) in item.parent_id for item in parents1)
