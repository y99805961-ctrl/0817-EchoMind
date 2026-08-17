from pathlib import Path

from rag.markdown_parser import parse_directory, parse_markdown, stable_doc_id


def test_required_corpus_has_ten_documents_and_stable_doc_ids():
    root = Path(__file__).parents[1] / "data" / "demo_docs" / "customer_kb_v2"
    documents = parse_directory(root)
    assert len(documents) == 10
    assert documents[0].doc_id == "order_logistics"
    assert all(document.parents for document in documents)


def test_long_h2_uses_h3_parent_boundaries(tmp_path):
    path = tmp_path / "demo.md"
    path.write_text("# Demo\n\n## Refund\n\n" + ("preamble " * 20) + "\n\n### Timing\n\n" + ("timing " * 250), encoding="utf-8")
    parsed = parse_markdown(path, max_parent_chars=100)
    assert any("timing" in parent.parent_id for parent in parsed.parents)
    assert all(parent.section_path[0] == "Demo" for parent in parsed.parents)


def test_unknown_document_id_does_not_depend_on_absolute_path():
    assert stable_doc_id(Path("C:/somewhere/12_my_policy.md")) == "my_policy"
