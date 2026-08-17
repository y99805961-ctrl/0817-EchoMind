from rag.dense_retriever import DenseRetriever
from rag.models import ChildChunk


class FakeEmbedding:
    def encode_batch(self, texts):
        return [[1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0] for text in texts]


def test_dense_uses_explicit_query_embeddings_and_child_ids():
    children = [
        ChildChunk("doc", "doc#refund", "doc#refund#c01", "Title", "Refund", ["Title", "Refund"], "doc.md", 0, "refund timing"),
        ChildChunk("doc", "doc#other", "doc#other#c01", "Title", "Other", ["Title", "Other"], "doc.md", 0, "account security"),
    ]
    retriever = DenseRetriever(children, embedding_service=FakeEmbedding())
    retriever.set_embeddings([[1.0, 0.0], [0.0, 1.0]])
    hits = retriever.search("refund")
    assert hits[0].child_id == "doc#refund#c01"
    assert hits[0].retriever == "dense"
