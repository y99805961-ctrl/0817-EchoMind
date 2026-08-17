# EchoMind RAG v1 Architecture

This document records the implementation that existed before RAG v2. It is a historical compatibility reference; the `knowledge_base` collection is not the v2 production corpus.

## Actual implementation

```text
Knowledge document
  -> roughly 500-character sentence/newline chunks
  -> Chroma collection: knowledge_base
  -> Chroma implicit embedding path (all-MiniLM-L6-v2 in the local implementation)
  -> query_texts semantic search
  -> MCPToolManager query rewrite (original + 3 rewrites)
  -> parallel tool calls
  -> md5(str(item)) content-based deduplication
  -> external LLM JSON index ordering reranker
  -> Top-K context in /chat
```

The old `KnowledgeBase` also auto-loaded six small fallback documents whenever the collection was empty. This behavior remains only in the legacy class for rollback compatibility; the v2 application path does not instantiate it and does not silently rebuild an index at startup.

## Audit answers

1. Collection name: `knowledge_base`.
2. Embedding: Chroma's implicit/default embedding path; the old module documents `all-MiniLM-L6-v2`.
3. Chunking: approximately 500 characters, assembled from sentence/newline pieces.
4. Metadata: `title`, `chunk_index`, and `total_chunks`.
5. IDs: an MD5 of title, chunk index, and the first 50 characters of the chunk on ingest.
6. Query rewrite: `MCPToolManager.rewrite_query()`.
7. Default rewrite count: three, with the original query retained.
8. Multi-query execution: parallel `asyncio.gather` calls through the tool manager.
9. Deduplication: MD5 of the string representation of each returned item.
10. Reranker: an LLM asked to return a relevance-ordered list of result indexes.
11. `/chat` gate: Intent result is passed to `_should_use_knowledge`; greeting/feedback/handoff/other are skipped while business intents search.
12. Agent context: the top results are formatted as title, similarity, and truncated content, then appended to memory context.
13. Knowledge APIs: `/knowledge/add` and `/knowledge/upload` immediately inserted legacy chunks; `/knowledge/stats` returned only the chunk count. `/search` exposed the rewrite/recall/rerank chain.
14. Evaluation: the existing evaluator judged end-to-end responses with an LLM, but did not calculate gold Child/Parent retrieval metrics.

## v2 boundary

The legacy collection is intentionally not deleted. New source content, stable IDs, BGE-M3 vectors, BM25 rows, and Parent context are owned by `rag/` and the `knowledge_base_v2_children` collection after an explicit rebuild.
