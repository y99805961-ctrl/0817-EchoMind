import { SearchOutlined, UploadOutlined, WarningOutlined } from "@ant-design/icons";
import { Button, Card, Collapse, Input, InputNumber, Modal, Skeleton, Space, Statistic, Tag, Upload, message } from "antd";
import { useEffect, useState } from "react";
import { debugRag, getKnowledgeStats, rebuildKnowledge, searchKnowledge, uploadKnowledge } from "../api/chat";
import { ApiError } from "../api/client";
import type { KnowledgeStats, RAGDebugResponse, SearchResponse } from "../types/api";

function valueOrNA(value: unknown): string | number {
  return value === undefined || value === null || value === "" ? "N/A" : typeof value === "number" ? value : String(value);
}

export function KnowledgePage() {
  const [messageApi, contextHolder] = message.useMessage();
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [query, setQuery] = useState("退款多久到账");
  const [topK, setTopK] = useState(5);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [debug, setDebug] = useState<RAGDebugResponse | null>(null);
  const [debugging, setDebugging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);

  useEffect(() => {
    void (async () => {
      try {
        setStats(await getKnowledgeStats());
      } catch (error) {
        messageApi.error(error instanceof Error ? error.message : "Knowledge stats unavailable");
      } finally {
        setStatsLoading(false);
      }
    })();
  }, [messageApi]);

  async function runSearch(value = query) {
    if (!value.trim()) return;
    setSearching(true);
    try {
      setSearchResult(await searchKnowledge(value.trim(), topK));
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }

  async function runDebug() {
    if (!query.trim()) return;
    setDebugging(true);
    try {
      setDebug(await debugRag(query.trim()));
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "RAG trace unavailable");
    } finally {
      setDebugging(false);
    }
  }

  async function handleUpload(file: File) {
    setUploading(true);
    try {
      const result = await uploadKnowledge(file);
      messageApi.success(result.message);
    } catch (error) {
      const detail = error instanceof ApiError && error.status === 413 ? "文件超过 10MB 限制" : error instanceof Error ? error.message : "Upload failed";
      messageApi.error(detail);
    } finally {
      setUploading(false);
    }
  }

  function confirmRebuild() {
    Modal.confirm({
      title: "Rebuild knowledge index?",
      icon: <WarningOutlined />,
      content: "这是高成本操作，会重建当前知识索引。确认继续吗？",
      okText: "Rebuild",
      cancelText: "Cancel",
      onOk: async () => {
        setRebuilding(true);
        try {
          const result = await rebuildKnowledge();
          messageApi.success(result.message);
          setStats(await getKnowledgeStats());
        } catch (error) {
          messageApi.error(error instanceof Error ? error.message : "Rebuild failed");
        } finally {
          setRebuilding(false);
        }
      },
    });
  }

  return (
    <div className="page-stack" data-testid="knowledge-page">
      {contextHolder}
      <div className="page-title-row"><div><span className="eyebrow">KNOWLEDGE SPACE</span><h1>Knowledge</h1><p>查看真实索引状态，搜索与调试 RAG v2 链路。</p></div><Tag className="soft-status">Production index · read first</Tag></div>
      <div className="stats-grid knowledge-stats-grid">
        <Card className="soft-card"><Statistic title="Index Status" value={statsLoading ? "…" : valueOrNA(stats?.index_status)} /></Card>
        <Card className="soft-card"><Statistic title="Parents" value={statsLoading ? "…" : valueOrNA(stats?.total_parents)} /></Card>
        <Card className="soft-card"><Statistic title="Children" value={statsLoading ? "…" : valueOrNA(stats?.total_children)} /></Card>
        <Card className="soft-card"><Statistic title="BM25 Docs" value={statsLoading ? "…" : valueOrNA(stats?.bm25_docs)} /></Card>
      </div>
      <div className="two-column-grid">
        <Card className="soft-card knowledge-card" title="Knowledge Search Playground">
          <Space.Compact block>
            <Input data-testid="knowledge-search-input" value={query} onChange={(event) => setQuery(event.target.value)} onPressEnter={() => void runSearch()} placeholder="输入 query" />
            <InputNumber min={1} max={20} value={topK} onChange={(value) => setTopK(value ?? 5)} aria-label="Top K" />
            <Button type="primary" icon={<SearchOutlined />} loading={searching} onClick={() => void runSearch()} data-testid="knowledge-search-button">Search</Button>
          </Space.Compact>
          {searchResult && <div className="search-output" data-testid="search-results">
            <div className="result-summary"><span>{searchResult.results.length} results</span><span>Reranked: {searchResult.reranked ? "Yes" : "No"}</span></div>
            {searchResult.results.map((result, index) => <article className="result-card" key={String(result.child_id ?? index)}><div className="result-heading"><Tag>{result.source ?? "N/A"}</Tag><span>Score {typeof result.score === "number" ? result.score.toFixed(4) : "N/A"}</span></div><strong>{result.parent_id ?? result.child_id ?? `Result ${index + 1}`}</strong><p>{result.content ?? "N/A"}</p></article>)}
            <div className="timing-row">Timing: {Object.entries(searchResult.timing).map(([key, value]) => <span key={key}>{key} {Math.round(value)} ms</span>)}</div>
            <div className="fallback-row">Fallback: {searchResult.fallbacks.length ? searchResult.fallbacks.join(" · ") : "None"}</div>
          </div>}
        </Card>
        <Card className="soft-card knowledge-card" title="Knowledge Status">
          {statsLoading ? <Skeleton active /> : <div className="detail-list">
            <div><span>Dense model</span><strong>{valueOrNA(stats?.dense_model)}</strong></div>
            <div><span>Reranker</span><strong>{valueOrNA(stats?.reranker_model)}</strong></div>
            <div><span>Device</span><strong>{valueOrNA(stats?.reranker_device)}</strong></div>
            <div><span>Chunk / overlap</span><strong>{valueOrNA(stats?.child_size)} / {valueOrNA(stats?.overlap)}</strong></div>
            <div><span>Collection</span><strong>{valueOrNA(stats?.dense_collection)}</strong></div>
          </div>}
        </Card>
      </div>
      <Card className="soft-card upload-card" title="Source Upload">
        <div className="upload-row"><div><strong>Save a Markdown source</strong><p>上传只保存 source，不会自动 rebuild production index。</p></div><Upload accept=".md" maxCount={1} showUploadList={false} beforeUpload={(file) => { void handleUpload(file); return false; }} disabled={uploading}><Button icon={<UploadOutlined />} loading={uploading}>Upload .md</Button></Upload></div>
        <Button danger type="text" loading={rebuilding} onClick={confirmRebuild}>Rebuild index</Button>
      </Card>
      <Card className="soft-card advanced-trace-card" title="Advanced Trace" extra={<Button size="small" loading={debugging} onClick={() => void runDebug()}>Run trace</Button>}>
        <Collapse items={[{ key: "trace", label: debug ? "Trace ready" : "Original Query → Rewrite → Dense/BM25 → RRF → Reranker → Parents", children: debug ? <div className="debug-trace"><div><span>Original Query</span><code>{debug.query}</code></div><div><span>Rewritten Queries</span><code>{debug.rewritten_queries.join(" · ") || "N/A"}</code></div><div><span>Selected Parents</span><code>{String(debug.selected_parents.length)}</code></div><div><span>Context Preview</span><pre>{debug.context_preview || "N/A"}</pre></div><div><span>Timing</span><code>{Object.entries(debug.timing).map(([key, value]) => `${key}: ${Math.round(value)} ms`).join(" · ")}</code></div><div><span>Fallback</span><code>{debug.fallbacks.join(" · ") || "None"}</code></div></div> : <span className="muted-value">Run a query to inspect the real RAG trace.</span> }]} />
      </Card>
    </div>
  );
}
