import { SearchOutlined, UploadOutlined, WarningOutlined } from "@ant-design/icons";
import { Button, Card, Collapse, Input, InputNumber, Modal, Skeleton, Space, Tag, Upload, message } from "antd";
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
        messageApi.error(error instanceof Error ? error.message : "知识库状态暂不可用");
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
      messageApi.error(error instanceof Error ? error.message : "知识检索失败");
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
      messageApi.error(error instanceof Error ? error.message : "检索链路暂不可用");
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
      const detail = error instanceof ApiError && error.status === 413 ? "文件超过 10MB 限制" : error instanceof Error ? error.message : "知识源上传失败";
      messageApi.error(detail);
    } finally {
      setUploading(false);
    }
  }

  function confirmRebuild() {
    Modal.confirm({
      title: "重建知识索引？",
      icon: <WarningOutlined />,
      content: "这是高成本操作，会重建当前知识索引。确认继续吗？",
      okText: "确认重建",
      cancelText: "取消",
      onOk: async () => {
        setRebuilding(true);
        try {
          const result = await rebuildKnowledge();
          messageApi.success(result.message);
          setStats(await getKnowledgeStats());
        } catch (error) {
          messageApi.error(error instanceof Error ? error.message : "索引重建失败");
        } finally {
          setRebuilding(false);
        }
      },
    });
  }

  return (
    <div className="page-stack" data-testid="knowledge-page">
      {contextHolder}
      <div className="page-title-row"><div><span className="eyebrow">KNOWLEDGE OPERATIONS</span><h1>企业知识</h1><p>维护 NexusOps 在运营协同过程中使用的企业规则与业务知识。</p></div><Tag className="soft-status">知识索引 · {statsLoading ? "检查中" : "Ready"}</Tag></div>

      <section className="knowledge-summary" aria-labelledby="knowledge-summary-title">
        <div className="summary-heading"><div><span className="eyebrow">KNOWLEDGE INDEX</span><h2 id="knowledge-summary-title">知识库状态</h2></div><span className="summary-ready"><i /> {statsLoading ? "正在检查" : "Ready"}</span></div>
        <div className="knowledge-kpi-grid">
          <div><span>Parents</span><strong>{statsLoading ? "…" : valueOrNA(stats?.total_parents)}</strong><small>父级知识片段</small></div>
          <div><span>Children</span><strong>{statsLoading ? "…" : valueOrNA(stats?.total_children)}</strong><small>检索子片段</small></div>
          <div><span>BM25 Docs</span><strong>{statsLoading ? "…" : valueOrNA(stats?.bm25_docs)}</strong><small>关键词文档</small></div>
          <div><span>Index</span><strong>{statsLoading ? "…" : valueOrNA(stats?.index_status)}</strong><small>当前索引状态</small></div>
        </div>
      </section>

      <div className="two-column-grid knowledge-main-grid">
        <Card className="soft-card knowledge-card" title={<div className="card-title-stack"><span>知识检索</span><small>SEARCH PLAYGROUND</small></div>}>
          <p className="card-intro">模拟真实运营请求，查看企业知识召回结果。</p>
          <Space.Compact block>
            <Input data-testid="knowledge-search-input" value={query} onChange={(event) => setQuery(event.target.value)} onPressEnter={() => void runSearch()} placeholder="输入运营请求" />
            <InputNumber min={1} max={20} value={topK} onChange={(value) => setTopK(value ?? 5)} aria-label="召回数量" />
            <Button type="primary" icon={<SearchOutlined />} loading={searching} onClick={() => void runSearch()} data-testid="knowledge-search-button">检索</Button>
          </Space.Compact>
          {searchResult && <div className="search-output" data-testid="search-results">
            <div className="result-summary"><span>{searchResult.results.length} 条召回结果</span><span>{searchResult.reranked ? "已完成重排" : "未启用重排"}</span></div>
            {searchResult.results.map((result, index) => <article className="result-card" key={String(result.child_id ?? index)}><div className="result-heading"><Tag>{result.source ?? "企业规则"}</Tag><span>相关度 {typeof result.score === "number" ? result.score.toFixed(4) : "N/A"}</span></div><strong>{result.parent_id ?? result.child_id ?? `结果 ${index + 1}`}</strong><p>{result.content ?? "暂无内容摘要"}</p></article>)}
            <div className="timing-row">检索耗时：{Object.entries(searchResult.timing).map(([key, value]) => <span key={key}>{key} {Math.round(value)} ms</span>)}</div>
            <div className="fallback-row">回退路径：{searchResult.fallbacks.length ? searchResult.fallbacks.join(" · ") : "无"}</div>
          </div>}
        </Card>
        <Card className="soft-card knowledge-card" title={<div className="card-title-stack"><span>知识库详情</span><small>INDEX DETAILS</small></div>}>
          {statsLoading ? <Skeleton active /> : <div className="detail-list">
            <div><span>向量模型</span><strong>{valueOrNA(stats?.dense_model)}</strong></div>
            <div><span>重排模型</span><strong>{valueOrNA(stats?.reranker_model)}</strong></div>
            <div><span>运行设备</span><strong>{valueOrNA(stats?.reranker_device)}</strong></div>
            <div><span>切片 / 重叠</span><strong>{valueOrNA(stats?.child_size)} / {valueOrNA(stats?.overlap)}</strong></div>
            <div><span>知识集合</span><strong>{valueOrNA(stats?.dense_collection)}</strong></div>
          </div>}
        </Card>
      </div>

      <Card className="soft-card upload-card" title={<div className="card-title-stack"><span>知识源管理</span><small>SOURCE MANAGEMENT</small></div>}>
        <div className="upload-row"><div><strong>上传新的企业知识文档</strong><p>知识源保存后需要显式重建索引，才会进入检索链路。</p></div><Upload accept=".md" maxCount={1} showUploadList={false} beforeUpload={(file) => { void handleUpload(file); return false; }} disabled={uploading}><Button icon={<UploadOutlined />} loading={uploading}>上传 Markdown</Button></Upload></div>
        <Button type="text" loading={rebuilding} onClick={confirmRebuild}>重建知识索引</Button>
      </Card>

      <Card className="soft-card advanced-trace-card" title={<div className="card-title-stack"><span>检索链路</span><small>RAG TRACE</small></div>} extra={<Button size="small" loading={debugging} onClick={() => void runDebug()}>运行链路</Button>}>
        <Collapse items={[{ key: "trace", label: debug ? "链路已就绪" : "原始请求 → 改写 → Dense / BM25 → RRF → Reranker → Parents", children: debug ? <div className="debug-trace"><div><span>原始请求</span><code>{debug.query}</code></div><div><span>改写请求</span><code>{debug.rewritten_queries.join(" · ") || "N/A"}</code></div><div><span>选中父片段</span><code>{String(debug.selected_parents.length)}</code></div><div><span>上下文预览</span><pre>{debug.context_preview || "N/A"}</pre></div><div><span>耗时</span><code>{Object.entries(debug.timing).map(([key, value]) => `${key}: ${Math.round(value)} ms`).join(" · ")}</code></div><div><span>回退路径</span><code>{debug.fallbacks.join(" · ") || "无"}</code></div></div> : <span className="muted-value">运行一次检索，查看真实 RAG 链路。</span> }]} />
      </Card>
    </div>
  );
}
