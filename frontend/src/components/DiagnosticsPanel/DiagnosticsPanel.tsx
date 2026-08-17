import { CaretRightOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { Collapse, Empty, Progress, Tag, Tooltip } from "antd";
import type { ChatResponse } from "../../types/api";
import { getIntentLabel } from "../../config/presentation";
import { AgentBadge } from "../AgentBadge/AgentBadge";

interface DiagnosticsPanelProps {
  diagnostics: ChatResponse | null;
}

function percent(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function DiagnosticsPanel({ diagnostics }: DiagnosticsPanelProps) {
  if (!diagnostics) {
    return (
      <aside className="diagnostics-panel" data-testid="diagnostics-panel">
        <div className="panel-heading"><div><span>协同轨迹</span><small>COLLABORATION TRACE</small></div><InfoCircleOutlined /></div>
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="发送一条消息后查看真实链路" />
      </aside>
    );
  }

  const entityEntries = Object.entries(diagnostics.entities ?? {});
  return (
    <aside className="diagnostics-panel" data-testid="diagnostics-panel">
      <div className="panel-heading"><div><span>协同轨迹</span><small>COLLABORATION TRACE</small></div><Tooltip title="数据来自最近一次 /api/chat 响应"><InfoCircleOutlined /></Tooltip></div>
      <section className="trace-card trace-intent">
        <div className="trace-label">理解结果</div>
        <div className="trace-value-row">
          <div className="intent-copy"><Tag className="intent-badge" data-testid="intent-badge">{getIntentLabel(diagnostics.intent || "")}</Tag><span className="raw-value">{diagnostics.intent || "N/A"}</span></div>
          <span className="confidence-number">{percent(diagnostics.intent_confidence)}%</span>
        </div>
        <Progress percent={percent(diagnostics.intent_confidence)} showInfo={false} strokeColor="#70a883" trailColor="#e3f1e7" size="small" />
      </section>
      <section className="trace-card">
        <div className="trace-label">主处理角色</div>
        <AgentBadge agent={diagnostics.primary_agent || diagnostics.agent_type} />
        <div className="trace-label supporting-label">协同角色</div>
        <div className="supporting-agents">
          {diagnostics.supporting_agents.length ? diagnostics.supporting_agents.map((agent) => <AgentBadge key={agent} agent={agent} compact />) : <span className="muted-value">暂无其他角色</span>}
        </div>
      </section>
      <section className="trace-grid">
        <div className="trace-card compact-trace"><div className="trace-label">路由置信度</div><strong>{percent(diagnostics.routing_confidence)}%</strong></div>
        <div className="trace-card compact-trace"><div className="trace-label">处理耗时</div><strong>{Math.round(diagnostics.latency_ms)} ms</strong></div>
      </section>
      <section className="trace-card">
        <div className="trace-label">知识增强</div>
        <Tag className={diagnostics.knowledge_used ? "knowledge-status used" : "knowledge-status direct"} data-testid="knowledge-used-badge">
          {diagnostics.knowledge_used ? "已使用企业知识" : "直接处理"}
        </Tag>
      </section>
      <section className="trace-card">
        <div className="trace-label">业务信息</div>
        {entityEntries.length ? <div className="entity-list">{entityEntries.map(([key, values]) => <Tag key={key}>{key}: {values.join("、") || "—"}</Tag>)}</div> : <span className="muted-value">暂未识别</span>}
      </section>
      <Collapse
        ghost
        className="advanced-diagnostics"
        expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
        items={[{
          key: "technical-details",
          label: "技术详情",
          children: <div className="technical-details"><div><span>意图来源评分</span><pre className="json-preview">{prettyJson(diagnostics.intent_source_scores)}</pre></div><div><span>协同依据</span><p className="routing-reason">{diagnostics.routing_reason || "N/A"}</p></div></div>,
        }]}
      />
    </aside>
  );
}
