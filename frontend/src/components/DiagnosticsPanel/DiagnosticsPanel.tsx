import { CaretRightOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { Collapse, Empty, Progress, Tag, Tooltip } from "antd";
import type { ChatResponse } from "../../types/api";
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
        <div className="panel-heading"><span>Agent Trace</span><InfoCircleOutlined /></div>
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="发送一条消息后查看真实链路" />
      </aside>
    );
  }

  const entityEntries = Object.entries(diagnostics.entities ?? {});
  return (
    <aside className="diagnostics-panel" data-testid="diagnostics-panel">
      <div className="panel-heading"><span>Agent Trace</span><Tooltip title="数据来自最近一次 /api/chat 响应"><InfoCircleOutlined /></Tooltip></div>
      <section className="trace-card">
        <div className="trace-label">Intent</div>
        <div className="trace-value-row">
          <Tag className="intent-badge" data-testid="intent-badge">{diagnostics.intent || "N/A"}</Tag>
          <span className="confidence-number">{percent(diagnostics.intent_confidence)}%</span>
        </div>
        <Progress percent={percent(diagnostics.intent_confidence)} showInfo={false} strokeColor="#bfcdbA" trailColor="#edf0ed" size="small" />
      </section>
      <section className="trace-card">
        <div className="trace-label">Primary Agent</div>
        <AgentBadge agent={diagnostics.primary_agent || diagnostics.agent_type} />
        <div className="trace-label supporting-label">Supporting</div>
        <div className="supporting-agents">
          {diagnostics.supporting_agents.length ? diagnostics.supporting_agents.map((agent) => <AgentBadge key={agent} agent={agent} compact />) : <span className="muted-value">—</span>}
        </div>
      </section>
      <section className="trace-grid">
        <div className="trace-card compact-trace"><div className="trace-label">Routing Confidence</div><strong>{percent(diagnostics.routing_confidence)}%</strong></div>
        <div className="trace-card compact-trace"><div className="trace-label">Latency</div><strong>{Math.round(diagnostics.latency_ms)} ms</strong></div>
      </section>
      <section className="trace-card">
        <div className="trace-label">Knowledge</div>
        <Tag className={diagnostics.knowledge_used ? "knowledge-status used" : "knowledge-status direct"} data-testid="knowledge-used-badge">
          {diagnostics.knowledge_used ? "Knowledge Used" : "Direct Answer"}
        </Tag>
      </section>
      <section className="trace-card">
        <div className="trace-label">Entities</div>
        {entityEntries.length ? <div className="entity-list">{entityEntries.map(([key, values]) => <Tag key={key}>{key}: {values.join("、") || "—"}</Tag>)}</div> : <span className="muted-value">N/A</span>}
      </section>
      <Collapse
        ghost
        className="advanced-diagnostics"
        expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
        items={[{
          key: "source-scores",
          label: "Intent Source Scores",
          children: <pre className="json-preview">{prettyJson(diagnostics.intent_source_scores)}</pre>,
        }, {
          key: "routing-reason",
          label: <Tooltip title={diagnostics.routing_reason || "N/A"}>Routing Reason</Tooltip>,
          children: <p className="routing-reason">{diagnostics.routing_reason || "N/A"}</p>,
        }]}
      />
    </aside>
  );
}
