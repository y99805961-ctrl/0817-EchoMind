import { Alert, Card, Empty, Progress, Skeleton, Statistic, Tag } from "antd";
import { useEffect, useState } from "react";
import { getMonitor } from "../api/chat";
import type { AgentStats, MonitorResponse } from "../types/api";

function display(value: unknown): string {
  return value === undefined || value === null || value === "" ? "N/A" : typeof value === "number" ? value.toFixed(1) : String(value);
}

function rate(value: unknown): number | null {
  return typeof value === "number" ? Math.round(Math.max(0, Math.min(1, value)) * 100) : null;
}

function StatCard({ name, stats }: { name: string; stats: AgentStats }) {
  const success = rate(stats.success_rate);
  return (
    <Card className="soft-card monitor-agent-card"><div className="monitor-card-heading"><strong>{name}</strong><Tag>{success === null ? "N/A" : `${success}% success`}</Tag></div><div className="monitor-stat-row"><span>Calls</span><strong>{display(stats.total_calls)}</strong></div><div className="monitor-stat-row"><span>Average latency</span><strong>{display(stats.avg_ms)} ms</strong></div>{success !== null && <Progress percent={success} showInfo={false} strokeColor="#afc3ce" trailColor="#edf1f1" size="small" />}</Card>
  );
}

export function MonitorPage() {
  const [data, setData] = useState<MonitorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        setData(await getMonitor());
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Monitor unavailable");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const agents = Object.entries(data?.agent_stats ?? {});
  const tools = Object.entries(data?.tool_stats ?? {});
  return (
    <div className="page-stack" data-testid="monitor-page">
      <div className="page-title-row"><div><span className="eyebrow">RUNTIME OBSERVABILITY</span><h1>Monitor</h1><p>在线 Agent、工具、告警和建议。字段缺失时保持 N/A。</p></div><Tag className="soft-status">Live API data</Tag></div>
      {error && <Alert type="warning" showIcon message="Monitor data unavailable" description={error} />}
      {loading ? <Skeleton active /> : <>
        <div className="stats-grid monitor-summary-grid"><Card className="soft-card"><Statistic title="Agents" value={agents.length} /></Card><Card className="soft-card"><Statistic title="Tools" value={tools.length} /></Card><Card className="soft-card"><Statistic title="Active alerts" value={data?.active_alerts.length ?? 0} /></Card><Card className="soft-card"><Statistic title="Suggestions" value={data?.suggestions.length ?? 0} /></Card></div>
        <section><div className="section-heading"><h2>Agent stats</h2><span>实时接口字段</span></div>{agents.length ? <div className="monitor-grid">{agents.map(([name, stats]) => <StatCard key={name} name={name} stats={stats} />)}</div> : <Empty description="N/A" />}</section>
        <section><div className="section-heading"><h2>Tool stats</h2><span>真实调用数据</span></div>{tools.length ? <div className="monitor-grid">{tools.map(([name, stats]) => <Card className="soft-card monitor-agent-card" key={name}><div className="monitor-card-heading"><strong>{name}</strong><Tag>{String(stats.circuit_state ?? "N/A")}</Tag></div><div className="monitor-stat-row"><span>Calls</span><strong>{display(stats.total_calls)}</strong></div><div className="monitor-stat-row"><span>Avg latency</span><strong>{display(stats.avg_latency_ms)} ms</strong></div><div className="monitor-stat-row"><span>Consecutive fails</span><strong>{display(stats.consecutive_fails)}</strong></div></Card>)}</div> : <Empty description="N/A" />}</section>
        <div className="two-column-grid"><Card className="soft-card" title="Active alerts">{data?.active_alerts.length ? data.active_alerts.map((alert) => <Alert key={`${alert.metric}-${alert.ts}`} type={alert.severity === "critical" || alert.severity === "error" ? "error" : "warning"} showIcon message={alert.message} description={`${alert.metric}: ${display(alert.value)} · threshold ${display(alert.threshold)}`} />) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No active alerts" />}</Card><Card className="soft-card" title="Recommendations">{data?.suggestions.length ? data.suggestions.map((suggestion) => <div className="suggestion-item" key={`${suggestion.title}-${suggestion.priority}`}><Tag>Priority {suggestion.priority}</Tag><strong>{suggestion.title}</strong><p>{suggestion.action}</p></div>) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="N/A" />}</Card></div>
      </>}
    </div>
  );
}
