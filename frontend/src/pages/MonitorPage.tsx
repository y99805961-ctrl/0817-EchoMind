import { Alert, Card, Empty, Progress, Skeleton, Tag } from "antd";
import { useEffect, useState } from "react";
import { getAgentPresentation } from "../config/presentation";
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
  const presentation = getAgentPresentation(name);
  return (
    <Card className={`soft-card monitor-agent-card monitor-${presentation.tone}`}>
      <div className="monitor-card-heading"><div><strong>{presentation.name}</strong><small>{name}</small></div><Tag>{success === null ? "N/A" : `${success}% 成功率`}</Tag></div>
      <div className="monitor-stat-row"><span>调用次数</span><strong>{display(stats.total_calls)}</strong></div>
      <div className="monitor-stat-row"><span>平均耗时</span><strong>{display(stats.avg_ms)} ms</strong></div>
      {success !== null && <Progress percent={success} showInfo={false} strokeColor={presentation.tone === "blue" ? "#79afc5" : presentation.tone === "apricot" ? "#e7a96b" : "#70a883"} trailColor="#e8f0e9" size="small" />}
    </Card>
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
        setError(reason instanceof Error ? reason.message : "运行观测暂不可用");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const agents = Object.entries(data?.agent_stats ?? {});
  const tools = Object.entries(data?.tool_stats ?? {});
  return (
    <div className="page-stack" data-testid="monitor-page">
      <div className="page-title-row"><div><span className="eyebrow">RUNTIME OBSERVABILITY</span><h1>系统运行观测</h1><p>观察专业 Agent、工具链和协同流程的实时运行状态。</p></div><Tag className="soft-status">实时运行数据</Tag></div>
      {error && <Alert type="warning" showIcon message="运行观测暂不可用" description={error} />}
      {loading ? <Skeleton active /> : <>
        <div className="stats-grid monitor-summary-grid"><div className="summary-stat"><span>专业 Agent</span><strong>{agents.length}</strong><small>当前协同角色</small></div><div className="summary-stat"><span>工具链</span><strong>{tools.length}</strong><small>已接入工具</small></div><div className="summary-stat"><span>活动告警</span><strong>{data?.active_alerts.length ?? 0}</strong><small>需要关注</small></div><div className="summary-stat"><span>运行建议</span><strong>{data?.suggestions.length ?? 0}</strong><small>系统建议</small></div></div>
        <section><div className="section-heading"><div><h2>Agent 状态</h2><span>专业角色的实时协同表现</span></div></div>{agents.length ? <div className="monitor-grid">{agents.map(([name, stats]) => <StatCard key={name} name={name} stats={stats} />)}</div> : <Empty description="暂无角色数据" />}</section>
        <section><div className="section-heading"><div><h2>工具状态</h2><span>真实调用与熔断状态</span></div></div>{tools.length ? <div className="monitor-grid">{tools.map(([name, stats]) => <Card className="soft-card monitor-agent-card" key={name}><div className="monitor-card-heading"><div><strong>{name}</strong><small>工具链节点</small></div><Tag>{String(stats.circuit_state ?? "N/A")}</Tag></div><div className="monitor-stat-row"><span>调用次数</span><strong>{display(stats.total_calls)}</strong></div><div className="monitor-stat-row"><span>平均耗时</span><strong>{display(stats.avg_latency_ms)} ms</strong></div><div className="monitor-stat-row"><span>连续失败</span><strong>{display(stats.consecutive_fails)}</strong></div></Card>)}</div> : <Empty description="暂无工具数据" />}</section>
        <div className="two-column-grid"><Card className="soft-card" title={<div className="card-title-stack"><span>活动告警</span><small>ACTIVE ALERTS</small></div>}>{data?.active_alerts.length ? data.active_alerts.map((alert) => <Alert key={`${alert.metric}-${alert.ts}`} type={alert.severity === "critical" || alert.severity === "error" ? "error" : "warning"} showIcon message={alert.message} description={`${alert.metric}: ${display(alert.value)} · threshold ${display(alert.threshold)}`} />) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前没有活动告警" />}</Card><Card className="soft-card" title={<div className="card-title-stack"><span>运行建议</span><small>RECOMMENDATIONS</small></div>}>{data?.suggestions.length ? data.suggestions.map((suggestion) => <div className="suggestion-item" key={`${suggestion.title}-${suggestion.priority}`}><Tag>优先级 {suggestion.priority}</Tag><strong>{suggestion.title}</strong><p>{suggestion.action}</p></div>) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无运行建议" />}</Card></div>
      </>}
    </div>
  );
}
