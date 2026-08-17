import { Tag } from "antd";
import { agentLabel, agentTone } from "./agent";

interface AgentBadgeProps {
  agent: string;
  compact?: boolean;
}

export function AgentBadge({ agent, compact = false }: AgentBadgeProps) {
  return (
    <Tag className={`agent-badge ${agentTone(agent)}`} data-testid={compact ? undefined : "primary-agent-badge"}>
      {compact ? agentLabel(agent) : `Agent · ${agentLabel(agent)}`}
    </Tag>
  );
}
