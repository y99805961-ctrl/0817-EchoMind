import { Tag } from "antd";
import { getAgentPresentation } from "../../config/presentation";
import { agentTone } from "./agent";

interface AgentBadgeProps {
  agent: string;
  compact?: boolean;
}

export function AgentBadge({ agent, compact = false }: AgentBadgeProps) {
  const presentation = getAgentPresentation(agent);
  return (
    <Tag className={`agent-badge ${agentTone(agent)}`} data-testid={compact ? undefined : "primary-agent-badge"}>
      {compact ? presentation.shortName : presentation.name}
    </Tag>
  );
}
