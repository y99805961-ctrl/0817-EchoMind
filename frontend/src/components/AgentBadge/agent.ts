const labels: Record<string, string> = {
  general: "General",
  technical: "Technical",
  billing: "Billing",
  escalation: "Escalation",
};

export function agentLabel(agent: string): string {
  return labels[agent.toLowerCase()] ?? (agent || "N/A");
}

export function agentTone(agent: string): string {
  switch (agent.toLowerCase()) {
    case "technical":
      return "agent-blue";
    case "billing":
      return "agent-apricot";
    case "escalation":
      return "agent-pink";
    default:
      return "agent-neutral";
  }
}
