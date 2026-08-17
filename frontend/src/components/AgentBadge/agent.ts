import { getAgentPresentation } from "../../config/presentation";

export function agentLabel(agent: string): string {
  return getAgentPresentation(agent).name;
}

export function agentTone(agent: string): string {
  return `agent-${getAgentPresentation(agent).tone}`;
}
