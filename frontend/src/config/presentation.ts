export interface AgentPresentation {
  name: string;
  shortName: string;
  tone: "green" | "blue" | "apricot" | "coral";
}

export const AGENT_PRESENTATION: Record<string, AgentPresentation> = {
  general: {
    name: "运营协调 Agent",
    shortName: "运营协调",
    tone: "green",
  },
  technical: {
    name: "技术可靠性 Agent",
    shortName: "技术可靠性",
    tone: "blue",
  },
  billing: {
    name: "收入与合规 Agent",
    shortName: "收入与合规",
    tone: "apricot",
  },
  escalation: {
    name: "运营升级通道",
    shortName: "升级通道",
    tone: "coral",
  },
};

export const INTENT_LABELS: Record<string, string> = {
  greeting: "通用咨询",
  refund: "退款处理",
  invoice: "发票处理",
  payment_issue: "支付异常",
  technical_login: "登录异常",
  technical_crash: "系统故障",
  order_status: "订单履约",
  logistics: "物流查询",
  human_handoff: "人工升级",
  escalation: "升级处理",
};

export function getAgentPresentation(agent: string): AgentPresentation {
  const key = agent.toLowerCase().replace(/_\d+$/, "");
  return AGENT_PRESENTATION[key] ?? {
    name: agent || "待识别角色",
    shortName: agent || "待识别角色",
    tone: "green",
  };
}

export function getIntentLabel(intent: string): string {
  return INTENT_LABELS[intent.toLowerCase()] ?? (intent || "待识别意图");
}
