import { NexusMark } from "../NexusMark/NexusMark";

interface EmptyStateProps {
  onSuggestion: (value: string) => void;
}

const suggestions = [
  { label: "订单履约", text: "订单显示已发货是什么意思？", tone: "green" },
  { label: "技术可靠性", text: "为什么登录一直提示 401？", tone: "blue" },
  { label: "收入与合规", text: "重复扣款怎么处理？", tone: "apricot" },
  { label: "财务运营", text: "电子发票怎么申请？", tone: "yellow" },
];

export function EmptyState({ onSuggestion }: EmptyStateProps) {
  return (
    <div className="empty-state" data-testid="empty-state">
      <div className="empty-mark"><NexusMark /></div>
      <span className="eyebrow">NEXUS OPERATIONS</span>
      <h2>今天需要协同处理什么？</h2>
      <p>输入企业运营请求，NexusOps 将自动理解意图并协调专业 Agent。</p>
      <div className="suggestion-grid">
        {suggestions.map((suggestion) => (
          <button className={`suggestion-card suggestion-${suggestion.tone}`} key={suggestion.text} type="button" onClick={() => onSuggestion(suggestion.text)}>
            <strong>{suggestion.label}</strong>
            <span>{suggestion.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
