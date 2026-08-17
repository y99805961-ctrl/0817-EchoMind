interface EmptyStateProps {
  onSuggestion: (value: string) => void;
}

const suggestions = [
  "我的退款什么时候到账？",
  "为什么登录一直失败？",
  "订单显示已发货是什么意思？",
  "电子发票怎么申请？",
];

export function EmptyState({ onSuggestion }: EmptyStateProps) {
  return (
    <div className="empty-state" data-testid="empty-state">
      <div className="bear-mark" aria-hidden="true">ʕ•ᴥ•ʔ</div>
      <h2>你好，我是 Echo</h2>
      <p>有什么可以帮你的？</p>
      <div className="suggestion-grid">
        {suggestions.map((suggestion) => (
          <button className="suggestion-chip" key={suggestion} type="button" onClick={() => onSuggestion(suggestion)}>
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
