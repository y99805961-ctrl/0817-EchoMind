export function LoadingBubble() {
  return (
    <div className="loading-row" aria-live="polite" data-testid="loading-bubble">
      <div className="assistant-avatar" aria-hidden="true">ʕ</div>
      <div className="loading-bubble">
        <span className="loading-dot" />
        <span className="loading-dot" />
        <span className="loading-dot" />
        <span>Echo 正在整理…</span>
      </div>
    </div>
  );
}
