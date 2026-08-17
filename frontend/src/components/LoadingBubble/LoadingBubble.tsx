import { NexusMark } from "../NexusMark/NexusMark";

export function LoadingBubble() {
  return (
    <div className="loading-row" aria-live="polite" data-testid="loading-bubble">
      <div className="assistant-avatar"><NexusMark small /></div>
      <div className="loading-bubble">
        <span className="loading-dot" />
        <span className="loading-dot" />
        <span className="loading-dot" />
        <span>正在协调专业 Agent…</span>
      </div>
    </div>
  );
}
