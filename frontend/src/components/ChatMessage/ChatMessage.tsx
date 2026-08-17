import { ReloadOutlined } from "@ant-design/icons";
import { Button, Typography } from "antd";
import ReactMarkdown from "react-markdown";
import type { ChatMessage as ChatMessageModel } from "../../stores/chatStore";
import { AgentBadge } from "../AgentBadge/AgentBadge";
import { NexusMark } from "../NexusMark/NexusMark";

interface ChatMessageProps {
  message: ChatMessageModel;
  onRetry?: (message: ChatMessageModel) => void;
}

export function ChatMessage({ message, onRetry }: ChatMessageProps) {
  const isUser = message.role === "user";
  return (
    <article className={`message-row ${isUser ? "message-user" : "message-assistant"}`} data-testid={`${isUser ? "user" : "assistant"}-message`}>
      {!isUser && <div className="assistant-avatar"><NexusMark small /></div>}
      <div className="message-content-wrap">
        {!isUser && <div className="message-author"><strong>NexusOps</strong><span>协同结果</span></div>}
        <div className="message-bubble">
          {isUser ? (
            <Typography.Paragraph>{message.content}</Typography.Paragraph>
          ) : (
            <div className="markdown-body">
              <ReactMarkdown skipHtml>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && message.diagnostics && (
          <div className="message-meta">
            <AgentBadge agent={message.diagnostics.primary_agent} compact />
            {message.diagnostics.knowledge_used && <span className="knowledge-pill">已使用企业知识</span>}
            <span>{Math.round(message.diagnostics.latency_ms)} ms</span>
          </div>
        )}
        {message.status === "error" && (
          <div className="message-error" role="alert">
            <span>{message.error ?? "本次协同暂未完成"}</span>
            {onRetry && <Button type="link" size="small" icon={<ReloadOutlined />} onClick={() => onRetry(message)}>重试</Button>}
          </div>
        )}
      </div>
    </article>
  );
}
