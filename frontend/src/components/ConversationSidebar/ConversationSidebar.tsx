import { FileTextOutlined, PlusOutlined } from "@ant-design/icons";
import { Button } from "antd";
import type { Conversation } from "../../stores/chatStore";

interface ConversationSidebarProps {
  conversations: Conversation[];
  activeId: string;
  onNew: () => void;
  onSelect: (id: string) => void;
}

export function ConversationSidebar({ conversations, activeId, onNew, onSelect }: ConversationSidebarProps) {
  const today = new Date().toDateString();
  const todayItems = conversations.filter((item) => new Date(item.updatedAt).toDateString() === today);
  const earlierItems = conversations.filter((item) => new Date(item.updatedAt).toDateString() !== today);

  const group = (title: string, items: Conversation[]) => (
    <div className="conversation-group" key={title}>
      <div className="conversation-group-title">{title}</div>
      {items.length === 0 ? <div className="conversation-empty">—</div> : items.map((conversation) => (
        <button
          className={`conversation-item ${conversation.id === activeId ? "is-active" : ""}`}
          data-testid="conversation-item"
          key={conversation.id}
          type="button"
          onClick={() => onSelect(conversation.id)}
        >
          <FileTextOutlined />
          <span>{conversation.title}</span>
        </button>
      ))}
    </div>
  );

  return (
    <aside className="conversation-sidebar" aria-label="业务请求">
      <div className="sidebar-heading"><span className="eyebrow">REQUEST HISTORY</span><h2>业务请求</h2><p>保留每一次协同上下文</p></div>
      <Button className="new-chat-button" icon={<PlusOutlined />} onClick={onNew} data-testid="new-chat-button">新建协同</Button>
      <div className="conversation-list">
        {group("今天", todayItems)}
        {group("更早", earlierItems)}
      </div>
    </aside>
  );
}
