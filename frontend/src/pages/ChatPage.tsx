import { MenuUnfoldOutlined, SettingOutlined } from "@ant-design/icons";
import { Button, Drawer } from "antd";
import { useEffect, useMemo, useState } from "react";
import { sendChat } from "../api/chat";
import type { ChatResponse } from "../types/api";
import { useAppStore } from "../stores/appStore";
import { useChatStore, type ChatMessage as ChatMessageModel } from "../stores/chatStore";
import { ChatComposer } from "../components/ChatComposer/ChatComposer";
import { ChatMessage } from "../components/ChatMessage/ChatMessage";
import { ConversationSidebar } from "../components/ConversationSidebar/ConversationSidebar";
import { DiagnosticsPanel } from "../components/DiagnosticsPanel/DiagnosticsPanel";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { LoadingBubble } from "../components/LoadingBubble/LoadingBubble";

function messageId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Echo 暂时没有响应";
}

export function ChatPage() {
  const conversations = useChatStore((state) => state.conversations);
  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const createConversation = useChatStore((state) => state.createConversation);
  const selectConversation = useChatStore((state) => state.selectConversation);
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const setConversationTitle = useChatStore((state) => state.setConversationTitle);
  const userId = useAppStore((state) => state.userId);
  const mobileDrawer = useAppStore((state) => state.mobileDrawer);
  const setMobileDrawer = useAppStore((state) => state.setMobileDrawer);
  const [loading, setLoading] = useState(false);

  const activeConversation = conversations.find((item) => item.id === activeConversationId) ?? conversations[0];
  useEffect(() => {
    if (!activeConversationId && activeConversation) selectConversation(activeConversation.id);
  }, [activeConversation, activeConversationId, selectConversation]);

  const latestDiagnostics = useMemo<ChatResponse | null>(() => {
    const messages = activeConversation?.messages ?? [];
    return [...messages].reverse().find((message) => message.diagnostics)?.diagnostics ?? null;
  }, [activeConversation]);

  async function sendMessage(text: string, retryMessageId?: string) {
    const conversation = activeConversation;
    if (!conversation || loading) return;
    const convId = conversation.id;
    if (retryMessageId) {
      updateMessage(convId, retryMessageId, { status: "sending", content: "Echo 正在整理…", error: undefined });
    } else {
      appendMessage(convId, {
        id: messageId(),
        role: "user",
        content: text,
        createdAt: new Date().toISOString(),
        status: "sent",
      });
      if (conversation.title === "新对话") setConversationTitle(convId, text);
    }
    setLoading(true);
    try {
      const response = await sendChat({ message: text, user_id: userId, conv_id: convId });
      if (retryMessageId) {
        updateMessage(convId, retryMessageId, { status: "sent", content: response.response, diagnostics: response, error: undefined });
      } else {
        appendMessage(convId, {
          id: messageId(),
          role: "assistant",
          content: response.response,
          createdAt: new Date().toISOString(),
          status: "sent",
          diagnostics: response,
        });
      }
    } catch (error) {
      const detail = errorMessage(error);
      if (retryMessageId) {
        updateMessage(convId, retryMessageId, { status: "error", content: "Echo 暂时没有响应", error: detail });
      } else {
        appendMessage(convId, {
          id: messageId(),
          role: "assistant",
          content: "Echo 暂时没有响应",
          createdAt: new Date().toISOString(),
          status: "error",
          error: detail,
          requestText: text,
        });
      }
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    createConversation();
    setMobileDrawer(null);
  }

  function selectChat(id: string) {
    selectConversation(id);
    setMobileDrawer(null);
  }

  return (
    <div className="chat-page" data-testid="chat-page">
      <div className="mobile-toolbar">
        <Button icon={<MenuUnfoldOutlined />} onClick={() => setMobileDrawer("conversations")} aria-label="打开会话列表" />
        <span>Chat workspace</span>
        <Button icon={<SettingOutlined />} onClick={() => setMobileDrawer("diagnostics")} aria-label="打开诊断面板" />
      </div>
      <div className="chat-workspace">
        <ConversationSidebar conversations={conversations} activeId={activeConversation?.id ?? ""} onNew={newChat} onSelect={selectChat} />
        <section className="chat-column">
          <div className="chat-heading">
            <div><span className="eyebrow">ECHO WORKSPACE</span><h1>Ask Echo anything</h1></div>
            <span className="chat-heading-note">Memory-aware · grounded when useful</span>
          </div>
          <div className="message-list" aria-live="polite">
            {!activeConversation?.messages.length ? <EmptyState onSuggestion={(value) => void sendMessage(value)} /> : activeConversation.messages.map((message) => (
              <ChatMessage key={message.id} message={message} onRetry={(item: ChatMessageModel) => item.requestText && void sendMessage(item.requestText, item.id)} />
            ))}
            {loading && <LoadingBubble />}
          </div>
          <div className="composer-dock">
            <ChatComposer disabled={loading || !activeConversation} onSend={(value) => void sendMessage(value)} />
            <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
          </div>
        </section>
        <DiagnosticsPanel diagnostics={latestDiagnostics} />
      </div>
      <Drawer title="Conversations" placement="left" open={mobileDrawer === "conversations"} onClose={() => setMobileDrawer(null)} width={300}>
        <ConversationSidebar conversations={conversations} activeId={activeConversation?.id ?? ""} onNew={newChat} onSelect={selectChat} />
      </Drawer>
      <Drawer title="Diagnostics" placement="right" open={mobileDrawer === "diagnostics"} onClose={() => setMobileDrawer(null)} width={340}>
        <DiagnosticsPanel diagnostics={latestDiagnostics} />
      </Drawer>
    </div>
  );
}
