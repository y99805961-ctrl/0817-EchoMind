import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatResponse } from "../types/api";

export type MessageRole = "user" | "assistant";
export type MessageStatus = "sending" | "sent" | "error";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  status: MessageStatus;
  error?: string;
  requestText?: string;
  diagnostics?: ChatResponse;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string;
  createConversation: () => string;
  selectConversation: (id: string) => void;
  appendMessage: (conversationId: string, message: ChatMessage) => void;
  updateMessage: (conversationId: string, messageId: string, update: Partial<ChatMessage>) => void;
  setConversationTitle: (conversationId: string, title: string) => void;
}

function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newConversation(): Conversation {
  const now = new Date().toISOString();
  return { id: uuid(), title: "新对话", createdAt: now, updatedAt: now, messages: [] };
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      conversations: [newConversation()],
      activeConversationId: "",
      createConversation: () => {
        const conversation = newConversation();
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: conversation.id,
        }));
        return conversation.id;
      },
      selectConversation: (activeConversationId) => set({ activeConversationId }),
      appendMessage: (conversationId, message) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? { ...conversation, messages: [...conversation.messages, message], updatedAt: new Date().toISOString() }
              : conversation,
          ),
        })),
      updateMessage: (conversationId, messageId, update) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  updatedAt: new Date().toISOString(),
                  messages: conversation.messages.map((message) =>
                    message.id === messageId ? { ...message, ...update } : message,
                  ),
                }
              : conversation,
          ),
        })),
      setConversationTitle: (conversationId, title) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId ? { ...conversation, title: title.slice(0, 28) } : conversation,
          ),
        })),
    }),
    {
      name: "echomind-conversations",
      partialize: (state) => ({ conversations: state.conversations, activeConversationId: state.activeConversationId }),
      onRehydrateStorage: () => (state) => {
        if (state && !state.activeConversationId && state.conversations[0]) {
          state.activeConversationId = state.conversations[0].id;
        }
      },
    },
  ),
);
