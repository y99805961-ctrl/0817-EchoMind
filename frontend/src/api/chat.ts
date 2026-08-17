import { apiGet, apiPost } from "./client";
import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  KnowledgeStats,
  MonitorResponse,
  RAGDebugResponse,
  RebuildResponse,
  SearchResponse,
  UploadResponse,
} from "../types/api";

export function sendChat(request: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  return apiPost<ChatResponse>("/chat", request, signal);
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health", signal);
}

export function getMonitor(signal?: AbortSignal): Promise<MonitorResponse> {
  return apiGet<MonitorResponse>("/monitor", signal);
}

export function getKnowledgeStats(signal?: AbortSignal): Promise<KnowledgeStats> {
  return apiGet<KnowledgeStats>("/knowledge/stats", signal);
}

export function searchKnowledge(query: string, topK = 5, signal?: AbortSignal): Promise<SearchResponse> {
  const params = new URLSearchParams({ query, top_k: String(topK) });
  return apiPost<SearchResponse>(`/search?${params.toString()}`, undefined, signal);
}

export function uploadKnowledge(file: File, signal?: AbortSignal): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return apiPost<UploadResponse>("/knowledge/upload", form, signal);
}

export function rebuildKnowledge(signal?: AbortSignal): Promise<RebuildResponse> {
  return apiPost<RebuildResponse>("/knowledge/rebuild", undefined, signal);
}

export function debugRag(query: string, signal?: AbortSignal): Promise<RAGDebugResponse> {
  return apiPost<RAGDebugResponse>("/rag/debug", { query }, signal);
}
