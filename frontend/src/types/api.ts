export type AgentName = string;

export interface AgentStats {
  total_calls?: number;
  success_calls?: number;
  success_rate?: number;
  avg_ms?: number;
  [key: string]: unknown;
}

export interface ChatRequest {
  message: string;
  user_id: string;
  conv_id?: string;
}

export interface ChatResponse {
  conv_id: string;
  response: string;
  intent: string;
  intent_group: string;
  agent_type: AgentName;
  agent_types: AgentName[];
  primary_agent: AgentName;
  supporting_agents: AgentName[];
  routing_reason: string;
  routing_confidence: number;
  escalated: boolean;
  latency_ms: number;
  knowledge_used: boolean;
  entities: Record<string, string[]>;
  intent_confidence: number;
  intent_source_scores: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  agents: Record<string, AgentStats>;
}

export interface MonitorResponse {
  agent_stats: Record<string, AgentStats>;
  tool_stats: Record<string, AgentStats & { circuit_state?: string; consecutive_fails?: number }>;
  active_alerts: AlertItem[];
  suggestions: SuggestionItem[];
}

export interface AlertItem {
  severity: string;
  metric: string;
  message: string;
  value: number;
  threshold: number;
  ts: string;
  resolved: boolean;
}

export interface SuggestionItem {
  title: string;
  action: string;
  priority: number;
}

export interface KnowledgeStats {
  index_status?: string;
  total_documents?: number;
  total_parents?: number;
  total_children?: number;
  dense_model?: string;
  dense_collection?: string;
  bm25_docs?: number;
  reranker_model?: string;
  reranker_device?: string;
  chunk_strategy?: string;
  child_size?: number;
  overlap?: number;
  [key: string]: unknown;
}

export interface SearchResult {
  child_id?: string;
  parent_id?: string;
  content?: string;
  source?: string;
  score?: number;
  rank?: number;
  retriever?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  reranked: boolean;
  timing: Record<string, number>;
  fallbacks: string[];
}

export interface UploadResponse {
  message: string;
  source?: string;
  saved_sources?: string[];
  requires_rebuild: boolean;
}

export interface RebuildResponse {
  message: string;
  manifest: Record<string, unknown>;
}

export interface RAGDebugResponse {
  query: string;
  intent: string | null;
  rewritten_queries: string[];
  dense_results: SearchResult[];
  bm25_results: SearchResult[];
  rrf_results: SearchResult[];
  reranked_results: SearchResult[];
  selected_parents: Array<Record<string, unknown>>;
  context_preview: string;
  timing: Record<string, number>;
  fallbacks: string[];
  reranker_status: string;
  index_status: string;
}
