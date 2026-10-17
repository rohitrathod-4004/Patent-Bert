export interface SearchRequest {
  query: string;
  candidate_k: number;
  top_k: number;
}

export interface PatentResult {
  patent_id: string;
  similarity: number;
  chunk_similarity: number;
  hybrid_score: number;
  ipc_class: string | null;
  subclass: string | null;
  label: number | null;
  matched_chunk: string | null;
}

export interface SearchResponse {
  query: string;
  candidate_k: number;
  top_k: number;
  total_results: number;
  results: PatentResult[];
}

export interface HealthResponse {
  message: string;
  status: string;
}

export type ActiveView =
  | 'overview'
  | 'search'
  | 'explorer'
  | 'analytics'
  | 'classification';

export interface GraphSimilarityRequest {
  patent_ids: string[];
  similarity_threshold: number;
}

export interface GraphNode {
  id: string;
  patent_id: string;
  ipc_class: string | null;
  subclass: string | null;
  label: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  similarity: number;
}

export interface GraphSimilarityResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}
