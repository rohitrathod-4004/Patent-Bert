import axios, { AxiosError } from 'axios';
import type {
  SearchRequest,
  SearchResponse,
  HealthResponse,
  GraphSimilarityRequest,
  GraphSimilarityResponse,
} from '../types/patent';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 45000, // 45s to allow for CPU chunk reranking if server is busy
});

export async function searchPatents(request: SearchRequest): Promise<SearchResponse> {
  try {
    const response = await apiClient.post<SearchResponse>('/search', request);
    return response.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      const axiosErr = err as AxiosError<{ detail?: string | Array<{ msg: string }> }>;
      if (axiosErr.response?.data?.detail) {
        const detail = axiosErr.response.data.detail;
        if (typeof detail === 'string') {
          throw new Error(detail, { cause: err });
        } else if (Array.isArray(detail) && detail.length > 0) {
          throw new Error(detail.map((d) => d.msg).join(', '), { cause: err });
        }
      }
      if (axiosErr.response?.status === 404) {
        throw new Error('Search endpoint not found. Ensure the backend server is running.', { cause: err });
      }
      if (axiosErr.code === 'ECONNABORTED') {
        throw new Error('Search request timed out. Please try a query with fewer candidates.', { cause: err });
      }
      if (!axiosErr.response) {
        throw new Error('Unable to connect to the backend API. Please verify the server is running on port 8000.', { cause: err });
      }
    }
    throw new Error('An unexpected error occurred while executing the patent search.', { cause: err });
  }
}

export async function getGraphSimilarity(
  request: GraphSimilarityRequest
): Promise<GraphSimilarityResponse> {
  try {
    const response = await apiClient.post<GraphSimilarityResponse>('/graph/similarity', request);
    return response.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      const axiosErr = err as AxiosError<{ detail?: string | Array<{ msg: string }> }>;
      if (axiosErr.response?.data?.detail) {
        const detail = axiosErr.response.data.detail;
        if (typeof detail === 'string') {
          throw new Error(detail, { cause: err });
        } else if (Array.isArray(detail) && detail.length > 0) {
          throw new Error(detail.map((d) => d.msg).join(', '), { cause: err });
        }
      }
      if (axiosErr.response?.status === 404) {
        throw new Error('One or more requested patent IDs were not found in the index.', { cause: err });
      }
      if (!axiosErr.response) {
        throw new Error('Unable to connect to backend API for graph similarity.', { cause: err });
      }
    }
    throw new Error('An unexpected error occurred while computing patent graph similarity.', { cause: err });
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/');
  return response.data;
}
