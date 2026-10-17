from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
from typing import Any, List, Optional

import faiss
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer

# Resolve project root and ensure imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.patent_search import patent_search

logger = logging.getLogger("patentbert")

INDEX_PATH = PROJECT_ROOT / "artifacts" / "patent_embeddings.index"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "patent_metadata.parquet"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EXPECTED_DIM = 768
REQUIRED_METADATA_COLUMNS = ["patent_id", "text", "ipc_class", "subclass", "label"]


# --- Schemas ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Patent search query text")
    candidate_k: int = Field(default=10, ge=1, le=20, description="Candidates for chunk reranking (1-20)")
    top_k: int = Field(default=5, ge=1, description="Number of results to return")

    @model_validator(mode="after")
    def validate_bounds(self):
        cleaned_query = self.query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty or whitespace only.")
        self.query = cleaned_query
        if self.top_k > self.candidate_k:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than candidate_k ({self.candidate_k}).")
        return self


class PatentResult(BaseModel):
    patent_id: str
    similarity: float
    chunk_similarity: float
    hybrid_score: float
    ipc_class: Optional[str] = None
    subclass: Optional[str] = None
    label: Optional[int] = None
    matched_chunk: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    candidate_k: int
    top_k: int
    total_results: int
    results: List[PatentResult]


# --- Graph Similarity Schemas ---
class GraphSimilarityRequest(BaseModel):
    patent_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=25,
        description="List of 2 to 25 unique patent ID strings",
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for graph edges (0.0 to 1.0)",
    )

    @model_validator(mode="after")
    def validate_ids(self):
        cleaned = [pid.strip() for pid in self.patent_ids if pid and isinstance(pid, str)]
        if len(cleaned) < len(self.patent_ids) or any(not pid for pid in cleaned):
            raise ValueError("patent_ids must not contain empty or blank strings.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("patent_ids must contain unique values without duplicates.")
        self.patent_ids = cleaned
        return self


class GraphNode(BaseModel):
    id: str
    patent_id: str
    ipc_class: Optional[str] = None
    subclass: Optional[str] = None
    label: Optional[int] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    similarity: float


class GraphSimilarityResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# --- Helpers for Nullable Field Handling ---
def _safe_str(val: Any) -> Optional[str]:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    return None if s.lower() == "nan" or s == "" else s


def _safe_int(val: Any) -> Optional[int]:
    if pd.isna(val) or val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    if pd.isna(val) or val is None:
        return default
    try:
        return round(float(val), 4)
    except (ValueError, TypeError):
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Startup] Using device: {device}")

    # 2. Verify artifact files exist
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"FAISS index file not found at: {INDEX_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found at: {METADATA_PATH}")

    # 3. Load FAISS index
    print(f"[Startup] Loading FAISS index from {INDEX_PATH.name}...")
    index = faiss.read_index(str(INDEX_PATH))
    if index.d != EXPECTED_DIM:
        raise ValueError(
            f"FAISS index dimension mismatch: expected {EXPECTED_DIM}, got {index.d}"
        )
    print(f"[Startup] FAISS index loaded: {index.ntotal:,} vectors (dim={index.d})")

    # 4. Load Metadata Parquet (only required columns)
    print(f"[Startup] Loading patent metadata from {METADATA_PATH.name}...")
    metadata = pd.read_parquet(METADATA_PATH, columns=REQUIRED_METADATA_COLUMNS)

    # 5. Validate metadata consistency
    if len(metadata) != index.ntotal:
        raise ValueError(
            f"Metadata row count ({len(metadata):,}) does not match FAISS index total ({index.ntotal:,})"
        )

    missing_cols = set(REQUIRED_METADATA_COLUMNS) - set(metadata.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_cols)}"
        )
    print(f"[Startup] Metadata loaded & validated: {len(metadata):,} rows")

    # 6. Build ID to Index Mapping for Graph & Explorer Lookups
    print(f"[Startup] Building patent ID index mapping...")
    id_to_idx = {pid: i for i, pid in enumerate(metadata["patent_id"])}
    print(f"[Startup] Mapped {len(id_to_idx):,} patent IDs to FAISS indices")

    # 7. Load Model & Tokenizer
    print(f"[Startup] Loading model '{MODEL_NAME}' onto {device}...")
    embed_model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[Startup] Loading tokenizer '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 8. Store objects on app.state
    app.state.device = device
    app.state.embed_model = embed_model
    app.state.tokenizer = tokenizer
    app.state.index = index
    app.state.metadata = metadata
    app.state.id_to_idx = id_to_idx
    print("[Startup] All PatentBERT resources loaded successfully.")

    yield

    # Clean up during shutdown
    print("[Shutdown] Cleaning up PatentBERT resources...")
    del app.state.embed_model
    del app.state.tokenizer
    del app.state.index
    del app.state.metadata
    del app.state.id_to_idx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="PatentBERT API",
    description="Semantic patent search using BGE embeddings and FAISS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "PatentBERT API is running",
        "status": "healthy",
    }


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request_data: SearchRequest, request: Request):
    try:
        embed_model = request.app.state.embed_model
        tokenizer = request.app.state.tokenizer
        index = request.app.state.index
        metadata = request.app.state.metadata
    except AttributeError as exc:
        logger.exception("PatentBERT app state resources are missing: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Search service is not ready. Application resources are not initialized.",
        )

    try:
        results_df = patent_search(
            query=request_data.query,
            embed_model=embed_model,
            index=index,
            metadata=metadata,
            tokenizer=tokenizer,
            candidate_k=request_data.candidate_k,
            top_k=request_data.top_k,
            alpha=0.5,
        )

        if results_df is None or results_df.empty:
            return SearchResponse(
                query=request_data.query,
                candidate_k=request_data.candidate_k,
                top_k=request_data.top_k,
                total_results=0,
                results=[],
            )

        output_results = []
        for _, row in results_df.iterrows():
            output_results.append(
                PatentResult(
                    patent_id=str(row["patent_id"]),
                    similarity=_safe_float(row.get("similarity")),
                    chunk_similarity=_safe_float(row.get("chunk_similarity")),
                    hybrid_score=_safe_float(row.get("hybrid_score")),
                    ipc_class=_safe_str(row.get("ipc_class")),
                    subclass=_safe_str(row.get("subclass")),
                    label=_safe_int(row.get("label")),
                    matched_chunk=_safe_str(row.get("matched_chunk")) or "",
                )
            )

        return SearchResponse(
            query=request_data.query,
            candidate_k=request_data.candidate_k,
            top_k=request_data.top_k,
            total_results=len(output_results),
            results=output_results,
        )
    except Exception as exc:
        logger.exception("Unexpected error occurred during patent search: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the search request.",
        )


@app.post("/graph/similarity", response_model=GraphSimilarityResponse)
def graph_similarity_endpoint(request_data: GraphSimilarityRequest, request: Request):
    try:
        index = request.app.state.index
        metadata = request.app.state.metadata
        id_to_idx = request.app.state.id_to_idx
    except AttributeError as exc:
        logger.exception("PatentBERT app state resources are missing: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Graph service is not ready. Application resources are not initialized.",
        )

    # Validate that all requested patent IDs exist in the index
    missing_ids = [pid for pid in request_data.patent_ids if pid not in id_to_idx]
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"The following patent IDs were not found in the index: {missing_ids}",
        )

    try:
        patent_ids = request_data.patent_ids
        n = len(patent_ids)
        row_indices = [id_to_idx[pid] for pid in patent_ids]

        # Reconstruct vectors for the requested patents (N, 768)
        vectors = np.empty((n, index.d), dtype=np.float32)
        for i, row_idx in enumerate(row_indices):
            vectors[i] = index.reconstruct(int(row_idx))

        # Ensure unit normalization for safe cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vectors = vectors / norms

        # Pairwise cosine similarity matrix: (N, N)
        similarity_matrix = norm_vectors @ norm_vectors.T
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

        # Build nodes from metadata
        nodes = []
        for i, pid in enumerate(patent_ids):
            row = metadata.iloc[row_indices[i]]
            nodes.append(
                GraphNode(
                    id=pid,
                    patent_id=pid,
                    ipc_class=_safe_str(row.get("ipc_class")),
                    subclass=_safe_str(row.get("subclass")),
                    label=_safe_int(row.get("label")),
                )
            )

        # Build edges for unordered pairs (i < j) meeting threshold
        edges = []
        threshold = request_data.similarity_threshold
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarity_matrix[i, j])
                if sim >= threshold:
                    edges.append(
                        GraphEdge(
                            source=patent_ids[i],
                            target=patent_ids[j],
                            similarity=round(sim, 4),
                        )
                    )

        return GraphSimilarityResponse(nodes=nodes, edges=edges)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during graph similarity computation: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while computing patent graph relationships.",
        )



