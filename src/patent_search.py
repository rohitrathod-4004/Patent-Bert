import numpy as np
import pandas as pd


def chunk_patent(text, tokenizer, chunk_size=400, overlap=80):
    if not text or not isinstance(text, str):
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if start + chunk_size >= len(tokens):
            break

    return chunks


def search_patents(query, embed_model, index, metadata, top_k=10):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    # Filter negative FAISS indices (e.g. -1 for missing neighbors)
    valid_mask = indices[0] >= 0
    valid_indices = indices[0][valid_mask]
    valid_scores = scores[0][valid_mask]

    if len(valid_indices) == 0:
        return pd.DataFrame(
            columns=["patent_id", "similarity", "ipc_class", "subclass", "label", "text"]
        )

    results = metadata.iloc[valid_indices].copy()
    results["similarity"] = valid_scores

    return results[
        ["patent_id", "similarity", "ipc_class", "subclass", "label", "text"]
    ].reset_index(drop=True)


def rerank_candidates_by_chunks(
    query, candidates, embed_model, tokenizer, chunk_size=400, overlap=80
):
    if candidates.empty:
        return pd.DataFrame(columns=["patent_id", "chunk_similarity", "matched_chunk"])

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)[0]

    reranked = []

    for _, row in candidates.iterrows():
        chunks = chunk_patent(
            row["text"], tokenizer, chunk_size, overlap
        )

        if not chunks:
            reranked.append({
                "patent_id": row["patent_id"],
                "chunk_similarity": float(row["similarity"]) if "similarity" in row and pd.notna(row["similarity"]) else 0.0,
                "matched_chunk": ""
            })
            continue

        chunk_embeddings = embed_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        chunk_scores = chunk_embeddings @ query_embedding

        reranked.append({
            "patent_id": row["patent_id"],
            "chunk_similarity": float(chunk_scores.max()),
            "matched_chunk": chunks[int(np.argmax(chunk_scores))]
        })

    return pd.DataFrame(reranked)


def patent_search(
    query, embed_model, index, metadata, tokenizer,
    candidate_k=10, top_k=5, alpha=0.5
):
    candidates = search_patents(
        query, embed_model, index, metadata, candidate_k
    )

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "patent_id", "similarity", "ipc_class", "subclass",
                "label", "text", "chunk_similarity", "matched_chunk", "hybrid_score"
            ]
        )

    chunk_results = rerank_candidates_by_chunks(
        query, candidates, embed_model, tokenizer
    )

    results = candidates.merge(
        chunk_results, on="patent_id", how="left"
    )

    results["hybrid_score"] = (
        alpha * results["similarity"]
        + (1 - alpha) * results["chunk_similarity"]
    )

    return results.sort_values(
        "hybrid_score", ascending=False
    ).head(top_k).reset_index(drop=True)

