# core/retrieval.py
from typing import Tuple
import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def embed_query_sentence_transformers(query: str, model_name: str, e5_prefix: bool = True) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    q = ("query: " + query) if e5_prefix else query
    vec = model.encode([q], normalize_embeddings=False)
    return vec.astype(np.float32)[0]


def topk_cosine(doc_emb_norm: np.ndarray, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_n = l2_normalize(query_emb.astype(np.float32), axis=0)
    scores = doc_emb_norm @ q_n  # (N,)

    n = scores.shape[0]
    if k >= n:
        idx = np.argsort(-scores)
    else:
        idx_part = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx_part[np.argsort(-scores[idx_part])]
    return idx, scores[idx]
