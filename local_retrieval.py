# local_retrieval.py
import argparse
import numpy as np
import pandas as pd

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def load_meta(meta_path: str) -> pd.DataFrame:
    # 윈도우 CSV BOM/인코딩 이슈 방지
    return pd.read_csv(meta_path, encoding="utf-8-sig")

def embed_query_sentence_transformers(query: str, model_name: str) -> np.ndarray:
    """
    ✅ 중요: 문서 임베딩을 만든 모델과 동일한 모델/전처리를 써야 함
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # e5 계열이면 일반적으로 "query: " 프리픽스 사용(문서쪽은 "passage: ")
    q = "query: " + query
    vec = model.encode([q], normalize_embeddings=False)
    return vec.astype(np.float32)[0]

def topk_cosine(
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int = 5
):
    # 코사인 유사도 = 정규화 후 내적
    doc_n = l2_normalize(doc_emb.astype(np.float32), axis=1)
    q_n = l2_normalize(query_emb.astype(np.float32), axis=0)

    scores = doc_n @ q_n  # (N,)
    if k >= scores.shape[0]:
        idx = np.argsort(-scores)
    else:
        # 빠른 top-k
        idx_part = np.argpartition(-scores, kth=k-1)[:k]
        idx = idx_part[np.argsort(-scores[idx_part])]
    return idx, scores[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="recipe_embeddings_e5_large.npy")
    ap.add_argument("--meta", default="recipe_meta.csv")
    ap.add_argument("--q", required=True, help="검색 질의문")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")  # 네 파일명이 e5_large라서 기본값
    ap.add_argument("--show_cols", default="", help="출력할 컬럼(콤마구분). 비우면 가능한 컬럼 자동 추정")
    args = ap.parse_args()

    doc_emb = np.load(args.emb)  # (N, D)
    meta = load_meta(args.meta)

    # (안전) 메타 행 수 일치 체크
    n = doc_emb.shape[0]
    if len(meta) != n:
        print(f"[WARN] meta rows({len(meta)}) != emb rows({n}). meta를 emb 개수에 맞춰 앞부분만 사용합니다.")
        meta = meta.iloc[:n].copy()

    # 쿼리 임베딩 생성 (sentence-transformers 사용)
    q_emb = embed_query_sentence_transformers(args.q, args.model)

    idx, scores = topk_cosine(doc_emb, q_emb, k=args.k)

    # 출력 컬럼 선택
    cols = [c.strip() for c in args.show_cols.split(",") if c.strip()]
    if not cols:
        # 흔한 컬럼 후보 자동 선택
        candidates = ["title", "name", "recipe_name", "food_name", "summary", "ingredients", "url", "link", "id"]
        cols = [c for c in candidates if c in meta.columns]
        if not cols:
            cols = meta.columns.tolist()[:6]  # 아무거나 앞 6개

    print("\n=== TOP-K RESULTS ===")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        row = meta.iloc[int(i)]
        print(f"\n[{rank}] idx={int(i)}  score={float(s):.4f}")
        for c in cols:
            val = row.get(c, "")
            # 너무 길면 자르기
            text = str(val)
            if len(text) > 200:
                text = text[:200] + "..."
            print(f" - {c}: {text}")

if __name__ == "__main__":
    main()
