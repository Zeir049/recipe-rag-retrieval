# core/engine.py
from __future__ import annotations

import os
import sys
import json
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 선택: pandas가 있으면 meta 로딩 편함
try:
    import pandas as pd
except Exception:
    pd = None

from .types import RunConfig


# ----------------------------
# internal cache (lazy loading)
# ----------------------------
_CACHE: Dict[str, Any] = {
    "config": None,
    "meta": None,            # pd.DataFrame or list/dict
    "embeddings": None,      # np.ndarray
    "embeddings_norm": None  # np.ndarray (L2 normalized)
}

def _select_with_diversity(
    candidates: List[Dict[str, Any]],
    k: int,
    diversify_method: bool,
    max_per_method: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    if not diversify_method:
        return candidates[:k]

    picked: List[Dict[str, Any]] = []
    method_cnt: Dict[str, int] = {}

    # 1차: 제한 지키면서 담기
    for it in candidates:
        m = str(it.get("method") or "-")
        if method_cnt.get(m, 0) >= max_per_method:
            continue
        picked.append(it)
        method_cnt[m] = method_cnt.get(m, 0) + 1
        if len(picked) >= k:
            return picked

    # 2차: 그래도 k가 안 채워지면 남은 것 점수순으로 채우기(제한 완화)
    if len(picked) < k:
        already = {id(x) for x in picked}
        for it in candidates:
            if id(it) in already:
                continue
            picked.append(it)
            if len(picked) >= k:
                break

    return picked


# ----------------------------
# utils
# ----------------------------
def _safe_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as e:
        raise RuntimeError(f"필수 모듈 import 실패: {module_name} / {e}") from e


def _flex_call(fn, *args, **kwargs):
    """
    fn 시그니처에 맞는 kwargs만 추려서 호출.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def _get_cfg() -> Dict[str, Any]:
    """
    core.io.load_config() 사용 시도. 실패하면 {}.
    """
    if _CACHE["config"] is not None:
        return _CACHE["config"]

    cfg = {}
    try:
        from .io import load_config  # 네 프로젝트에 존재(이미 확인됨)
        # load_config가 인자 없이 되거나, 경로 필요할 수도 있으니 유연하게
        try:
            sig = inspect.signature(load_config)
            if len(sig.parameters) == 0:
                cfg = load_config() or {}
            else:
                # 흔한 파일명 후보만 시도
                for p in ("config.json", "config.yaml", "config.yml", "settings.json", "rag_config.json"):
                    if os.path.exists(p):
                        try:
                            cfg = load_config(p) or {}
                            break
                        except TypeError:
                            # 키워드 인자 변형
                            for kw in ("path", "config_path", "filepath", "file"):
                                try:
                                    cfg = load_config(**{kw: p}) or {}
                                    break
                                except TypeError:
                                    continue
        except Exception:
            cfg = {}
    except Exception:
        cfg = {}

    if not isinstance(cfg, dict):
        cfg = {}

    _CACHE["config"] = cfg
    return cfg


def _pick_path(cfg: Dict[str, Any], keys: List[str], fallbacks: List[str]) -> Optional[str]:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str) and v.strip() and os.path.exists(v):
            return v
    for p in fallbacks:
        if os.path.exists(p):
            return p
    return None


def _load_meta(cfg: Dict[str, Any]):
    """
    meta csv 로딩. (pandas 있으면 DataFrame)
    """
    if _CACHE["meta"] is not None:
        return _CACHE["meta"]

    meta_path = _pick_path(
        cfg,
        keys=["meta_path", "meta_csv", "meta_file", "recipe_meta", "menu_meta"],
        fallbacks=["recipe_meta.csv", "menu_meta.csv", "meta.csv"]
    )
    if meta_path is None:
        raise RuntimeError("meta csv 파일을 찾지 못했습니다. (recipe_meta.csv 등)")

    if pd is None:
        # pandas 없으면 최소 로딩(헤더/행 dict 리스트)
        import csv
        with open(meta_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            meta = list(reader)
    else:
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")

    _CACHE["meta"] = meta
    return meta


def load_embeddings_norm(cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    임베딩 로딩 + L2 정규화 캐시.
    """
    if _CACHE["embeddings_norm"] is not None:
        return _CACHE["embeddings_norm"]

    cfg = cfg or _get_cfg()
    emb_path = _pick_path(
        cfg,
        keys=["embeddings_path", "embed_path", "vector_path", "recipe_embeddings", "menu_embeddings"],
        fallbacks=["recipe_embeddings_e5_large.npy", "menu_embeddings.npy", "embeddings.npy"]
    )
    if emb_path is None:
        raise RuntimeError("임베딩 npy 파일을 찾지 못했습니다. (recipe_embeddings_e5_large.npy 등)")

    emb = np.load(emb_path)
    emb = emb.astype(np.float32, copy=False)

    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_norm = emb / norms

    _CACHE["embeddings"] = emb
    _CACHE["embeddings_norm"] = emb_norm
    return emb_norm


def _embed_query(query: str, cfg: Dict[str, Any]) -> np.ndarray:
    """
    쿼리 임베딩: 먼저 core.retrieval 쪽 helper가 있으면 그걸 사용.
    없으면 sentence-transformers로 시도(설치/모델 필요).
    """
    retrieval = _safe_import("core.retrieval")

    # 1) retrieval 모듈에 query embedding 함수가 있으면 사용
    for fn_name in ("embed_query", "encode_query", "encode", "embed", "get_query_embedding"):
        if hasattr(retrieval, fn_name) and callable(getattr(retrieval, fn_name)):
            fn = getattr(retrieval, fn_name)
            try:
                out = _flex_call(fn, query=query, q=query, text=query, cfg=cfg)
                vec = np.asarray(out, dtype=np.float32)
                return vec
            except Exception:
                pass

    # 2) sentence_transformers fallback
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "쿼리 임베딩 함수를 찾지 못했고, sentence_transformers도 없습니다.\n"
            "core/retrieval.py에 embed_query(또는 encode) 같은 함수를 제공하거나, sentence_transformers 설치/모델 설정이 필요합니다."
        ) from e

    model_name = cfg.get("embed_model") or cfg.get("model_name") or "intfloat/multilingual-e5-large"
    model = SentenceTransformer(model_name)
    vec = model.encode([query], normalize_embeddings=False)[0]
    return np.asarray(vec, dtype=np.float32)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return vec
    return vec / n


# ----------------------------
# core pipeline pieces (public)
# ----------------------------
def build_candidates(
    query: str,
    k: int = 7,
    min_score: float = 0.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    1) query embedding
    2) cosine top-k
    3) candidate dict list 반환 (idx/score 포함)
    """
    cfg = cfg or _get_cfg()
    emb_norm = load_embeddings_norm(cfg)
    meta = _load_meta(cfg)

    qv = _embed_query(query, cfg)
    qv = _normalize_vec(qv)

    # cosine: (N, D) dot (D,)
    sims = emb_norm @ qv
    # top-k 후보는 selection/diversify에서 더 필요할 수 있어 약간 여유로 뽑는 걸 권장
    probe_k = max(k * 5, k)
    idxs = np.argpartition(-sims, min(probe_k, len(sims) - 1))[:probe_k]
    idxs = idxs[np.argsort(-sims[idxs])]

    candidates: List[Dict[str, Any]] = []
    for idx in idxs:
        score = float(sims[idx])
        if score < float(min_score):
            continue

        item: Dict[str, Any] = {"idx": int(idx), "score": score}

        # meta 붙이기(가능하면)
        try:
            if pd is not None and hasattr(meta, "iloc"):
                row = meta.iloc[int(idx)].to_dict()
                item.update(row)
            elif isinstance(meta, list) and int(idx) < len(meta):
                item.update(meta[int(idx)])
        except Exception:
            pass

        candidates.append(item)

        if len(candidates) >= probe_k:
            break

    return candidates


def make_rule_answer(item: Dict[str, Any], query: str, cfg: Dict[str, Any]) -> str:
    """
    rule_engine 기반 ‘추천 이유’ 생성.
    (rule_engine의 함수명이 달라도 최대한 찾아서 호출)
    """
    rule_engine = _safe_import("core.rule_engine")

    # 흔한 함수명 후보
    for fn_name in ("make_rule_answer", "build_rule_answer", "rule_answer", "generate_reason", "make_reason"):
        if hasattr(rule_engine, fn_name) and callable(getattr(rule_engine, fn_name)):
            fn = getattr(rule_engine, fn_name)
            try:
                return str(_flex_call(fn, item=item, query=query, cfg=cfg))
            except Exception:
                pass

    # fallback: 최소 텍스트
    name = item.get("name") or item.get("menu") or item.get("title") or "메뉴"
    return f"{name}는 요청과 잘 맞는 후보예요."


def run_engine(
    query: str,
    k: int = 7,
    min_score: float = 0.0,
    generate_rule: bool = False,
    only_answer: bool = False,
    diversify_method: bool = False,
    max_per_method: int = 999,
    diversify_tag: bool = False,
    max_per_tag: int = 999,
) -> Dict[str, Any]:
    """
    기존 코어 파이프라인 진입점(너 파일에 원래 있던 이름).
    - candidates 생성
    - select로 최종 k개 선택/다양화
    - enrich로 출력용 필드 보강
    - (옵션) rule_engine로 추천 이유 생성
    - dict 반환
    """
    cfg = _get_cfg()
    meta = _load_meta(cfg)

    guard_kind = None
    guard_situation = None
    try:
        rule_engine = _safe_import("core.rule_engine")
        for fn_name in ("extract_guardrail", "infer_guardrail", "make_guardrail", "get_guardrail"):
            if hasattr(rule_engine, fn_name) and callable(getattr(rule_engine, fn_name)):
                g = _flex_call(getattr(rule_engine, fn_name), query=query, cfg=cfg)
                if isinstance(g, dict):
                    guard_kind = g.get("kind")
                    guard_situation = g.get("situation")
                break
    except Exception:
        pass


    candidates = build_candidates(query=query, k=k, min_score=min_score, cfg=cfg)

    # --- select 단계 ---
    select_mod = _safe_import("core.select")
    selected: List[Dict[str, Any]] = candidates
    #
    # select 단계가 실패/미적용이면 fallback 적용
# (selected가 그냥 candidates 그대로인 경우가 많아서, diversify_method 켰을 때만 fallback)
    # select 모듈에 선택 함수가 있으면 사용(없으면 candidates에서 앞 k개)
    for fn_name in ("select", "select_items", "pick", "choose", "run_select"):
        if hasattr(select_mod, fn_name) and callable(getattr(select_mod, fn_name)):
            fn = getattr(select_mod, fn_name)
            try:
                selected = _flex_call(
                    fn,
                    candidates=candidates,
                    items=candidates,
                    k=k,
                    diversify_method=diversify_method,
                    max_per_method=max_per_method,
                    diversify_tag=diversify_tag,
                    max_per_tag=max_per_tag,
                    guard_kind=guard_kind,
                    guard_situation=guard_situation,

                )
                break
            except Exception:
                continue

    if not isinstance(selected, list):
        selected = candidates
# select 모듈이 다양화를 안 해주거나 실패했을 때를 대비한 fallback
    if diversify_method:
        selected = _select_with_diversity(
            candidates=candidates,
            k=int(k),
            diversify_method=bool(diversify_method),
            max_per_method=int(max_per_method),
    )
    else:
        selected = selected[: int(k)]


    # --- enrich 단계 ---
    enrich_mod = _safe_import("core.enrich")
    for fn_name in ("enrich", "enrich_items", "apply", "run_enrich"):
        if hasattr(enrich_mod, fn_name) and callable(getattr(enrich_mod, fn_name)):
            fn = getattr(enrich_mod, fn_name)
            try:
                enriched = _flex_call(fn, items=selected, selected=selected, meta=meta, cfg=cfg, query=query)
                if isinstance(enriched, list):
                    selected = enriched
                break
            except Exception:
                continue

    # --- rule answer ---
    if generate_rule:
        for it in selected:
            # 이미 reason이 있으면 유지
            if "추천 이유" in it and it["추천 이유"]:
                continue
            if "reason" in it and it["reason"]:
                continue
            reason = make_rule_answer(it, query=query, cfg=cfg)
            it["reason"] = reason

    # --- guardrail(가능하면 rule_engine에서) ---
    guardrail = None
    try:
        rule_engine = _safe_import("core.rule_engine")
        for fn_name in ("extract_guardrail", "infer_guardrail", "make_guardrail", "get_guardrail"):
            if hasattr(rule_engine, fn_name) and callable(getattr(rule_engine, fn_name)):
                guardrail = _flex_call(getattr(rule_engine, fn_name), query=query, cfg=cfg)
                break
    except Exception:
        guardrail = None

    debug = {
        "returned": len(selected),
    }

    return {
        "query": query,
        "selected": selected,
        "guardrail": guardrail,
        "debug": debug,
    }


# ============================================================
# Unified entrypoint for CLI/batch
# ============================================================
def run(query: str, cfg: RunConfig):
    """
    UnifiedEngine이 호출할 표준 진입점.
    내부적으로 기존 run_engine을 그대로 호출한다.
    """
    return run_engine(
        query,
        k=cfg.k,
        min_score=cfg.min_score,
        generate_rule=cfg.generate_rule,
        only_answer=cfg.only_answer,
        diversify_method=cfg.diversify_method,
        max_per_method=cfg.max_per_method,
        diversify_tag=cfg.diversify_tag,
        max_per_tag=cfg.max_per_tag,
    )
