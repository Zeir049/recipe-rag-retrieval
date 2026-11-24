# core/engine.py
"""
Retrieval pipeline used by local_menu_rag.py and batch_eval.py.

    query
      -> "query: " + query, multilingual-e5 embedding          (core.retrieval)
      -> cosine top-(k * pool_multiplier) over the recipe matrix
      -> clean ingredients, key ingredients                     (core.enrich)
      -> rule-based tags                                        (core.rule_engine)
      -> query-triggered filters + dedupe + method diversity    (core.select)
      -> rule-based reason + evidence-tagged passage per result

Everything that is not code (paths, field names, filters, rules, passage
template) comes from configs/menu_rag.json.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .enrich import build_menu_to_ingredients_and_vocab, enrich_doc, safe_str
from .io import load_config
from .retrieval import embed_query_sentence_transformers, l2_normalize, topk_cosine
from .rule_engine import infer_tags, pick_best_reason
from .select import attach_filter_ok, infer_auto_filters, select_guarded
from .types import RunConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "menu_rag.json")
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
FALLBACK_REASON = "질문과 의미적으로 가까운 후보로 검색되어 추천해요."

# Loaded once per process and reused for every query.
_CACHE: Dict[str, Any] = {}


# ----------------------------
# loading
# ----------------------------
def _resolve(path: str) -> str:
    """Paths in the config are relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG
    key = ("config", path)
    if key not in _CACHE:
        _CACHE[key] = load_config(path)
    return _CACHE[key]


def _load_data(cfg: Dict[str, Any]):
    data = cfg.get("data", {})
    key = ("data", data.get("emb_path"), data.get("meta_path"))
    if key in _CACHE:
        return _CACHE[key]

    emb = np.load(_resolve(data.get("emb_path", "recipe_embeddings_e5_large.npy"))).astype(np.float32)
    emb_norm = l2_normalize(emb, axis=1)
    meta = pd.read_csv(_resolve(data.get("meta_path", "recipe_meta.csv")),
                       encoding=data.get("meta_encoding", "utf-8-sig"))
    if len(meta) != emb.shape[0]:
        raise RuntimeError(f"meta rows ({len(meta)}) != embedding rows ({emb.shape[0]})")

    f = cfg.get("fields", {})
    # used by enrich to fill in ingredients for recipes that have none
    menu_to_ing, vocab = build_menu_to_ingredients_and_vocab(
        meta, title_col=f.get("title", "CKG_NM"), ingredients_col=f.get("ingredients", "ingredients_keywords")
    )
    _CACHE[key] = (emb_norm, meta, menu_to_ing, vocab)
    return _CACHE[key]


# ----------------------------
# pipeline steps
# ----------------------------
def build_candidates(query: str, pool_k: int, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Embed the query and return the pool_k most similar recipes as docs."""
    emb_norm, meta, menu_to_ing, vocab = _load_data(cfg)
    f = cfg.get("fields", {})
    out_cfg = cfg.get("output", {})
    model_name = cfg.get("data", {}).get("embed_model", DEFAULT_MODEL)

    qv = embed_query_sentence_transformers(query, model_name, e5_prefix=True)
    idxs, scores = topk_cosine(emb_norm, qv, k=pool_k)

    docs = []
    for idx, score in zip(idxs, scores):
        row = meta.iloc[int(idx)]
        doc = {
            "idx": int(idx),
            "score": float(score),
            "title": safe_str(row.get(f.get("title", "CKG_NM"))),
            "kind": safe_str(row.get(f.get("kind", "kind"))),
            "method": safe_str(row.get(f.get("method", "method"))),
            "situation": safe_str(row.get(f.get("situation", "situation"))),
            "doc_id": safe_str(row.get(f.get("doc_id", "RCP_SNO"))),
            "ingredients": row.get(f.get("ingredients", "ingredients_keywords")),
        }
        enrich_doc(
            doc, menu_to_ing, vocab,
            max_ing_in_passage=int(cfg.get("passage", {}).get("key_ingredients_max", 10)),
            key_ing_topn=int(out_cfg.get("key_ingredients_topn", 5)),
        )
        tags = infer_tags(doc, cfg.get("rules", {}).get("tags", []))
        doc["tags"] = tags
        doc["primary_tag"] = tags[0] if tags else ""
        docs.append(doc)
    return docs


def make_reason(doc: Dict[str, Any], selected: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    """Highest-priority matching reason rule, plus a note when the cooking method is unique."""
    rule = pick_best_reason(doc, cfg.get("rules", {}).get("reasons", []))
    text = str(rule["text"]) if rule else FALLBACK_REASON

    if cfg.get("output", {}).get("include_method_choice_line", False):
        method = doc.get("method", "")
        if method and sum(1 for d in selected if d.get("method") == method) == 1:
            text += f" 조리 방법이 **{method}**이라 다른 후보와 스타일이 달라 선택폭이 넓어져요."
    return text


def make_passage(doc: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Evidence-tagged passage for a generator: ends with the recipe ID it came from."""
    template = cfg.get("passage", {}).get(
        "template",
        "[MENU] {title}\n- 종류={kind} | 조리={method} | 상황={situation}\n"
        "- 핵심 재료: {ingredients_compact}\n- 근거 ID: {doc_id}\n",
    )
    fields = {k: (doc.get(k) or "-") for k in ("title", "kind", "method", "situation", "ingredients_compact", "doc_id")}
    return template.format(**fields).rstrip()


def run_engine(query: str, rc: RunConfig, config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = get_config(config_path)
    sel_cfg = cfg.get("selection", {})
    div_cfg = cfg.get("diversity", {})

    k = int(rc.k)
    pool_k = k * int(sel_cfg.get("pool_multiplier", 10))
    candidates = build_candidates(query, pool_k, cfg)

    # query-triggered filters, e.g. "샐러드" in the query -> kind contains "샐러드"
    applied = infer_auto_filters(query, cfg.get("auto_filters", []))
    attach_filter_ok(candidates, applied)

    selected = select_guarded(
        candidates,
        k=k,
        dedupe_field=sel_cfg.get("dedupe_field", "title"),
        diversify=bool(rc.diversify_method),
        diversify_field=div_cfg.get("field", "method"),
        max_per_value=int(rc.max_per_method),
        min_score=float(rc.min_score),
        diversify_tag=bool(rc.diversify_tag),
        max_per_tag=int(rc.max_per_tag),
        tag_field="primary_tag",
    )

    for doc in selected:
        if rc.generate_rule:
            doc["reason"] = make_reason(doc, selected, cfg)
        doc["passage"] = make_passage(doc, cfg)

    return {
        "query": query,
        "selected": selected,
        "guardrail": {"applied_filters": [list(x) for x in applied]} if applied else None,
        "debug": {
            "returned": len(selected),
            "pool": len(candidates),
            "filter_ok_in_pool": sum(1 for d in candidates if d.get("filter_ok")),
        },
    }


# ============================================================
# Entry point used by core.unified_engine.UnifiedEngine
# ============================================================
def run(query: str, cfg: RunConfig):
    return run_engine(query, cfg, config_path=(cfg.extra or {}).get("config_path"))
