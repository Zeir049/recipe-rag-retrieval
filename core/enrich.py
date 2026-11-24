# core/enrich.py
import re
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


_STOP_PHRASES = {
    "약간", "적당량", "조금", "조금씩", "기호에따라", "기호에 따라",
    "숟가락", "숟갈", "스푼", "큰", "작은", "중간",
    "대", "소", "개", "쪽", "줌", "컵", "ml", "l", "g", "kg",
    "것", "그램", "그람", "지",
    "통", "않은",
}


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def compact_ingredients(raw: Any, max_items: int = 10) -> str:
    s = safe_str(raw)
    if not s:
        return ""

    parts = [p.strip() for p in s.split(",") if p.strip()]
    cleaned: List[str] = []

    for p in parts:
        p2 = re.sub(r"\s+", " ", p)
        p2 = re.sub(r"[0-9]+(\.[0-9]+)?", "", p2).strip()

        tokens = [t for t in p2.split(" ") if t and t not in _STOP_PHRASES]
        if not tokens:
            continue
        cleaned.append(" ".join(tokens))

    uniq = []
    seen = set()
    for item in cleaned:
        if item not in seen:
            uniq.append(item)
            seen.add(item)

    return ", ".join(uniq[:max_items])


def build_menu_to_ingredients_and_vocab(
    meta_df: pd.DataFrame,
    title_col: str,
    ingredients_col: str
) -> Tuple[Dict[str, str], Set[str]]:
    menu_to_ing: Dict[str, str] = {}
    vocab: Set[str] = set()

    for _, r in meta_df.iterrows():
        title = safe_str(r.get(title_col)).strip().lower()
        ing_comp = compact_ingredients(r.get(ingredients_col), max_items=50)

        if title and ing_comp and title not in menu_to_ing:
            menu_to_ing[title] = ing_comp

        if ing_comp:
            for tok in [t.strip() for t in ing_comp.split(",") if t.strip()]:
                if len(tok) >= 2:
                    vocab.add(tok)

    return menu_to_ing, vocab


def guess_ingredients_from_title(title: str, vocab: Set[str], max_n: int = 5) -> str:
    t = safe_str(title)
    if not t:
        return ""

    found = []
    for tok in sorted(vocab, key=len, reverse=True):
        if len(found) >= max_n:
            break
        if not tok or len(tok) < 2:
            continue
        if tok in t and tok not in found:
            found.append(tok)

    return ", ".join(found[:max_n])


def top_n_ingredients_list(ingredients_compact: str, n: int) -> List[str]:
    s = safe_str(ingredients_compact)
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts[:n]


def enrich_doc(
    doc: Dict[str, Any],
    menu_to_ing: Dict[str, str],
    vocab: Set[str],
    max_ing_in_passage: int,
    key_ing_topn: int
) -> Dict[str, Any]:
    title = safe_str(doc.get("title"))
    raw_ing = doc.get("ingredients")

    ing_compact = compact_ingredients(raw_ing, max_items=max_ing_in_passage)

    if not ing_compact:
        ing_compact = menu_to_ing.get(title.strip().lower(), "")

    if not ing_compact:
        ing_compact = guess_ingredients_from_title(title, vocab, max_n=min(5, max_ing_in_passage))

    doc["ingredients_compact"] = ing_compact
    doc["key_ingredients_list"] = top_n_ingredients_list(ing_compact, key_ing_topn)
    doc["key_ingredients"] = ", ".join(doc["key_ingredients_list"])  # 규칙 평가용
    return doc
