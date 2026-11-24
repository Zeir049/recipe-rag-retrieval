# core/select.py
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def _contains(field_value: str, required_value: str) -> bool:
    if not required_value:
        return True
    fv = (field_value or "").strip().lower()
    rv = (required_value or "").strip().lower()
    if not fv or not rv:
        return False
    return rv in fv


def infer_auto_filters(query: str, auto_filters: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    q = (query or "")
    applied: List[Tuple[str, str]] = []
    for af in auto_filters or []:
        triggers = af.get("trigger_contains", []) or []
        field = str(af.get("field", "")).strip()
        value = str(af.get("value", "")).strip()
        if not field or not value:
            continue
        if any(t in q for t in triggers):
            applied.append((field, value))
    return applied


def attach_filter_ok(docs: List[Dict[str, Any]], required: List[Tuple[str, str]]) -> None:
    if not required:
        for d in docs:
            d["filter_ok"] = True
        return

    for d in docs:
        ok = True
        for field, value in required:
            ok = ok and _contains(str(d.get(field, "")), value)
        d["filter_ok"] = bool(ok)


def select_guarded(
    candidates: List[Dict[str, Any]],
    k: int,
    dedupe_field: str,
    diversify: bool,
    diversify_field: str,
    max_per_value: int,
    min_score: float,
    diversify_tag: bool = False,
    max_per_tag: int = 2,
    tag_field: str = "primary_tag",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    cnt_field = defaultdict(int)
    cnt_tag = defaultdict(int)

    def key(it: Dict[str, Any]) -> str:
        v = str(it.get(dedupe_field, "")).strip().lower()
        return v if v else f"__idx__{it.get('idx')}"

    def can_take(it: Dict[str, Any], enforce_filter: bool, enforce_div: bool, enforce_tag: bool) -> bool:
        if float(it.get("score", 0.0)) < float(min_score):
            return False
        if enforce_filter and not bool(it.get("filter_ok", True)):
            return False

        k_ = key(it)
        if k_ in seen:
            return False

        if enforce_div and diversify:
            v = str(it.get(diversify_field, "")).strip()
            if cnt_field[v] >= max_per_value:
                return False

        if enforce_tag and diversify_tag:
            t = str(it.get(tag_field, "")).strip()
            if t:
                if cnt_tag[t] >= max_per_tag:
                    return False

        return True

    def take(it: Dict[str, Any]):
        out.append(it)
        seen.add(key(it))

        v = str(it.get(diversify_field, "")).strip()
        cnt_field[v] += 1

        t = str(it.get(tag_field, "")).strip()
        if t:
            cnt_tag[t] += 1

    # 단계적으로 완화하면서 k개 채우기
    phases = [
        (True,  True,  True),   # filter + div + tag
        (True,  True,  False),  # filter + div
        (True,  False, True),   # filter + tag
        (True,  False, False),  # filter only
        (False, True,  True),   # div + tag
        (False, True,  False),  # div only
        (False, False, True),   # tag only
        (False, False, False)   # fallback
    ]

    for enforce_filter, enforce_div, enforce_tag in phases:
        if len(out) >= k:
            break
        for it in candidates:
            if len(out) >= k:
                break
            if can_take(it, enforce_filter, enforce_div, enforce_tag):
                take(it)

    return out[:k]
