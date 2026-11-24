# core/rule_engine.py
import re
from typing import Any, Dict, List, Optional


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join([str(x) for x in v])
    if isinstance(v, dict):
        return " ".join([f"{k}:{v[k]}" for k in v])
    return str(v)


def resolve_field(doc: Dict[str, Any], field: str) -> Any:
    # extra.xxx 형태 지원
    if field.startswith("extra."):
        k = field.split(".", 1)[1]
        extra = doc.get("extra", {}) or {}
        return extra.get(k)
    return doc.get(field)


def eval_condition(doc: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    field = cond.get("field", "")
    v = resolve_field(doc, field)
    text = _to_text(v).lower().strip()

    if "contains_any" in cond:
        needles = [str(x).lower() for x in cond["contains_any"]]
        return any(n in text for n in needles)

    if "contains" in cond:
        n = str(cond["contains"]).lower()
        return n in text

    if "equals" in cond:
        n = str(cond["equals"]).lower().strip()
        return text == n

    if "regex" in cond:
        pat = str(cond["regex"])
        try:
            return re.search(pat, _to_text(v), flags=re.IGNORECASE) is not None
        except re.error:
            return False

    if "in" in cond:
        arr = [str(x).lower().strip() for x in cond["in"]]
        return text in arr

    return False


def eval_when(doc: Dict[str, Any], when: Dict[str, Any]) -> bool:
    if not when:
        return True

    if "any" in when:
        return any(_eval_node(doc, node) for node in when["any"])

    if "all" in when:
        return all(_eval_node(doc, node) for node in when["all"])

    # 단일 cond
    return eval_condition(doc, when)


def _eval_node(doc: Dict[str, Any], node: Dict[str, Any]) -> bool:
    # node가 group(any/all)인지 cond인지 구분
    if "any" in node or "all" in node:
        return eval_when(doc, node)
    return eval_condition(doc, node)


def pick_best_reason(doc: Dict[str, Any], reason_rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    matched = []
    for r in reason_rules:
        when = r.get("when", {})
        if eval_when(doc, when):
            matched.append(r)

    if not matched:
        return None

    matched.sort(key=lambda x: int(x.get("priority", 0)), reverse=True)
    return matched[0]


def infer_tags(doc: Dict[str, Any], tag_rules: List[Dict[str, Any]]) -> List[str]:
    tags = []
    for r in tag_rules:
        when = r.get("when", {})
        if eval_when(doc, when):
            name = str(r.get("name", "")).strip()
            if name:
                tags.append(name)
    return tags
