from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None

from .types import RunResult, RunConfig


def _is_na(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd is not None and pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _get_name(item: Dict[str, Any]) -> str:
    return (
        item.get("name")
        or item.get("CKG_NM")
        or item.get("menu")
        or item.get("title")
        or "UNKNOWN"
    )


def _get_id(item: Dict[str, Any]) -> Any:
    return item.get("RCP_SNO") or item.get("id") or item.get("recipe_id") or item.get("idx")


def _get_method(item: Dict[str, Any]) -> str:
    return (
        item.get("method")
        or item.get("cook_method")
        or item.get("조리")
        or item.get("조리방법")
        or "-"
    )


def _get_kind(item: Dict[str, Any]) -> str:
    return item.get("kind") or item.get("종류") or "-"


def _get_situation(item: Dict[str, Any]) -> str:
    return item.get("situation") or item.get("상황") or "-"


def _get_tags(item: Dict[str, Any]) -> List[str]:
    # 너 데이터에는 tag가 없을 수도 있으니 최대한 유연하게
    t = item.get("tags") or item.get("tag") or item.get("태그")
    if _is_na(t):
        return []
    if isinstance(t, list):
        return [str(x).strip() for x in t if not _is_na(x)]
    if isinstance(t, str):
        # "단백질, 채소위주" 형태면 분리
        parts = [p.strip() for p in t.replace("/", ",").split(",")]
        return [p for p in parts if p]
    return [str(t).strip()]


def _get_ingredients(item: Dict[str, Any]) -> str:
    # ingredients_keywords가 string이면 그대로
    ing = item.get("ingredients_keywords") or item.get("키 재료") or item.get("ingredients")
    if _is_na(ing):
        return "-"
    return str(ing).strip()


def _get_reason(item: Dict[str, Any]) -> str:
    return (
        item.get("reason")
        or item.get("추천 이유")
        or item.get("why")
        or "-"
    )


def render(result: RunResult, cfg: Optional[RunConfig] = None) -> str:
    cfg = cfg or RunConfig()
    lines: List[str] = []

    lines.append(f"질문: {result.query}")
    lines.append("")

    # guardrail이 있으면 우선, 없으면 1번 아이템에서 추정
    guard = result.guardrail
    if not guard and result.selected:
        first = result.selected[0]
        kind = _get_kind(first)
        sit = _get_situation(first)
        if kind != "-" or sit != "-":
            guard = {"kind": kind, "situation": sit}

    if guard:
        # 예전 출력 스타일 비슷하게
        kind = guard.get("kind") if isinstance(guard, dict) else None
        sit = guard.get("situation") if isinstance(guard, dict) else None
        if kind and sit:
            lines.append(f"선정 기준: (자동 가드레일) kind='{kind}', situation='{sit}'")
        else:
            lines.append(f"선정 기준: {guard}")
        lines.append("")

    lines.append("추천 메뉴:")
    lines.append("")

    for i, item in enumerate(result.selected, 1):
        name = _get_name(item)
        kind = _get_kind(item)
        method = _get_method(item)
        sit = _get_situation(item)
        tags = _get_tags(item)
        ing = _get_ingredients(item)
        reason = _get_reason(item)
        rid = _get_id(item)

        lines.append(f"{i}) {name}")
        lines.append(f"   - 종류: {kind} / 조리: {method} / 상황: {sit}")
        if tags:
            lines.append(f"   - 태그: {', '.join(tags)}")
        lines.append(f"   - 키 재료: {ing}")
        lines.append(f"   - 추천 이유: {reason}")
        lines.append(f"   - 근거 ID: {rid}")
        lines.append("")

    return "\n".join(lines).rstrip()


# 혹시 local_menu_rag.py가 render_result를 부르면 대응
def render_result(result: RunResult, cfg: Optional[RunConfig] = None) -> str:
    return render(result, cfg=cfg)
