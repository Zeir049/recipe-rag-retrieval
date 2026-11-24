from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import RunConfig, RunResult


def _filters_line(guardrail: Optional[Dict[str, Any]]) -> Optional[str]:
    if not guardrail or not guardrail.get("applied_filters"):
        return None
    parts = [f"{field}='{value}'" for field, value in guardrail["applied_filters"]]
    return "선정 기준: (자동 필터) " + ", ".join(parts)


def render(result: RunResult, cfg: Optional[RunConfig] = None, show_tags: bool = True) -> str:
    """Human-readable answer: one block per recommended recipe, ending with its evidence ID."""
    lines: List[str] = [f"질문: {result.query}", ""]

    header = _filters_line(result.guardrail)
    if header:
        lines += [header, ""]

    lines += ["추천 메뉴:", ""]
    for i, item in enumerate(result.selected, 1):
        key_ing = ", ".join(item.get("key_ingredients_list") or []) or "재료 정보 없음"
        lines.append(f"{i}) {item.get('title') or '-'}")
        lines.append(f"   - 종류: {item.get('kind') or '-'} / 조리: {item.get('method') or '-'} / 상황: {item.get('situation') or '-'}")
        if show_tags and item.get("tags"):
            lines.append(f"   - 태그: {', '.join(item['tags'])}")
        lines.append(f"   - 키 재료: {key_ing}")
        if item.get("reason"):
            lines.append(f"   - 추천 이유: {item['reason']}")
        lines.append(f"   - 근거 ID: {item.get('doc_id') or '-'}")
        lines.append("")

    return "\n".join(lines).rstrip()


def render_passages(result: RunResult) -> str:
    """The evidence-tagged passages a generator would receive as context."""
    return "\n\n".join(item.get("passage", "") for item in result.selected)


# kept for callers that use the old name
def render_result(result: RunResult, cfg: Optional[RunConfig] = None) -> str:
    return render(result, cfg=cfg)
