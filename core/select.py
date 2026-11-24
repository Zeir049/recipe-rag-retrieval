from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _to_str(x: Any, default: str = "-") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _normalize_tags(item: Dict[str, Any]) -> List[str]:
    """
    태그 후보:
    - tags / tag / 태그  (list or str)
    - 없으면 main_material_group를 태그처럼 사용
    """
    t = item.get("tags") or item.get("tag") or item.get("태그")

    if isinstance(t, list):
        tags = [_to_str(x) for x in t if _to_str(x) != "-"]
        return tags

    if isinstance(t, str):
        parts = [p.strip() for p in t.replace("/", ",").split(",")]
        tags = [p for p in parts if p]
        return tags

    fallback = item.get("main_material_group")
    if fallback is not None:
        return [_to_str(fallback)]

    return []


def _primary_tag(item: Dict[str, Any]) -> str:
    tags = _normalize_tags(item)
    return tags[0] if tags else "-"


def select(
    candidates: List[Dict[str, Any]],
    k: int = 7,
    diversify_method: bool = False,
    max_per_method: int = 999,
    diversify_tag: bool = False,
    max_per_tag: int = 999,
    guard_kind: str | None = None,
    guard_situation: str | None = None,
) -> List[Dict[str, Any]]:

        # 0) guardrail 필터: kind/situation이 지정되면 그 범위만 먼저 선택
    filtered = sorted_candidates
    if guard_kind:
        filtered = [it for it in filtered if _to_str(it.get("kind")) == guard_kind]
    if guard_situation:
        filtered = [it for it in filtered if _to_str(it.get("situation")) == guard_situation]

    # guardrail로 너무 줄어서 k를 못 채우면, 원본으로 fallback
    base = filtered if len(filtered) >= kk else sorted_candidates

    """
    후보(candidates)에서 최종 k개 선택.
    - 점수(score) 높은 순으로 우선 선택
    - diversify_method면 method 중복을 max_per_method 이하로 제한
    - diversify_tag면 (태그 or main_material_group) 중복을 max_per_tag 이하로 제한
    - 제한 때문에 k개가 안 채워지면, 남은 후보로 제한을 완화해 채움
    """
    if not candidates:
        return []

    kk = int(k)

    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    method_cnt = Counter()
    tag_cnt = Counter()

    picked: List[Dict[str, Any]] = []
    picked_ids = set()

    # 1차: 제한을 모두 지키면서 채우기
    for it in sorted_candidates:
        obj_id = id(it)
        if obj_id in picked_ids:
            continue

        m = _to_str(it.get("method"))
        t = _primary_tag(it)

        if diversify_method and method_cnt[m] >= int(max_per_method):
            continue
        if diversify_tag and tag_cnt[t] >= int(max_per_tag):
            continue

        picked.append(it)
        picked_ids.add(obj_id)

        if diversify_method:
            method_cnt[m] += 1
        if diversify_tag:
            tag_cnt[t] += 1

        if len(picked) >= kk:
            return picked

    # 2차: k가 안 채워지면 제한을 완화해서 채우기(중복 허용)
    if len(picked) < kk:
        for it in sorted_candidates:
            obj_id = id(it)
            if obj_id in picked_ids:
                continue
            picked.append(it)
            picked_ids.add(obj_id)
            if len(picked) >= kk:
                break

    return picked
