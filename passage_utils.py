# passage_utils.py
import re
import pandas as pd

# 너무 흔한 단위/수량 표현(완벽하진 않지만 컨텍스트를 짧게 만드는 용도)
_STOP_PHRASES = {
    "약간", "적당량", "조금", "조금씩", "기호에따라", "기호에 따라",
    "숟가락", "숟갈", "스푼", "큰 것", "작은 것", "중간것", "중간 것",
    "대", "소", "개", "쪽", "줌", "컵", "ml", "l", "g", "kg",
}

def _safe_str(x) -> str:
    if x is None:
        return ""
    # pandas NaN 처리
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def _compact_ingredients(ingredients_keywords: str, max_items: int = 10) -> str:
    s = _safe_str(ingredients_keywords)
    if not s:
        return ""

    # 쉼표 기준 분리
    parts = [p.strip() for p in s.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        # 괄호/슬래시 같은 잡표현 살짝 정리
        p2 = re.sub(r"\s+", " ", p)
        # 너무 흔한 단위/수량표현만 제거(완전히 제거하는게 아니라 토큰 단위로 필터)
        tokens = [t for t in p2.split(" ") if t and t not in _STOP_PHRASES]
        if not tokens:
            continue
        cleaned.append(" ".join(tokens))

    # 중복 제거(순서 유지)
    uniq = []
    seen = set()
    for item in cleaned:
        if item not in seen:
            uniq.append(item)
            seen.add(item)

    return ", ".join(uniq[:max_items])

def build_menu_passage(row: "pd.Series", max_ing: int = 10) -> str:
    """
    ✅ 메뉴 추천 목적의 passage
    - '레시피 상세'가 아니라 '메뉴 선택'에 필요한 축(종류/방법/상황/재료) 중심
    """
    menu = _safe_str(row.get("CKG_NM"))
    kind = _safe_str(row.get("kind"))
    method = _safe_str(row.get("method"))
    situation = _safe_str(row.get("situation"))
    rcp_sno = _safe_str(row.get("RCP_SNO"))

    ing = _compact_ingredients(row.get("ingredients_keywords"), max_items=max_ing)

    lines = []
    # 메뉴명은 최상단에 강하게
    if menu:
        lines.append(f"[MENU] {menu}")
    else:
        lines.append("[MENU] (이름없음)")

    # 선택축(다양화에 쓰일 메타)
    meta_parts = []
    if kind: meta_parts.append(f"종류={kind}")
    if method: meta_parts.append(f"조리={method}")
    if situation: meta_parts.append(f"상황={situation}")
    if meta_parts:
        lines.append("- " + " | ".join(meta_parts))

    # 재료는 요약만 (선택폭 확대 목적)
    if ing:
        lines.append(f"- 핵심 재료: {ing}")

    # 추적용 id (근거 표시/디버깅)
    if rcp_sno:
        lines.append(f"- 근거 ID: {rcp_sno}")

    return "\n".join(lines)
