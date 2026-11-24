import argparse
import json
import re
import pandas as pd


def safe_str(x) -> str:
    """None/NaN/'nan' 문자열을 빈 문자열로 정리"""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def to_nullable(s: str):
    """빈 문자열이면 None(=결측)으로 변환, 아니면 그대로"""
    s = safe_str(s)
    return None if s == "" else s


def parse_line(text: str, key: str) -> str:
    """doc_text에서 '키: 값' 형태 라인을 찾아 값만 반환 (nan 정리 포함)"""
    if not text:
        return ""
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return safe_str(m.group(1).strip()) if m else ""


def parse_category_from_text(text: str):
    """
    카테고리: <kind> / <situation> / <method>
    ⚠ kind에 '국/탕'처럼 '/'가 포함될 수 있으므로 구분자는 반드시 ' / ' 사용.
    """
    line = parse_line(text, "카테고리")
    if not line:
        return "", "", ""
    parts = [safe_str(p.strip()) for p in line.split(" / ")]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def sanitize_doc_text(text: str) -> str:
    """doc_text 내부의 nan 제거 + 빈 라인 제거"""
    if not text:
        return ""
    s = str(text)

    cleaned_lines = []
    for ln in s.splitlines():
        # "키: nan" 제거
        if re.search(r"(?i)^\s*[^:]+:\s*nan\s*$", ln):
            continue
        # 라인 전체가 nan이면 제거
        if ln.strip().lower() == "nan":
            continue

        ln = re.sub(r"(?i)\bnan\b", "", ln).strip()
        if ln:
            cleaned_lines.append(ln)

    return "\n".join(cleaned_lines).strip()


def build_doc_text_from_metadata(m: dict) -> str:
    """metadata로 doc_text 새로 만들기(결측 라인은 아예 제외)"""
    dish = safe_str(m.get("dish"))
    kind = safe_str(m.get("kind"))
    situation = safe_str(m.get("situation"))
    method = safe_str(m.get("method"))
    main_group = safe_str(m.get("main_material_group"))
    servings = safe_str(m.get("servings"))
    ing_kw = safe_str(m.get("ingredients_keywords"))

    lines = [
        f"요리명: {dish}",
        f"카테고리: {kind} / {situation} / {method}",
        f"주재료군: {main_group}",
    ]
    if ing_kw:
        lines.append(f"재료: {ing_kw}")
    if servings:
        lines.append(f"인분: {servings}")

    return sanitize_doc_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (recipes_rag_with_metadata.jsonl)")
    ap.add_argument("--out_csv", default="recipes_output_fixed.csv", help="Output CSV path")
    ap.add_argument(
        "--use_text_field",
        action="store_true",
        help="JSONL의 top-level 'text'를 doc_text로 그대로 사용 (없으면 metadata로 생성)"
    )
    ap.add_argument(
        "--keep_nulls",
        action="store_true",
        help="결측값을 빈문자('')가 아니라 None(=CSV에서 빈칸, pandas로 읽으면 NaN)으로 유지"
    )
    args = ap.parse_args()

    results = []
    print(f"작업 시작: {args.input} 읽는 중...")

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"{i}행 JSON 파싱 실패: {e}")
                continue

            rid = data.get("id")
            m = data.get("metadata", {}) or {}

            # doc_text 결정
            if args.use_text_field:
                doc_text = sanitize_doc_text(safe_str(data.get("text")))
                if not doc_text:
                    doc_text = build_doc_text_from_metadata(m)
            else:
                doc_text = build_doc_text_from_metadata(m)

            # 카테고리는 doc_text에서 다시 파싱(국/탕 이슈 회피)
            kind, situation, method = parse_category_from_text(doc_text)

            # 다른 필드도 doc_text 우선
            dish = parse_line(doc_text, "요리명") or safe_str(m.get("dish"))
            main_group = parse_line(doc_text, "주재료군") or safe_str(m.get("main_material_group"))
            servings = parse_line(doc_text, "인분") or safe_str(m.get("servings"))
            ing_kw = parse_line(doc_text, "재료") or safe_str(m.get("ingredients_keywords"))

            # ✅ 결측 유지 옵션 적용
            if args.keep_nulls:
                dish_out = to_nullable(dish)
                ing_kw_out = to_nullable(ing_kw)
                kind_out = to_nullable(kind)
                situation_out = to_nullable(situation)
                method_out = to_nullable(method)
                main_group_out = to_nullable(main_group)
                servings_out = to_nullable(servings)
            else:
                dish_out = safe_str(dish)
                ing_kw_out = safe_str(ing_kw)
                kind_out = safe_str(kind)
                situation_out = safe_str(situation)
                method_out = safe_str(method)
                main_group_out = safe_str(main_group)
                servings_out = safe_str(servings)

            row = {
                "RCP_SNO": rid,                 # 내부 식별자(권장)
                "CKG_NM": dish_out,
                "ingredients_keywords": ing_kw_out,
                "doc_text": doc_text,

                "kind": kind_out,
                "situation": situation_out,
                "method": method_out,
                "main_material_group": main_group_out,
                "servings": servings_out,
            }
            results.append(row)

    if not results:
        print("❌ 처리된 데이터가 없습니다. 입력 파일 내용을 확인해주세요.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 성공! {len(df)}개의 레시피가 {args.out_csv}에 저장되었습니다.")


if __name__ == "__main__":
    main()
