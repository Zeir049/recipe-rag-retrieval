# make_passages.py
import argparse
import pandas as pd
from passage_utils import build_menu_passage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="recipe_meta.csv")
    ap.add_argument("--out", default="menu_passages.jsonl")
    ap.add_argument("--max_ing", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.meta, encoding="utf-8-sig")

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            passage = build_menu_passage(row, max_ing=args.max_ing)
            # jsonl (간단 버전: 큰따옴표 이스케이프 최소화)
            passage_escaped = passage.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'{{"idx": {idx}, "RCP_SNO": "{row.get("RCP_SNO")}", "CKG_NM": "{row.get("CKG_NM")}", "passage": "{passage_escaped}"}}\n')

    print(f"saved: {args.out}  (rows={len(df)})")

if __name__ == "__main__":
    main()
