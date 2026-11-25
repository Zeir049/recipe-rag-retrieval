# make_passages.py
import argparse
import json
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
            # one JSON object per line; json.dumps escapes the newlines inside the passage
            rec = {"idx": int(idx), "RCP_SNO": str(row.get("RCP_SNO")), "CKG_NM": str(row.get("CKG_NM")), "passage": passage}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"saved: {args.out}  (rows={len(df)})")

if __name__ == "__main__":
    main()
