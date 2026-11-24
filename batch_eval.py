import argparse
import json
import os
from datetime import datetime

from core.engine import DEFAULT_CONFIG, get_config
from core.render import render
from core.types import RunConfig
from core.unified_engine import UnifiedEngine


def read_jsonl(path: str):
    """One JSON object per line; blank lines and lines starting with '#' are skipped."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Run a JSONL file of test queries.")
    ap.add_argument("--queries", required=True, help="test_queries.jsonl")
    ap.add_argument("--generate_rule", action="store_true", help="force rule-based reasons for every query")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    cfg_dict = get_config(args.config)
    default_min_score = float(cfg_dict.get("selection", {}).get("min_score_default", 0.0))
    default_max_per_method = int(cfg_dict.get("diversity", {}).get("max_per_value_default", 1))
    default_max_per_tag = int(cfg_dict.get("diversity_tags", {}).get("max_per_tag_default", 2))

    engine = UnifiedEngine()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    run_path = os.path.join(args.out_dir, f"run_{run_ts}.jsonl")
    summary_path = os.path.join(args.out_dir, f"run_{run_ts}_summary.json")

    records = []
    for idx, obj in enumerate(read_jsonl(args.queries), 1):
        q = obj.get("q") or obj.get("query")
        if not q:
            raise ValueError(f"query line without 'q': {obj}")

        cfg = RunConfig(
            k=int(obj.get("k", 7)),
            min_score=float(obj.get("min_score", default_min_score)),
            generate_rule=bool(args.generate_rule or obj.get("generate_rule", False)),
            diversify_method=bool(obj.get("diversify_method", False)),
            max_per_method=int(obj.get("max_per_method", default_max_per_method)),
            diversify_tag=bool(obj.get("diversify_tag", False)),
            max_per_tag=int(obj.get("max_per_tag", default_max_per_tag)),
            extra={"config_path": args.config},
        )
        result = engine.run(q, cfg)

        items = result.selected
        scores = [float(it["score"]) for it in items]
        applied = (result.guardrail or {}).get("applied_filters", [])
        rec = {
            "query": q,
            "k": cfg.k,
            "min_score": cfg.min_score,
            "returned": len(items),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "uniq_methods": len({it.get("method") for it in items}),
            "applied_filters": applied,
            "items_passing_filters": sum(1 for it in items if it.get("filter_ok")) if applied else None,
            "items": [
                {key: it.get(key) for key in
                 ("idx", "score", "title", "kind", "method", "situation", "doc_id",
                  "key_ingredients_list", "tags", "reason", "passage")}
                for it in items
            ],
            "answer": render(result, cfg=cfg),
        }
        records.append(rec)

        top1 = items[0]["title"] if items else "-"
        print(f"[{idx}] returned={rec['returned']}/{cfg.k} top1='{top1}' "
              f"score={scores[0] if scores else '-'} uniq_methods={rec['uniq_methods']} filters={applied}")

    with open(run_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    with_items = [r for r in records if r["avg_score"] is not None]
    summary = {
        "run_file": run_path,
        "queries_file": args.queries,
        "count": n,
        "avg_returned": sum(r["returned"] for r in records) / n if n else 0.0,
        "avg_avg_score": sum(r["avg_score"] for r in with_items) / len(with_items) if with_items else None,
        "fail_count": sum(1 for r in records if r["returned"] < r["k"]),  # fewer than k results
        "filtered_queries": sum(1 for r in records if r["applied_filters"]),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
