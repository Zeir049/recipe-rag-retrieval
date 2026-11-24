import argparse
import json
import os
import sys
from datetime import datetime

from core.types import RunConfig
from core.unified_engine import UnifiedEngine

# local_menu_rag.py와 동일한 safe_load_config 로직(복사)
import inspect


def safe_load_config(config_path: str | None = None) -> dict:
    try:
        from core.io import load_config
    except Exception as e:
        print(f"[WARN] core.io import 실패: {e}", file=sys.stderr)
        return {}

    candidates: list[str] = []
    if config_path:
        candidates.append(config_path)
    candidates += ["config.json", "config.yaml", "config.yml", "settings.json", "rag_config.json"]

    try:
        sig = inspect.signature(load_config)
        if len(sig.parameters) == 0:
            return load_config() or {}
    except Exception:
        pass

    for p in candidates:
        try:
            if not p:
                continue
            if not os.path.exists(p):
                continue

            try:
                return load_config(p) or {}
            except TypeError:
                pass

            for kw in ("path", "config_path", "filepath", "file"):
                try:
                    return load_config(**{kw: p}) or {}
                except TypeError:
                    continue

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"[WARN] load_config 실패 ({p}): {e}", file=sys.stderr)
            return {}

    return {}


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="test_queries.jsonl 경로")
    ap.add_argument("--generate_rule", action="store_true")
    ap.add_argument("--config", default=None, help="(선택) config 파일 경로")
    args = ap.parse_args()

    cfg_dict = safe_load_config(args.config)
    default_k = int(cfg_dict.get("k", 7))
    default_min_score = float(cfg_dict.get("min_score", 0.0))

    engine = UnifiedEngine()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir("runs")
    run_path = os.path.join("runs", f"run_{run_ts}.jsonl")
    summary_path = os.path.join("runs", f"run_{run_ts}_summary.json")

    records = []
    total = 0

    for idx, obj in enumerate(read_jsonl(args.queries), 1):
        q = obj.get("q") or obj.get("query")
        if not q:
            raise ValueError(f"queries jsonl에서 q/query 필드가 없습니다: {obj}")

        k = int(obj.get("k", default_k))
        min_score = float(obj.get("min_score", default_min_score))

        cfg = RunConfig(
            k=k,
            min_score=min_score,
            generate_rule=bool(args.generate_rule),
            only_answer=False,
            diversify_method=bool(obj.get("diversify_method", cfg_dict.get("diversify_method", False))),
            max_per_method=int(obj.get("max_per_method", cfg_dict.get("max_per_method", 999))),
            diversify_tag=bool(obj.get("diversify_tag", cfg_dict.get("diversify_tag", False))),
            max_per_tag=int(obj.get("max_per_tag", cfg_dict.get("max_per_tag", 999))),
        )

        result = engine.run(q, cfg)
        total += 1

        returned = len(result.selected)
        top1_item = result.selected[0] if returned > 0 else {}
        top1 = top1_item.get("name") or top1_item.get("menu") or top1_item.get("title") or "-"
        score = top1_item.get("score") or top1_item.get("sim") or top1_item.get("similarity")

        methods = []
        for it in result.selected:
            m = it.get("method") or it.get("cook_method") or it.get("조리") or it.get("조리방법")
            if m is not None:
                methods.append(str(m))
        uniq_methods = len(set(methods)) if methods else 0

        print(f"[{idx}] returned={returned}/{k} top1='{top1}' score={score if score is not None else '-'} uniq_methods={uniq_methods}")

        rec = {
            "idx": idx,
            "q": q,
            "k": k,
            "min_score": min_score,
            "returned": returned,
            "top1": top1,
            "top1_score": score,
            "uniq_methods": uniq_methods,
            "guardrail": result.guardrail,
            "debug": result.debug,
        }
        records.append(rec)

    with open(run_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total": total,
        "avg_returned": (sum(r["returned"] for r in records) / total) if total else 0.0,
        "runs_file": run_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"- run:     {run_path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
