import argparse

from core.engine import DEFAULT_CONFIG, get_config
from core.render import render, render_passages
from core.types import RunConfig
from core.unified_engine import UnifiedEngine


def main():
    ap = argparse.ArgumentParser(description="Retrieve recipes for one query.")
    ap.add_argument("--q", required=True, help="질문 (query)")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--min_score", type=float, default=None)

    ap.add_argument("--generate_rule", action="store_true", help="rule-based reason for each result")
    ap.add_argument("--passages", action="store_true", help="print the evidence-tagged passages instead")

    ap.add_argument("--diversify_method", action="store_true")
    ap.add_argument("--max_per_method", type=int, default=None)

    ap.add_argument("--diversify_tag", action="store_true")
    ap.add_argument("--max_per_tag", type=int, default=None)

    ap.add_argument("--config", default=DEFAULT_CONFIG, help="config file (default: configs/menu_rag.json)")
    args = ap.parse_args()

    cfg_dict = get_config(args.config)

    def pick(cli_value, default):
        return cli_value if cli_value is not None else default

    cfg = RunConfig(
        k=args.k,
        min_score=pick(args.min_score, float(cfg_dict.get("selection", {}).get("min_score_default", 0.0))),
        generate_rule=bool(args.generate_rule),
        diversify_method=bool(args.diversify_method),
        max_per_method=pick(args.max_per_method, int(cfg_dict.get("diversity", {}).get("max_per_value_default", 1))),
        diversify_tag=bool(args.diversify_tag),
        max_per_tag=pick(args.max_per_tag, int(cfg_dict.get("diversity_tags", {}).get("max_per_tag_default", 2))),
        extra={"config_path": args.config},
    )

    result = UnifiedEngine().run(args.q, cfg)
    if args.passages:
        print(render_passages(result))
    else:
        print(render(result, cfg=cfg, show_tags=bool(cfg_dict.get("output", {}).get("show_tags", True))))


if __name__ == "__main__":
    main()
