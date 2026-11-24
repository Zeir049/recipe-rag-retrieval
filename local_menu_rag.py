import argparse
import inspect
import os
import sys

from core.types import RunConfig
from core.unified_engine import UnifiedEngine


def safe_load_config(config_path: str | None = None) -> dict:
    """
    core.io.load_config가:
    - 인자 없이 동작하는 경우
    - 경로 인자를 요구하는 경우
    - config 파일이 없거나 파싱 실패하는 경우
    모두 안전하게 처리하고 {}를 반환한다.
    """
    try:
        from core.io import load_config  # 기존 코드 유지
    except Exception as e:
        print(f"[WARN] core.io import 실패: {e}", file=sys.stderr)
        return {}

    # 시도할 후보 경로들
    candidates: list[str] = []
    if config_path:
        candidates.append(config_path)
    candidates += [
        "config.json",
        "config.yaml",
        "config.yml",
        "settings.json",
        "rag_config.json",
    ]

    # 1) 인자 없이 호출 시도
    try:
        sig = inspect.signature(load_config)
        if len(sig.parameters) == 0:
            return load_config() or {}
    except Exception:
        # signature 조회 실패/호출 실패면 아래로
        pass

    # 2) 경로 인자를 넣어 호출 시도 (존재하는 파일만)
    for p in candidates:
        try:
            if not p:
                continue
            if not os.path.exists(p):
                continue

            # (a) positional
            try:
                return load_config(p) or {}
            except TypeError:
                pass

            # (b) keyword들 (load_config(path=...), load_config(config_path=...) 등)
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

    # 아무 것도 못 찾으면 빈 config
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="질문")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--min_score", type=float, default=None)

    ap.add_argument("--generate_rule", action="store_true")
    ap.add_argument("--only_answer", action="store_true")

    ap.add_argument("--diversify_method", action="store_true")
    ap.add_argument("--max_per_method", type=int, default=999)

    ap.add_argument("--diversify_tag", action="store_true")
    ap.add_argument("--max_per_tag", type=int, default=999)

    ap.add_argument("--config", default=None, help="(선택) config 파일 경로")
    args = ap.parse_args()

    cfg_dict = safe_load_config(args.config)

    # config 파일에 기본값이 있으면 사용하고, CLI가 우선
    k = args.k if args.k is not None else int(cfg_dict.get("k", 7))
    min_score = args.min_score if args.min_score is not None else float(cfg_dict.get("min_score", 0.0))

    cfg = RunConfig(
        k=k,
        min_score=min_score,
        generate_rule=bool(args.generate_rule),
        only_answer=bool(args.only_answer),
        diversify_method=bool(args.diversify_method),
        max_per_method=int(args.max_per_method),
        diversify_tag=bool(args.diversify_tag),
        max_per_tag=int(args.max_per_tag),
    )

    engine = UnifiedEngine()
    result = engine.run(args.q, cfg)

    # core.render가 있으면 사용, 없으면 fallback
    try:
        from core import render as render_mod

        if hasattr(render_mod, "render") and callable(getattr(render_mod, "render")):
            print(render_mod.render(result, cfg=cfg))
            return
        if hasattr(render_mod, "render_result") and callable(getattr(render_mod, "render_result")):
            print(render_mod.render_result(result, cfg=cfg))
            return

        raise AttributeError("core.render에 render/render_result가 없습니다.")
    except Exception:
        # fallback: 최소 출력
        print(f"질문: {result.query}\n")
        if result.guardrail:
            print(f"선정 기준: {result.guardrail}")
        print("추천 메뉴:\n")
        for i, item in enumerate(result.selected, 1):
            name = item.get("name") or item.get("menu") or item.get("title") or str(item)
            print(f"{i}) {name}")


if __name__ == "__main__":
    main()
