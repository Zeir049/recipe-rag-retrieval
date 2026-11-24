from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple

from .types import RunConfig, RunResult


def _as_run_result(query: str, out: Any) -> RunResult:
    """core.engine이 무엇을 반환하든 RunResult로 정규화."""
    if isinstance(out, RunResult):
        return out

    if isinstance(out, dict):
        selected = out.get("selected") or out.get("items") or out.get("results") or out.get("menus") or []
        guardrail = out.get("guardrail")
        debug = out.get("debug", {})
        return RunResult(query=query, selected=list(selected), guardrail=guardrail, debug=dict(debug))

    if isinstance(out, list):
        return RunResult(query=query, selected=list(out), guardrail=None, debug={})

    raise TypeError(f"Unsupported engine output type: {type(out)}")


def _score_name(name: str) -> int:
    """
    core.engine 내부에서 '진입점'일 가능성이 높은 이름을 우선시하기 위한 점수.
    """
    name_l = name.lower()
    score = 0
    for kw, w in [
        ("run", 50),
        ("recommend", 45),
        ("engine", 40),
        ("pipeline", 35),
        ("rag", 30),
        ("infer", 25),
        ("predict", 25),
        ("query", 20),
        ("execute", 20),
        ("main", 10),
    ]:
        if kw in name_l:
            score += w
    # 너무 일반적인 이름은 약간 감점
    if name_l in ("test", "demo", "example"):
        score -= 30
    return score


class UnifiedEngine:
    """
    목표: local_menu_rag.py / batch_eval.py가 '항상 같은 엔진'을 호출하게 고정.
    core.engine.py 구현이 어떤 형태여도 최대한 자동으로 진입점을 찾아 호출한다.
    """

    def __init__(self):
        self._engine_mod = importlib.import_module("core.engine")
        self._entry = self._resolve_entrypoint()

    def run(self, query: str, cfg: RunConfig) -> RunResult:
        fn, kind = self._entry  # kind는 디버그용 문자열
        out = self._call_flex(fn, query, cfg)
        return _as_run_result(query, out)

    def _resolve_entrypoint(self) -> Tuple[Any, str]:
        eng = self._engine_mod

        # 0) 명시적 지정: ENTRYPOINT(문자열/콜러블), ENGINE(콜러블) 지원
        if hasattr(eng, "ENTRYPOINT"):
            ep = getattr(eng, "ENTRYPOINT")
            if callable(ep):
                return ep, "module.ENTRYPOINT(callable)"
            if isinstance(ep, str) and hasattr(eng, ep) and callable(getattr(eng, ep)):
                return getattr(eng, ep), f"module.ENTRYPOINT('{ep}')"

        if hasattr(eng, "ENGINE") and callable(getattr(eng, "ENGINE")):
            return getattr(eng, "ENGINE"), "module.ENGINE(callable)"

        # 1) 가장 정석 패턴
        for name in ("run", "recommend"):
            if hasattr(eng, name) and callable(getattr(eng, name)):
                return getattr(eng, name), f"module.{name}()"

        # 2) 클래스 패턴 (Engine/CoreEngine)
        for cls_name in ("Engine", "CoreEngine"):
            if hasattr(eng, cls_name) and inspect.isclass(getattr(eng, cls_name)):
                cls = getattr(eng, cls_name)
                try:
                    inst = cls()
                except Exception:
                    inst = None
                if inst is not None:
                    for meth in ("run", "recommend", "__call__"):
                        if hasattr(inst, meth) and callable(getattr(inst, meth)):
                            return getattr(inst, meth), f"{cls_name}.{meth}()"

        # 3) 자동 탐색: core.engine 안의 함수들 중 “실행 함수”로 보이는 걸 찾아 호출 가능한지 테스트
        candidates: List[Tuple[int, str, Any]] = []

        for name, fn in inspect.getmembers(eng, inspect.isfunction):
            if fn.__module__ != eng.__name__:
                continue
            candidates.append((_score_name(name), name, fn))

        # 점수 높은 것부터 시도
        candidates.sort(key=lambda x: x[0], reverse=True)

        working: List[Tuple[int, str, Any]] = []
        for score, name, fn in candidates:
            # 점수가 너무 낮으면(가능성 낮음) 스킵
            if score <= 0:
                continue
            try:
                out = self._call_flex(fn, "__probe__", RunConfig())  # 시그니처만 통과하는지 체크
                # out이 None이면 진짜 실행 함수가 아닐 수도 있음 → 일단 스킵
                if out is None:
                    continue
                # 형태가 dict/list/RunResult면 진입점 가능성이 높음
                if isinstance(out, (dict, list, RunResult)):
                    working.append((score, name, fn))
            except TypeError:
                continue
            except Exception:
                # 실제 실행 중 리소스가 필요해서 실패할 수 있으니, 완전 제외하지 않고 후보로 남겨둠
                # (단, 너무 위험하니 score만으로 선택하되, 여기선 working엔 안 넣음)
                continue

        if working:
            # 제일 점수 높은 것 선택
            working.sort(key=lambda x: x[0], reverse=True)
            best = working[0]
            return best[2], f"auto_detected:{best[1]}()"

        # 여기까지 왔으면 진짜로 진입점을 못 찾은 것
        available = []
        for name, obj in inspect.getmembers(eng):
            if callable(obj) and not name.startswith("_"):
                available.append(name)
        available = sorted(set(available))

        raise RuntimeError(
            "core.engine에서 호출 가능한 엔진을 찾지 못했습니다.\n"
            "지원 패턴: run(...), recommend(...), Engine.run(), Engine.recommend(), CoreEngine.run(), CoreEngine.recommend(), "
            "또는 ENTRYPOINT/ENGINE 지정\n"
            f"core.engine 내 callable 후보(참고): {available}"
        )

    @staticmethod
    def _call_flex(fn, query: str, cfg: RunConfig):
        """
        기존 core.engine 구현 시그니처가 달라도 웬만하면 호출되게 유연 호출.
        """
        sig = inspect.signature(fn)
        params = sig.parameters

        # A) (query, cfg)
        try:
            if len(params) >= 2:
                return fn(query, cfg)
        except TypeError:
            pass

        # B) keyword: query/q, cfg/config
        kw = {}
        if "query" in params:
            kw["query"] = query
        elif "q" in params:
            kw["q"] = query

        if "cfg" in params:
            kw["cfg"] = cfg
        elif "config" in params:
            kw["config"] = cfg

        if kw:
            try:
                return fn(**kw)
            except TypeError:
                pass

        # C) (query, k=..., min_score=..., ...)
        try:
            return fn(
                query,
                k=cfg.k,
                min_score=cfg.min_score,
                generate_rule=cfg.generate_rule,
                only_answer=cfg.only_answer,
                diversify_method=cfg.diversify_method,
                max_per_method=cfg.max_per_method,
                diversify_tag=cfg.diversify_tag,
                max_per_tag=cfg.max_per_tag,
            )
        except TypeError:
            pass

        # D) all kwargs
        try:
            return fn(
                query=query,
                k=cfg.k,
                min_score=cfg.min_score,
                generate_rule=cfg.generate_rule,
                only_answer=cfg.only_answer,
                diversify_method=cfg.diversify_method,
                max_per_method=cfg.max_per_method,
                diversify_tag=cfg.diversify_tag,
                max_per_tag=cfg.max_per_tag,
            )
        except TypeError as e:
            raise TypeError(f"엔진 호출 실패: {fn} / signature={sig} / error={e}") from e
