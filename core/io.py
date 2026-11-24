# core/io.py
import json
import os
from typing import Any, Dict

import pandas as pd


def load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ext in (".yml", ".yaml"):
        # YAML은 optional dependency로 처리
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config를 쓰려면 PyYAML이 필요해요. "
                "pip install pyyaml 로 설치하거나 .json config를 사용하세요."
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config extension: {ext} (use .json or .yaml)")


def load_meta(meta_path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    return pd.read_csv(meta_path, encoding=encoding)
