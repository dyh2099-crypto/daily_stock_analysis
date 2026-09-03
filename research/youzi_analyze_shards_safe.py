#!/usr/bin/env python3
"""Safe launcher for shard analysis; tolerates headerless empty audit CSVs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("youzi_analyze_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_read_many(root: Path, pattern: str, **kwargs: Any) -> pd.DataFrame:
    frames = []
    for file in sorted(root.rglob(pattern)):
        try:
            frame = pd.read_csv(file, **kwargs)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    implementation = load_module(Path(__file__).with_name("youzi_analyze_shards.py"))
    implementation.read_many = safe_read_many
    return int(implementation.main())


if __name__ == "__main__":
    raise SystemExit(main())
