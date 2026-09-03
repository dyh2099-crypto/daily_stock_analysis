#!/usr/bin/env python3
"""Audited v2 wrapper around the TDX-only Youzi strategy backtest.

Changes versus v1:
1. Uses a much more conservative price-behaviour ST proxy.
2. Masks the corporate-action discontinuity session and the following 60
   sessions from signals and market-sentiment inputs.
3. Corrects audit labels so proxy fields are not described as vendor fields.
4. Supports ST_FILTER_MODE=conservative and ST_FILTER_MODE=none for a
   sensitivity check.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).with_name("youzi_tdx_only_backtest.py")
spec = importlib.util.spec_from_file_location("tdx_v1", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import TDX-only v1 backtest")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ST_FILTER_MODE = os.environ.get("ST_FILTER_MODE", "conservative").strip().lower()
if ST_FILTER_MODE not in {"conservative", "none"}:
    raise ValueError(f"unsupported ST_FILTER_MODE={ST_FILTER_MODE!r}")


def suspected_historical_st_v2(day: pd.DataFrame, board: str) -> Tuple[bool, int, int]:
    """Infer only clear historical 5% price-limit behaviour.

    An ordinary main-board stock can coincidentally close at exactly +5%, so
    the previous two-hit rule was too aggressive. V2 requires one of:
    - a one-price session at a reconstructed +/-5% cap;
    - at least two capped-up and two capped-down sessions; or
    - at least eight capped sessions in total.

    This remains a proxy, not an official point-in-time ST flag.
    """
    if ST_FILTER_MODE == "none" or board not in ("sh_main", "sz_main") or len(day) < 2:
        return False, 0, 0
    d = day.copy()
    d["preclose"] = d["close"].shift(1)
    d = d[
        d["date"].between(base.core.SIGNAL_START, pd.Timestamp("2026-07-03"))
        & d["preclose"].notna()
    ]
    if d.empty:
        return False, 0, 0
    up5 = base.rounded_price(d["preclose"] * 1.05)
    down5 = base.rounded_price(d["preclose"] * 0.95)
    tol = 0.0011
    up_hit = (d["close"].sub(up5).abs() <= tol) & (d["high"].sub(up5).abs() <= tol)
    down_hit = (d["close"].sub(down5).abs() <= tol) & (d["low"].sub(down5).abs() <= tol)
    capped = up_hit | down_hit
    one_price = capped & (d["high"].sub(d["low"]).abs() <= tol)
    up_count = int(up_hit.sum())
    down_count = int(down_hit.sum())
    total_count = up_count + down_count
    one_count = int(one_price.sum())
    inferred = one_count >= 1 or (up_count >= 2 and down_count >= 2) or total_count >= 8
    return bool(inferred), total_count, one_count


_original_prepare = base.core.prepare_data


def prepare_data_v2(raw: pd.DataFrame, industries: pd.DataFrame):
    raw = raw.copy()
    thresholds = raw["board"].map({
        "sh_main": 0.12,
        "sz_main": 0.12,
        "star": 0.22,
        "chinext": 0.22,
        "bse": 0.32,
    }).fillna(0.12)
    discontinuity = (raw["open"] / raw["preclose"] - 1.0).abs() > thresholds
    raw["corp_action_discontinuity"] = discontinuity.fillna(False)
    # A backward-looking rolling window marks the event itself and the next
    # 60 observed sessions, because the event stays inside the trailing window.
    raw["corp_action_recent60"] = (
        raw.assign(_event=raw["corp_action_discontinuity"].astype(int))
        .groupby("code", sort=False)["_event"]
        .transform(lambda s: s.rolling(61, min_periods=1).max())
        .fillna(0)
        .astype(bool)
    )
    frame, raw_rows = _original_prepare(raw, industries)
    mask = frame["corp_action_recent60"].fillna(True).astype(bool)
    base.AUDIT_EXTRA["corporate_action_event_and_following60_rows_masked"] = int(mask.sum())
    for col in ("ret1", "ret3", "ret10", "close_location", "stock_ret10_percentile"):
        if col in frame.columns:
            frame.loc[mask, col] = np.nan
    for col in ("limit_up", "limit_down", "touched_up", "one_price_up", "one_price_down"):
        if col in frame.columns:
            frame.loc[mask, col] = False
    if "prev_limit_up" in frame.columns:
        frame["prev_limit_up"] = (
            frame.sort_values(["code", "date"])
            .groupby("code", sort=False)["limit_up"]
            .shift(1)
            .fillna(False)
            .astype(bool)
        )
    return frame, raw_rows


def patch_audit_v2(output_dir: Path) -> None:
    audit_path = output_dir / "data_audit.json"
    headline_path = output_dir / "headline.json"
    if not audit_path.exists() or not headline_path.exists():
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "st_filter_mode": ST_FILTER_MODE,
        "historical_isST_field_present": False,
        "tradestatus_field_present": False,
        "daily_security_status": "volume>0 proxy; no vendor point-in-time security-status table",
        "st_exclusion_rule": (
            "no ST exclusion in sensitivity run" if ST_FILTER_MODE == "none" else
            "conservative behavioural proxy: remove whole main-board security after >=1 one-price reconstructed 5% cap, or >=2 capped-up and >=2 capped-down sessions, or >=8 total reconstructed 5% capped sessions in 2023-09-04..2026-07-03"
        ),
        "st_proxy_warning": "not an official point-in-time ST history; conservative mode may still miss quiet ST periods or exclude rare ordinary-price coincidences",
        "industry_row_coverage": 0.0,
        "theme_method": "market-wide new-leadership cohort proxy; no historical concept/industry membership was used",
        "corporate_action_mask": "event session plus following 60 observed sessions removed from signal eligibility and return/limit inputs to market sentiment",
    })
    audit_path.write_text(json.dumps(base.core.json_clean(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    headline["audit"] = audit
    headline_path.write_text(json.dumps(base.core.json_clean(headline), ensure_ascii=False, indent=2), encoding="utf-8")
    report = output_dir / "REPORT.md"
    if report.exists():
        report.write_text(
            report.read_text(encoding="utf-8")
            + "\n\n## V2 audit correction\n\n"
            + f"ST filter mode: `{ST_FILTER_MODE}`. Historical ST and daily security status are behavioural proxies, not vendor point-in-time fields. Corporate-action event sessions and the following 60 observed sessions were masked from signals and market-sentiment return/limit inputs.\n",
            encoding="utf-8",
        )


def main() -> int:
    base.suspected_historical_st = suspected_historical_st_v2
    base.core.prepare_data = prepare_data_v2
    rc = base.main()
    output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    patch_audit_v2(output_dir)
    if (output_dir / "headline.json").exists():
        print(f"===TDX_V2_{ST_FILTER_MODE.upper()}_HEADLINE_BEGIN===")
        print((output_dir / "headline.json").read_text(encoding="utf-8"))
        print(f"===TDX_V2_{ST_FILTER_MODE.upper()}_HEADLINE_END===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
