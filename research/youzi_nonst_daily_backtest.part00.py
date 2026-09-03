#!/usr/bin/env python3
"""Point-in-time non-ST daily proxy backtest for A-share 'hot money sayings'.

Primary data: BaoStock daily forward-adjusted (adjustflag=2) OHLCV, which also
contains point-in-time isST and tradestatus. The historical universe comes from
query_stock_basic(), including delisted securities where BaoStock has records.
Industry snapshots are requested with historical dates and used as a coarse,
explicitly-labelled proxy for themes. No parameter is tuned on the result.

This is intentionally a DAILY proxy, not a minute/tick reconstruction:
- signals are formed after the close;
- entries occur no earlier than the next tradable session open;
- T+1 is enforced;
- one-price limit-up entries are rejected;
- one-price limit-down exits are deferred;
- when stop and target are both touched in one daily bar, stop is assumed first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    read_start: str = "2023-01-03"
    signal_start: str = "2023-09-04"
    end_date: str = "2026-09-02"  # last completed session before 2026-09-03 close
    min_calendar_listing_days: int = 180
    min_median_amount20: float = 50_000_000.0
    min_price: float = 2.0
    sentiment_lookback: int = 120
    sentiment_min_history: int = 40
    max_entry_gap: float = 0.05
    buy_commission: float = 0.0003
    sell_commission: float = 0.0003
    stamp_duty_sell: float = 0.0005
    transfer_fee_each_side: float = 0.00001
    slippage_each_side: float = 0.0005
    industry_snapshot: str = "monthly"
    factor_query: bool = True


STRATEGY_LABELS = {
    "leader_continuation": "热点龙头延续（日线代理）",
    "pullback_rebound": "强势股回撤反抽（日线代理）",
    "panic_rebound_confirmation": "恐慌超跌确认反弹（日线代理）",
    "second_board_after_close": "二板收盘确认后接力（日线代理）",
    "new_hot_industry_proxy": "新热点行业切换（题材代理）",
}


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def resultset_to_frame(rs: Any) -> pd.DataFrame:
    rows: List[List[str]] = []
    while getattr(rs, "error_code", "1") == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=getattr(rs, "fields", []))


def bs_login(bs: Any, attempts: int = 5) -> None:
    last = None
    for attempt in range(1, attempts + 1):
        lg = bs.login()
        if getattr(lg, "error_code", "1") == "0":
            print(f"BaoStock login success on attempt {attempt}", flush=True)
            return
        last = f"{getattr(lg, 'error_code', '?')} {getattr(lg, 'error_msg', '')}"
        time.sleep(min(30, attempt * 5))
    raise RuntimeError(f"BaoStock login failed: {last}")


def query_with_retry(bs: Any, fn_name: str, kwargs: Mapping[str, Any], attempts: int = 4) -> pd.DataFrame:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            fn = getattr(bs, fn_name)
            rs = fn(**dict(kwargs))
            code = getattr(rs, "error_code", "1")
            if code == "0":
                return resultset_to_frame(rs)
            last = f"{code} {getattr(rs, 'error_msg', '')}"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        try:
            bs.logout()
        except Exception:
            pass
        time.sleep(min(20, 2 ** attempt))
        bs_login(bs, attempts=2)
    raise RuntimeError(f"{fn_name} failed after retries: {last}; kwargs={kwargs}")


def classify_board(code: str) -> Optional[str]:
    if code.startswith("sh."):
        c = code[3:]
        if c.startswith(("600", "601", "603", "605")):
            return "MAIN"
        if c.startswith(("688", "689")):
            return "STAR"
    if code.startswith("sz."):
        c = code[3:]
        if c.startswith(("000", "001", "002", "003")):
            return "MAIN"
        if c.startswith(("300", "301")):
            return "CHINEXT"
    if code.startswith("bj."):
        return "BSE"
    return None


def limit_ratio_for_board(board: str) -> float:
    return {"MAIN": 0.10, "STAR": 0.20, "CHINEXT": 0.20, "BSE": 0.30}[board]


def choose_snapshot_dates(dates: Sequence[pd.Timestamp], mode: str) -> List[pd.Timestamp]:
    ds = pd.Series(pd.to_datetime(pd.Index(dates).unique())).sort_values()
    if mode == "weekly":
        return ds.groupby(ds.dt.to_period("W")).first().tolist()
    return ds.groupby(ds.dt.to_period("M")).first().tolist()


def normalize_daily(raw: pd.DataFrame, sid: int, board: str, ipo_date: pd.Timestamp) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST", "adjustflag"]
    for c in numeric:
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    required = ["date", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg", "tradestatus", "isST"]
    raw = raw.dropna(subset=required)
    raw = raw[(raw["date"] >= pd.Timestamp("2023-01-03")) & (raw["date"] <= pd.Timestamp("2026-09-02"))]
    if raw.empty:
        return raw
    out = raw[["date", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST", "adjustflag"]].copy()
    out["sid"] = np.int32(sid)
    out["board"] = board
    out["ipo_date"] = ipo_date
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    for c in ["open", "high", "low", "close", "preclose", "turn", "pctChg"]:
        out[c] = out[c].astype("float32")
    out["volume"] = out["volume"].astype("float64")
    out["amount"] = out["amount"].astype("float64")
    out["tradestatus"] = out["tradestatus"].astype("int8")
    out["isST"] = out["isST"].astype("int8")
    out["adjustflag"] = out["adjustflag"].astype("int8")
    return out


def download_data(cfg: Config, outdir: Path, max_symbols: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    import baostock as bs  # type: ignore

    bs_login(bs)
    basic = query_with_retry(bs, "query_stock_basic", {})
    if basic.empty:
        raise RuntimeError("query_stock_basic returned no rows")
    basic.to_csv(outdir / "stock_basic_all.csv", index=False)
    for c in ["ipoDate", "outDate"]:
        basic[c] = pd.to_datetime(basic[c], errors="coerce")
    basic["board"] = basic["code"].map(classify_board)
    basic["type"] = pd.to_numeric(basic["type"], errors="coerce")
    overlap = (basic["ipoDate"].isna() | (basic["ipoDate"] <= pd.Timestamp(cfg.end_date))) & (
        basic["outDate"].isna() | (basic["outDate"] >= pd.Timestamp(cfg.read_start))
    )
    stocks = basic[(basic["type"] == 1) & basic["board"].notna() & overlap].copy()
    # BaoStock historically focuses on Shanghai/Shenzhen. Keep BSE only if actually returned.
    stocks = stocks.sort_values("code").reset_index(drop=True)
    if max_symbols:
        stocks = stocks.head(max_symbols).copy()
    stocks["sid"] = np.arange(len(stocks), dtype=np.int32)
    stocks.to_csv(outdir / "stock_universe_selected.csv", index=False)
    print(f"Selected stock_basic rows={len(stocks)} boards={stocks['board'].value_counts().to_dict()}", flush=True)

    frames: List[pd.DataFrame] = []
    factor_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    for i, row in stocks.iterrows():
        code = str(row["code"])
        try:
            raw = query_with_retry(bs, "query_history_k_data_plus", {
                "code": code,
                "fields": fields,
                "start_date": cfg.read_start,
                "end_date": cfg.end_date,
                "frequency": "d",
                "adjustflag": "2",
            })
            norm = normalize_daily(raw, int(row["sid"]), str(row["board"]), pd.Timestamp(row["ipoDate"]))
            if not norm.empty:
                frames.append(norm)
            else:
                failures.append({"code": code, "stage": "history", "error": "empty"})
            if cfg.factor_query:
                fac = query_with_retry(bs, "query_adjust_factor", {
                    "code": code,
                    "start_date": cfg.read_start,
                    "end_date": cfg.end_date,
                }, attempts=3)
                if not fac.empty:
                    fac["sid"] = int(row["sid"])
                    factor_frames.append(fac)
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "stage": "history_or_factor", "error": repr(exc)})
        if (i + 1) % 100 == 0 or (i + 1) == len(stocks):
            rows_so_far = sum(len(x) for x in frames)
            print(f"history progress {i+1}/{len(stocks)} rows={rows_so_far:,} failures={len(failures)}", flush=True)

    if not frames:
        raise RuntimeError("No BaoStock daily histories were downloaded")
    daily = pd.concat(frames, ignore_index=True).sort_values(["sid", "date"]).reset_index(drop=True)
    factors = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    if not factors.empty:
        factors.to_csv(outdir / "adjust_factors.csv.gz", index=False, compression="gzip")
    pd.DataFrame(failures).to_csv(outdir / "download_failures.csv", index=False)

    unique_dates = sorted(daily["date"].unique())
    snapshot_dates = choose_snapshot_dates([pd.Timestamp(d) for d in unique_dates], cfg.industry_snapshot)
    industry_frames: List[pd.DataFrame] = []
    for j, d in enumerate(snapshot_dates, 1):
        try:
            ind = query_with_retry(bs, "query_stock_industry", {"date": pd.Timestamp(d).strftime("%Y-%m-%d")}, attempts=3)
            if not ind.empty:
                ind["snapshot_date"] = pd.Timestamp(d)
                industry_frames.append(ind)
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": "ALL", "stage": f"industry_{d}", "error": repr(exc)})
        print(f"industry snapshot {j}/{len(snapshot_dates)} date={pd.Timestamp(d).date()} rows={len(industry_frames[-1]) if industry_frames else 0}", flush=True)
    industries = pd.concat(industry_frames, ignore_index=True) if industry_frames else pd.DataFrame()
    if not industries.empty:
        industries.to_csv(outdir / "industry_snapshots.csv.gz", index=False, compression="gzip")

    try:
        bs.logout()
    except Exception:
        pass

    matched_sids = set(daily["sid"].unique().tolist())
    selected = stocks.copy()
    selected["history_ok"] = selected["sid"].isin(matched_sids)
    audit = {
        "stock_basic_total_rows": int(len(basic)),
        "selected_stock_codes": int(len(stocks)),
        "history_success_codes": int(selected["history_ok"].sum()),
        "history_failure_codes": int((~selected["history_ok"]).sum()),
        "history_coverage": float(selected["history_ok"].mean()),
        "daily_rows": int(len(daily)),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "board_counts_selected": stocks["board"].value_counts().to_dict(),
        "board_counts_with_history": stocks[stocks["history_ok"]]["board"].value_counts().to_dict(),
        "delisted_codes_selected": int(stocks["outDate"].notna().sum()),
        "factor_rows": int(len(factors)),
        "factor_symbol_coverage": float(factors["sid"].nunique() / max(1, daily["sid"].nunique())) if not factors.empty else 0.0,
        "industry_snapshot_dates": int(industries["snapshot_date"].nunique()) if not industries.empty else 0,
    }
    return daily, stocks, industries, audit


def expanding_past_percentile(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    for i, v in enumerate(a):
        if not np.isfinite(v):
            continue
        past = a[max(0, i - window):i]
        past = past[np.isfinite(past)]
        if len(past) >= min_periods:
            out[i] = (np.sum(past <= v) + 0.5) / (len(past) + 1.0)
    return pd.Series(out, index=s.index)


def assign_point_in_time_industry(daily: pd.DataFrame, stocks: pd.DataFrame, industries: pd.DataFrame, outdir: Path) -> pd.DataFrame:
