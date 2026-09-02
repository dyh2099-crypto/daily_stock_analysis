#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-share short-term breakout backtest based on the Tianrongxin setup.

Primary universe: Shanghai/Shenzhen 10% price-limit main-board A shares.
Signal: a close above a compact 10-session resistance band, confirmed by
turnover/volume expansion, a strong close, and a positive MA20/MA60 trend.
Execution: signal at t close; buy no earlier than t+1 open. A conservative
variant waits up to 3 sessions for a low-volume retest that holds resistance.

The script downloads point-in-time daily bars from BaoStock, including
tradestatus and isST, then saves fully reproducible audit files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import baostock as bs
except ImportError as exc:
    raise SystemExit("baostock is required: pip install baostock==0.9.3") from exc

STOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,isST"
)
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
NUMERIC_FIELDS = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "tradestatus", "pctChg", "isST",
]

@dataclass(frozen=True)
class Config:
    start: str = "2023-09-01"
    end: str = "2026-08-31"
    warmup_start: str = "2022-12-01"
    min_list_sessions: int = 120
    coverage_window: int = 60
    min_trade_sessions: int = 54
    liquidity_median_amount: float = 50_000_000.0
    resistance_window: int = 10
    lower_band_window: int = 5
    max_band_width: float = 0.04
    amount_ratio: float = 1.50
    volume_ratio: float = 1.30
    min_breakout_buffer: float = 0.003
    max_breakout_extension: float = 0.040
    min_day_return: float = 0.015
    max_day_return: float = 0.102
    min_close_location: float = 0.70
    max_close_to_ma20: float = 1.12
    recent_pullback_window: int = 3
    min_pullback_to_resistance: float = 0.90
    max_pullback_to_lower_band: float = 0.995
    cooldown_bars: int = 10
    max_entry_above_resistance: float = 1.05
    min_entry_to_resistance: float = 0.985
    retest_window: int = 3
    retest_touch_above: float = 1.015
    retest_close_floor: float = 0.995
    retest_close_ceiling: float = 1.040
    retest_max_amount_fraction: float = 0.80
    retest_min_close_location: float = 0.50
    buy_commission_bp: float = 3.0
    buy_slippage_bp: float = 5.0
    sell_commission_bp: float = 3.0
    sell_slippage_bp: float = 5.0
    stamp_duty_bp: float = 5.0
    stop_loss: float = 0.03
    take_profit: float = 0.05
    max_holding_days: int = 5
    max_exit_delay: int = 5

    @property
    def buy_cost(self) -> float:
        return (self.buy_commission_bp + self.buy_slippage_bp) / 10_000.0

    @property
    def sell_cost(self) -> float:
        return (
            self.sell_commission_bp + self.sell_slippage_bp + self.stamp_duty_bp
        ) / 10_000.0

@dataclass
class DownloadAudit:
    baostock_version: str
    requested_start: str
    requested_end: str
    universe_count: int
    downloaded_count: int
    empty_count: int
    failed_count: int
    failures: List[Dict[str, str]]
    calendar_rows: int
    calendar_min: Optional[str]
    calendar_max: Optional[str]
    data_last_date_counts: Dict[str, int]

def log(message: str) -> None:
    print(message, flush=True)

def result_to_frame(rs) -> pd.DataFrame:
    rows: List[List[str]] = []
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock error {rs.error_code}: {rs.error_msg}")
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)

def login() -> None:
    response = bs.login()
    if response.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {response.error_code} {response.error_msg}")

def reconnect() -> None:
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(2)
    login()

def query_history(code: str, fields: str, start: str, end: str, adjustflag: str, retries: int = 5) -> pd.DataFrame:
    last: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=start, end_date=end,
                frequency="d", adjustflag=adjustflag,
            )
            return result_to_frame(rs)
        except BaseException as exc:
            last = exc
            if attempt == retries:
                break
            log(f"  query retry {attempt}/{retries} for {code}: {exc}")
            reconnect()
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"history query failed for {code}") from last

def is_mainboard_code(code: str) -> bool:
    code = str(code).lower().strip()
    return bool(re.fullmatch(r"sh\.60\d{4}", code) or re.fullmatch(r"sz\.00\d{4}", code))

def query_basic_universe() -> pd.DataFrame:
    basic = result_to_frame(bs.query_stock_basic())
    if basic.empty:
        raise RuntimeError("query_stock_basic returned no rows")
    for col in basic.columns:
        basic[col] = basic[col].astype(str).str.strip()
    if "type" in basic.columns:
        basic = basic[basic["type"].eq("1")]
    return basic[basic["code"].map(is_mainboard_code)].drop_duplicates("code", keep="last")

def query_market_calendar(start: str, end: str) -> pd.DataFrame:
    frame = query_history("sh.000001", INDEX_FIELDS, start, end, adjustflag="3")
    if frame.empty:
        raise RuntimeError("failed to obtain SSE trading calendar")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

def query_historical_universe(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    snapshots: List[pd.DataFrame] = []
    sample_dates = list(calendar[::20])
    if len(calendar) and (not sample_dates or sample_dates[-1] != calendar[-1]):
        sample_dates.append(calendar[-1])
    for i, dt in enumerate(sample_dates, 1):
        ds = dt.strftime("%Y-%m-%d")
        try:
            frame = result_to_frame(bs.query_all_stock(day=ds))
            if not frame.empty and "code" in frame.columns:
                frame = frame[frame["code"].map(is_mainboard_code)].copy()
                frame["snapshot_date"] = ds
                snapshots.append(frame)
        except Exception as exc:
            log(f"snapshot universe failed at {ds}: {exc}")
        if i % 10 == 0:
            log(f"historical universe snapshots: {i}/{len(sample_dates)}")
    if not snapshots:
        return pd.DataFrame(columns=["code", "code_name", "snapshot_date"])
    return pd.concat(snapshots, ignore_index=True, sort=False).drop_duplicates("code", keep="last")

def canonical_cache_path(cache_dir: Path, code: str) -> Path:
    return cache_dir / "bars" / f"{code.replace('.', '_')}.csv.gz"

def normalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in NUMERIC_FIELDS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return (
        frame.dropna(subset=["date", "code"])
        .sort_values("date")
        .drop_duplicates(["code", "date"], keep="last")
    )

def cached_file_is_complete(path: Path, end: str) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        dates = pd.read_csv(path, usecols=["date"], compression="gzip")
        return not dates.empty and str(dates["date"].iloc[-1]) >= end
    except Exception:
        return False

def download_bars(cfg: Config, cache_dir: Path, max_codes: Optional[int] = None) -> DownloadAudit:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "bars").mkdir(parents=True, exist_ok=True)
    login()
    try:
        calendar_df = query_market_calendar(cfg.warmup_start, cfg.end)
        calendar = pd.DatetimeIndex(calendar_df["date"])
        basic = query_basic_universe()
        historical = query_historical_universe(calendar)
        codes = sorted(set(basic["code"].tolist()) | set(historical.get("code", [])))
        codes = [c for c in codes if is_mainboard_code(c)]
        if max_codes is not None:
            codes = (["sz.002212"] + [c for c in codes if c != "sz.002212"])[:max_codes]
        log(f"main-board universe: {len(codes)} codes")
        basic.to_csv(cache_dir / "stock_basic.csv", index=False, encoding="utf-8-sig")
        historical.to_csv(cache_dir / "historical_universe.csv", index=False, encoding="utf-8-sig")
        calendar_df.to_csv(cache_dir / "market_calendar.csv", index=False, encoding="utf-8-sig")
        failures: List[Dict[str, str]] = []
        downloaded = empty = 0
        last_date_counts: Dict[str, int] = {}
        for i, code in enumerate(codes, 1):
            path = canonical_cache_path(cache_dir, code)
            if cached_file_is_complete(path, cfg.end):
                try:
                    last_date = str(pd.read_csv(path, usecols=["date"], compression="gzip")["date"].iloc[-1])
                    last_date_counts[last_date] = last_date_counts.get(last_date, 0) + 1
                except Exception:
                    pass
                continue
            try:
                frame = query_history(code, STOCK_FIELDS, cfg.warmup_start, cfg.end, adjustflag="2")
                if frame.empty:
                    empty += 1
                    continue
                frame = normalize_bar_frame(frame)
                if frame.empty:
                    empty += 1
                    continue
                frame.to_csv(path, index=False, compression="gzip")
                downloaded += 1
                last_date = frame["date"].max().strftime("%Y-%m-%d")
                last_date_counts[last_date] = last_date_counts.get(last_date, 0) + 1
            except Exception as exc:
                failures.append({"code": code, "error": repr(exc)})
                log(f"FAILED {code}: {exc}")
            if i == 1 or i % 50 == 0 or i == len(codes):
                log(f"download progress {i}/{len(codes)}; new={downloaded}, empty={empty}, failed={len(failures)}")
        audit = DownloadAudit(
            baostock_version=str(getattr(bs, "__version__", "unknown")),
            requested_start=cfg.warmup_start,
            requested_end=cfg.end,
            universe_count=len(codes),
            downloaded_count=downloaded,
            empty_count=empty,
            failed_count=len(failures),
            failures=failures,
            calendar_rows=len(calendar),
            calendar_min=None if not len(calendar) else calendar.min().strftime("%Y-%m-%d"),
            calendar_max=None if not len(calendar) else calendar.max().strftime("%Y-%m-%d"),
            data_last_date_counts=dict(sorted(last_date_counts.items())),
        )
        (cache_dir / "download_audit.json").write_text(
            json.dumps(asdict(audit), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return audit
    finally:
        try:
            bs.logout()
        except Exception:
            pass

def load_cached_metadata(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    basic = pd.read_csv(cache_dir / "stock_basic.csv", dtype=str)
    hist_path = cache_dir / "historical_universe.csv"
    historical = pd.read_csv(hist_path, dtype=str) if hist_path.exists() else pd.DataFrame()
    cal = pd.read_csv(cache_dir / "market_calendar.csv")
    dates = pd.to_datetime(cal["date"], errors="coerce").dropna().sort_values().drop_duplicates()
    return basic, historical, pd.DatetimeIndex(dates)

def safe_close_location(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    span = high - low
    out = (close - low) / span.replace(0, np.nan)
    return out.where(span > 0, 1.0).clip(0, 1)

def locked_limit_bar(row: pd.Series, previous_close: float, direction: str) -> bool:
    if not np.isfinite(previous_close) or previous_close <= 0:
        return False
    values = [float(row.get(k, np.nan)) for k in ("open", "high", "low", "close")]
    if not all(np.isfinite(v) for v in values):
        return False
    one_price = max(values) / min(values) - 1 <= 0.0005
    gap = values[0] / previous_close - 1
    return one_price and (gap >= 0.095 if direction == "up" else gap <= -0.095)

def next_tradable_position(cal_frame: pd.DataFrame, start_pos: int, max_delay: int) -> Optional[int]:
    last = min(len(cal_frame) - 1, start_pos + max_delay)
    for pos in range(start_pos, last + 1):
        if bool(cal_frame.iloc[pos].get("trade_ok", False)):
            return pos
    return None

def after_cost_return(entry: float, exit_: float, cfg: Config) -> float:
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0 or exit_ <= 0:
        return math.nan
    return (exit_ * (1.0 - cfg.sell_cost)) / (entry * (1.0 + cfg.buy_cost)) - 1.0

def build_trade_outcomes(cal_frame: pd.DataFrame, entry_pos: int, cfg: Config, benchmark: Optional[pd.DataFrame]) -> Dict[str, object]:
    entry = float(cal_frame.iloc[entry_pos]["open"])
    result: Dict[str, object] = {
        "entry_date": cal_frame.index[entry_pos].strftime("%Y-%m-%d"),
        "entry_open": entry,
    }
    def benchmark_return(exit_date: pd.Timestamp) -> float:
        if benchmark is None or benchmark.empty:
            return math.nan
        entry_date = cal_frame.index[entry_pos]
        if entry_date not in benchmark.index or exit_date not in benchmark.index:
            return math.nan
        bo = float(benchmark.loc[entry_date, "open"])
        bx = float(benchmark.loc[exit_date, "close"])
        return bx / bo - 1.0 if np.isfinite(bo) and np.isfinite(bx) and bo > 0 else math.nan
    for horizon in (1, 3, 5):
        scheduled = entry_pos + horizon
        exit_pos = None if scheduled >= len(cal_frame) else next_tradable_position(cal_frame, scheduled, cfg.max_exit_delay)
        if exit_pos is None:
            result.update({f"ret_{horizon}d": math.nan, f"exit_{horizon}d": None, f"bench_{horizon}d": math.nan, f"excess_{horizon}d": math.nan})
            continue
        exit_date = cal_frame.index[exit_pos]
        ret = after_cost_return(entry, float(cal_frame.iloc[exit_pos]["close"]), cfg)
        bret = benchmark_return(exit_date)
        result.update({
            f"ret_{horizon}d": ret,
            f"exit_{horizon}d": exit_date.strftime("%Y-%m-%d"),
            f"bench_{horizon}d": bret,
            f"excess_{horizon}d": ret - bret if np.isfinite(bret) else math.nan,
        })
    stop, target = entry * (1.0 - cfg.stop_loss), entry * (1.0 + cfg.take_profit)
    end_pos = min(len(cal_frame) - 1, entry_pos + cfg.max_holding_days)
    risk_exit_pos: Optional[int] = None
    risk_exit_price = math.nan
    reason = "time"
    for pos in range(entry_pos + 1, end_pos + 1):
        row = cal_frame.iloc[pos]
        if not bool(row.get("trade_ok", False)):
            continue
        op, lo, hi = float(row["open"]), float(row["low"]), float(row["high"])
        if op <= stop:
            risk_exit_pos, risk_exit_price, reason = pos, op, "gap_stop"; break
        if op >= target:
            risk_exit_pos, risk_exit_price, reason = pos, op, "gap_target"; break
        low_hit, high_hit = lo <= stop, hi >= target
        if low_hit and high_hit:
            risk_exit_pos, risk_exit_price, reason = pos, stop, "both_stop_first"; break
        if low_hit:
            risk_exit_pos, risk_exit_price, reason = pos, stop, "stop"; break
        if high_hit:
            risk_exit_pos, risk_exit_price, reason = pos, target, "target"; break
    if risk_exit_pos is None:
        risk_exit_pos = next_tradable_position(cal_frame, end_pos, cfg.max_exit_delay)
        if risk_exit_pos is not None:
            risk_exit_price = float(cal_frame.iloc[risk_exit_pos]["close"])
    if risk_exit_pos is None:
        result.update({"risk_ret": math.nan, "risk_exit_date": None, "risk_exit_reason": "no_exit", "risk_benchmark": math.nan, "risk_excess": math.nan})
    else:
        risk_date = cal_frame.index[risk_exit_pos]
        rr = after_cost_return(entry, risk_exit_price, cfg)
        br = benchmark_return(risk_date)
        result.update({
            "risk_ret": rr,
            "risk_exit_date": risk_date.strftime("%Y-%m-%d"),
            "risk_exit_reason": reason,
            "risk_benchmark": br,
            "risk_excess": rr - br if np.isfinite(br) else math.nan,
        })
    path = cal_frame.iloc[entry_pos:end_pos + 1]
    lows, highs = pd.to_numeric(path["low"], errors="coerce"), pd.to_numeric(path["high"], errors="coerce")
    result["mae_5d"] = float(lows.min() / entry - 1) if lows.notna().any() else math.nan
    result["mfe_5d"] = float(highs.max() / entry - 1) if highs.notna().any() else math.nan
    return result

def prepare_benchmark(cache_dir: Path, cfg: Config, calendar: pd.DatetimeIndex) -> Tuple[str, pd.DataFrame]:
    login()
    chosen, benchmark = "none", pd.DataFrame()
    try:
        for code in ("sh.000985", "sh.000300", "sh.000001"):
            try:
                frame = normalize_bar_frame(query_history(code, INDEX_FIELDS, cfg.warmup_start, cfg.end, adjustflag="3"))
                if len(frame) >= int(len(calendar) * 0.8):
                    chosen = code
                    benchmark = frame.set_index("date").reindex(calendar)
                    break
            except Exception as exc:
                log(f"benchmark candidate {code} failed: {exc}")
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if not benchmark.empty:
        benchmark[["open", "close"]] = benchmark[["open", "close"]].apply(pd.to_numeric, errors="coerce")
        benchmark.to_csv(cache_dir / f"benchmark_{chosen.replace('.', '_')}.csv", encoding="utf-8-sig")
    return chosen, benchmark

def metadata_maps(basic: pd.DataFrame, historical: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, pd.Timestamp]]:
    name_map: Dict[str, str] = {}
    ipo_map: Dict[str, pd.Timestamp] = {}
    for frame in (historical, basic):
        if frame is None or frame.empty or "code" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            code = str(row.get("code", "")).strip()
            if not code:
                continue
            if "code_name" in frame.columns and pd.notna(row.get("code_name")):
                name_map[code] = str(row.get("code_name")).strip()
            if "ipoDate" in frame.columns and pd.notna(row.get("ipoDate")):
                value = pd.to_datetime(row.get("ipoDate"), errors="coerce")
                if pd.notna(value):
                    ipo_map[code] = value
    return name_map, ipo_map

def add_signal_features(valid: pd.DataFrame, cfg: Config, resistance_window: int, amount_ratio: float) -> pd.DataFrame:
    x = valid.copy()
    x["bar_no"] = np.arange(1, len(x) + 1)
    x["ret_day"] = x["close"].pct_change(fill_method=None)
    x["ma20"] = x["close"].rolling(20, min_periods=20).mean()
    x["ma60"] = x["close"].rolling(60, min_periods=60).mean()
    x["resistance"] = x["high"].shift(1).rolling(resistance_window, min_periods=resistance_window).max()
    x["lower_band"] = x["close"].shift(1).rolling(cfg.lower_band_window, min_periods=cfg.lower_band_window).max()
    x["amount_med20"] = x["amount"].shift(1).rolling(20, min_periods=20).median()
    x["amount_ratio"] = x["amount"] / x["amount_med20"].replace(0, np.nan)
    x["volume_avg20"] = x["volume"].shift(1).rolling(20, min_periods=20).mean()
    x["volume_ratio"] = x["volume"] / x["volume_avg20"].replace(0, np.nan)
    x["close_location"] = safe_close_location(x["high"], x["low"], x["close"])
    x["band_width"] = x["resistance"] / x["lower_band"].replace(0, np.nan) - 1.0
    x["recent_min_close"] = x["close"].shift(1).rolling(cfg.recent_pullback_window, min_periods=cfg.recent_pullback_window).min()
    x["recent_mean_amount"] = x["amount"].shift(1).rolling(cfg.recent_pullback_window, min_periods=cfg.recent_pullback_window).mean()
    base = (
        x["eligible_common"]
        & (x["close"] > x["resistance"] * (1.0 + cfg.min_breakout_buffer))
        & (x["close"] <= x["resistance"] * (1.0 + cfg.max_breakout_extension))
        & x["ret_day"].between(0.0, cfg.max_day_return)
        & ~x["one_price_limit_up"]
    )
    confirmed = (
        base
        & x["ret_day"].between(cfg.min_day_return, cfg.max_day_return)
        & (x["amount_ratio"] >= amount_ratio)
        & (x["volume_ratio"] >= cfg.volume_ratio)
        & (x["close_location"] >= cfg.min_close_location)
        & (x["close"] > x["ma20"])
        & (x["ma20"] > x["ma60"])
        & (x["close"] / x["ma20"] <= cfg.max_close_to_ma20)
        & x["band_width"].between(0.0, cfg.max_band_width)
        & (x["recent_min_close"] >= x["resistance"] * cfg.min_pullback_to_resistance)
        & (x["recent_min_close"] <= x["lower_band"] * cfg.max_pullback_to_lower_band)
        & (x["recent_mean_amount"] < x["amount"])
    )
    x["signal_simple"] = base.fillna(False)
    x["signal_confirmed"] = confirmed.fillna(False)
    return x

def apply_cooldown(signal: pd.Series, bars: pd.Series, cooldown: int) -> List[int]:
    positions: List[int] = []
    last_bar = -10**9
    for idx in np.flatnonzero(signal.to_numpy(dtype=bool)):
        bar = int(bars.iloc[idx])
        if bar - last_bar > cooldown:
            positions.append(int(idx)); last_bar = bar
    return positions

def build_calendar_frame(raw: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    raw = raw.copy().set_index("date").sort_index()
    cal = raw[~raw.index.duplicated(keep="last")].reindex(calendar)
    for col in NUMERIC_FIELDS:
        if col in cal.columns:
            cal[col] = pd.to_numeric(cal[col], errors="coerce")
    cal["trade_ok"] = (
        cal.get("tradestatus", pd.Series(1.0, index=cal.index)).fillna(0).eq(1)
        & cal["open"].gt(0) & cal["high"].gt(0) & cal["low"].gt(0)
        & cal["close"].gt(0) & cal["volume"].gt(0) & cal["amount"].gt(0)
    )
    cal["coverage_60"] = cal["trade_ok"].rolling(60, min_periods=1).sum()
    return cal

def event_entry_ok(cal: pd.DataFrame, entry_pos: int, resistance: float, cfg: Config) -> bool:
    if entry_pos <= 0 or entry_pos >= len(cal):
        return False
    row = cal.iloc[entry_pos]
    if not bool(row.get("trade_ok", False)):
        return False
    prev_close = float(cal.iloc[entry_pos - 1].get("close", math.nan))
    if locked_limit_bar(row, prev_close, "up") or locked_limit_bar(row, prev_close, "down"):
        return False
    op = float(row["open"])
    return resistance * cfg.min_entry_to_resistance <= op <= resistance * cfg.max_entry_above_resistance

def process_stock(raw: pd.DataFrame, code: str, name: str, ipo_date: Optional[pd.Timestamp], calendar: pd.DatetimeIndex, cfg: Config, benchmark: Optional[pd.DataFrame]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    cal = build_calendar_frame(raw, calendar)
    valid = cal[cal["trade_ok"]].copy()
    if valid.empty or len(valid) < 80:
        return [], {"raw": len(raw), "valid": len(valid), "eligible": 0, "signals": 0}
    valid["coverage_60"] = cal.loc[valid.index, "coverage_60"]
    valid["is_st_flag"] = valid.get("isST", 0).fillna(0).ne(0)
    if ipo_date is not None and pd.notna(ipo_date):
        valid["listed_sessions"] = np.searchsorted(calendar.values, valid.index.values) - np.searchsorted(calendar.values, np.datetime64(ipo_date), side="left") + 1
    else:
        valid["listed_sessions"] = np.arange(1, len(valid) + 1)
    prev_close = valid["close"].shift(1)
    one_price = (valid["high"] / valid["low"].replace(0, np.nan) - 1).abs() <= 0.0005
    valid["one_price_limit_up"] = one_price & (valid["open"] / prev_close - 1 >= 0.095)
    valid["eligible_common"] = (
        ~valid["is_st_flag"]
        & (valid["coverage_60"] >= cfg.min_trade_sessions)
        & (valid["listed_sessions"] >= cfg.min_list_sessions)
    )
    variants = [
        ("simple_10d_next_open", 10, 1.50, "simple", "next"),
        ("confirm_10d_next_open", 10, 1.50, "confirmed", "next"),
        ("confirm_10d_retest", 10, 1.50, "confirmed", "retest"),
        ("confirm_10d_vol2_next_open", 10, 2.00, "confirmed", "next"),
        ("confirm_20d_next_open", 20, 1.50, "confirmed", "next"),
    ]
    trades: List[Dict[str, object]] = []
    total_signals = eligible_count = 0
    start_ts, end_ts = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    for strategy, window, amt_ratio, signal_kind, entry_kind in variants:
        feat = add_signal_features(valid, cfg, window, amt_ratio)
        feat["eligible_common"] &= feat["amount_med20"] >= cfg.liquidity_median_amount
        feat = add_signal_features(feat, cfg, window, amt_ratio)
        eligible_count = max(eligible_count, int(feat["eligible_common"].sum()))
        signal_col = "signal_simple" if signal_kind == "simple" else "signal_confirmed"
        candidate_positions = apply_cooldown(feat[signal_col], feat["bar_no"], cfg.cooldown_bars)
        total_signals += len(candidate_positions)
        for feat_pos in candidate_positions:
            signal_date = feat.index[feat_pos]
            if not (start_ts <= signal_date <= end_ts):
                continue
            resistance = float(feat.iloc[feat_pos]["resistance"])
            signal_amount = float(feat.iloc[feat_pos]["amount"])
            global_signal_pos = int(calendar.get_indexer([signal_date])[0])
            if global_signal_pos < 0:
                continue
            retest_date: Optional[pd.Timestamp] = None
            if entry_kind == "next":
                entry_pos = global_signal_pos + 1
                if not event_entry_ok(cal, entry_pos, resistance, cfg):
                    continue
            else:
                entry_pos = -1
                for offset in range(1, cfg.retest_window + 1):
                    rp = global_signal_pos + offset
                    if rp >= len(cal):
                        break
                    row = cal.iloc[rp]
                    if not bool(row.get("trade_ok", False)):
                        continue
                    span = float(row["high"] - row["low"])
                    close_loc = 1.0 if span <= 0 else float((row["close"] - row["low"]) / span)
                    holds = (
                        float(row["low"]) <= resistance * cfg.retest_touch_above
                        and float(row["close"]) >= resistance * cfg.retest_close_floor
                        and float(row["close"]) <= resistance * cfg.retest_close_ceiling
                        and float(row["amount"]) <= signal_amount * cfg.retest_max_amount_fraction
                        and close_loc >= cfg.retest_min_close_location
                    )
                    if holds and event_entry_ok(cal, rp + 1, resistance, cfg):
                        retest_date, entry_pos = calendar[rp], rp + 1
                        break
                if entry_pos < 0:
                    continue
            if entry_pos + cfg.max_holding_days >= len(cal):
                continue
            outcome = build_trade_outcomes(cal, entry_pos, cfg, benchmark)
            row = feat.iloc[feat_pos]
            record: Dict[str, object] = {
                "strategy": strategy, "code": code, "name": name,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "retest_date": None if retest_date is None else retest_date.strftime("%Y-%m-%d"),
                "resistance_window": window, "resistance": resistance,
                "lower_band": float(row["lower_band"]), "signal_close": float(row["close"]),
                "signal_return": float(row["ret_day"]), "amount_ratio": float(row["amount_ratio"]),
                "volume_ratio": float(row["volume_ratio"]), "close_location": float(row["close_location"]),
                "band_width": float(row["band_width"]), "ma20": float(row["ma20"]),
                "ma60": float(row["ma60"]), "listed_sessions": int(row["listed_sessions"]),
                "coverage_60": int(row["coverage_60"]),
            }
            record.update(outcome)
            trades.append(record)
    return trades, {"raw": int(len(raw)), "valid": int(len(valid)), "eligible": int(eligible_count), "signals": int(total_signals)}

def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = wins / n; den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)

def summarize_return(df: pd.DataFrame, col: str, excess_col: Optional[str] = None) -> Dict[str, object]:
    r = pd.to_numeric(df[col], errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    wins = int((r > 0).sum()); lo, hi = wilson_interval(wins, len(r))
    pos, neg = r[r > 0], r[r <= 0]
    profit_factor = pos.sum() / abs(neg.sum()) if len(neg) and neg.sum() < 0 else math.inf
    out: Dict[str, object] = {
        "n": int(len(r)), "win_rate": float(wins / len(r)), "win_ci_low": lo, "win_ci_high": hi,
        "mean_return": float(r.mean()), "median_return": float(r.median()),
        "mean_win": float(pos.mean()) if len(pos) else math.nan,
        "mean_loss": float(neg.mean()) if len(neg) else math.nan,
        "payoff_ratio": float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) and neg.mean() < 0 else math.nan,
        "profit_factor": float(profit_factor), "p_ge_3pct": float((r >= 0.03).mean()),
        "p_le_minus3pct": float((r <= -0.03).mean()),
    }
    basket = df.assign(_r=pd.to_numeric(df[col], errors="coerce")).groupby("entry_date", observed=True)["_r"].mean().dropna()
    out.update({
        "signal_day_basket_n": int(len(basket)),
        "signal_day_basket_win_rate": float((basket > 0).mean()) if len(basket) else math.nan,
        "signal_day_basket_mean": float(basket.mean()) if len(basket) else math.nan,
    })
    if excess_col and excess_col in df.columns:
        ex = pd.to_numeric(df[excess_col], errors="coerce").dropna()
        out.update({
            "mean_excess": float(ex.mean()) if len(ex) else math.nan,
            "median_excess": float(ex.median()) if len(ex) else math.nan,
            "excess_win_rate": float((ex > 0).mean()) if len(ex) else math.nan,
        })
    return out

def fiscal_period_label(date: pd.Timestamp) -> str:
    year = date.year if date.month >= 9 else date.year - 1
    return f"{year}-09~{year + 1}-08"

def make_summaries(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []
    reason_rows: List[Dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    trades = trades.copy()
    trades["period"] = pd.to_datetime(trades["entry_date"]).map(fiscal_period_label)
    specs = [("1d", "ret_1d", "excess_1d"), ("3d", "ret_3d", "excess_3d"), ("5d", "ret_5d", "excess_5d"), ("risk_3stop_5target", "risk_ret", "risk_excess")]
    for strategy, group in trades.groupby("strategy", observed=True):
        for horizon, col, excol in specs:
            summary_rows.append({"strategy": strategy, "horizon": horizon, **summarize_return(group, col, excol)})
        for reason, n in group["risk_exit_reason"].value_counts(dropna=False).items():
            reason_rows.append({"strategy": strategy, "risk_exit_reason": str(reason), "n": int(n), "fraction": float(n / len(group))})
        for period, pg in group.groupby("period", observed=True):
            for horizon, col, excol in specs:
                period_rows.append({"strategy": strategy, "period": period, "horizon": horizon, **summarize_return(pg, col, excol)})
    return pd.DataFrame(summary_rows), pd.DataFrame(period_rows), pd.DataFrame(reason_rows)

def make_markdown_report(cfg: Config, summary: pd.DataFrame, periods: pd.DataFrame, audit: Dict[str, object], benchmark_code: str) -> str:
    lines = [
        "# 天融信式放量突破策略：A股主板三年回测", "",
        f"- 回测区间：{cfg.start} 至 {cfg.end}；预热数据始于 {cfg.warmup_start}。",
        "- 股票池：沪深 10% 涨跌幅主板 A 股；逐日剔除 ST、停牌/零成交、上市不足 120 个市场交易日、近 60 个市场日可交易不足 54 日，以及 20 日成交额中位数低于 5000 万元的股票。",
        "- 信号时点：t 日收盘确认；最早 t+1 开盘买入，不使用同一收盘价成交。",
        f"- 成本：买入 {cfg.buy_cost:.2%}，卖出 {cfg.sell_cost:.2%}，往返约 {(cfg.buy_cost + cfg.sell_cost):.2%}。",
        f"- 对照基准：{benchmark_code}。", "", "## 预注册交易规则", "",
        "主信号 `confirm_10d_next_open`：收盘突破过去 10 个交易日最高价 0.3% 以上、但不超过 4%；当日成交额至少为此前 20 日中位数 1.5 倍，成交量至少为此前 20 日均量 1.3 倍；收盘位于日内振幅上部 30%；MA20>MA60；价格距离 MA20 不超过 12%；过去 5 日收盘高点与 10 日最高价构成宽度不超过 4% 的压力带，且前三日曾回踩但未远离压力带。", "",
        "`confirm_10d_retest` 在主信号后再等 1—3 日：股价回踩压力位附近并收回，回踩日成交额不超过突破日的 80%，随后下一交易日开盘买入。", "",
        "固定持有口径分别在入场后第 1、3、5 个市场交易日收盘退出；风控口径遵守 T+1，从次日起执行 -3% 止损、+5% 止盈，最多持有 5 个市场交易日；同日同时触发止盈止损时，保守按先止损处理。", "", "## 汇总结果", "",
    ]
    if summary.empty:
        lines.append("无可用结果。")
    else:
        cols = ["strategy", "horizon", "n", "win_rate", "win_ci_low", "win_ci_high", "mean_return", "median_return", "payoff_ratio", "profit_factor", "signal_day_basket_win_rate", "mean_excess", "excess_win_rate"]
        lines.append(summary[[c for c in cols if c in summary.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## 分年度稳定性", ""]
    if not periods.empty:
        primary = periods[periods["strategy"].isin(["confirm_10d_next_open", "confirm_10d_retest"]) & periods["horizon"].isin(["3d", "risk_3stop_5target"])]
        cols = ["strategy", "period", "horizon", "n", "win_rate", "mean_return", "median_return", "profit_factor", "mean_excess"]
        lines.append(primary[[c for c in cols if c in primary.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## 数据与执行审计", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2, default=str), "```", "", "说明：交易级胜率会受到同日行业/题材信号相关性的影响，因此同时给出按入场日等权合并后的‘信号日篮子胜率’。单看胜率不足以证明策略有效，还需同时看扣费后期望收益、盈亏比、利润因子、年度稳定性和相对基准超额。"]
    return "\n".join(lines)

def run_backtest(cfg: Config, cache_dir: Path, output_dir: Path, max_codes: Optional[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    download_bars(cfg, cache_dir, max_codes=max_codes)
    basic, historical, calendar = load_cached_metadata(cache_dir)
    name_map, ipo_map = metadata_maps(basic, historical)
    benchmark_code, benchmark = prepare_benchmark(cache_dir, cfg, calendar)
    bar_files = sorted((cache_dir / "bars").glob("*.csv.gz"))
    all_trades: List[Dict[str, object]] = []
    stock_audit_rows: List[Dict[str, object]] = []
    for i, path in enumerate(bar_files, 1):
        raw = normalize_bar_frame(pd.read_csv(path, compression="gzip", low_memory=False))
        if raw.empty:
            continue
        code = str(raw["code"].iloc[0])
        trades, aud = process_stock(raw, code, name_map.get(code, ""), ipo_map.get(code), calendar, cfg, benchmark if not benchmark.empty else None)
        all_trades.extend(trades)
        stock_audit_rows.append({"code": code, "name": name_map.get(code, ""), **aud})
        if i == 1 or i % 100 == 0 or i == len(bar_files):
            log(f"backtest progress {i}/{len(bar_files)}; trades={len(all_trades)}")
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["strategy", "entry_date", "code"]).reset_index(drop=True)
    stock_audit = pd.DataFrame(stock_audit_rows)
    summary, periods, reasons = make_summaries(trades_df)
    trades_df.to_csv(output_dir / "01_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "02_summary.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(output_dir / "03_period_stability.csv", index=False, encoding="utf-8-sig")
    reasons.to_csv(output_dir / "04_risk_exit_reasons.csv", index=False, encoding="utf-8-sig")
    stock_audit.to_csv(output_dir / "05_stock_audit.csv", index=False, encoding="utf-8-sig")
    download_audit_path = cache_dir / "download_audit.json"
    download_audit = json.loads(download_audit_path.read_text(encoding="utf-8")) if download_audit_path.exists() else {}
    run_audit: Dict[str, object] = {
        "config": asdict(cfg), "python": sys.version, "pandas": pd.__version__,
        "numpy": np.__version__, "baostock": getattr(bs, "__version__", "unknown"),
        "benchmark_code": benchmark_code,
        "calendar_start": calendar.min().strftime("%Y-%m-%d") if len(calendar) else None,
        "calendar_end": calendar.max().strftime("%Y-%m-%d") if len(calendar) else None,
        "bar_files": len(bar_files), "trade_rows": len(trades_df), "stock_audit_rows": len(stock_audit),
        "download": download_audit,
    }
    (output_dir / "00_run_audit.json").write_text(json.dumps(run_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = make_markdown_report(cfg, summary, periods, run_audit, benchmark_code)
    (output_dir / "06_report.md").write_text(report, encoding="utf-8")
    if max_codes is None:
        if len(bar_files) < 1500:
            raise RuntimeError(f"too few main-board stock files: {len(bar_files)}")
        if len(trades_df) < 100:
            raise RuntimeError(f"too few strategy trades: {len(trades_df)}")
    log(report)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-09-01")
    p.add_argument("--end", default="2026-08-31")
    p.add_argument("--warmup-start", default="2022-12-01")
    p.add_argument("--cache-dir", default=".cache/baostock_mainboard_20260901")
    p.add_argument("--output-dir", default="artifacts/tianrongxin_breakout_backtest")
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = Config(start=args.start, end=args.end, warmup_start=args.warmup_start)
    try:
        run_backtest(cfg, Path(args.cache_dir), Path(args.output_dir), args.max_codes)
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
