#!/usr/bin/env python3
"""TDX-only fallback for the five Youzi-sayings strategy proxies.

The script reuses the strategy engine in ``youzi_nonst_daily_backtest.py`` but
replaces BaoStock data access with the official TongdaXin complete daily
package. It is intentionally conservative: suspected historical ST securities
are removed using repeated 5% price-limit behaviour; raw-price corporate-action
discontinuities contaminate the following 60 sessions; and the fifth strategy
is labelled a market-wide new-leadership-cohort proxy rather than a true
point-in-time concept-theme rotation test.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).with_name("youzi_nonst_daily_backtest.py")
spec = importlib.util.spec_from_file_location("youzi_core", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import core backtest")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

TDX_ROOT = Path(os.environ.get("TDX_ROOT", "/tmp/tdx"))
DAY = struct.Struct("<5If2I")
CODE_RE = re.compile(r"^(sh|sz|bj)(\d{6})\.day$", re.I)
AUDIT_EXTRA: Dict[str, object] = {}


class _OkLogin:
    error_code = "0"
    error_msg = "TDX-only mode; BaoStock login bypassed"


def classify_tdx(exchange: str, digits: str) -> Optional[str]:
    return core.classify_board(f"{exchange.lower()}.{digits}")


def iter_stock_files() -> Iterator[Tuple[Path, str, str]]:
    rows: List[Tuple[Path, str, str]] = []
    for path in TDX_ROOT.rglob("*.day"):
        match = CODE_RE.match(path.name)
        if not match:
            continue
        exchange, digits = match.group(1).lower(), match.group(2)
        board = classify_tdx(exchange, digits)
        if board is None:
            continue
        rows.append((path, f"{exchange}.{digits}", board))
    rows.sort(key=lambda x: x[1])
    yield from rows


def read_day(path: Path, start: int = 20230103, end: int = 20260902) -> pd.DataFrame:
    raw = path.read_bytes()
    usable = len(raw) - len(raw) % DAY.size
    records: List[Tuple[object, ...]] = []
    append = records.append
    for offset in range(0, usable, DAY.size):
        date, op, hi, lo, close, amount, volume, _ = DAY.unpack_from(raw, offset)
        if date < start or date > end or min(op, hi, lo, close) <= 0:
            continue
        append((date, op / 100.0, hi / 100.0, lo / 100.0, close / 100.0,
                float(amount), float(volume)))
    if not records:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "amount", "volume"])
    frame = pd.DataFrame.from_records(
        records,
        columns=["date", "open", "high", "low", "close", "amount", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def fetch_basic_override() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for n, (path, code, board) in enumerate(iter_stock_files(), 1):
        day = read_day(path)
        if day.empty:
            continue
        rows.append({
            "code": code, "code_name": "", "ipoDate": day["date"].min(),
            "outDate": pd.NaT, "type": "1", "status": "", "board": board,
            "path": str(path), "first_date": day["date"].min(),
            "last_date": day["date"].max(),
        })
        if n % 1000 == 0:
            print(f"MASTER {n} files scanned", flush=True)
    master = pd.DataFrame(rows)
    if master.empty:
        raise RuntimeError("TDX package contained no usable A-share day files")
    AUDIT_EXTRA["tdx_scanned_stock_files"] = int(len(master))
    AUDIT_EXTRA["tdx_first_available_date"] = str(master["first_date"].min().date())
    AUDIT_EXTRA["tdx_last_available_date"] = str(master["last_date"].max().date())
    AUDIT_EXTRA["delisted_or_inactive_by_last_date"] = int((master["last_date"] < core.SIGNAL_END).sum())
    return master.sort_values("code").reset_index(drop=True)


def rounded_price(series: pd.Series) -> pd.Series:
    return np.floor(series * 100.0 + 0.5) / 100.0


def suspected_historical_st(day: pd.DataFrame, board: str) -> Tuple[bool, int, int]:
    if board not in ("sh_main", "sz_main") or len(day) < 2:
        return False, 0, 0
    d = day.copy()
    d["preclose"] = d["close"].shift(1)
    d = d[d["date"].between(core.SIGNAL_START, pd.Timestamp("2026-07-03")) & d["preclose"].notna()]
    if d.empty:
        return False, 0, 0
    up5 = rounded_price(d["preclose"] * 1.05)
    down5 = rounded_price(d["preclose"] * 0.95)
    tol = 0.011
    up_hit = (d["close"].sub(up5).abs() <= tol) & (d["high"].sub(up5).abs() <= tol)
    down_hit = (d["close"].sub(down5).abs() <= tol) & (d["low"].sub(down5).abs() <= tol)
    capped = up_hit | down_hit
    one_price = capped & (d["high"].sub(d["low"]).abs() <= tol)
    capped_count = int(capped.sum())
    one_price_count = int(one_price.sum())
    return bool(capped_count >= 2 or one_price_count >= 1), capped_count, one_price_count


def fetch_history_override(basic: pd.DataFrame, max_symbols: int, progress_every: int):
    histories: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    ever_st: List[str] = []
    st_evidence_rows: List[Dict[str, object]] = []
    contaminated_rows = 0
    work = basic.head(max_symbols) if max_symbols > 0 else basic
    for n, row in enumerate(work.itertuples(index=False), 1):
        try:
            day = read_day(Path(row.path))
        except Exception as exc:
            failures.append({"code": row.code, "stage": "tdx_read", "error": repr(exc)})
            continue
        if day.empty:
            failures.append({"code": row.code, "stage": "tdx_read", "error": "empty"})
            continue
        is_st, capped_count, one_price_count = suspected_historical_st(day, row.board)
        if is_st:
            ever_st.append(str(row.code))
            st_evidence_rows.append({
                "code": row.code, "five_pct_capped_sessions": capped_count,
                "five_pct_one_price_sessions": one_price_count,
            })
            continue
        day["code"] = row.code
        day["preclose"] = day["close"].shift(1)
        day["pctChg"] = (day["close"] / day["preclose"] - 1.0) * 100.0
        day["tradestatus"] = (day["volume"] > 0).astype(int)
        day["isST"] = 0
        day["adjustflag"] = 3
        day["turn"] = np.nan
        day["board"] = row.board
        day["ipoDate"] = row.ipoDate
        day["outDate"] = row.outDate
        day["code_name"] = ""
        gap = (day["open"] / day["preclose"] - 1.0).abs()
        threshold = {"sh_main": 0.12, "sz_main": 0.12, "star": 0.22,
                     "chinext": 0.22, "bse": 0.32}[row.board]
        discontinuity = gap > threshold
        day["corp_action_recent60"] = discontinuity.shift(1).rolling(60, min_periods=1).max().fillna(0).astype(bool)
        contaminated_rows += int(day["corp_action_recent60"].sum())
        day = day[day["preclose"].notna()].copy()
        histories.append(day)
        if progress_every and n % progress_every == 0:
            print(f"TDX {n}/{len(work)} histories={len(histories)} suspected_st={len(ever_st)} failures={len(failures)}", flush=True)
    if not histories:
        raise RuntimeError("no usable TDX histories after filtering")
    AUDIT_EXTRA["suspected_historical_st_removed"] = int(len(ever_st))
    AUDIT_EXTRA["st_proxy_evidence_rows"] = st_evidence_rows[:100]
    AUDIT_EXTRA["corporate_action_contaminated_rows"] = int(contaminated_rows)
    raw = pd.concat(histories, ignore_index=True)
    return raw, pd.DataFrame(), pd.DataFrame(failures), ever_st, 0, int(len(work))


def empty_industries(_periods) -> pd.DataFrame:
    return pd.DataFrame(columns=["code", "industry", "period", "requested_date", "snapshot_fresh"])


def build_marketwide_signals(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_cols = ["date", "sentiment", "sentiment_prev5_max", "q_market_ret10",
                   "q_limitdown_stress", "market_regime", "median_ret3"]
    df = df.merge(market[market_cols], on="date", how="left")
    for src, dst in (("ret3", "rank_ret3"), ("limit5", "rank_limit5"), ("amount3", "rank_amount3")):
        df[dst] = df.groupby("date")[src].rank(pct=True, method="average")
    df["leader_score"] = df[["rank_ret3", "rank_limit5", "rank_amount3"]].mean(axis=1)
    df["leader_percentile"] = df.groupby("date")["leader_score"].rank(pct=True, method="average")
    df["leader_top"] = (df["leader_percentile"] >= 0.995) & ((df["limit5"] >= 1) | (df["ret3"] >= 0.08))
    df = df.sort_values(["code", "date"])
    df["was_leader5"] = df.groupby("code")["leader_top"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).max()).fillna(0).astype(bool)
    df["prior20_leadermax"] = df.groupby("code")["leader_percentile"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    df["new_leader"] = (df["leader_percentile"] >= 0.98) & (df["prior20_leadermax"] < 0.80) & (df["ret3"] > 0)
    new_count = df.groupby("date")["new_leader"].transform("sum")
    df["new_leader_cohort"] = df["new_leader"] & (new_count >= 3)
    df["stock_ret10_percentile"] = df.groupby("date")["ret10"].rank(pct=True, method="average")
    df["drawdown_atr"] = (df["recent_high5"] - df["close"]) / df["atr14"].replace(0, np.nan)
    contamination = df.get("corp_action_recent60", pd.Series(False, index=df.index)).fillna(True).astype(bool)
    eligible = (df["date"].between(core.SIGNAL_START, core.SIGNAL_END)
                & (df["listing_days"] >= 180) & (df["amount_med20"] >= 50_000_000)
                & (df["close"] >= 2) & df["ma20"].notna() & (~contamination))
    df["sig_leader_continuation"] = eligible & (df["sentiment"] >= 60) & df["leader_top"] & (df["ret1"] > 0) & (df["close"] > df["ma20"])
    df["sig_pullback_rebound"] = eligible & df["sentiment"].between(35, 60, inclusive="left") & (df["sentiment_prev5_max"] >= 60) & df["was_leader5"] & df["drawdown_atr"].between(1.0, 2.5) & df["prev_down_count3"].between(1, 3) & (df["ret1"] > 0) & (df["close_location"] >= 0.60)
    df["sig_panic_rebound_confirmation"] = eligible & (df["sentiment"] < 35) & (df["q_market_ret10"] <= 0.20) & (df["q_limitdown_stress"] >= 0.95) & (df["stock_ret10_percentile"] <= 0.10) & (df["ret1"] > 0) & (df["close"] > df["open"]) & (~df["limit_down"])
    df["sig_second_board_after_close"] = eligible & (df["sentiment"] >= 60) & df["limit_up"] & df["prev_limit_up"] & (~df["one_price_up"])
    df["sig_new_hot_industry_proxy"] = eligible & (df["sentiment"] >= 60) & df["new_leader_cohort"]
    df["industry"] = "MARKET_WIDE_PROXY"
    df["hot_score"] = df["leader_score"]
    df["hot_percentile"] = df["leader_percentile"]
    df["hot"] = df["leader_percentile"] >= 0.98
    df["new_hot"] = df["new_leader_cohort"]
    df["leader_top2"] = df["leader_top"]
    return df


def patch_outputs(out: Path) -> None:
    audit_path = out / "data_audit.json"
    headline_path = out / "headline.json"
    if not audit_path.exists() or not headline_path.exists():
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(AUDIT_EXTRA)
    audit.update({
        "requested_scope": "TDX-available Shanghai, Shenzhen and Beijing A-share day files; historical inactive files retained",
        "data_source": "TongdaXin official hsjday complete daily package",
        "daily_security_status": "volume>0 proxy; no vendor point-in-time status table",
        "st_exclusion_rule": "remove whole main-board security after >=2 reconstructed 5% capped closes or >=1 one-price 5% session in 2023-09-04..2026-07-03",
        "price_adjustment": "raw TDX prices; signals blocked for 60 sessions after abnormal overnight discontinuity; no complete adjustment-factor table",
        "adjust_factor_query_success_stocks": 0,
        "adjust_factor_rows": 0,
        "limit_price_method": "board-rule reconstruction from prior raw close; not official daily stk_limit table",
        "theme_method": "market-wide new leadership cohort proxy; not point-in-time concept-theme membership",
    })
    audit_path.write_text(json.dumps(core.json_clean(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    headline["audit"] = audit
    headline_path.write_text(json.dumps(core.json_clean(headline), ensure_ascii=False, indent=2), encoding="utf-8")
    report = out / "REPORT.md"
    if report.exists():
        report.write_text(report.read_text(encoding="utf-8") + "\n\n## TDX-only correction\n\nThis run did not use BaoStock. Suspected historical ST securities were removed using repeated 5% capped-price behaviour. The new-theme strategy is a market-wide new-leadership-cohort proxy and must not be interpreted as a historical concept-theme test.\n", encoding="utf-8")


def main() -> int:
    core.bs.login = lambda: _OkLogin()
    core.bs.logout = lambda: None
    core.fetch_basic = fetch_basic_override
    core.fetch_history_and_factors = fetch_history_override
    core.fetch_industry_periods = empty_industries
    core.build_industry_and_signals = build_marketwide_signals
    rc = core.main()
    output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    patch_outputs(output_dir)
    if (output_dir / "headline.json").exists():
        print("===TDX_ONLY_HEADLINE_BEGIN===")
        print((output_dir / "headline.json").read_text(encoding="utf-8"))
        print("===TDX_ONLY_HEADLINE_END===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
