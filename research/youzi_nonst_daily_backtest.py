#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import baostock as bs
import numpy as np
import pandas as pd

READ_START = "2023-01-03"
SIGNAL_START = pd.Timestamp("2023-09-04")
SIGNAL_END = pd.Timestamp("2026-09-02")
QUERY_END = "2026-09-02"

BUY_COST = 0.0003 + 0.0005 + 0.00001
SELL_COST = 0.0003 + 0.0005 + 0.0005 + 0.00001

STRATEGIES = {
    "leader_continuation": {"hold": 5, "stop": -0.03, "target": 0.05},
    "pullback_rebound": {"hold": 3, "stop": -0.03, "target": 0.05},
    "panic_rebound_confirmation": {"hold": 2, "stop": -0.03, "target": 0.04},
    "second_board_after_close": {"hold": 3, "stop": -0.03, "target": 0.05},
    "new_hot_industry_proxy": {"hold": 5, "stop": -0.03, "target": 0.05},
}


def consume(rs) -> pd.DataFrame:
    rows: List[List[str]] = []
    if rs is None or str(getattr(rs, "error_code", "-1")) != "0":
        return pd.DataFrame()
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=list(rs.fields)) if rows else pd.DataFrame(columns=list(rs.fields))


def classify_board(code: str) -> Optional[str]:
    c = str(code).lower()
    digits = c.split(".")[-1]
    if c.startswith("sh."):
        if digits.startswith(("600", "601", "603", "605")):
            return "sh_main"
        if digits.startswith(("688", "689")):
            return "star"
    if c.startswith("sz."):
        if digits.startswith(("000", "001", "002", "003")):
            return "sz_main"
        if digits.startswith(("300", "301")):
            return "chinext"
    if c.startswith("bj.") and digits.startswith(("4", "8", "9")):
        return "bse"
    return None


def limit_ratio(board: str) -> float:
    if board in ("star", "chinext"):
        return 0.20
    if board == "bse":
        return 0.30
    return 0.10


def rolling_last_percentile(s: pd.Series, window: int = 120, min_periods: int = 60) -> pd.Series:
    def f(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return np.nan
        return float(np.mean(a <= a[-1]))
    return s.rolling(window, min_periods=min_periods).apply(f, raw=True)


def wilson(success: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return center - half, center + half


def fetch_basic() -> pd.DataFrame:
    df = consume(bs.query_stock_basic())
    if df.empty:
        raise RuntimeError("query_stock_basic returned no rows")
    for col in ("type", "status"):
        if col not in df.columns:
            df[col] = ""
    df = df[df["type"].astype(str).eq("1")].copy()
    df["board"] = df["code"].map(classify_board)
    df = df[df["board"].notna()].copy()
    for col in ("ipoDate", "outDate"):
        if col not in df.columns:
            df[col] = ""
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.drop_duplicates("code").reset_index(drop=True)


def fetch_history_and_factors(basic: pd.DataFrame, max_symbols: int, progress_every: int):
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    histories: List[pd.DataFrame] = []
    factors: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    ever_st: List[str] = []
    factor_query_success = 0
    codes = basic["code"].tolist()
    if max_symbols > 0:
        codes = codes[:max_symbols]
    basic_map = basic.set_index("code")
    for idx, code in enumerate(codes, 1):
        rs = bs.query_history_k_data_plus(
            code, fields, start_date=READ_START, end_date=QUERY_END,
            frequency="d", adjustflag="2"
        )
        h = consume(rs)
        if h.empty:
            failures.append({"code": code, "stage": "history", "error": str(getattr(rs, "error_msg", "empty"))})
            continue
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "tradestatus", "pctChg", "isST"):
            h[col] = pd.to_numeric(h[col], errors="coerce")
        h = h[h["date"].notna()].copy()
        if h.empty:
            failures.append({"code": code, "stage": "history_parse", "error": "no valid dates"})
            continue
        sample_mask = h["date"].between(pd.Timestamp(READ_START), SIGNAL_END)
        if bool((h.loc[sample_mask, "isST"].fillna(0) == 1).any()):
            ever_st.append(code)
            continue
        meta = basic_map.loc[code]
        h["board"] = str(meta["board"])
        h["ipoDate"] = meta["ipoDate"]
        h["outDate"] = meta["outDate"]
        h["code_name"] = str(meta.get("code_name", ""))
        histories.append(h)

        frs = bs.query_adjust_factor(code, start_date=READ_START, end_date=QUERY_END)
        if str(getattr(frs, "error_code", "-1")) == "0":
            factor_query_success += 1
            f = consume(frs)
            if not f.empty:
                f["queried_code"] = code
                factors.append(f)
        else:
            failures.append({"code": code, "stage": "adjust_factor", "error": str(getattr(frs, "error_msg", "failed"))})
        if progress_every and idx % progress_every == 0:
            print(f"FETCH {idx}/{len(codes)} histories={len(histories)} ever_st={len(ever_st)} failures={len(failures)}", flush=True)
    if not histories:
        raise RuntimeError("No usable non-ST history")
    return pd.concat(histories, ignore_index=True), (pd.concat(factors, ignore_index=True) if factors else pd.DataFrame()), pd.DataFrame(failures), ever_st, factor_query_success, len(codes)


def fetch_industry_periods(periods: Iterable[pd.Period]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    last_valid: Dict[str, str] = {}
    for p in periods:
        query_date = p.start_time.strftime("%Y-%m-%d")
        rs = bs.query_stock_industry(date=query_date)
        snap = consume(rs)
        current: Dict[str, str] = {}
        if not snap.empty and "code" in snap.columns and "industry" in snap.columns:
            if "updateDate" in snap.columns:
                upd = pd.to_datetime(snap["updateDate"], errors="coerce")
                snap = snap[(upd.isna()) | (upd <= p.end_time)]
            snap = snap[snap["industry"].astype(str).str.len() > 0]
            current = dict(zip(snap["code"].astype(str), snap["industry"].astype(str)))
            if current:
                last_valid = current
        if last_valid:
            part = pd.DataFrame({"code": list(last_valid.keys()), "industry": list(last_valid.values())})
            part["period"] = str(p)
            part["requested_date"] = query_date
            part["snapshot_fresh"] = bool(current)
            rows.append(part)
        print(f"INDUSTRY {p} rows={len(current)} carried={len(last_valid)}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["code", "industry", "period", "requested_date", "snapshot_fresh"])


def prepare_data(raw: pd.DataFrame, industry_periods: pd.DataFrame):
    audit_status_rows = len(raw)
    raw = raw.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    raw["status_valid"] = raw["tradestatus"].eq(1)
    df = raw[raw["status_valid"] & raw["close"].gt(0) & raw["preclose"].gt(0) & raw["volume"].gt(0)].copy()
    df["limit_ratio"] = df["board"].map(limit_ratio)
    df["ret1"] = df["pctChg"] / 100.0
    missing_ret = ~np.isfinite(df["ret1"])
    df.loc[missing_ret, "ret1"] = df.loc[missing_ret, "close"] / df.loc[missing_ret, "preclose"] - 1
    df["limit_up"] = df["ret1"] >= (df["limit_ratio"] - 0.003)
    df["limit_down"] = df["ret1"] <= (-df["limit_ratio"] + 0.003)
    df["touched_up"] = df["high"] / df["preclose"] >= (1 + df["limit_ratio"] - 0.003)
    df["one_price_up"] = df["limit_up"] & np.isclose(df["high"], df["low"], rtol=0, atol=1e-8)
    df["one_price_down"] = df["limit_down"] & np.isclose(df["high"], df["low"], rtol=0, atol=1e-8)
    df["derived_limit_up"] = df["preclose"] * (1 + df["limit_ratio"])
    df["derived_limit_down"] = df["preclose"] * (1 - df["limit_ratio"])
    df["listing_days"] = (df["date"] - df["ipoDate"]).dt.days
    df["bar_range"] = df["high"] - df["low"]
    df["close_location"] = (df["close"] - df["low"]) / df["bar_range"].replace(0, np.nan)

    g = df.groupby("code", sort=False)
    df["ret3"] = g["close"].pct_change(3)
    df["ret10"] = g["close"].pct_change(10)
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["ma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    df["amount_med20"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).median())
    df["amount3"] = g["amount"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    df["limit5"] = g["limit_up"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["preclose"]).abs(),
        (df["low"] - df["preclose"]).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=10).mean())
    df["recent_high5"] = g["high"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).max())
    df["prev_limit_up"] = g["limit_up"].shift(1).fillna(False).astype(bool)
    df["prev_down_count3"] = g["ret1"].transform(lambda s: s.shift(1).lt(0).rolling(3, min_periods=1).sum())

    df["period"] = df["date"].dt.to_period("M").astype(str)
    if not industry_periods.empty:
        ind = industry_periods[["code", "period", "industry", "snapshot_fresh"]].drop_duplicates(["code", "period"], keep="last")
        df = df.merge(ind, on=["code", "period"], how="left")
    else:
        df["industry"] = np.nan
        df["snapshot_fresh"] = False
    return df, audit_status_rows


def build_market(df: pd.DataFrame) -> pd.DataFrame:
    base = df.groupby("date").agg(
        stock_count=("code", "size"),
        limit_up_count=("limit_up", "sum"),
        touched_up_count=("touched_up", "sum"),
        limit_down_count=("limit_down", "sum"),
        median_ret10=("ret10", "median"),
        median_ret3=("ret3", "median"),
    ).reset_index()
    base["broken_rate"] = (base["touched_up_count"] - base["limit_up_count"]) / base["touched_up_count"].replace(0, np.nan)
    base["limit_down_share"] = base["limit_down_count"] / base["stock_count"].replace(0, np.nan)
    prev = df[df["prev_limit_up"]].groupby("date").agg(
        prev_limit_median_return=("ret1", "median"),
        promotion_rate=("limit_up", "mean"),
        prev_limit_count=("code", "size"),
    ).reset_index()
    m = base.merge(prev, on="date", how="left").sort_values("date")
    for c in ("prev_limit_median_return", "promotion_rate", "broken_rate", "limit_down_share"):
        m[c] = m[c].fillna(0.0)
    m["q_positive_feedback"] = rolling_last_percentile(m["prev_limit_median_return"])
    m["q_promotion"] = rolling_last_percentile(m["promotion_rate"])
    m["q_broken"] = rolling_last_percentile(m["broken_rate"])
    m["q_limitdown"] = rolling_last_percentile(m["limit_down_share"])
    m["sentiment"] = 25 * (m["q_positive_feedback"] + m["q_promotion"] + 1 - m["q_broken"] + 1 - m["q_limitdown"])
    m["sentiment_prev5_max"] = m["sentiment"].shift(1).rolling(5, min_periods=1).max()
    m["q_market_ret10"] = rolling_last_percentile(m["median_ret10"])
    m["q_limitdown_stress"] = rolling_last_percentile(m["limit_down_share"])
    m["market_regime"] = np.select([m["sentiment"] >= 60, m["sentiment"] < 35], ["strong", "weak"], default="divergent")
    return m


def build_industry_and_signals(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(market[["date", "sentiment", "sentiment_prev5_max", "q_market_ret10", "q_limitdown_stress", "market_regime", "median_ret3"]], on="date", how="left")
    known = df[df["industry"].notna()].copy()
    if not known.empty:
        ind = known.groupby(["date", "industry"]).agg(
            industry_size=("code", "size"),
            industry_ret3=("ret3", "median"),
            industry_limitups=("limit_up", "sum"),
            industry_amount=("amount", "sum"),
        ).reset_index().sort_values(["industry", "date"])
        ind["industry_excess3"] = ind["industry_ret3"] - ind.merge(market[["date", "median_ret3"]], on="date", how="left")["median_ret3"].to_numpy()
        ind["industry_limit_density"] = ind["industry_limitups"] / ind["industry_size"].replace(0, np.nan)
        ind["industry_amount_med20"] = ind.groupby("industry")["industry_amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
        ind["industry_activity"] = ind["industry_amount"] / ind["industry_amount_med20"].replace(0, np.nan)
        for src, dst in (("industry_excess3", "rank_excess"), ("industry_limit_density", "rank_limit"), ("industry_activity", "rank_activity")):
            ind[dst] = ind.groupby("date")[src].rank(pct=True, method="average")
        ind["hot_score"] = ind[["rank_excess", "rank_limit", "rank_activity"]].mean(axis=1)
        ind["hot_percentile"] = ind.groupby("date")["hot_score"].rank(pct=True, method="average")
        ind["prior20_hotmax"] = ind.groupby("industry")["hot_percentile"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
        ind["hot"] = ind["hot_percentile"] >= 0.90
        ind["new_hot"] = ind["hot"] & (ind["prior20_hotmax"] < 0.80) & (ind["industry_limitups"] >= 3)
        keep = ["date", "industry", "industry_size", "industry_limitups", "hot_score", "hot_percentile", "hot", "new_hot"]
        df = df.merge(ind[keep], on=["date", "industry"], how="left")
        df["rank_ret3"] = df.groupby(["date", "industry"])["ret3"].rank(pct=True, method="average")
        df["rank_limit5"] = df.groupby(["date", "industry"])["limit5"].rank(pct=True, method="average")
        df["rank_amount3"] = df.groupby(["date", "industry"])["amount3"].rank(pct=True, method="average")
        df["leader_score"] = df[["rank_ret3", "rank_limit5", "rank_amount3"]].mean(axis=1)
        df["leader_order"] = df.groupby(["date", "industry"])["leader_score"].rank(ascending=False, method="first")
        df["leader_top2"] = (df["leader_order"] <= 2) & (df["industry_size"] >= 3)
    else:
        for c in ("industry_size", "industry_limitups", "hot_score", "hot_percentile", "leader_score", "leader_order"):
            df[c] = np.nan
        df["hot"] = False
        df["new_hot"] = False
        df["leader_top2"] = False
    df["hot"] = df["hot"].fillna(False).astype(bool)
    df["new_hot"] = df["new_hot"].fillna(False).astype(bool)
    df["leader_top2"] = df["leader_top2"].fillna(False).astype(bool)
    df = df.sort_values(["code", "date"])
    df["was_leader5"] = df.groupby("code")["leader_top2"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).max()).fillna(0).astype(bool)
    df["stock_ret10_percentile"] = df.groupby("date")["ret10"].rank(pct=True, method="average")
    df["drawdown_atr"] = (df["recent_high5"] - df["close"]) / df["atr14"].replace(0, np.nan)
    eligible = (
        df["date"].between(SIGNAL_START, SIGNAL_END)
        & (df["listing_days"] >= 180)
        & (df["amount_med20"] >= 50_000_000)
        & (df["close"] >= 2)
        & df["ma20"].notna()
    )
    df["sig_leader_continuation"] = eligible & (df["sentiment"] >= 60) & df["hot"] & df["leader_top2"] & (df["close"] > df["preclose"]) & (df["close"] > df["ma20"])
    df["sig_pullback_rebound"] = eligible & df["sentiment"].between(35, 60, inclusive="left") & (df["sentiment_prev5_max"] >= 60) & df["was_leader5"] & df["drawdown_atr"].between(1.0, 2.5) & df["prev_down_count3"].between(1, 3) & (df["ret1"] > 0) & (df["close_location"] >= 0.60)
    df["sig_panic_rebound_confirmation"] = eligible & (df["sentiment"] < 35) & (df["q_market_ret10"] <= 0.20) & (df["q_limitdown_stress"] >= 0.95) & (df["stock_ret10_percentile"] <= 0.10) & (df["ret1"] > 0) & (df["close"] > df["open"]) & (~df["limit_down"])
    df["sig_second_board_after_close"] = eligible & (df["sentiment"] >= 60) & df["hot"] & df["limit_up"] & df["prev_limit_up"] & (~df["one_price_up"])
    df["sig_new_hot_industry_proxy"] = eligible & (df["sentiment"] >= 60) & df["new_hot"] & df["leader_top2"]
    return df


def net_return(entry: float, exit_price: float) -> float:
    return float(exit_price * (1 - SELL_COST) / (entry * (1 + BUY_COST)) - 1)


def simulate_one(code_df: pd.DataFrame, signal_i: int, strategy: str):
    cfg = STRATEGIES[strategy]
    if signal_i + 1 >= len(code_df):
        return None, "no_next_session"
    signal = code_df.iloc[signal_i]
    entry_i = signal_i + 1
    entry_row = code_df.iloc[entry_i]
    entry = float(entry_row["open"])
    if not np.isfinite(entry) or entry <= 0 or float(entry_row["volume"]) <= 0:
        return None, "invalid_entry"
    if bool(entry_row["one_price_up"]):
        return None, "entry_one_price_limit_up"
    entry_gap = entry / float(entry_row["preclose"]) - 1
    if entry_gap >= float(entry_row["limit_ratio"]) - 0.003:
        return None, "entry_at_limit_up"
    if strategy == "second_board_after_close" and entry_gap > 0.05:
        return None, "second_board_gap_over_5pct"

    stop = entry * (1 + float(cfg["stop"]))
    target = entry * (1 + float(cfg["target"]))
    if strategy == "pullback_rebound":
        signal_low = float(signal["low"])
        if signal_low < entry:
            stop = max(signal_low, entry * 0.95)
        recent_high = float(signal["recent_high5"])
        if np.isfinite(recent_high) and recent_high > entry * 1.005:
            target = recent_high
    elif strategy == "panic_rebound_confirmation":
        signal_low = float(signal["low"])
        if signal_low < entry:
            stop = max(signal_low, entry * 0.94)
        atr = float(signal["atr14"])
        atr_ret = atr / entry if np.isfinite(atr) and entry > 0 else 0.04
        target = entry * (1 + min(0.08, max(0.03, atr_ret)))

    planned_last = min(len(code_df) - 1, entry_i + int(cfg["hold"]) - 1)
    if planned_last <= entry_i:
        return None, "insufficient_future"
    exit_i: Optional[int] = None
    exit_price = np.nan
    reason = ""
    for k in range(entry_i + 1, planned_last + 1):
        row = code_df.iloc[k]
        op, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
        if bool(row["one_price_down"]):
            continue
        if op <= stop:
            exit_i, exit_price, reason = k, op, "stop_gap"
            break
        if op >= target:
            exit_i, exit_price, reason = k, op, "target_gap"
            break
        stop_hit, target_hit = lo <= stop, hi >= target
        if stop_hit and target_hit:
            exit_i, exit_price, reason = k, stop, "both_touched_stop_first"
            break
        if stop_hit:
            exit_i, exit_price, reason = k, stop, "stop"
            break
        if target_hit:
            exit_i, exit_price, reason = k, target, "target"
            break
    if exit_i is None:
        for k in range(planned_last, min(len(code_df), planned_last + 21)):
            row = code_df.iloc[k]
            if not bool(row["one_price_down"]):
                exit_i, exit_price = k, float(row["close"])
                reason = "time_exit" if k == planned_last else "time_exit_deferred_limitdown"
                break
    if exit_i is None:
        return None, "unresolved_limitdown"
    score = float(signal.get("leader_score", np.nan))
    if strategy == "panic_rebound_confirmation":
        score = -float(signal.get("ret10", 0.0))
    return {
        "strategy": strategy,
        "code": str(signal["code"]),
        "code_name": str(signal.get("code_name", "")),
        "board": str(signal["board"]),
        "industry": str(signal.get("industry", "")),
        "signal_date": signal["date"],
        "entry_date": entry_row["date"],
        "exit_date": code_df.iloc[exit_i]["date"],
        "entry_price": entry,
        "exit_price": float(exit_price),
        "net_return": net_return(entry, float(exit_price)),
        "exit_reason": reason,
        "holding_sessions": int(exit_i - entry_i + 1),
        "signal_score": score if np.isfinite(score) else 0.0,
        "sentiment": float(signal.get("sentiment", np.nan)),
        "market_regime": str(signal.get("market_regime", "unknown")),
        "hot_score": float(signal.get("hot_score", np.nan)),
    }, "ok"


def build_trades(df: pd.DataFrame):
    trades: List[Dict[str, object]] = []
    skipped: Dict[str, Dict[str, int]] = {s: {} for s in STRATEGIES}
    signal_cols = {s: f"sig_{s}" for s in STRATEGIES}
    candidate_counts = {s: int(df[c].fillna(False).sum()) for s, c in signal_cols.items()}
    for n, (code, cdf0) in enumerate(df.groupby("code", sort=False), 1):
        cdf = cdf0.sort_values("date").reset_index(drop=True)
        for strategy, col in signal_cols.items():
            indices = np.flatnonzero(cdf[col].fillna(False).to_numpy(bool))
            blocked_until = -1
            for i0 in indices:
                i = int(i0)
                if i <= blocked_until:
                    skipped[strategy]["overlap_same_stock"] = skipped[strategy].get("overlap_same_stock", 0) + 1
                    continue
                trade, reason = simulate_one(cdf, i, strategy)
                if trade is None:
                    skipped[strategy][reason] = skipped[strategy].get(reason, 0) + 1
                    continue
                blocked_until = int(cdf.index[cdf["date"].eq(trade["exit_date"])][0])
                trades.append(trade)
        if n % 500 == 0:
            print(f"SIM {n} stocks trades={len(trades)}", flush=True)
    t = pd.DataFrame(trades)
    if not t.empty:
        t = t.sort_values(["strategy", "entry_date", "signal_score"], ascending=[True, True, False]).reset_index(drop=True)
    return t, skipped, candidate_counts


def summarize_group(x: pd.DataFrame) -> Dict[str, object]:
    n = len(x)
    if n == 0:
        return {"trades": 0}
    r = x["net_return"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    low, high = wilson(int((r > 0).sum()), n)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": n,
        "win_rate": float((r > 0).mean()),
        "win_rate_ci95_low": low,
        "win_rate_ci95_high": high,
        "mean_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "positive_3pct_rate": float((r >= 0.03).mean()),
        "negative_3pct_rate": float((r <= -0.03).mean()),
        "median_holding_sessions": float(x["holding_sessions"].median()),
        "target_exit_rate": float(x["exit_reason"].str.startswith("target").mean()),
        "stop_exit_rate": float(x["exit_reason"].str.contains("stop").mean()),
    }


def build_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for strategy, all_s in trades.groupby("strategy"):
        for universe, x in (
            ("all_available_boards", all_s),
            ("sh_sz_mainboard", all_s[all_s["board"].isin(["sh_main", "sz_main"])]),
        ):
            row = {"strategy": strategy, "universe": universe}
            row.update(summarize_group(x))
            rows.append(row)
    return pd.DataFrame(rows)


def build_annual(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    rows = []
    for (strategy, year), x in t.groupby(["strategy", "year"]):
        for universe, y in (("all_available_boards", x), ("sh_sz_mainboard", x[x["board"].isin(["sh_main", "sz_main"])])):
            row = {"strategy": strategy, "year": int(year), "universe": universe}
            row.update(summarize_group(y))
            rows.append(row)
    return pd.DataFrame(rows)


def build_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, regime), x in trades.groupby(["strategy", "market_regime"]):
        for universe, y in (("all_available_boards", x), ("sh_sz_mainboard", x[x["board"].isin(["sh_main", "sz_main"])])):
            row = {"strategy": strategy, "market_regime": regime, "universe": universe}
            row.update(summarize_group(y))
            rows.append(row)
    return pd.DataFrame(rows)


def slot_portfolio(trades: pd.DataFrame, slots: int = 10) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for strategy, all_s in trades.groupby("strategy"):
        for universe, x in (("all_available_boards", all_s), ("sh_sz_mainboard", all_s[all_s["board"].isin(["sh_main", "sz_main"])])):
            capitals = np.full(slots, 1.0 / slots)
            available = np.array([np.datetime64("1900-01-01")] * slots)
            accepted = 0
            x = x.sort_values(["entry_date", "signal_score"], ascending=[True, False])
            for tr in x.itertuples(index=False):
                entry_date = np.datetime64(pd.Timestamp(tr.entry_date).date())
                candidates = np.flatnonzero(available < entry_date)
                if len(candidates) == 0:
                    continue
                slot = int(candidates[np.argmax(capitals[candidates])])
                capitals[slot] *= 1 + float(tr.net_return)
                available[slot] = np.datetime64(pd.Timestamp(tr.exit_date).date())
                accepted += 1
            terminal = float(capitals.sum())
            years = max((SIGNAL_END - SIGNAL_START).days / 365.25, 0.01)
            rows.append({
                "strategy": strategy, "universe": universe, "slots": slots,
                "accepted_trades": accepted, "terminal_equity": terminal,
                "total_return": terminal - 1, "annualized_return": terminal ** (1 / years) - 1,
                "note": "10 independent equal-capital slots; realized at exit; no mark-to-market drawdown",
            })
    return pd.DataFrame(rows)


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    return obj


def write_report(out: Path, audit: Dict[str, object], summary: pd.DataFrame, portfolio: pd.DataFrame):
    lines = [
        "# A股游资语录五策略：非ST日线近似三年回测",
        "",
        f"- 信号期：{SIGNAL_START.date()} 至 {SIGNAL_END.date()}。",
        "- 数据：BaoStock前复权日线、逐日isST与tradestatus；样本期任一日为ST的股票整只剔除。",
        "- 涨跌停价：因无供应商逐日stk_limit表，按当日板块规则和前收盘重建；上市不足180日全部剔除。",
        "- 题材：使用历史行业分类作为题材代理，不能等同于实时概念题材。",
        "- 信号：收盘后确认，下一可交易日开盘成交；T+1；一字涨停不买、一字跌停不强平。",
        "",
        "## 数据审计",
        "",
        "```json",
        json.dumps(json_clean(audit), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 策略汇总",
        "",
        summary.to_markdown(index=False) if not summary.empty else "无交易结果。",
        "",
        "## 10槽位资金约束近似",
        "",
        portfolio.to_markdown(index=False) if not portfolio.empty else "无组合结果。",
        "",
        "## 解释边界",
        "",
        "这是日线可执行代理回测，不是分钟级排队成交重放。二板策略在第二个涨停收盘确认后，次日才尝试买入；行业策略不是概念题材事件回测。",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    login = bs.login()
    if str(getattr(login, "error_code", "-1")) != "0":
        raise RuntimeError(f"baostock login failed: {getattr(login, 'error_msg', '')}")
    try:
        basic = fetch_basic()
        basic.to_csv(out / "security_master.csv", index=False)
        raw, factors, failures, ever_st, factor_success, attempted = fetch_history_and_factors(basic, args.max_symbols, args.progress_every)
        if not factors.empty:
            factors.to_csv(out / "adjust_factors.csv.gz", index=False, compression="gzip")
        failures.to_csv(out / "data_failures.csv", index=False)
        pd.DataFrame({"code": ever_st}).to_csv(out / "ever_st_removed.csv", index=False)
        periods = pd.period_range(pd.Period(READ_START, freq="M"), pd.Period(SIGNAL_END, freq="M"), freq="M")
        industry_periods = fetch_industry_periods(periods)
        industry_periods.to_csv(out / "industry_snapshots.csv.gz", index=False, compression="gzip")
        df, raw_status_rows = prepare_data(raw, industry_periods)
        market = build_market(df)
        market.to_csv(out / "market_sentiment.csv", index=False)
        df = build_industry_and_signals(df, market)
        trades, skipped, candidates = build_trades(df)
        summary = build_summary(trades)
        annual = build_annual(trades)
        regime = build_regime(trades)
        portfolio = slot_portfolio(trades)
        trades.to_csv(out / "trades.csv.gz", index=False, compression="gzip")
        summary.to_csv(out / "summary.csv", index=False)
        annual.to_csv(out / "annual_summary.csv", index=False)
        regime.to_csv(out / "sentiment_regime_summary.csv", index=False)
        portfolio.to_csv(out / "portfolio_slots.csv", index=False)
        (out / "skip_reasons.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"strategy": k, "raw_candidates": v} for k, v in candidates.items()]).to_csv(out / "candidate_counts.csv", index=False)

        latest = df["date"].max()
        basic_attempted = basic.head(attempted) if args.max_symbols > 0 else basic
        delisted = int(basic_attempted["outDate"].notna().sum())
        board_counts = df.groupby("board")["code"].nunique().to_dict()
        industry_known = float(df["industry"].notna().mean())
        audit = {
            "requested_scope": "all A shares; actual boards reported below",
            "period": {"read_start": READ_START, "signal_start": str(SIGNAL_START.date()), "signal_end": str(SIGNAL_END.date())},
            "basic_stock_count": int(len(basic_attempted)),
            "history_attempted": attempted,
            "non_st_history_success": int(df["code"].nunique()),
            "ever_st_stocks_removed": int(len(ever_st)),
            "delisted_records_in_basic": delisted,
            "board_stock_counts": board_counts,
            "raw_history_rows_before_tradestatus_filter": int(raw_status_rows),
            "usable_trading_rows": int(len(df)),
            "latest_data_date": str(pd.Timestamp(latest).date()),
            "reached_requested_end": bool(pd.Timestamp(latest) >= SIGNAL_END),
            "tradestatus_field_present": True,
            "historical_isST_field_present": True,
            "st_exclusion_rule": "remove the entire stock if isST=1 on any sample-period row",
            "price_adjustment": "BaoStock query_history_k_data_plus adjustflag=2 (forward adjusted / qfq)",
            "adjust_factor_query_success_stocks": int(factor_success),
            "adjust_factor_rows": int(len(factors)),
            "industry_row_coverage": industry_known,
            "limit_price_method": "derived from point-in-time board rule and preclose; not vendor official stk_limit table",
            "new_listing_filter": "listing_days >= 180; avoids ordinary IPO no-limit window",
            "transaction_cost_round_trip_approx": BUY_COST + SELL_COST,
            "runtime_seconds": time.time() - started,
        }
        (out / "data_audit.json").write_text(json.dumps(json_clean(audit), ensure_ascii=False, indent=2), encoding="utf-8")
        headline = {
            "audit": audit,
            "candidate_counts": candidates,
            "summary": json_clean(summary.to_dict("records")),
            "portfolio": json_clean(portfolio.to_dict("records")),
        }
        (out / "headline.json").write_text(json.dumps(json_clean(headline), ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(out, audit, summary, portfolio)
        print("===HEADLINE_JSON_BEGIN===")
        print(json.dumps(json_clean(headline), ensure_ascii=False, indent=2))
        print("===HEADLINE_JSON_END===")
        if pd.Timestamp(latest) < pd.Timestamp("2026-08-01") or len(df) < 500_000:
            print("ERROR: data coverage too low", file=sys.stderr)
            return 3
        return 0
    finally:
        bs.logout()


if __name__ == "__main__":
    raise SystemExit(main())
