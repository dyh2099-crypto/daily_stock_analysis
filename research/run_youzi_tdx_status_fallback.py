#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import baostock as bs
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
CODE_RE = re.compile(r"^(sh|sz)(\d{6})\.day$", re.I)
STATUS_AUDIT: Dict[str, object] = {}


def iter_stock_files() -> Iterator[Tuple[Path, str, str]]:
    rows = []
    for p in TDX_ROOT.rglob("*.day"):
        m = CODE_RE.match(p.name)
        if not m:
            continue
        ex, digits = m.group(1).lower(), m.group(2)
        code = f"{ex}.{digits}"
        board = core.classify_board(code)
        if board in ("sh_main", "sz_main", "star", "chinext"):
            rows.append((p, code, board))
    rows.sort(key=lambda x: x[1])
    yield from rows


def read_day(path: Path, start: int = 20230103, end: int = 20260902) -> pd.DataFrame:
    raw = path.read_bytes()
    usable = len(raw) - len(raw) % DAY.size
    recs = []
    for off in range(0, usable, DAY.size):
        date, op, hi, lo, close, amount, vol, _ = DAY.unpack_from(raw, off)
        if start <= date <= end and op > 0 and close > 0:
            recs.append((date, op / 100.0, hi / 100.0, lo / 100.0, close / 100.0, float(amount), float(vol)))
    if not recs:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "amount", "volume"])
    d = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "amount", "volume"])
    d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d", errors="coerce")
    return d.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def scan_file_meta() -> pd.DataFrame:
    rows = []
    for p, code, board in iter_stock_files():
        d = read_day(p)
        if d.empty:
            continue
        rows.append({"code": code, "board": board, "first_date": d["date"].min(), "last_date": d["date"].max(), "path": str(p)})
    return pd.DataFrame(rows)


def fetch_basic_override() -> pd.DataFrame:
    meta = scan_file_meta()
    if meta.empty:
        raise RuntimeError("TDX package has no Shanghai/Shenzhen stock files")
    b = core.consume(bs.query_stock_basic())
    if not b.empty:
        for c in ("type", "status"):
            if c not in b:
                b[c] = ""
        b = b[b["type"].astype(str).eq("1")].copy()
        b["board"] = b["code"].map(core.classify_board)
        b = b[b["board"].isin(["sh_main", "sz_main", "star", "chinext"])]
        for c in ("ipoDate", "outDate"):
            if c not in b:
                b[c] = ""
            b[c] = pd.to_datetime(b[c], errors="coerce")
        keep = [c for c in ["code", "code_name", "ipoDate", "outDate", "type", "status", "board"] if c in b.columns]
        b = b[keep].drop_duplicates("code")
    else:
        b = pd.DataFrame(columns=["code", "code_name", "ipoDate", "outDate", "type", "status", "board"])
    merged = meta.merge(b.drop(columns=["board"], errors="ignore"), on="code", how="left")
    merged["board"] = meta.set_index("code").loc[merged["code"], "board"].to_numpy()
    merged["ipoDate"] = pd.to_datetime(merged["ipoDate"], errors="coerce").fillna(merged["first_date"])
    merged["outDate"] = pd.to_datetime(merged["outDate"], errors="coerce")
    merged["code_name"] = merged["code_name"].fillna("")
    merged["type"] = "1"
    merged["status"] = merged["status"].fillna("")
    global STATUS_AUDIT
    STATUS_AUDIT["tdx_scanned_stock_files"] = int(len(meta))
    STATUS_AUDIT["tdx_first_date"] = str(meta["first_date"].min().date())
    STATUS_AUDIT["tdx_last_date"] = str(meta["last_date"].max().date())
    return merged[["code", "code_name", "ipoDate", "outDate", "type", "status", "board", "path", "first_date", "last_date"]].sort_values("code").reset_index(drop=True)


def calendar_dates() -> List[pd.Timestamp]:
    candidates = sorted(TDX_ROOT.rglob("sh000001.day"))
    if not candidates:
        all_dates = set()
        for p, _, _ in list(iter_stock_files())[:30]:
            all_dates.update(read_day(p)["date"].tolist())
        return sorted(pd.Timestamp(x) for x in all_dates)
    return read_day(candidates[0])["date"].tolist()


def fetch_daily_status(dates: List[pd.Timestamp]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    success_days = 0
    for n, dt in enumerate(dates, 1):
        if dt < pd.Timestamp("2023-01-03") or dt > pd.Timestamp("2026-09-02"):
            continue
        rs = bs.query_all_stock(day=dt.strftime("%Y-%m-%d"))
        x = core.consume(rs)
        if x.empty or "code" not in x.columns:
            continue
        success_days += 1
        trade_col = "tradeStatus" if "tradeStatus" in x.columns else ("tradestatus" if "tradestatus" in x.columns else None)
        name_col = "code_name" if "code_name" in x.columns else None
        x = x[x["code"].astype(str).str.startswith(("sh.", "sz."))].copy()
        x["date"] = dt
        x["tradestatus"] = pd.to_numeric(x[trade_col], errors="coerce") if trade_col else 1
        x["status_code_name"] = x[name_col].astype(str) if name_col else ""
        x["isST"] = x["status_code_name"].str.upper().str.contains("ST", regex=False).astype(int)
        parts.append(x[["date", "code", "tradestatus", "isST", "status_code_name"]])
        if n % 50 == 0:
            print(f"STATUS {n}/{len(dates)} success_days={success_days}", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "code", "tradestatus", "isST", "status_code_name"])
    STATUS_AUDIT["status_calendar_days_requested"] = int(sum(pd.Timestamp("2023-01-03") <= d <= pd.Timestamp("2026-09-02") for d in dates))
    STATUS_AUDIT["status_query_success_days"] = int(success_days)
    STATUS_AUDIT["status_rows"] = int(len(out))
    return out


def fetch_history_override(basic: pd.DataFrame, max_symbols: int, progress_every: int):
    dates = calendar_dates()
    status = fetch_daily_status(dates)
    if not status.empty:
        status = status.drop_duplicates(["code", "date"], keep="last").set_index(["code", "date"])
    histories: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    ever_st: List[str] = []
    unknown_status_rows = 0
    corporate_action_rows_removed = 0
    work = basic.head(max_symbols) if max_symbols > 0 else basic
    for n, row in enumerate(work.itertuples(index=False), 1):
        p = Path(row.path)
        try:
            h = read_day(p)
        except Exception as exc:
            failures.append({"code": row.code, "stage": "tdx_read", "error": repr(exc)})
            continue
        if h.empty:
            failures.append({"code": row.code, "stage": "tdx_read", "error": "empty"})
            continue
        h["code"] = row.code
        if not status.empty:
            idx = pd.MultiIndex.from_arrays([h["code"], h["date"]], names=["code", "date"])
            aligned = status.reindex(idx)
            h["tradestatus"] = aligned["tradestatus"].to_numpy()
            h["isST"] = aligned["isST"].to_numpy()
            h["status_code_name"] = aligned["status_code_name"].to_numpy()
        else:
            h["tradestatus"] = np.nan
            h["isST"] = np.nan
            h["status_code_name"] = ""
        missing = h["tradestatus"].isna()
        unknown_status_rows += int(missing.sum())
        h.loc[missing, "tradestatus"] = (h.loc[missing, "volume"] > 0).astype(int)
        current_st = "ST" in str(row.code_name).upper()
        h.loc[h["isST"].isna(), "isST"] = int(current_st)
        h["isST"] = pd.to_numeric(h["isST"], errors="coerce").fillna(0)
        if bool((h.loc[h["date"].between(core.SIGNAL_START, core.SIGNAL_END), "isST"] == 1).any()):
            ever_st.append(row.code)
            continue
        h["preclose"] = h["close"].shift(1)
        h["pctChg"] = (h["close"] / h["preclose"] - 1) * 100
        h["adjustflag"] = 3
        h["turn"] = np.nan
        h["board"] = row.board
        h["ipoDate"] = row.ipoDate
        h["outDate"] = row.outDate
        h["code_name"] = row.code_name
        # Raw TDX prices are not adjusted. Remove the 60-session window after abnormal
        # discontinuities that exceed ordinary board limits and are likely corporate actions.
        gap = (h["open"] / h["preclose"] - 1).abs()
        threshold = 0.15 if row.board in ("sh_main", "sz_main") else 0.25
        extreme = gap > threshold
        contaminated = extreme.shift(1).rolling(60, min_periods=1).max().fillna(0).astype(bool)
        corporate_action_rows_removed += int(contaminated.sum())
        h = h[~contaminated & h["preclose"].notna()].copy()
        histories.append(h)
        if progress_every and n % progress_every == 0:
            print(f"TDX {n}/{len(work)} histories={len(histories)} ever_st={len(ever_st)}", flush=True)
    STATUS_AUDIT["status_unknown_rows_filled_from_volume"] = int(unknown_status_rows)
    STATUS_AUDIT["corporate_action_contaminated_rows_removed"] = int(corporate_action_rows_removed)
    if not histories:
        raise RuntimeError("no TDX histories after non-ST filtering")
    return pd.concat(histories, ignore_index=True), pd.DataFrame(), pd.DataFrame(failures), ever_st, 0, int(len(work))


def patch_outputs(out: Path) -> None:
    audit_path = out / "data_audit.json"
    headline_path = out / "headline.json"
    if not audit_path.exists() or not headline_path.exists():
        return
    audit = json.loads(audit_path.read_text())
    audit.update(STATUS_AUDIT)
    audit["requested_scope"] = "Shanghai and Shenzhen A shares, including main board, STAR and ChiNext; BSE excluded because daily historical status was unavailable"
    audit["data_source"] = "TongdaXin official hsjday complete daily package + BaoStock query_all_stock point-in-time status/name"
    audit["price_adjustment"] = "TDX raw prices; 60-session windows after abnormal discontinuities filtered; no complete adjustment-factor table in fallback"
    audit["adjust_factor_query_success_stocks"] = 0
    audit["adjust_factor_rows"] = 0
    audit["limit_price_method"] = "reconstructed from board rule and raw preclose; not official daily stk_limit table"
    audit_path.write_text(json.dumps(core.json_clean(audit), ensure_ascii=False, indent=2))
    headline = json.loads(headline_path.read_text())
    headline["audit"] = audit
    headline_path.write_text(json.dumps(core.json_clean(headline), ensure_ascii=False, indent=2))
    report = out / "REPORT.md"
    if report.exists():
        report.write_text(report.read_text() + "\n\n## Fallback data source correction\n\nThis run used TDX raw daily prices plus BaoStock point-in-time daily status. It did not obtain a complete adjustment-factor table; abnormal corporate-action windows were filtered.\n", encoding="utf-8")


def main() -> int:
    core.fetch_basic = fetch_basic_override
    core.fetch_history_and_factors = fetch_history_override
    rc = core.main()
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    patch_outputs(out)
    if (out / "headline.json").exists():
        print("===PATCHED_HEADLINE_BEGIN===")
        print((out / "headline.json").read_text())
        print("===PATCHED_HEADLINE_END===")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
