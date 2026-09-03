#!/usr/bin/env python3
"""Parallel BaoStock downloader for point-in-time non-ST A-share backtest."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

READ_START = "2023-01-03"
END_DATE = "2026-09-02"
FACTOR_START = "1990-01-01"


def resultset_to_frame(rs: Any) -> pd.DataFrame:
    rows: List[List[str]] = []
    while getattr(rs, "error_code", "1") == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=getattr(rs, "fields", []))


def login(bs: Any, attempts: int = 6) -> None:
    last = None
    for attempt in range(1, attempts + 1):
        lg = bs.login()
        if getattr(lg, "error_code", "1") == "0":
            print(f"login success attempt={attempt}", flush=True)
            return
        last = f"{getattr(lg, 'error_code', '?')} {getattr(lg, 'error_msg', '')}"
        time.sleep(min(20, attempt * 3))
    raise RuntimeError(f"BaoStock login failed: {last}")


def query_retry(bs: Any, fn_name: str, kwargs: Mapping[str, Any], attempts: int = 5) -> pd.DataFrame:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            rs = getattr(bs, fn_name)(**dict(kwargs))
            if getattr(rs, "error_code", "1") == "0":
                return resultset_to_frame(rs)
            last = f"{getattr(rs, 'error_code', '?')} {getattr(rs, 'error_msg', '')}"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(min(20, 2**attempt))
        login(bs, attempts=2)
    raise RuntimeError(f"{fn_name} failed: {last}; kwargs={kwargs}")


def classify_board(code: str) -> Optional[str]:
    if code.startswith("sh."):
        raw = code[3:]
        if raw.startswith(("600", "601", "603", "605")):
            return "MAIN"
        if raw.startswith(("688", "689")):
            return "STAR"
    if code.startswith("sz."):
        raw = code[3:]
        if raw.startswith(("000", "001", "002", "003")):
            return "MAIN"
        if raw.startswith(("300", "301")):
            return "CHINEXT"
    if code.startswith("bj."):
        return "BSE"
    return None


def normalize_history(raw: pd.DataFrame, sid: int, board: str, ipo_date: pd.Timestamp) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric = [
        "open", "high", "low", "close", "preclose", "volume", "amount",
        "turn", "pctChg", "tradestatus", "isST", "adjustflag",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = [
        "date", "open", "high", "low", "close", "preclose", "volume",
        "amount", "pctChg", "tradestatus", "isST", "adjustflag",
    ]
    frame = frame.dropna(subset=required)
    frame = frame[
        frame["date"].between(pd.Timestamp(READ_START), pd.Timestamp(END_DATE))
    ]
    if frame.empty:
        return frame
    columns = [
        "date", "open", "high", "low", "close", "preclose", "volume",
        "amount", "turn", "pctChg", "tradestatus", "isST", "adjustflag",
    ]
    out = frame[columns].copy()
    out["sid"] = np.int32(sid)
    out["board"] = board
    out["ipo_date"] = ipo_date
    return out.sort_values("date").drop_duplicates("date", keep="last")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--skip-factors", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    import baostock as bs  # type: ignore

    login(bs)
    basic = query_retry(bs, "query_stock_basic", {})
    if basic.empty:
        raise RuntimeError("query_stock_basic returned no rows")
    basic["ipoDate"] = pd.to_datetime(basic["ipoDate"], errors="coerce")
    basic["outDate"] = pd.to_datetime(basic["outDate"], errors="coerce")
    basic["type"] = pd.to_numeric(basic["type"], errors="coerce")
    basic["board"] = basic["code"].map(classify_board)
    overlap = (
        (basic["ipoDate"].isna() | (basic["ipoDate"] <= pd.Timestamp(END_DATE)))
        & (basic["outDate"].isna() | (basic["outDate"] >= pd.Timestamp(READ_START)))
    )
    universe = basic[(basic["type"] == 1) & basic["board"].notna() & overlap].copy()
    universe = universe.sort_values("code").reset_index(drop=True)
    universe["sid"] = np.arange(len(universe), dtype=np.int32)
    selected = universe.iloc[args.shard_index :: args.shard_count].copy()
    selected.to_csv(outdir / f"basic_shard_{args.shard_index:02d}.csv", index=False)
    print(
        f"universe={len(universe)} shard={args.shard_index}/{args.shard_count} "
        f"selected={len(selected)} boards={selected['board'].value_counts().to_dict()}",
        flush=True,
    )

    fields = (
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
        "turn,tradestatus,pctChg,isST"
    )
    daily_frames: List[pd.DataFrame] = []
    factor_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []

    for number, row in enumerate(selected.itertuples(index=False), start=1):
        code = str(row.code)
        try:
            history = query_retry(
                bs,
                "query_history_k_data_plus",
                {
                    "code": code,
                    "fields": fields,
                    "start_date": READ_START,
                    "end_date": END_DATE,
                    "frequency": "d",
                    "adjustflag": "2",
                },
            )
            normalized = normalize_history(
                history, int(row.sid), str(row.board), pd.Timestamp(row.ipoDate)
            )
            if normalized.empty:
                failures.append({"code": code, "stage": "history", "error": "empty"})
            else:
                daily_frames.append(normalized)
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "stage": "history", "error": repr(exc)})

        if not args.skip_factors:
            try:
                factors = query_retry(
                    bs,
                    "query_adjust_factor",
                    {
                        "code": code,
                        "start_date": FACTOR_START,
                        "end_date": END_DATE,
                    },
                    attempts=3,
                )
                if not factors.empty:
                    factors["sid"] = int(row.sid)
                    factor_frames.append(factors)
            except Exception as exc:  # noqa: BLE001
                failures.append({"code": code, "stage": "factor", "error": repr(exc)})

        if number % 25 == 0 or number == len(selected):
            rows = sum(len(frame) for frame in daily_frames)
            print(
                f"progress={number}/{len(selected)} rows={rows:,} failures={len(failures)}",
                flush=True,
            )

    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass

    if not daily_frames:
        raise RuntimeError("shard returned no history")
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(
        outdir / f"daily_shard_{args.shard_index:02d}.csv.gz",
        index=False,
        compression="gzip",
    )
    if factor_frames:
        factors = pd.concat(factor_frames, ignore_index=True)
        factors.to_csv(
            outdir / f"factors_shard_{args.shard_index:02d}.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(failures).to_csv(
        outdir / f"failures_shard_{args.shard_index:02d}.csv", index=False
    )
    manifest = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "universe_rows": int(len(universe)),
        "selected_rows": int(len(selected)),
        "history_symbols": int(daily["sid"].nunique()),
        "daily_rows": int(len(daily)),
        "factor_symbols": int(
            pd.concat(factor_frames, ignore_index=True)["sid"].nunique()
        ) if factor_frames else 0,
        "failure_rows": int(len(failures)),
        "first_date": str(pd.to_datetime(daily["date"]).min().date()),
        "last_date": str(pd.to_datetime(daily["date"]).max().date()),
    }
    (outdir / f"manifest_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
