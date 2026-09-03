#!/usr/bin/env python3
"""Combine parallel BaoStock shards and run the fixed daily-proxy strategies."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_core(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("youzi_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load core module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_many(root: Path, pattern: str, **kwargs: Any) -> pd.DataFrame:
    files = sorted(root.rglob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(file, **kwargs) for file in files]
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def normalize_loaded(daily: pd.DataFrame, stocks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty or stocks.empty:
        raise RuntimeError("missing daily or stock basic shard data")
    daily = daily.copy()
    stocks = stocks.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["ipo_date"] = pd.to_datetime(daily["ipo_date"], errors="coerce")
    numeric = [
        "open", "high", "low", "close", "preclose", "volume", "amount",
        "turn", "pctChg", "tradestatus", "isST", "adjustflag", "sid",
    ]
    for column in numeric:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.dropna(
        subset=[
            "date", "open", "high", "low", "close", "preclose", "volume",
            "amount", "pctChg", "tradestatus", "isST", "adjustflag", "sid",
        ]
    )
    daily["sid"] = daily["sid"].astype(np.int32)
    daily["tradestatus"] = daily["tradestatus"].astype(np.int8)
    daily["isST"] = daily["isST"].astype(np.int8)
    daily["adjustflag"] = daily["adjustflag"].astype(np.int8)
    daily = daily.sort_values(["sid", "date"]).drop_duplicates(["sid", "date"], keep="last")

    stocks["sid"] = pd.to_numeric(stocks["sid"], errors="coerce")
    stocks = stocks.dropna(subset=["sid", "code"]).copy()
    stocks["sid"] = stocks["sid"].astype(np.int32)
    stocks["ipoDate"] = pd.to_datetime(stocks["ipoDate"], errors="coerce")
    stocks["outDate"] = pd.to_datetime(stocks["outDate"], errors="coerce")
    stocks = stocks.sort_values("sid").drop_duplicates("sid", keep="last")
    return daily.reset_index(drop=True), stocks.reset_index(drop=True)


def download_industries(core: Any, daily: pd.DataFrame, cfg: Any, outdir: Path) -> pd.DataFrame:
    import baostock as bs  # type: ignore

    core.bs_login(bs)
    dates = core.choose_snapshot_dates(
        [pd.Timestamp(value) for value in sorted(daily["date"].unique())],
        cfg.industry_snapshot,
    )
    frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    for index, date in enumerate(dates, start=1):
        try:
            frame = core.query_with_retry(
                bs,
                "query_stock_industry",
                {"date": pd.Timestamp(date).strftime("%Y-%m-%d")},
                attempts=3,
            )
            if not frame.empty:
                frame["snapshot_date"] = pd.Timestamp(date)
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            failures.append({"snapshot_date": str(pd.Timestamp(date).date()), "error": repr(exc)})
        if index % 6 == 0 or index == len(dates):
            print(
                f"industry progress={index}/{len(dates)} successful={len(frames)}",
                flush=True,
            )
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass
    pd.DataFrame(failures).to_csv(outdir / "industry_failures.csv", index=False)
    industries = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not industries.empty:
        industries.to_csv(
            outdir / "industry_snapshots.csv.gz", index=False, compression="gzip"
        )
    return industries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core-script", required=True)
    parser.add_argument("--industry-snapshot", choices=["monthly", "weekly"], default="monthly")
    args = parser.parse_args()

    core = load_core(args.core_script)
    root = Path(args.input_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = core.Config(factor_query=True, industry_snapshot=args.industry_snapshot)
    (outdir / "config.json").write_text(
        json.dumps(core.asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    daily = read_many(root, "daily_shard_*.csv.gz")
    stocks = read_many(root, "basic_shard_*.csv")
    factors = read_many(root, "factors_shard_*.csv.gz")
    failures = read_many(root, "failures_shard_*.csv")
    manifests = []
    for file in sorted(root.rglob("manifest_shard_*.json")):
        manifests.append(json.loads(file.read_text(encoding="utf-8")))
    daily, stocks = normalize_loaded(daily, stocks)
    stocks.to_csv(outdir / "stock_universe_selected.csv", index=False)
    if not factors.empty:
        factors.to_csv(outdir / "adjust_factors.csv.gz", index=False, compression="gzip")
    failures.to_csv(outdir / "download_failures.csv", index=False)
    (outdir / "shard_manifests.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    industries = download_industries(core, daily, cfg, outdir)
    matched = set(daily["sid"].astype(int).unique())
    audit: Dict[str, Any] = {
        "selected_stock_codes": int(len(stocks)),
        "history_success_codes": int(len(matched)),
        "history_failure_codes": int(len(stocks) - len(matched)),
        "history_coverage": float(len(matched) / max(1, len(stocks))),
        "daily_rows": int(len(daily)),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "board_counts_selected": stocks["board"].value_counts().to_dict(),
        "board_counts_with_history": stocks[stocks["sid"].isin(matched)]["board"].value_counts().to_dict(),
        "delisted_codes_selected": int(stocks["outDate"].notna().sum()),
        "factor_rows": int(len(factors)),
        "factor_symbol_coverage": float(factors["sid"].nunique() / max(1, len(matched))) if not factors.empty else 0.0,
        "factor_query_failure_rows": int((failures.get("stage", pd.Series(dtype=str)) == "factor").sum()) if not failures.empty else 0,
        "history_query_failure_rows": int((failures.get("stage", pd.Series(dtype=str)) == "history").sum()) if not failures.empty else 0,
        "industry_snapshot_dates": int(pd.to_datetime(industries["snapshot_date"]).nunique()) if not industries.empty else 0,
        "shards_expected": int(manifests[0]["shard_count"]) if manifests else 0,
        "shards_received": int(len(manifests)),
    }

    enriched = core.assign_point_in_time_industry(daily, stocks, industries, outdir)
    featured, market, industry_daily = core.add_features(enriched, cfg)
    audit = core.validation_report(featured, stocks, industries, audit, cfg)
    checks = audit["validation_checks"]
    audit["strict_historical_universe_gate"] = bool(checks.get("delisted_symbols_included", False))
    audit["ready_for_nonst_daily_proxy"] = bool(
        checks.get("history_coverage_ge_90pct", False)
        and checks.get("qfq_flag_coverage_ge_99pct", False)
        and checks.get("st_flag_present", False)
        and checks.get("trade_status_present", False)
        and checks.get("date_reaches_target", False)
        and audit.get("shards_received", 0) == audit.get("shards_expected", -1)
    )
    (outdir / "data_audit.json").write_text(
        json.dumps(core.json_clean(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    market.to_csv(outdir / "market_sentiment_daily.csv", index=False)
    industry_daily.to_csv(
        outdir / "industry_hotspots_daily.csv.gz", index=False, compression="gzip"
    )
    if not audit["ready_for_nonst_daily_proxy"]:
        print("RELAXED_DATA_GATE_FAILED", json.dumps(core.json_clean(audit), ensure_ascii=False), flush=True)
        return 4

    candidates = core.signal_candidates(featured, cfg)
    candidate_counts = {key: int(len(value)) for key, value in candidates.items()}
    pd.DataFrame(
        [
            {
                "strategy": key,
                "strategy_cn": core.STRATEGY_LABELS[key],
                "candidates": value,
            }
            for key, value in candidate_counts.items()
        ]
    ).to_csv(outdir / "candidate_counts.csv", index=False)

    arrays = core.build_symbol_arrays(featured)
    trade_frames = []
    for strategy, frame in candidates.items():
        print(f"simulate strategy={strategy} candidates={len(frame):,}", flush=True)
        trades = core.run_strategy(strategy, frame, arrays, stocks, cfg)
        print(f"completed strategy={strategy} trades={len(trades):,}", flush=True)
        if not trades.empty:
            trade_frames.append(trades)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    all_trades.to_csv(outdir / "trades.csv.gz", index=False, compression="gzip")
    summary, annual, regimes = core.make_summaries(all_trades)
    summary.to_csv(outdir / "summary.csv", index=False)
    annual.to_csv(outdir / "annual_summary.csv", index=False)
    regimes.to_csv(outdir / "sentiment_regime_summary.csv", index=False)
    core.write_markdown_report(
        outdir, cfg, audit, summary, annual, regimes, candidate_counts
    )

    status = "PASS" if not summary.empty else "NO_TRADES"
    headline = {
        "status": status,
        "scope": (
            "Shanghai/Shenzhen A-shares in BaoStock historical universe; "
            "point-in-time isST=0 and tradestatus=1; ChiNext/STAR included; "
            "BSE only if returned by source"
        ),
        "period": {
            "read_start": cfg.read_start,
            "signal_start": cfg.signal_start,
            "end_date": cfg.end_date,
        },
        "data_audit": audit,
        "candidate_counts": candidate_counts,
        "summary": summary.to_dict("records"),
        "limitations": [
            "Daily proxy, not minute/tick reconstruction; no first-touch, reseal, VWAP, or queue-priority claims.",
            "Limit status is reconstructed from source pctChg, board limit ratio, and OHLC; it is not a dedicated exchange limit-price table.",
            "Historical Shenwan industry is a theme proxy, not a timestamped concept-event database.",
            "Trade-event statistics are not a capital-constrained portfolio CAGR.",
            "If the delisted-universe check is false, results are explicitly exposed as non-ST source-universe estimates rather than strict full-history-universe results.",
        ],
    }
    (outdir / "headline.json").write_text(
        json.dumps(core.json_clean(headline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("===HEADLINE_JSON_BEGIN===", flush=True)
    print(json.dumps(core.json_clean(headline), ensure_ascii=False, indent=2), flush=True)
    print("===HEADLINE_JSON_END===", flush=True)
    print("===SUMMARY_CSV_BEGIN===", flush=True)
    print(summary.to_csv(index=False), flush=True)
    print("===SUMMARY_CSV_END===", flush=True)
    return 0 if status == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
