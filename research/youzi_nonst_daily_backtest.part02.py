                break
            if target_hit:
                exit_pos, exit_price, reason = j, target, "target"
                break

        # State failure is known only after this close; execute next open.
        if j < hard_last:
            if strategy == "leader_continuation":
                if (pd.notna(row["sentiment_score"]) and float(row["sentiment_score"]) < 35) or (
                    pd.notna(row["leader_rank"]) and float(row["leader_rank"]) > 10 and float(row["close"]) < float(row["prev_low"])
                ):
                    scheduled_open_exit = j + 1
            elif strategy == "new_hot_industry_proxy":
                if int(row["industry_id"]) != signal_industry or not bool(row["is_hotspot"]) or (
                    pd.notna(row["leader_rank"]) and float(row["leader_rank"]) > 2
                ):
                    scheduled_open_exit = j + 1
            elif strategy == "second_board_after_close" and j == entry_pos + 1 and float(row["close"]) < signal_close:
                # T+1 permits selling at the first holding session close.
                exit_pos, exit_price, reason = j, float(row["close"]), "next_session_weakness"
                break

        if j >= last_planned:
            exit_pos, exit_price, reason = j, float(row["close"]), "time_exit"
            break

    if exit_pos is None or exit_price is None:
        return None
    net = cost_adjusted_return(entry_price, exit_price, cfg)
    return {
        "entry_date": entry["date"],
        "entry_price": entry_price,
        "entry_gap": entry_gap,
        "exit_date": arr.iloc[exit_pos]["date"],
        "exit_price": exit_price,
        "exit_reason": reason,
        "holding_sessions": int(exit_pos - entry_pos + 1),
        "net_return": net,
        "win": bool(net > 0),
        "target_hit": reason.startswith("target"),
        "stop_hit": "stop" in reason,
        "exit_pos": exit_pos,
    }


def run_strategy(strategy: str, candidates: pd.DataFrame, arrays: Dict[int, pd.DataFrame], stocks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    code_map = stocks.set_index("sid")["code"].to_dict()
    board_map = stocks.set_index("sid")["board"].to_dict()
    name_map = stocks.set_index("sid")["code_name"].to_dict()
    trades: List[Dict[str, Any]] = []
    for sid, cdf in candidates.sort_values(["sid", "date"]).groupby("sid", sort=False):
        sid = int(sid)
        arr = arrays.get(sid)
        if arr is None:
            continue
        next_allowed = -1
        for row in cdf.itertuples(index=False):
            pos = int(getattr(row, "row_in_symbol"))
            entry_pos = pos + 1
            if entry_pos <= next_allowed:
                continue
            sr = pd.Series(row._asdict())
            result = exit_trade(arr, entry_pos, strategy, sr, cfg)
            if result is None:
                continue
            next_allowed = int(result.pop("exit_pos"))
            trade = {
                "strategy": strategy,
                "strategy_cn": STRATEGY_LABELS[strategy],
                "sid": sid,
                "code": code_map.get(sid, str(sid)),
                "code_name": name_map.get(sid, ""),
                "board": board_map.get(sid, ""),
                "signal_date": pd.Timestamp(getattr(row, "date")),
                "signal_close": float(getattr(row, "close")),
                "sentiment_score": float(getattr(row, "sentiment_score")) if pd.notna(getattr(row, "sentiment_score")) else np.nan,
                "industry_id": int(getattr(row, "industry_id")),
                "leader_rank": float(getattr(row, "leader_rank")) if pd.notna(getattr(row, "leader_rank")) else np.nan,
                "volume_ratio": float(getattr(row, "volume_ratio")) if pd.notna(getattr(row, "volume_ratio")) else np.nan,
                "signal_rank": float(getattr(row, "signal_rank")) if pd.notna(getattr(row, "signal_rank")) else np.nan,
            }
            trade.update(result)
            trades.append(trade)
    return pd.DataFrame(trades)


def summarize_group(g: pd.DataFrame) -> Dict[str, Any]:
    n = len(g)
    if n == 0:
        return {"trades": 0}
    r = g["net_return"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    ci = wilson_interval(int((r > 0).sum()), n)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(n),
        "win_rate": float((r > 0).mean()),
        "win_rate_ci95_low": ci[0],
        "win_rate_ci95_high": ci[1],
        "mean_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "p_ge_3pct": float((r >= 0.03).mean()),
        "p_le_minus_3pct": float((r <= -0.03).mean()),
        "median_holding_sessions": float(g["holding_sessions"].median()),
        "target_hit_rate": float(g["target_hit"].mean()),
        "stop_hit_rate": float(g["stop_hit"].mean()),
    }


def make_summaries(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    annual = []
    regimes = []
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    trades["sentiment_regime"] = pd.cut(trades["sentiment_score"], [-np.inf, 35, 60, np.inf], labels=["weak", "divergent", "strong"], right=False)
    for strategy, g in trades.groupby("strategy"):
        row = {"strategy": strategy, "strategy_cn": STRATEGY_LABELS[strategy]}
        row.update(summarize_group(g))
        rows.append(row)
        for year, gy in g.groupby("year"):
            rr = {"strategy": strategy, "strategy_cn": STRATEGY_LABELS[strategy], "year": int(year)}
            rr.update(summarize_group(gy))
            annual.append(rr)
        for regime, gr in g.groupby("sentiment_regime", observed=True):
            rr = {"strategy": strategy, "strategy_cn": STRATEGY_LABELS[strategy], "sentiment_regime": str(regime)}
            rr.update(summarize_group(gr))
            regimes.append(rr)
    return pd.DataFrame(rows), pd.DataFrame(annual), pd.DataFrame(regimes)


def validation_report(x: pd.DataFrame, stocks: pd.DataFrame, industries: pd.DataFrame, audit: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    signal_rows = x[x["date"].between(pd.Timestamp(cfg.signal_start), pd.Timestamp(cfg.end_date))]
    nonst_eligible = signal_rows[signal_rows["eligible"]]
    delisted_sids = set(stocks[stocks["outDate"].notna()]["sid"].astype(int))
    audit.update({
        "analysis_daily_rows": int(len(x)),
        "analysis_symbols": int(x["sid"].nunique()),
        "signal_period_rows": int(len(signal_rows)),
        "eligible_nonst_rows": int(len(nonst_eligible)),
        "st_rows_excluded": int(signal_rows["isST"].eq(1).sum()),
        "ever_st_symbols": int(signal_rows.loc[signal_rows["isST"].eq(1), "sid"].nunique()),
        "suspended_rows": int(signal_rows["tradestatus"].eq(0).sum()),
        "qfq_flag_coverage": float(x["adjustflag"].eq(2).mean()),
        "industry_coverage_eligible": float(nonst_eligible["industry_id"].ge(0).mean()) if len(nonst_eligible) else 0.0,
        "delisted_symbols_with_rows": int(x[x["sid"].isin(delisted_sids)]["sid"].nunique()),
        "median_eligible_names_per_day": float(nonst_eligible.groupby("date")["sid"].nunique().median()) if len(nonst_eligible) else 0.0,
        "min_eligible_names_per_day": int(nonst_eligible.groupby("date")["sid"].nunique().min()) if len(nonst_eligible) else 0,
        "last_complete_date": str(x["date"].max().date()),
    })
    checks = {
        "history_coverage_ge_90pct": audit.get("history_coverage", 0) >= 0.90,
        "qfq_flag_coverage_ge_99pct": audit["qfq_flag_coverage"] >= 0.99,
        "st_flag_present": x["isST"].notna().mean() >= 0.999,
        "trade_status_present": x["tradestatus"].notna().mean() >= 0.999,
        "delisted_symbols_included": audit["delisted_symbols_with_rows"] > 0,
        "industry_coverage_ge_80pct": audit["industry_coverage_eligible"] >= 0.80,
        "date_reaches_target": x["date"].max() >= pd.Timestamp(cfg.end_date),
    }
    audit["validation_checks"] = checks
    audit["ready_for_daily_proxy"] = bool(all(v for k, v in checks.items() if k != "industry_coverage_ge_80pct"))
    return audit


def write_markdown_report(outdir: Path, cfg: Config, audit: Dict[str, Any], summary: pd.DataFrame, annual: pd.DataFrame, regime: pd.DataFrame, candidate_counts: Mapping[str, int]) -> None:
    def pct(v: Any) -> str:
        return "—" if pd.isna(v) else f"{float(v)*100:.3f}%"
    lines = [
        "# 游资语录量化：沪深全A非ST三年日线代理回测",
        "",
        "## 结论表",
        "",
        "| 策略 | 交易数 | 胜率 | 平均净收益 | 中位数 | 利润因子 | 中位持有期 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.strategy_cn} | {int(r.trades):,} | {pct(r.win_rate)} | {pct(r.mean_net_return)} | {pct(r.median_net_return)} | {float(r.profit_factor):.3f} | {float(r.median_holding_sessions):.1f} |")
    lines += [
        "",
        "## 数据门禁",
        "",
        f"- 区间：{cfg.signal_start} 至 {cfg.end_date}；读取窗口自 {cfg.read_start}。",
        f"- 历史股票池：证券基本资料选中 {audit.get('selected_stock_codes', 0):,} 只；成功取得历史行情 {audit.get('history_success_codes', 0):,} 只，覆盖率 {audit.get('history_coverage', 0)*100:.2f}%。",
        f"- 历史退市股票：选中池含 {audit.get('delisted_codes_selected', 0):,} 只；实际取得行情 {audit.get('delisted_symbols_with_rows', 0):,} 只。",
        f"- ST处理：信号期剔除 {audit.get('st_rows_excluded', 0):,} 个ST股票日，涉及 {audit.get('ever_st_symbols', 0):,} 只股票。",
        f"- 复权：分析行情 adjustflag=2 覆盖率 {audit.get('qfq_flag_coverage', 0)*100:.3f}%；单独复权因子覆盖率 {audit.get('factor_symbol_coverage', 0)*100:.2f}%。",
        f"- 行业历史快照：{audit.get('industry_snapshot_dates', 0)} 个日期；非ST可交易样本行业覆盖率 {audit.get('industry_coverage_eligible', 0)*100:.2f}%。",
        f"- 最后完整交易日：{audit.get('last_complete_date')}。2026-09-03尚未收盘，不纳入。",
        "",
        "## 成交与偏差控制",
        "",
        "- 所有日线信号在收盘后确认，最早下一可交易日开盘成交。",
        "- 执行T+1；一字涨停不买，一字跌停不假设能卖；同一日同时触及止盈止损时按止损先发生。",
        "- 净收益扣除双边佣金、卖出印花税、双边过户费与双边滑点；忽略最低5元佣金，因为事件研究按约10万元单笔名义仓位解释。",
        "- 五类规则是对游资语言的预先固定量化翻译，没有基于本次结果调参。",
        "- 第五类使用申万行业历史快照代理题材，不能等同于带事件时间戳的概念题材；二板策略是收盘确认后次日接力，不是盘中首次触板或回封策略。",
        "",
        "## 候选信号数量（成交过滤前）",
        "",
    ]
    for k, v in candidate_counts.items():
        lines.append(f"- {STRATEGY_LABELS[k]}：{v:,}")
    lines += ["", "## 分年度结果", "", annual.to_markdown(index=False) if not annual.empty else "无", "", "## 情绪状态拆分", "", regime.to_markdown(index=False) if not regime.empty else "无"]
    (outdir / "回测报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--no-factor-query", action="store_true")
    p.add_argument("--industry-snapshot", choices=["monthly", "weekly"], default="monthly")
    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(factor_query=not args.no_factor_query, industry_snapshot=args.industry_snapshot)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    daily, stocks, industries, audit = download_data(cfg, outdir, max_symbols=args.max_symbols)
    daily = assign_point_in_time_industry(daily, stocks, industries, outdir)
    x, market, ind = add_features(daily, cfg)
    audit = validation_report(x, stocks, industries, audit, cfg)
    (outdir / "data_audit.json").write_text(json.dumps(json_clean(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    market.to_csv(outdir / "market_sentiment_daily.csv", index=False)
    ind.to_csv(outdir / "industry_hotspots_daily.csv.gz", index=False, compression="gzip")
    if not audit["ready_for_daily_proxy"]:
        print("DATA_GATE_FAILED", json.dumps(json_clean(audit), ensure_ascii=False), flush=True)
        return 4

    candidates = signal_candidates(x, cfg)
    candidate_counts = {k: int(len(v)) for k, v in candidates.items()}
    pd.DataFrame([{"strategy": k, "strategy_cn": STRATEGY_LABELS[k], "candidates": v} for k, v in candidate_counts.items()]).to_csv(outdir / "candidate_counts.csv", index=False)
    arrays = build_symbol_arrays(x)
    trade_frames = []
    for strategy, frame in candidates.items():
        print(f"simulate {strategy} candidates={len(frame):,}", flush=True)
        t = run_strategy(strategy, frame, arrays, stocks, cfg)
        print(f"completed {strategy} trades={len(t):,}", flush=True)
        if not t.empty:
            trade_frames.append(t)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades.to_csv(outdir / "trades.csv.gz", index=False, compression="gzip")
    summary, annual, regime = make_summaries(trades)
    summary.to_csv(outdir / "summary.csv", index=False)
    annual.to_csv(outdir / "annual_summary.csv", index=False)
    regime.to_csv(outdir / "sentiment_regime_summary.csv", index=False)
    write_markdown_report(outdir, cfg, audit, summary, annual, regime, candidate_counts)

    headline = {
        "status": "PASS" if not summary.empty else "NO_TRADES",
        "scope": "Shanghai and Shenzhen A-shares, point-in-time non-ST daily rows; ChiNext and STAR included; BSE only if BaoStock returns it",
        "period": {"signal_start": cfg.signal_start, "end_date": cfg.end_date, "read_start": cfg.read_start},
        "data_audit": audit,
        "candidate_counts": candidate_counts,
        "summary": summary.to_dict("records"),
        "limitations": [
            "Daily proxy: cannot identify intraday first touch, reseal, VWAP reclaim, or queue priority.",
            "Industry is a historical Shenwan-industry proxy, not timestamped concept/theme events.",
            "BaoStock coverage determines whether BSE appears; the primary comparable pool is Shanghai/Shenzhen.",
            "Event-level trade statistics are not a capital-constrained portfolio CAGR.",
        ],
    }
    (outdir / "headline.json").write_text(json.dumps(json_clean(headline), ensure_ascii=False, indent=2), encoding="utf-8")
    print("===HEADLINE_JSON_BEGIN===", flush=True)
    print(json.dumps(json_clean(headline), ensure_ascii=False, indent=2), flush=True)
    print("===HEADLINE_JSON_END===", flush=True)
    print("===SUMMARY_CSV_BEGIN===", flush=True)
    print(summary.to_csv(index=False), flush=True)
    print("===SUMMARY_CSV_END===", flush=True)
    return 0 if not summary.empty else 5


if __name__ == "__main__":
    raise SystemExit(main())
