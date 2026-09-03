    daily = daily.copy()
    daily["industry_id"] = np.int16(-1)
    if industries.empty:
        return daily
    code_to_sid = stocks.set_index("code")["sid"].to_dict()
    ind = industries.copy()
    ind["sid"] = ind["code"].map(code_to_sid)
    ind = ind[ind["sid"].notna()].copy()
    ind["sid"] = ind["sid"].astype(int)
    ind["snapshot_date"] = pd.to_datetime(ind["snapshot_date"])
    ind["industry"] = ind["industry"].fillna("").replace("", "UNKNOWN")
    names = sorted(x for x in ind["industry"].unique() if x != "UNKNOWN")
    name_to_id = {name: i for i, name in enumerate(names)}
    ind["industry_id"] = ind["industry"].map(name_to_id).fillna(-1).astype("int16")
    pd.DataFrame([{"industry_id": v, "industry": k} for k, v in name_to_id.items()]).sort_values("industry_id").to_csv(outdir / "industry_lookup.csv", index=False)

    snapshots: Dict[pd.Timestamp, Dict[int, int]] = {}
    for d, g in ind.groupby("snapshot_date"):
        snapshots[pd.Timestamp(d)] = dict(zip(g["sid"].astype(int), g["industry_id"].astype(int)))
    snap_dates = sorted(snapshots)
    current: Dict[int, int] = {}
    ptr = 0
    values = np.full(len(daily), -1, dtype=np.int16)
    for d, idx in daily.groupby("date", sort=True).groups.items():
        d = pd.Timestamp(d)
        while ptr < len(snap_dates) and snap_dates[ptr] <= d:
            current.update(snapshots[snap_dates[ptr]])
            ptr += 1
        sids = daily.loc[idx, "sid"].astype(int)
        values[np.asarray(idx, dtype=int)] = sids.map(current).fillna(-1).astype(np.int16).to_numpy()
    daily["industry_id"] = values
    return daily


def add_features(daily: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = daily.sort_values(["sid", "date"]).reset_index(drop=True).copy()
    x["row_in_symbol"] = x.groupby("sid", sort=False).cumcount().astype("int16")
    x["listing_age_days"] = (x["date"] - x["ipo_date"]).dt.days
    x["non_st"] = x["isST"].eq(0)
    x["tradable"] = x["tradestatus"].eq(1) & x["volume"].gt(0) & x["open"].gt(0)
    x["ret1"] = x["pctChg"].astype(float) / 100.0
    fallback = x["close"].astype(float) / x["preclose"].replace(0, np.nan).astype(float) - 1.0
    x["ret1"] = x["ret1"].where(np.isfinite(x["ret1"]), fallback)
    x["close_location"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    x["limit_ratio"] = x["board"].map(limit_ratio_for_board).astype(float)
    x["high_ret_ref"] = x["high"].astype(float) / x["preclose"].replace(0, np.nan).astype(float) - 1.0
    x["low_ret_ref"] = x["low"].astype(float) / x["preclose"].replace(0, np.nan).astype(float) - 1.0
    # 0.5 percentage-point tolerance accommodates tick rounding at low prices.
    x["limit_up_close"] = x["non_st"] & (x["ret1"] >= x["limit_ratio"] - 0.005) & (x["close"] >= x["high"] * 0.999)
    x["limit_down_close"] = x["non_st"] & (x["ret1"] <= -x["limit_ratio"] + 0.005) & (x["close"] <= x["low"] * 1.001)
    x["touched_limit_up"] = x["non_st"] & (x["high_ret_ref"] >= x["limit_ratio"] - 0.005)
    x["one_price_limit_up"] = x["limit_up_close"] & ((x["high"] - x["low"]).abs() <= x["close"].abs() * 1e-6)
    x["one_price_limit_down"] = x["limit_down_close"] & ((x["high"] - x["low"]).abs() <= x["close"].abs() * 1e-6)

    g = x.groupby("sid", sort=False, group_keys=False)
    x["ret3"] = g["close"].pct_change(3, fill_method=None)
    x["ret10"] = g["close"].pct_change(10, fill_method=None)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - x["preclose"]).abs(),
        (x["low"] - x["preclose"]).abs(),
    ], axis=1).max(axis=1)
    x["atr20"] = tr.groupby(x["sid"]).transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    x["amount_med20"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).median())
    x["volume_mean20"] = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    x["volume_ratio"] = x["volume"] / x["volume_mean20"].replace(0, np.nan)
    x["amount_activity"] = x["amount"] / x["amount_med20"].replace(0, np.nan)
    x["amount3"] = g["amount"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    x["limitup5"] = g["limit_up_close"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    x["prior_high5"] = g["high"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).max())
    x["prior_low3"] = g["low"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).min())
    x["prev_limitup"] = g["limit_up_close"].shift(1).fillna(False).astype(bool)
    x["prev2_limitup"] = g["limit_up_close"].shift(2).fillna(False).astype(bool)
    x["prev_close_actual"] = g["close"].shift(1)
    x["prev_low"] = g["low"].shift(1)

    x["eligible"] = (
        x["non_st"] & x["tradable"] &
        x["listing_age_days"].ge(cfg.min_calendar_listing_days) &
        x["amount_med20"].ge(cfg.min_median_amount20) &
        x["close"].ge(cfg.min_price) &
        x["adjustflag"].eq(2)
    )

    e = x[x["eligible"]].copy()
    market = e.groupby("date").agg(
        eligible_names=("sid", "size"),
        limitdown_share=("limit_down_close", "mean"),
        market_ret10_median=("ret10", "median"),
        touched_limitup_names=("touched_limit_up", "sum"),
        close_limitup_names=("limit_up_close", "sum"),
    )
    prior_lu = e[e["prev_limitup"]].groupby("date")
    market["prior_limitup_return_median"] = prior_lu["ret1"].median()
    market["prior_limitup_names"] = prior_lu.size()
    promo_num = e[e["prev_limitup"] & e["limit_up_close"]].groupby("date").size()
    market["promotion_rate"] = promo_num / market["prior_limitup_names"].replace(0, np.nan)
    broken_num = e[e["touched_limit_up"] & ~e["limit_up_close"]].groupby("date").size()
    market["broken_board_rate"] = broken_num / market["touched_limitup_names"].replace(0, np.nan)
    market = market.reset_index().sort_values("date")
    p = expanding_past_percentile(market["prior_limitup_return_median"], cfg.sentiment_lookback, cfg.sentiment_min_history)
    j = expanding_past_percentile(market["promotion_rate"], cfg.sentiment_lookback, cfg.sentiment_min_history)
    z = expanding_past_percentile(market["broken_board_rate"], cfg.sentiment_lookback, cfg.sentiment_min_history)
    d = expanding_past_percentile(market["limitdown_share"], cfg.sentiment_lookback, cfg.sentiment_min_history)
    mr = expanding_past_percentile(market["market_ret10_median"], cfg.sentiment_lookback, cfg.sentiment_min_history)
    market["sentiment_score"] = 25.0 * (p + j + 1.0 - z + 1.0 - d)
    market["market_ret10_percentile"] = mr
    market["limitdown_percentile"] = d
    market["panic_environment"] = (market["market_ret10_percentile"] <= 0.20) & (market["limitdown_percentile"] >= 0.95)
    x = x.merge(market[["date", "sentiment_score", "panic_environment", "market_ret10_percentile", "limitdown_percentile"]], on="date", how="left", validate="many_to_one")

    eg = x[x["eligible"] & x["industry_id"].ge(0)].copy()
    ind = eg.groupby(["date", "industry_id"]).agg(
        industry_ret3=("ret3", "median"),
        limitup_density=("limit_up_close", "mean"),
        limitup_names=("limit_up_close", "sum"),
        industry_activity=("amount_activity", "median"),
        names=("sid", "size"),
    ).reset_index()
    for src, dst in [("industry_ret3", "r_ret"), ("limitup_density", "r_lu"), ("industry_activity", "r_act")]:
        ind[dst] = ind.groupby("date")[src].rank(pct=True, method="average")
    ind["hotspot_score"] = ind[["r_ret", "r_lu", "r_act"]].mean(axis=1)
    ind["is_hotspot"] = ind["hotspot_score"] >= 0.90
    ind = ind.sort_values(["industry_id", "date"])
    ind["hotspot_prev20"] = ind.groupby("industry_id")["is_hotspot"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    )
    ind["new_hot_industry"] = ind["is_hotspot"] & ~ind["hotspot_prev20"] & ind["limitup_names"].ge(3)
    x = x.merge(ind[["date", "industry_id", "hotspot_score", "is_hotspot", "new_hot_industry", "limitup_names"]], on=["date", "industry_id"], how="left", validate="many_to_one")
    x["is_hotspot"] = x["is_hotspot"].fillna(False).astype(bool)
    x["new_hot_industry"] = x["new_hot_industry"].fillna(False).astype(bool)

    eligible_ind = x["eligible"] & x["industry_id"].ge(0)
    zf = x.loc[eligible_ind].copy()
    group_keys = [zf["date"], zf["industry_id"]]
    zf["rank_ret3"] = zf.groupby(["date", "industry_id"])["ret3"].rank(pct=True)
    zf["rank_lu5"] = zf.groupby(["date", "industry_id"])["limitup5"].rank(pct=True)
    zf["rank_amount3"] = zf.groupby(["date", "industry_id"])["amount3"].rank(pct=True)
    zf["leader_score"] = zf[["rank_ret3", "rank_lu5", "rank_amount3"]].mean(axis=1)
    zf["leader_rank"] = zf.groupby(["date", "industry_id"])["leader_score"].rank(method="first", ascending=False)
    x["leader_score"] = np.nan
    x["leader_rank"] = np.nan
    x.loc[zf.index, "leader_score"] = zf["leader_score"]
    x.loc[zf.index, "leader_rank"] = zf["leader_rank"]
    x["is_leader"] = x["eligible"] & x["is_hotspot"] & x["leader_rank"].le(2)
    x["was_recent_leader"] = x.groupby("sid")["is_leader"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    )
    x["stock_ret10_pct_rank"] = x[x["eligible"]].groupby("date")["ret10"].rank(pct=True).reindex(x.index)

    return x, market, ind


def signal_candidates(x: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    signal_window = x["date"].between(pd.Timestamp(cfg.signal_start), pd.Timestamp(cfg.end_date))
    close_loc = x["close_location"].fillna(0)

    leader = x[
        signal_window & x["eligible"] & x["is_leader"] & x["sentiment_score"].ge(60) &
        x["close"].gt(x["prior_high5"]) & x["ret1"].gt(0) & close_loc.ge(0.70)
    ].copy()
    leader["signal_rank"] = -leader["leader_score"].fillna(-999)

    peak = x["prior_high5"]
    frozen_low = pd.concat([x["low"], x["prior_low3"]], axis=1).min(axis=1)
    depth_atr = (peak - frozen_low) / x["atr20"].replace(0, np.nan)
    pullback = x[
        signal_window & x["eligible"] & x["was_recent_leader"] &
        x["sentiment_score"].ge(35) & x["sentiment_score"].lt(60) &
        depth_atr.between(1.0, 2.5) & x["close"].lt(peak) &
        x["close"].gt(x["open"]) & x["close"].gt(x["prev_close_actual"]) & close_loc.ge(0.60)
    ].copy()
    pullback["target_price"] = peak.loc[pullback.index]
    pullback["stop_price"] = frozen_low.loc[pullback.index]
    pullback["signal_rank"] = depth_atr.loc[pullback.index]

    panic_base = (
        signal_window & x["eligible"] & x["panic_environment"].fillna(False) &
        x["stock_ret10_pct_rank"].le(0.10)
    )
    x2 = x.copy()
    gb = x2.groupby("sid", sort=False)
    panic_series = pd.Series(panic_base.to_numpy(dtype=bool), index=x2.index)
    x2["prev_panic_base"] = panic_series.groupby(x2["sid"], sort=False).shift(1).fillna(False).astype(bool)
    x2["prev_panic_low"] = gb["low"].shift(1)
    x2["prev_panic_close"] = gb["close"].shift(1)
    x2["prev_panic_atr"] = gb["atr20"].shift(1)
    panic = x2[
        signal_window & x2["eligible"] & x2["prev_panic_base"] &
        x2["low"].ge(x2["prev_panic_low"] * 0.995) &
        x2["close"].gt(x2["open"]) & x2["close"].gt(x2["prev_panic_close"]) &
        x2["close_location"].fillna(0).ge(0.60)
    ].copy()
    panic["stop_price"] = panic["prev_panic_low"]
    panic["target_price"] = panic["close"] + panic["prev_panic_atr"]
    panic["signal_rank"] = panic["stock_ret10_pct_rank"]

    second = x[
        signal_window & x["eligible"] & x["limit_up_close"] & x["prev_limitup"] & ~x["prev2_limitup"] &
        x["sentiment_score"].ge(60) & x["volume_ratio"].ge(1.5)
    ].copy()
    second["signal_rank"] = -second["volume_ratio"]

    new_hot = x[
        signal_window & x["eligible"] & x["new_hot_industry"] & x["leader_rank"].le(2) &
        x["sentiment_score"].ge(60) & close_loc.ge(0.60)
    ].copy()
    new_hot["signal_rank"] = -new_hot["leader_score"].fillna(-999)

    return {
        "leader_continuation": leader,
        "pullback_rebound": pullback,
        "panic_rebound_confirmation": panic,
        "second_board_after_close": second,
        "new_hot_industry_proxy": new_hot,
    }


def cost_adjusted_return(entry: float, exit_: float, cfg: Config) -> float:
    buy = entry * (1.0 + cfg.slippage_each_side + cfg.buy_commission + cfg.transfer_fee_each_side)
    sell = exit_ * (1.0 - cfg.slippage_each_side - cfg.sell_commission - cfg.stamp_duty_sell - cfg.transfer_fee_each_side)
    return sell / buy - 1.0


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return center - half, center + half


def build_symbol_arrays(x: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    cols = [
        "date", "open", "high", "low", "close", "ret1", "tradable", "non_st", "eligible",
        "one_price_limit_up", "one_price_limit_down", "sentiment_score", "leader_rank", "is_hotspot",
        "industry_id", "volume_ratio", "limit_up_close", "limit_down_close", "atr20", "prev_low",
    ]
    return {int(sid): g[cols].reset_index(drop=True) for sid, g in x.groupby("sid", sort=False)}


def exit_trade(arr: pd.DataFrame, entry_pos: int, strategy: str, signal_row: pd.Series, cfg: Config) -> Optional[Dict[str, Any]]:
    if entry_pos >= len(arr):
        return None
    entry = arr.iloc[entry_pos]
    if not bool(entry["tradable"]) or not bool(entry["non_st"]) or float(entry["open"]) <= 0:
        return None
    if bool(entry["one_price_limit_up"]):
        return None
    entry_gap = float(entry["open"]) / float(signal_row["close"]) - 1.0
    if entry_gap > cfg.max_entry_gap:
        return None
    entry_price = float(entry["open"])
    max_hold = {
        "leader_continuation": 5,
        "pullback_rebound": 3,
        "panic_rebound_confirmation": 2,
        "second_board_after_close": 3,
        "new_hot_industry_proxy": 5,
    }[strategy]
    target = float(signal_row.get("target_price", np.nan))
    stop = float(signal_row.get("stop_price", np.nan))
    signal_industry = int(signal_row.get("industry_id", -1))
    signal_close = float(signal_row["close"])

    scheduled_open_exit: Optional[int] = None
    last_planned = min(len(arr) - 1, entry_pos + max_hold - 1)
    hard_last = min(len(arr) - 1, last_planned + 10)
    exit_pos: Optional[int] = None
    exit_price: Optional[float] = None
    reason = ""

    for j in range(entry_pos + 1, hard_last + 1):  # T+1: no same-session exit
        row = arr.iloc[j]
        if scheduled_open_exit == j:
            if bool(row["tradable"]) and bool(row["non_st"]) and not bool(row["one_price_limit_down"]):
                exit_pos, exit_price, reason = j, float(row["open"]), "state_exit_next_open"
                break
            scheduled_open_exit = j + 1 if j < hard_last else None

        if not bool(row["tradable"]) or bool(row["one_price_limit_down"]):
            continue

        if np.isfinite(stop) or np.isfinite(target):
            op, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
            if np.isfinite(stop) and op <= stop:
                exit_pos, exit_price, reason = j, op, "stop_gap"
                break
            if np.isfinite(target) and op >= target:
                exit_pos, exit_price, reason = j, op, "target_gap"
                break
            stop_hit = np.isfinite(stop) and lo <= stop
            target_hit = np.isfinite(target) and hi >= target
            if stop_hit and target_hit:
                exit_pos, exit_price, reason = j, stop, "both_touched_stop_first"
                break
            if stop_hit:
                exit_pos, exit_price, reason = j, stop, "stop"
