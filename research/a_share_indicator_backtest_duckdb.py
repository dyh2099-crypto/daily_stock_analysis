#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股技术指标全市场真实回测（GitHub Actions / DuckDB）。

严格历史时点股票池：剔除 ST、当日停牌/零成交、最近60个市场交易日
可交易不足54日、上市不足120个市场交易日，以及20日成交额中位数位于
当日横截面后20%的股票。t日收盘形成信号，t+1收盘买入，t+h+1收盘卖出。
"""
from __future__ import annotations
import argparse, json, math, os, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests

DATASET = "ellendan/a-share-21"
API = "https://datasets-server.huggingface.co/parquet"
SIGNALS: Dict[str, str] = {
    "动量5日": "mom5", "动量20日": "mom20", "动量60日": "mom60",
    "反转5日": "rev5", "反转20日": "rev20", "均线20_60趋势": "ma2060",
    "20日突破": "break20", "RSI14顺势": "rsi_trend", "RSI14反转": "rsi_rev",
    "布林Z顺势": "boll_trend", "布林Z反转": "boll_rev",
    "相对成交额20日": "rel_amt", "低波动20日": "low_vol",
}
HORIZONS = (1, 5, 20)

@dataclass(frozen=True)
class Cfg:
    start: str = "2021-01-01"
    end: str = "2025-02-27"
    min_list_sessions: int = 120
    suspension_window: int = 60
    min_trade_sessions: int = 54
    amount_window: int = 20
    min_amount_obs: int = 15
    amount_pct: float = .20
    min_cs: int = 50
    commission_bp: float = 3.0
    slippage_bp: float = 5.0
    stamp_old_bp: float = 10.0
    stamp_new_bp: float = 5.0
    stamp_change: str = "2023-08-28"


def log(s: str) -> None:
    print(s, flush=True)


def get_json(url: str, params=None, retries=6):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90,
                             headers={"User-Agent": "a-share-backtest/1.0"})
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"请求失败: {url}") from last


def download_data(data_dir: Path) -> List[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = get_json(API, {"dataset": DATASET})
    items = [x for x in payload.get("parquet_files", []) if x.get("split") == "train"]
    if not items: raise RuntimeError(f"Parquet API 无文件: {payload}")
    log(f"Parquet 分片 {len(items)} 个，API标注 {sum(int(x.get('size') or 0) for x in items)/1e9:.2f}GB")
    out = []
    for i, item in enumerate(items):
        p = data_dir / f"part-{i:03d}.parquet"; size = int(item.get("size") or 0)
        if p.exists() and (not size or p.stat().st_size == size): out.append(p); continue
        tmp = p.with_suffix(".part")
        for attempt in range(6):
            try:
                with requests.get(item["url"], stream=True, timeout=(60, 240),
                                  headers={"User-Agent": "a-share-backtest/1.0"}) as r:
                    r.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in r.iter_content(8 << 20):
                            if chunk: f.write(chunk)
                if size and tmp.stat().st_size != size:
                    raise IOError(f"文件大小 {tmp.stat().st_size} != {size}")
                tmp.replace(p); break
            except Exception:
                if attempt == 5: raise
                time.sleep(min(90, 2 ** (attempt + 1)))
        log(f"已下载 {p.name}: {p.stat().st_size/1e6:.0f}MB"); out.append(p)
    return out


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_features(con, glob: str, cfg: Cfg) -> dict:
    log("构建历史时点合格股票池与指标……")
    con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet({q(glob)}, union_by_name=true)")
    con.execute(f"""
    CREATE OR REPLACE TABLE raw AS
    SELECT CAST(code AS VARCHAR) code, CAST(date AS DATE) date,
      TRY_CAST(open AS DOUBLE) open, TRY_CAST(high AS DOUBLE) high,
      TRY_CAST(low AS DOUBLE) low, TRY_CAST(close AS DOUBLE) close,
      TRY_CAST(turnover AS DOUBLE) amount, TRY_CAST(volume AS DOUBLE) volume,
      COALESCE(TRY_CAST(is_paused AS DOUBLE),0) is_paused,
      COALESCE(TRY_CAST(is_st AS DOUBLE),0) is_st,
      CAST(name AS VARCHAR) name, CAST(market AS VARCHAR) market,
      CAST(exchange AS VARCHAR) exchange, TRY_CAST(list_date AS DATE) list_date
    FROM src
    WHERE TRY_CAST(date AS DATE) BETWEEN DATE {q(cfg.start)} AND DATE {q(cfg.end)}
      AND TRY_CAST(close AS DOUBLE)>0
    QUALIFY row_number() OVER(PARTITION BY CAST(code AS VARCHAR),CAST(date AS DATE))=1
    """)
    con.execute("""
    CREATE OR REPLACE TABLE cal AS
    SELECT date, CAST(row_number() OVER(ORDER BY date)-1 AS INTEGER) session
    FROM (SELECT date FROM raw GROUP BY date HAVING count(DISTINCT code)>=100) ORDER BY date
    """)
    con.execute("""
    CREATE OR REPLACE TABLE meta AS
    SELECT s.code, min(c.session) list_session
    FROM (SELECT code,min(list_date) list_date FROM raw GROUP BY code) s
    LEFT JOIN cal c ON c.date>=s.list_date GROUP BY s.code
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE feat AS
    WITH x00 AS (
      SELECT r.*, c.session, m.list_session,
        (c.session-m.list_session+1) listed_sessions,
        (r.is_paused=0 AND r.volume>0 AND r.amount>0) trade_ok,
        (r.is_st<>0 OR regexp_matches(upper(coalesce(r.name,'')),'(^|\\*)ST|退')) st_flag,
        lag(r.close) OVER w pclose,
        lag(r.close,5) OVER w c5, lag(r.close,20) OVER w c20, lag(r.close,60) OVER w c60,
        avg(r.close) OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) ma20,
        avg(r.close) OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) ma60,
        max(r.high) OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) prev_high20,
        avg(r.close) OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) bma20,
        stddev_samp(r.close) OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) bsd20,
        median(CASE WHEN r.is_paused=0 AND r.volume>0 AND r.amount>0 THEN r.amount END)
          OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) med_amt20,
        count(CASE WHEN r.is_paused=0 AND r.volume>0 AND r.amount>0 THEN 1 END)
          OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) amt_n20,
        sum(CASE WHEN r.is_paused=0 AND r.volume>0 AND r.amount>0 THEN 1 ELSE 0 END)
          OVER (PARTITION BY r.code ORDER BY c.session ROWS BETWEEN {cfg.suspension_window-1} PRECEDING AND CURRENT ROW) trade_n60,
        lead(r.close,1) OVER w e1, lead(c.session,1) OVER w es1,
        lead((r.is_paused=0 AND r.volume>0 AND r.amount>0),1) OVER w et1,
        lead(r.close,2) OVER w exit1, lead(c.session,2) OVER w xs1,
        lead((r.is_paused=0 AND r.volume>0 AND r.amount>0),2) OVER w xt1,
        lead(r.close,6) OVER w exit5, lead(c.session,6) OVER w xs5,
        lead((r.is_paused=0 AND r.volume>0 AND r.amount>0),6) OVER w xt5,
        lead(r.close,21) OVER w exit20, lead(c.session,21) OVER w xs20,
        lead((r.is_paused=0 AND r.volume>0 AND r.amount>0),21) OVER w xt20
      FROM raw r JOIN cal c USING(date) LEFT JOIN meta m USING(code)
      WINDOW w AS (PARTITION BY r.code ORDER BY c.session)
    ), x0 AS (
      SELECT *, ln(close/nullif(pclose,0)) lr,
        stddev_samp(ln(close/nullif(pclose,0))) OVER
          (PARTITION BY code ORDER BY session ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) vol20,
        avg(greatest(close-pclose,0)) OVER
          (PARTITION BY code ORDER BY session ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) gain14,
        avg(greatest(pclose-close,0)) OVER
          (PARTITION BY code ORDER BY session ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) loss14
      FROM x00
    ), x1 AS (
      SELECT *, close/c5-1 mom5, close/c20-1 mom20, close/c60-1 mom60,
        ma20/ma60-1 ma2060, close/prev_high20-1 break20,
        (close-bma20)/nullif(bsd20,0) boll,
        amount/nullif(med_amt20,0) rel_amt,
        CASE WHEN loss14=0 THEN 100 ELSE 100-100/(1+gain14/nullif(loss14,0)) END rsi14,
        quantile_cont(med_amt20,{cfg.amount_pct}) OVER(PARTITION BY date) amt_cut
      FROM x0
      WHERE NOT st_flag AND trade_ok AND listed_sessions>={cfg.min_list_sessions}
        AND trade_n60>={cfg.min_trade_sessions} AND amt_n20>={cfg.min_amount_obs}
    )
    SELECT *, -mom5 rev5, -mom20 rev20, rsi14 rsi_trend, -rsi14 rsi_rev,
      boll boll_trend, -boll boll_rev, -vol20 low_vol,
      CASE WHEN es1=session+1 AND xs1=session+2 AND et1 AND xt1 THEN exit1/e1-1 END ret1,
      CASE WHEN es1=session+1 AND xs5=session+6 AND et1 AND xt5 THEN exit5/e1-1 END ret5,
      CASE WHEN es1=session+1 AND xs20=session+21 AND et1 AND xt20 THEN exit20/e1-1 END ret20,
      (med_amt20>=amt_cut) eligible,
      CASE WHEN market LIKE '%创业%' OR code LIKE '300%' OR code LIKE '301%' THEN '创业板'
           WHEN market LIKE '%科创%' OR code LIKE '688%' THEN '科创板'
           WHEN exchange LIKE '%BSE%' OR code LIKE '4%' OR code LIKE '8%' THEN '北交所'
           ELSE '主板' END board
    FROM x1
    """)
    return con.execute("""
      SELECT count(*) rows,count(DISTINCT code) stocks,min(date) min_date,max(date) max_date,
        sum(eligible::INT) eligible_rows,count(DISTINCT CASE WHEN eligible THEN code END) eligible_stocks
      FROM feat
    """).fetchdf().iloc[0].to_dict()


def nw_t(x: pd.Series, lag: int) -> Tuple[float,float]:
    a=np.asarray(x.dropna(),float); n=len(a)
    if n<20:return math.nan,math.nan
    u=a-a.mean(); g0=float(u@u/n); v=g0
    for k in range(1,min(lag,n-2)+1):
        g=float(u[k:]@u[:-k]/n); v+=2*(1-k/(lag+1))*g
    se=math.sqrt(max(v,0)/n); t=float(a.mean()/se) if se else math.nan
    p=math.erfc(abs(t)/math.sqrt(2)) if math.isfinite(t) else math.nan
    return t,p


def perf(g: pd.DataFrame,h:int) -> dict:
    r=g.sort_values('date')['net'].dropna().astype(float)
    if len(r)<10:return {}
    ppy=252/h; mu=r.mean(); sd=r.std(ddof=1)
    wealth=(1+r).cumprod(); dd=wealth/wealth.cummax()-1
    return {'n_periods':len(r),'ann_net':(1+mu)**ppy-1 if mu>-1 else -1,
            'sharpe':mu/sd*math.sqrt(ppy) if sd>0 else math.nan,
            'max_dd':dd.min(),'win_rate':(r>0).mean(),'avg_turnover':g['turnover'].mean()}


def bh(p: pd.Series) -> pd.Series:
    x=p.astype(float); ok=x.notna(); vals=x[ok].values; out=pd.Series(np.nan,index=x.index)
    if not len(vals):return out
    order=np.argsort(vals); ranked=vals[order]; adj=np.minimum.accumulate((ranked*len(vals)/np.arange(1,len(vals)+1))[::-1])[::-1]
    back=np.empty(len(vals)); back[order]=np.minimum(adj,1); out.loc[ok]=back; return out


def evaluate(con,cfg:Cfg,segment:str,label:str,col:str,h:int):
    seg="" if segment=='ALL' else f"AND board={q(segment)}"
    ret=f"ret{h}"
    sql=f"""
    WITH b AS (
      SELECT date,session,code,{col} signal,{ret} fwd,
        rank() OVER(PARTITION BY date ORDER BY {col}) sr,
        rank() OVER(PARTITION BY date ORDER BY {ret}) rr,
        ntile(5) OVER(PARTITION BY date ORDER BY {col}) q,
        count(*) OVER(PARTITION BY date) n
      FROM feat WHERE eligible AND {col} IS NOT NULL AND {ret} IS NOT NULL {seg}
    ), good AS (SELECT * FROM b WHERE n>={cfg.min_cs}),
    d AS (
      SELECT date,min(session) session,corr(sr,rr) ic,avg(fwd) benchmark,
        avg(fwd) FILTER(WHERE q=5) top,avg(fwd) FILTER(WHERE q=1) bottom,count(*) n
      FROM good GROUP BY date
    ), topcodes AS (SELECT date,session,code FROM good WHERE q=5),
    dates AS (
      SELECT date,session,session%{h} offs,count(*) ncur,
        lag(date) OVER(PARTITION BY session%{h} ORDER BY date) pdate,
        lag(count(*)) OVER(PARTITION BY session%{h} ORDER BY date) nprev
      FROM topcodes GROUP BY date,session
    ), trn AS (
      SELECT d.date,d.offs,d.ncur,d.nprev,count(p.code) overlap,
        CASE WHEN d.pdate IS NULL THEN 1.0
          ELSE 1-count(p.code)*least(1.0/d.ncur,1.0/d.nprev) END turnover
      FROM dates d JOIN topcodes t ON t.date=d.date
      LEFT JOIN topcodes p ON p.date=d.pdate AND p.code=t.code
      GROUP BY d.date,d.offs,d.ncur,d.nprev,d.pdate
    )
    SELECT d.*,t.offs,t.turnover,
      d.top-d.benchmark gross,
      d.top-d.benchmark-t.turnover*((2*({cfg.commission_bp}+{cfg.slippage_bp})+
        CASE WHEN d.date<DATE {q(cfg.stamp_change)} THEN {cfg.stamp_old_bp} ELSE {cfg.stamp_new_bp} END)/10000.0) net
    FROM d JOIN trn t USING(date) ORDER BY date
    """
    df=con.execute(sql).fetchdf()
    if df.empty:return None,None
    t,p=nw_t(df.ic,h)
    rows=[]
    for off,g in df.groupby('offs'):
        z=perf(g,h)
        if z:z.update(offset=int(off));rows.append(z)
    od=pd.DataFrame(rows)
    if od.empty:return None,None
    years=(df.assign(year=pd.to_datetime(df.date).dt.year).groupby(['offs','year']).net.mean()>0).mean()
    s={'segment':segment,'signal':label,'horizon':h,'days':len(df),'mean_ic':df.ic.mean(),
       'ic_t':t,'ic_p':p,'gross_ann_simple':df.gross.mean()*252/h,
       'median_ann_net':od.ann_net.median(),'median_sharpe':od.sharpe.median(),
       'median_max_dd':od.max_dd.median(),'positive_year_offset':years,
       'avg_turnover':od.avg_turnover.mean()}
    df.insert(0,'segment',segment);df.insert(1,'signal',label);df.insert(2,'horizon',h)
    return s,df


def classify(s:pd.DataFrame)->pd.DataFrame:
    s=s.copy();s['fdr_q']=bh(s.ic_p)
    def f(r):
        if r.fdr_q<=.10 and r.mean_ic>0 and r.median_ann_net>0 and r.median_sharpe>=.5 and r.positive_year_offset>=.60:
            return '未失效（稳健通过）'
        if r.fdr_q<=.10 and r.mean_ic<0:return '当前方向失效（反向关系显著）'
        if r.mean_ic>0 and r.median_ann_net>0 and (r.fdr_q<=.20 or r.median_sharpe>0):return '条件有效/较弱'
        return '失效或不可交易'
    s['classification']=s.apply(f,axis=1);return s


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='artifacts/a_share_backtest')
    ap.add_argument('--data-dir',default='.cache/a_share21');ap.add_argument('--work-dir',default='.cache/a_share_work')
    a=ap.parse_args();cfg=Cfg();out=Path(a.output);data=Path(a.data_dir);work=Path(a.work_dir)
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    paths=download_data(data);(work/'tmp').mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(work/'backtest.duckdb'))
    con.execute("SET memory_limit='5GB'");con.execute("SET threads=2");con.execute(f"SET temp_directory={q(str(work/'tmp'))}")
    stats=build_features(con,str(data/'part-*.parquet'),cfg)
    audit=con.execute("""SELECT date,count(*) raw_rows,sum(eligible::INT) eligible,
      count(DISTINCT CASE WHEN eligible THEN code END) eligible_stocks FROM feat GROUP BY date ORDER BY date""").fetchdf()
    boards=con.execute("""SELECT board FROM
      (SELECT date,board,count(*) n FROM feat WHERE eligible GROUP BY date,board)
      GROUP BY board HAVING median(n)>=50 ORDER BY board""").fetchdf()
    segments=['ALL']+boards.board.astype(str).tolist()
    summaries=[];daily=[];total=len(segments)*len(SIGNALS)*len(HORIZONS);k=0
    for seg in segments:
      for label,col in SIGNALS.items():
       for h in HORIZONS:
        k+=1;log(f"[{k}/{total}] {seg}/{label}/{h}日")
        s,d=evaluate(con,cfg,seg,label,col,h)
        if s:summaries.append(s);daily.append(d)
    summ=classify(pd.DataFrame(summaries));day=pd.concat(daily,ignore_index=True) if daily else pd.DataFrame()
    audit.to_csv(out/'01_universe_audit.csv',index=False,encoding='utf-8-sig')
    summ.to_csv(out/'02_indicator_classification.csv',index=False,encoding='utf-8-sig')
    day.to_csv(out/'03_daily_results.csv.gz',index=False,compression='gzip')
    run={'dataset':DATASET,'config':asdict(cfg),'stats':{k:str(v) for k,v in stats.items()},'segments':segments}
    (out/'00_run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    allr=summ[summ.segment=='ALL'].sort_values(['classification','median_sharpe'],ascending=[True,False])
    lines=['# A股技术指标真实回测结果','',f"数据：{DATASET}，{stats['min_date']} 至 {stats['max_date']}；原始清洗后 {int(stats['rows']):,} 行、{int(stats['stocks']):,} 只股票。",
      f"每日合格股票中位数 {audit.eligible_stocks.median():.0f}；板块：{', '.join(segments)}。",'',
      '口径：逐日剔除ST、停牌/零成交、近60日可交易不足54日、上市不足120日、20日成交额中位数处于后20%的股票。t日收盘信号，t+1收盘买入，t+h+1收盘卖出；扣佣金、滑点及卖出印花税。','',
      '## 全市场分类','',allr[['signal','horizon','classification','mean_ic','ic_t','fdr_q','median_ann_net','median_sharpe','positive_year_offset','avg_turnover']].to_markdown(index=False),
      '','## 限制','',
      '数据截止2025-02-27，样本约四年；第三方公开数据可能有字段误差。收盘成交模型没有逐笔盘口、冲击成本及涨跌停排队，因此极端行情实盘通常更差。分类方向事先固定，未事后翻转信号。']
    (out/'A股技术指标真实回测报告.md').write_text('\n'.join(lines),encoding='utf-8')
    headline={'stats':run['stats'],'median_eligible':float(audit.eligible_stocks.median()),
      'all_market':allr.replace({np.nan:None}).to_dict(orient='records')}
    (out/'headline.json').write_text(json.dumps(headline,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('RESULT_JSON_START');print(json.dumps(headline,ensure_ascii=False,default=str));print('RESULT_JSON_END')
    con.close();return 0

if __name__=='__main__': raise SystemExit(main())
