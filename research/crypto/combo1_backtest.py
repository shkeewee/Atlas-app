#!/usr/bin/env python3
"""COMBO-1 walk-forward backtest using free Binance Vision archives.

Frozen research spec (2026-08-26):
- Universe: BTC/ETH/BNB/SOL/XRP/DOGE USDT-M perpetuals.
- Test: 2025-08-26 through 2026-08-25; 90d warm-up.
- ECON-2-M: 4 economic primitives (price, leverage, taker flow, carry),
  cross-sectional ranks + interactions with 2 latent market PCs, Ridge alpha=1.
- STATE-RISK-1: expanding GMM market-state model; state is tradable only when
  its training mean next-day cross-sectional dispersion exceeds training mean.
- OOD abstention: GMM sample log-likelihood below the 1st percentile of the
  contemporaneous training distribution.
- Trade: long highest ECON prediction, short lowest, 00:00->00:00 UTC.
- Cost: 40bp total pair round-trip. Historical funding included.

Important: Binance Vision's metrics archive exposes a 5m taker buy/sell *ratio*,
not underlying buy/sell volumes. Therefore F is frozen here as the mean log 5m
sum_taker_long_short_vol_ratio over the prior 7 complete UTC days. This is the
closest leakage-free historical analogue to the recent-API 7d cumulative taker
buy/sell ratio and is explicitly reported as such.
"""
from __future__ import annotations

import io, json, math, os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"]
BASE = "https://data.binance.vision/data/futures/um"
TEST_START = pd.Timestamp("2025-08-26", tz="UTC")
TEST_END = pd.Timestamp("2026-08-25", tz="UTC")
WARM_START = TEST_START - pd.Timedelta(days=90)
DOWNLOAD_START = WARM_START - pd.Timedelta(days=15)
DOWNLOAD_END = TEST_END + pd.Timedelta(days=2)
CACHE = Path(os.environ.get("COMBO_CACHE", ".combo1_cache"))
OUT = Path(os.environ.get("COMBO_OUT", "combo1_results"))
COST = 0.004
ALPHA = 1.0
N_PC = 2
RNG = np.random.default_rng(42)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"combo1-research/1.0"})


def get_bytes(url: str, retries: int = 5) -> bytes | None:
    for k in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if k == retries-1:
                return None
            time.sleep(0.5 * (2**k))
    return None


def cached_zip_csv(url: str, key: str) -> pd.DataFrame | None:
    p = CACHE / key
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = p.read_bytes() if p.exists() else get_bytes(url)
    if raw is None:
        return None
    if not p.exists():
        p.write_bytes(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names: return None
            return pd.read_csv(z.open(names[0]))
    except Exception:
        return None


def months(a: pd.Timestamp, b: pd.Timestamp):
    cur = pd.Timestamp(a.year, a.month, 1, tz="UTC")
    end = pd.Timestamp(b.year, b.month, 1, tz="UTC")
    while cur <= end:
        yield cur.strftime("%Y-%m")
        cur += pd.offsets.MonthBegin(1)


def load_monthly(symbol: str, kind: str, interval: str | None = None) -> pd.DataFrame:
    parts=[]
    for ym in months(DOWNLOAD_START, DOWNLOAD_END):
        if kind == "klines":
            fn=f"{symbol}-{interval}-{ym}.zip"
            url=f"{BASE}/monthly/klines/{symbol}/{interval}/{fn}"
        elif kind == "premiumIndexKlines":
            fn=f"{symbol}-{interval}-{ym}.zip"
            url=f"{BASE}/monthly/premiumIndexKlines/{symbol}/{interval}/{fn}"
        elif kind == "fundingRate":
            fn=f"{symbol}-fundingRate-{ym}.zip"
            url=f"{BASE}/monthly/fundingRate/{symbol}/{fn}"
        else: raise ValueError(kind)
        df=cached_zip_csv(url, f"monthly/{kind}/{symbol}/{fn}")
        if df is not None and len(df): parts.append(df)
    if not parts: raise RuntimeError(f"No {kind} data for {symbol}")
    return pd.concat(parts, ignore_index=True)


def metric_url(symbol: str, day: pd.Timestamp):
    ds=day.strftime("%Y-%m-%d")
    fn=f"{symbol}-metrics-{ds}.zip"
    return f"{BASE}/daily/metrics/{symbol}/{fn}", f"daily/metrics/{symbol}/{fn}"


def load_metrics_symbol(symbol: str) -> pd.DataFrame:
    days=pd.date_range(DOWNLOAD_START.floor("D"), DOWNLOAD_END.floor("D"), freq="D", tz="UTC")
    def one(d):
        url,key=metric_url(symbol,d)
        df=cached_zip_csv(url,key)
        if df is None or not len(df): return None
        df["archive_date"]=d
        return df
    parts=[]
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs={ex.submit(one,d): d for d in days}
        for f in as_completed(futs):
            x=f.result()
            if x is not None: parts.append(x)
    if not parts: raise RuntimeError(f"No metrics for {symbol}")
    return pd.concat(parts, ignore_index=True)


def find_col(df, candidates):
    low={str(c).lower():c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low: return low[cand.lower()]
    for c in df.columns:
        lc=str(c).lower()
        if any(cand.lower() in lc for cand in candidates): return c
    return None


def parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    # Vision files can have headers or Binance's positional schema.
    if "open_time" in [str(c).lower() for c in df.columns]:
        t=find_col(df,["open_time"]); op=find_col(df,["open"]); hi=find_col(df,["high"])
        lo=find_col(df,["low"]); cl=find_col(df,["close"]); vol=find_col(df,["quote_volume","quote asset volume"])
        if vol is None: vol=find_col(df,["volume"])
    else:
        # pd.read_csv may have consumed first data row as header. Re-read impossible here;
        # most current Vision archives are headered. Fail loudly if key columns absent.
        cols=list(df.columns)
        if len(cols) < 8: raise RuntimeError(f"Unexpected kline schema {cols}")
        t,op,hi,lo,cl,vol=cols[0],cols[1],cols[2],cols[3],cols[4],cols[7]
    out=pd.DataFrame({"ts":pd.to_datetime(pd.to_numeric(df[t],errors="coerce"),unit="ms",utc=True),
                      "open":pd.to_numeric(df[op],errors="coerce"),"high":pd.to_numeric(df[hi],errors="coerce"),
                      "low":pd.to_numeric(df[lo],errors="coerce"),"close":pd.to_numeric(df[cl],errors="coerce"),
                      "quote_vol":pd.to_numeric(df[vol],errors="coerce")})
    return out.dropna(subset=["ts","open","close"]).drop_duplicates("ts").sort_values("ts")


def parse_premium(df: pd.DataFrame) -> pd.DataFrame:
    k=parse_klines(df)
    return k[["ts","close"]].rename(columns={"close":"premium"})


def parse_funding(df: pd.DataFrame) -> pd.DataFrame:
    tc=find_col(df,["fundingTime","funding_time","calc_time","time"])
    rc=find_col(df,["fundingRate","funding_rate","lastFundingRate"])
    if tc is None or rc is None:
        cols=list(df.columns)
        # common Vision funding CSV order: calc_time,funding_interval_hours,last_funding_rate
        if len(cols)>=3: tc,rc=cols[0],cols[-1]
        else: raise RuntimeError(f"Unexpected funding schema {cols}")
    return pd.DataFrame({"ts":pd.to_datetime(pd.to_numeric(df[tc],errors="coerce"),unit="ms",utc=True),
                         "funding":pd.to_numeric(df[rc],errors="coerce")}).dropna().drop_duplicates("ts").sort_values("ts")


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    tc=find_col(df,["create_time","timestamp","time"])
    oi=find_col(df,["sum_open_interest"])
    tr=find_col(df,["sum_taker_long_short_vol_ratio","taker_long_short_vol_ratio"])
    if oi is None or tr is None: raise RuntimeError(f"Metrics schema lacks OI/taker ratio: {list(df.columns)}")
    if tc is None:
        # fallback: date column sometimes encoded as create_time string
        tc=df.columns[0]
    rawt=df[tc]
    num=pd.to_numeric(rawt,errors="coerce")
    if num.notna().mean()>.9:
        unit="ms" if num.dropna().median()>1e11 else "s"
        ts=pd.to_datetime(num,unit=unit,utc=True,errors="coerce")
    else:
        ts=pd.to_datetime(rawt,utc=True,errors="coerce")
    out=pd.DataFrame({"ts":ts,"oi":pd.to_numeric(df[oi],errors="coerce"),"taker_ratio":pd.to_numeric(df[tr],errors="coerce")})
    return out.dropna(subset=["ts","oi","taker_ratio"]).drop_duplicates("ts").sort_values("ts")


def cs_rank(s: pd.Series) -> pd.Series:
    n=s.notna().sum()
    if n<=1: return pd.Series(0.0,index=s.index)
    r=s.rank(method="average")
    return 2*(r-1)/(n-1)-1


def max_drawdown(r: pd.Series):
    eq=(1+r.fillna(0)).cumprod(); dd=eq/eq.cummax()-1
    return float(dd.min())


def perf(name, r: pd.Series, active: pd.Series):
    r=r.fillna(0).astype(float); a=active.astype(bool)
    total=float((1+r).prod()-1)
    ann=float((1+total)**(365/max(len(r),1))-1) if total>-1 else -1
    sd=r.std(ddof=1); sharpe=float(r.mean()/sd*np.sqrt(365)) if sd>0 else np.nan
    ar=r[a]
    return {"name":name,"days":int(len(r)),"active_days":int(a.sum()),"total_return":total,"annualized_return":ann,
            "sharpe":sharpe,"max_drawdown":max_drawdown(r),"active_hit_rate":float((ar>0).mean()) if len(ar) else np.nan,
            "mean_active":float(ar.mean()) if len(ar) else np.nan,"median_active":float(ar.median()) if len(ar) else np.nan}


def main():
    CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    asset={}; quality={}
    print("Downloading/reading Binance Vision archives...", flush=True)
    for s in SYMS:
        print(" ",s,flush=True)
        k=parse_klines(load_monthly(s,"klines","1d"))
        p=parse_premium(load_monthly(s,"premiumIndexKlines","1d"))
        f=parse_funding(load_monthly(s,"fundingRate"))
        m=parse_metrics(load_metrics_symbol(s))
        quality[s]={"kline_rows":len(k),"metric_rows":len(m),"metric_start":str(m.ts.min()),"metric_end":str(m.ts.max()),
                    "metric_duplicate_ts":int(m.ts.duplicated().sum())}
        # Daily boundary table indexed by UTC date. At boundary d use last metric strictly before d.
        idx=pd.date_range(DOWNLOAD_START.floor("D"), DOWNLOAD_END.floor("D"),freq="D",tz="UTC")
        d=pd.DataFrame(index=idx)
        ko=k.set_index(k.ts.dt.floor("D"))
        d["open"]=ko["open"].groupby(level=0).first(); d["high"]=ko["high"].groupby(level=0).max(); d["low"]=ko["low"].groupby(level=0).min()
        d["close"]=ko["close"].groupby(level=0).last(); d["quote_vol"]=ko["quote_vol"].groupby(level=0).sum()
        pp=p.set_index(p.ts.dt.floor("D")); d["premium_close"]=pp["premium"].groupby(level=0).last()
        mm=m.copy(); mm["day"]=mm.ts.dt.floor("D")
        # last OI from each complete day; boundary d sees day d-1
        lastoi=mm.groupby("day")["oi"].last(); d["oi_prev"]=lastoi.shift(1)
        # 5m log taker ratio aggregated over each complete day; boundary d sees d-1 etc.
        mm=mm[(mm.taker_ratio>0)&np.isfinite(mm.taker_ratio)]
        flowday=mm.assign(logtr=np.log(mm.taker_ratio)).groupby("day")["logtr"].mean()
        d["flow_day_prev"]=flowday.shift(1)
        # funding paid within [d,d+1)
        ff=f.copy(); ff["day"]=ff.ts.dt.floor("D")
        d["funding_day"]=ff.groupby("day")["funding"].sum()
        asset[s]=d

    # common panel features
    dates=pd.date_range(DOWNLOAD_START.floor("D")+pd.Timedelta(days=14), TEST_END+pd.Timedelta(days=1),freq="D",tz="UTC")
    rows=[]
    for s,d in asset.items():
        x=d.reindex(dates).copy()
        x["P"]=np.log(x.open/x.open.shift(7))
        x["L"]=np.log(x.oi_prev/x.oi_prev.shift(7))
        x["F"]=x.flow_day_prev.rolling(7,min_periods=7).mean()
        x["C"]=x.premium_close.shift(1).rolling(7,min_periods=7).mean()
        ret=np.log(x.open/x.open.shift(1))
        x["rv7"]=ret.shift(1).rolling(7,min_periods=7).std(ddof=1)*np.sqrt(365)
        x["vol_surprise"]=np.log(x.quote_vol.shift(1)/x.quote_vol.shift(2).rolling(7,min_periods=4).median())
        x["fwd_ret"]=x.open.shift(-1)/x.open-1
        x["funding_next"]=x.funding_day.fillna(0)
        x["symbol"]=s; x["date"]=x.index
        rows.append(x.reset_index(drop=True))
    panel=pd.concat(rows,ignore_index=True).sort_values(["date","symbol"])
    panel["excess"]=panel.fwd_ret-panel.groupby("date").fwd_ret.transform("mean")
    for c in ["P","L","F","C"]:
        panel[c+"r"]=panel.groupby("date")[c].transform(cs_rank)

    # market-state features, all observable at boundary date
    g=panel.groupby("date")
    market=pd.DataFrame(index=sorted(panel.date.unique()))
    market["mkt_trend"]=g.P.mean(); market["btc_trend"]=panel[panel.symbol=="BTCUSDT"].set_index("date").P
    market["mkt_rv"]=g.rv7.mean(); market["breadth"]=g.P.apply(lambda s: (s>0).mean())
    market["dispersion_signal"]=g.P.std(ddof=1); market["median_oi"]=g.L.median(); market["median_flow"]=g.F.median()
    market["premium"]=g.C.mean(); market["turnover"]=g.quote_vol.apply(lambda x: np.log(x.sum()) if x.notna().any() and x.sum()>0 else np.nan)
    market["volume_surprise"]=g.vol_surprise.mean()
    # realized target for STATE-RISK: next-day cross-sectional return dispersion
    market["next_dispersion"]=g.fwd_ret.std(ddof=1)

    test_dates=pd.date_range(TEST_START,TEST_END,freq="D",tz="UTC")
    out=[]
    mcols=["mkt_trend","btc_trend","mkt_rv","breadth","dispersion_signal","median_oi","median_flow","premium","turnover","volume_surprise"]
    for t in test_dates:
        # Expanding walk-forward: only rows with outcome known at t are trainable: feature date < t.
        trdates=market.index[(market.index>=WARM_START)&(market.index<t)]
        trm=market.loc[trdates,mcols+['next_dispersion']].dropna()
        curm=market.loc[[t],mcols].dropna()
        if len(trm)<45 or curm.empty: continue
        scaler=StandardScaler().fit(trm[mcols]); Z=scaler.transform(trm[mcols]); zt=scaler.transform(curm[mcols])
        pca=PCA(n_components=N_PC,random_state=42).fit(Z); pcs=pca.transform(Z); pct=pca.transform(zt)[0]
        pcdf=pd.DataFrame(pcs,index=trm.index,columns=[f"pc{i+1}" for i in range(N_PC)])
        # directional training panel
        trp=panel[(panel.date.isin(trm.index))].dropna(subset=["Pr","Lr","Fr","Cr","excess"])
        # attach historical PCs
        trp=trp.join(pcdf,on="date")
        base=["Pr","Lr","Fr","Cr"]
        feats=base.copy()
        for b in base:
            for j in range(N_PC):
                nm=f"{b}_x_pc{j+1}"; trp[nm]=trp[b]*trp[f"pc{j+1}"]; feats.append(nm)
        model=Ridge(alpha=ALPHA).fit(trp[feats],trp.excess)
        cp=panel[panel.date==t].dropna(subset=base).copy()
        if len(cp)!=len(SYMS): continue
        for j in range(N_PC): cp[f"pc{j+1}"]=pct[j]
        for b in base:
            for j in range(N_PC): cp[f"{b}_x_pc{j+1}"]=cp[b]*pct[j]
        cp["pred"]=model.predict(cp[feats])
        long=cp.loc[cp.pred.idxmax()]; short=cp.loc[cp.pred.idxmin()]

        # STATE-RISK expanding GMM, k selected by training BIC only.
        ks=range(2,min(6,max(3,len(trm)//12+1)))
        gmms=[]
        for k in ks:
            try:
                gm=GaussianMixture(n_components=k,covariance_type="full",reg_covar=1e-5,random_state=42,n_init=10).fit(Z)
                gmms.append((gm.bic(Z),gm))
            except Exception: pass
        if not gmms: continue
        gm=min(gmms,key=lambda z:z[0])[1]
        labels=gm.predict(Z); lab=int(gm.predict(zt)[0]); scores=gm.score_samples(Z); cur_score=float(gm.score_samples(zt)[0])
        overall=float(trm.next_dispersion.mean())
        state_mean={j:float(trm.next_dispersion.values[labels==j].mean()) for j in np.unique(labels)}
        high=state_mean.get(lab,-np.inf)>overall
        ood=cur_score < float(np.quantile(scores,0.01))
        active=bool(high and not ood)

        spread=float(long.fwd_ret-short.fwd_ret)
        funding_adj=float(-long.funding_next + short.funding_next)
        net=float(spread+funding_adj-COST) if active else 0.0
        econ_net=float(spread+funding_adj-COST)
        # 50/50 gross-capital normalized return; keep spread convention too.
        net_half=net/2
        out.append({"date":str(t.date()),"active":active,"high_disp_state":bool(high),"ood":bool(ood),"gmm_k":gm.n_components,
                    "state":lab,"state_pred_disp":state_mean.get(lab,np.nan),"train_mean_disp":overall,
                    "long":long.symbol,"short":short.symbol,"long_pred":float(long.pred),"short_pred":float(short.pred),
                    "long_ret":float(long.fwd_ret),"short_ret":float(short.fwd_ret),"funding_adj":funding_adj,
                    "gross_spread":spread,"net_spread":net,"net_50_50":net_half,"econ_everyday_net":econ_net,
                    "actual_dispersion":float(market.loc[t,"next_dispersion"]),"ood_score":cur_score})

    res=pd.DataFrame(out)
    if res.empty: raise RuntimeError("No walk-forward results")
    res["date"]=pd.to_datetime(res.date,utc=True); res=res.set_index("date")

    summary=[]
    summary.append(perf("COMBO-1 spread convention",res.net_spread,res.active))
    summary.append(perf("COMBO-1 50/50 capital normalized",res.net_50_50,res.active))
    summary.append(perf("ECON-2-M every day",res.econ_everyday_net,pd.Series(True,index=res.index)))
    # simple 7d momentum control, every day and on same gate
    mom=[]
    for t in res.index:
        cp=panel[panel.date==t].dropna(subset=["P","fwd_ret"])
        lg=cp.loc[cp.P.idxmax()]; sh=cp.loc[cp.P.idxmin()]
        r=float(lg.fwd_ret-sh.fwd_ret-lg.funding_next+sh.funding_next-COST)
        mom.append(r)
    res["momentum_net"]=mom; res["momentum_gated"]=np.where(res.active,res.momentum_net,0)
    summary.append(perf("7d momentum every day",res.momentum_net,pd.Series(True,index=res.index)))
    summary.append(perf("7d momentum same gate",res.momentum_gated,res.active))

    # STATE-RISK gate value: random-day controls using the ECON every-day return series and same trade count.
    n=int(res.active.sum()); econ=res.econ_everyday_net.values; obs=float((1+res.net_spread).prod()-1)
    mc=[]
    if n>0:
        for _ in range(5000):
            pick=RNG.choice(len(econ),size=n,replace=False); rr=np.zeros(len(econ)); rr[pick]=econ[pick]; mc.append(np.prod(1+rr)-1)
    gate_p=float((1+np.sum(np.asarray(mc)>=obs))/(len(mc)+1)) if mc else np.nan

    # stress remove best 1/3/5 active COMBO trades
    stress={}
    active_returns=res.loc[res.active,"net_spread"].sort_values(ascending=False)
    for k in [0,1,3,5]:
        rem=active_returns.iloc[k:] if k else active_returns
        stress[f"remove_best_{k}"]=float((1+rem).prod()-1) if len(rem) else np.nan

    # information coefficients of directional model on all test asset-days, reconstructed from long/short only not enough;
    # report gate/dispersion association directly.
    rho,pv=spearmanr(res.state_pred_disp,res.actual_dispersion,nan_policy="omit")
    diagnostics={"state_risk_spearman":float(rho),"state_risk_p":float(pv),"random_gate_one_sided_p":gate_p,
                 "active_days":n,"stress":stress,"data_quality":quality,
                 "spec_notes":{"flow":"mean log 5m Binance Vision taker ratio over prior 7 complete days",
                               "ood":"current GMM log-likelihood below training 1st percentile","cost":COST,"ridge_alpha":ALPHA,"market_pcs":N_PC}}

    # yearly/quarter splits
    tmp=res.copy(); tmp["quarter"]=tmp.index.to_period("Q").astype(str)
    quarters={q:{"active":int(z.active.sum()),"net":float((1+z.net_spread).prod()-1),"econ_all":float((1+z.econ_everyday_net).prod()-1)} for q,z in tmp.groupby("quarter")}
    diagnostics["quarters"]=quarters

    res.to_csv(OUT/"daily_results.csv")
    (OUT/"summary.json").write_text(json.dumps({"summary":summary,"diagnostics":diagnostics},indent=2,default=str))
    lines=["# COMBO-1 one-year walk-forward backtest","",f"Test: {TEST_START.date()} to {TEST_END.date()}","",
           "| Model | Active days | Total | Ann. | Sharpe | Max DD | Hit |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['name']} | {s['active_days']} | {s['total_return']:.2%} | {s['annualized_return']:.2%} | {s['sharpe']:.2f} | {s['max_drawdown']:.2%} | {s['active_hit_rate']:.1%} |")
    lines += ["","## Robustness",f"- STATE-RISK predicted-vs-realized dispersion Spearman: {rho:.3f} (p={pv:.4g})",
              f"- Random-day gate one-sided Monte Carlo p: {gate_p:.4g}",f"- Stress: {stress}","","## Important implementation note",
              "Historical Binance Vision metrics contain the taker buy/sell ratio but not the underlying taker volumes. The historical flow primitive is therefore the 7-day mean of log 5-minute taker ratios. This was fixed before running the one-year outcomes and is not an optimized substitution.","",
              "## Quarterly", "```json", json.dumps(quarters,indent=2), "```"]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
