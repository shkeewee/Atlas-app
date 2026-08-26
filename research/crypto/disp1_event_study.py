#!/usr/bin/env python3
"""DISP-1 event study: can frozen STATE-RISK monetize cross-sectional dispersion?

Pre-registered primary question (2026-08-26, before viewing these intraday outcomes):
- Reuse the exact one-year STATE-RISK gate produced by COMBO-1 run 32982359677.
- Universe: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT.
- Test dates: whatever valid dates are in that frozen daily_results.csv.
- At 00:00 UTC no directional coin forecast is used.
- Measure cross-sectional dispersion at 1h, 2h, 4h, and 24h.
- Primary monetization probe: at 02:00 UTC, rank already-realized *relative* returns
  since 00:00; long equal-weight top 2 and short equal-weight bottom 2 until next
  00:00. This tests whether early divergence continues. The exact reversal is
  reported symmetrically, not selected based on outcome.
- Costs: 40 bp total long-short round-trip spread convention. Historical funding
  between entry and exit is included when available.
- Secondary nearby entry horizons: 01:00 and 04:00 UTC, reported without selecting
  the best horizon as the 'answer'.
- Compare STATE-RISK active days with the same rule on all days and with random
  matched-frequency gates. No threshold search or parameter optimization.

This is retrospective mechanism research. Positive results are candidates, not a
proven trading edge.
"""
from __future__ import annotations

import io, json, math, os, time, zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

SYMS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"]
BASE="https://data.binance.vision/data/futures/um"
COST=0.004
CACHE=Path(os.environ.get("DISP_CACHE",".disp1_cache"))
OUT=Path(os.environ.get("DISP_OUT","disp1_results"))
INPUT=Path(os.environ.get("COMBO_DAILY","combo1_results/daily_results.csv"))
RNG=np.random.default_rng(20260826)
S=requests.Session(); S.headers.update({"User-Agent":"disp1-research/1.0"})


def get_bytes(url,retries=4):
    for k in range(retries):
        try:
            r=S.get(url,timeout=30)
            if r.status_code==404: return None
            r.raise_for_status(); return r.content
        except Exception:
            if k==retries-1: return None
            time.sleep(0.5*(2**k))
    return None


def zip_csv(url,key):
    p=CACHE/key; p.parent.mkdir(parents=True,exist_ok=True)
    raw=p.read_bytes() if p.exists() else get_bytes(url)
    if raw is None: return None
    if not p.exists(): p.write_bytes(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv')]
            if not names: return None
            b=z.open(names[0]).read()
        df=pd.read_csv(io.BytesIO(b),header=None)
        if df.empty: return df
        first=[str(v).strip().lower() for v in df.iloc[0].tolist()]
        toks=("open_time","funding_time","fundingrate","funding_rate","calc_time","symbol")
        if any(any(tok in v for tok in toks) for v in first):
            cols=[str(v).strip() for v in df.iloc[0].tolist()]
            df=df.iloc[1:].reset_index(drop=True); df.columns=cols
        return df
    except Exception:
        return None


def dtcol(x):
    num=pd.to_numeric(x,errors='coerce')
    med=num.dropna().median() if num.notna().any() else np.nan
    if np.isfinite(med):
        unit='us' if med>1e14 else ('ms' if med>1e11 else 's')
        return pd.to_datetime(num,unit=unit,utc=True,errors='coerce')
    return pd.to_datetime(x,utc=True,errors='coerce')


def months(a,b):
    cur=pd.Timestamp(a.year,a.month,1,tz='UTC'); end=pd.Timestamp(b.year,b.month,1,tz='UTC')
    while cur<=end:
        yield cur.strftime('%Y-%m'); cur += pd.offsets.MonthBegin(1)


def load_hourly(symbol,start,end):
    parts=[]
    for ym in months(start,end):
        fn=f"{symbol}-1h-{ym}.zip"; url=f"{BASE}/monthly/klines/{symbol}/1h/{fn}"
        x=zip_csv(url,f"monthly/klines/{symbol}/{fn}")
        if x is not None and len(x): parts.append(x)
    # Current partial month fallback and overlap for robustness.
    d0=max(start.floor('D'),end.floor('D')-pd.Timedelta(days=45))
    for d in pd.date_range(d0,end.floor('D')+pd.Timedelta(days=1),freq='D',tz='UTC'):
        ds=d.strftime('%Y-%m-%d'); fn=f"{symbol}-1h-{ds}.zip"
        url=f"{BASE}/daily/klines/{symbol}/1h/{fn}"
        x=zip_csv(url,f"daily/klines/{symbol}/{fn}")
        if x is not None and len(x): parts.append(x)
    if not parts: raise RuntimeError(f"No hourly data for {symbol}")
    df=pd.concat(parts,ignore_index=True,sort=False)
    cols=list(df.columns)
    low={str(c).lower():c for c in cols}
    if 'open_time' in low:
        tc=low['open_time']; oc=low.get('open'); cc=low.get('close')
    else:
        if len(cols)<5: raise RuntimeError(f"Bad hourly schema {cols}")
        tc,oc,cc=cols[0],cols[1],cols[4]
    out=pd.DataFrame({'ts':dtcol(df[tc]),'open':pd.to_numeric(df[oc],errors='coerce'),'close':pd.to_numeric(df[cc],errors='coerce')})
    return out.dropna().drop_duplicates('ts').sort_values('ts').set_index('ts')


def load_funding(symbol,start,end):
    parts=[]
    for ym in months(start,end):
        fn=f"{symbol}-fundingRate-{ym}.zip"; url=f"{BASE}/monthly/fundingRate/{symbol}/{fn}"
        x=zip_csv(url,f"monthly/funding/{symbol}/{fn}")
        if x is not None and len(x): parts.append(x)
    d0=max(start.floor('D'),end.floor('D')-pd.Timedelta(days=45))
    for d in pd.date_range(d0,end.floor('D')+pd.Timedelta(days=1),freq='D',tz='UTC'):
        ds=d.strftime('%Y-%m-%d'); fn=f"{symbol}-fundingRate-{ds}.zip"
        url=f"{BASE}/daily/fundingRate/{symbol}/{fn}"
        x=zip_csv(url,f"daily/funding/{symbol}/{fn}")
        if x is not None and len(x): parts.append(x)
    if not parts: return pd.Series(dtype=float)
    df=pd.concat(parts,ignore_index=True,sort=False)
    cols=list(df.columns); low={str(c).lower():c for c in cols}
    tc=None; rc=None
    for k in ('fundingtime','funding_time','calc_time','time'):
        if k in low: tc=low[k]; break
    for k in ('fundingrate','funding_rate','last_funding_rate','lastfundingrate'):
        if k in low: rc=low[k]; break
    if tc is None or rc is None:
        if len(cols)>=3: tc,rc=cols[0],cols[-1]
        elif len(cols)>=2: tc,rc=cols[0],cols[1]
        else: return pd.Series(dtype=float)
    out=pd.DataFrame({'ts':dtcol(df[tc]),'rate':pd.to_numeric(df[rc],errors='coerce')}).dropna().drop_duplicates('ts').sort_values('ts')
    return out.set_index('ts').rate


def cs_rel(vals: Dict[str,float]):
    s=pd.Series(vals,dtype=float).dropna(); return s-s.mean()


def spread_trade(rel_entry: pd.Series, px_entry: Dict[str,float], px_exit: Dict[str,float], funding: Dict[str,pd.Series], entry_ts, exit_ts):
    rel_entry=rel_entry.dropna()
    if len(rel_entry)<6: return None
    longs=list(rel_entry.nlargest(2).index); shorts=list(rel_entry.nsmallest(2).index)
    lr=np.mean([px_exit[s]/px_entry[s]-1 for s in longs]); sr=np.mean([px_exit[s]/px_entry[s]-1 for s in shorts])
    gross=float(lr-sr)
    # Long pays positive funding, short receives positive funding. Equal weight within each side.
    fad=0.0
    for s in longs:
        fs=funding.get(s,pd.Series(dtype=float)); fad -= 0.5*float(fs[(fs.index>entry_ts)&(fs.index<=exit_ts)].sum())
    for s in shorts:
        fs=funding.get(s,pd.Series(dtype=float)); fad += 0.5*float(fs[(fs.index>entry_ts)&(fs.index<=exit_ts)].sum())
    return {'longs':','.join(longs),'shorts':','.join(shorts),'gross':gross,'funding_adj':fad,'net':gross+fad-COST,'reversal_net':-gross-fad-COST}


def comp(r):
    x=pd.Series(r,dtype=float).dropna(); return float((1+x).prod()-1) if len(x) else np.nan


def sharpe(r):
    x=pd.Series(r,dtype=float).dropna(); sd=x.std(ddof=1)
    return float(x.mean()/sd*np.sqrt(365)) if len(x)>1 and sd>0 else np.nan


def maxdd(r):
    x=pd.Series(r,dtype=float).fillna(0); eq=(1+x).cumprod(); dd=eq/eq.cummax()-1
    return float(dd.min()) if len(x) else np.nan


def gate_mc(all_returns: pd.Series, active_mask: pd.Series, obs, nperm=10000):
    z=all_returns.dropna(); mask=active_mask.reindex(z.index).fillna(False); n=int(mask.sum())
    if n<2 or n>=len(z): return np.nan
    arr=z.values; sims=np.empty(nperm)
    for i in range(nperm):
        idx=RNG.choice(len(arr),size=n,replace=False); sims[i]=np.prod(1+arr[idx])-1
    return float((1+np.sum(sims>=obs))/(nperm+1))


def mean_diff_perm(x,gate,nperm=10000):
    df=pd.DataFrame({'x':x,'g':gate}).dropna(); n=int(df.g.sum())
    if n<2 or n>=len(df): return (np.nan,np.nan)
    obs=float(df.loc[df.g,'x'].mean()-df.loc[~df.g,'x'].mean()); vals=df.x.values
    sims=np.empty(nperm)
    for i in range(nperm):
        idx=RNG.choice(len(vals),size=n,replace=False); m=np.zeros(len(vals),dtype=bool); m[idx]=True
        sims[i]=vals[m].mean()-vals[~m].mean()
    p=float((1+np.sum(np.abs(sims)>=abs(obs)))/(nperm+1)); return obs,p


def main():
    CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    sig=pd.read_csv(INPUT,parse_dates=['date']).set_index('date').sort_index()
    if sig.index.tz is None: sig.index=sig.index.tz_localize('UTC')
    start=sig.index.min()-pd.Timedelta(days=2); end=sig.index.max()+pd.Timedelta(days=2)
    hourly={}; funding={}
    print('Loading intraday/funding archives...',flush=True)
    for s in SYMS:
        print(' ',s,flush=True); hourly[s]=load_hourly(s,start,end); funding[s]=load_funding(s,start,end)

    rows=[]
    for d,row in sig.iterrows():
        t0=d.floor('D'); exit_ts=t0+pd.Timedelta(days=1)
        opens={}; closes={h:{} for h in (1,2,4,24)}
        ok=True
        for s in SYMS:
            hdf=hourly[s]
            try:
                opens[s]=float(hdf.loc[t0,'open'])
                closes[1][s]=float(hdf.loc[t0,'close'])
                closes[2][s]=float(hdf.loc[t0+pd.Timedelta(hours=1),'close'])
                closes[4][s]=float(hdf.loc[t0+pd.Timedelta(hours=3),'close'])
                # Prefer next day open to align exactly with 24h boundary.
                closes[24][s]=float(hdf.loc[exit_ts,'open'])
            except Exception:
                ok=False; break
        if not ok: continue
        rel={}; disp={}
        for h in (1,2,4,24):
            rr={s:closes[h][s]/opens[s]-1 for s in SYMS}; rel[h]=cs_rel(rr); disp[h]=float(rel[h].std(ddof=1))
        # Maximum absolute relative excursion over 24 hourly closes.
        maxexc=0.0
        for hh in range(1,25):
            vals={}
            for s in SYMS:
                ts=t0+pd.Timedelta(hours=hh-1)
                try: vals[s]=float(hourly[s].loc[ts,'close'])/opens[s]-1
                except Exception: vals={}; break
            if len(vals)==6:
                maxexc=max(maxexc,float(cs_rel(vals).abs().max()))
        # Continuation IC: relative first-2h move vs relative 2h->24h move.
        rrest={s:closes[24][s]/closes[2][s]-1 for s in SYMS}; relrest=cs_rel(rrest)
        ic2=float(spearmanr(rel[2],relrest).statistic)
        # Primary 02:00 trade and nearby 01:00/04:00 entries.
        trades={}
        for h in (1,2,4):
            entry_ts=t0+pd.Timedelta(hours=h)
            pxe={s:closes[h][s] for s in SYMS}; pxx={s:closes[24][s] for s in SYMS}
            trades[h]=spread_trade(rel[h],pxe,pxx,funding,entry_ts,exit_ts)
        rec={'date':d,'active':bool(row.get('active',False)),'high_disp_state':bool(row.get('high_disp_state',False)),'ood':bool(row.get('ood',False)),
             'state_pred_disp':float(row.get('state_pred_disp',np.nan)),'disp1':disp[1],'disp2':disp[2],'disp4':disp[4],'disp24':disp[24],
             'max_abs_rel_excursion':maxexc,'cont_ic_2h':ic2}
        for h,tr in trades.items():
            if tr:
                for k,v in tr.items(): rec[f'{k}_{h}h']=v
        rows.append(rec)
    ev=pd.DataFrame(rows).set_index('date').sort_index()
    ev.to_csv(OUT/'event_days.csv')

    # Core descriptive statistics.
    stats={}
    for col in ('disp1','disp2','disp4','disp24','max_abs_rel_excursion','cont_ic_2h'):
        diff,p=mean_diff_perm(ev[col],ev.active)
        rho,rp=spearmanr(ev.state_pred_disp,ev[col],nan_policy='omit')
        stats[col]={'active_mean':float(ev.loc[ev.active,col].mean()),'inactive_mean':float(ev.loc[~ev.active,col].mean()),
                    'active_minus_inactive':diff,'perm_two_sided_p':p,'pred_spearman':float(rho),'pred_spearman_p':float(rp)}

    strategies={}
    for h in (1,2,4):
        for side in ('net','reversal_net'):
            col=f'{side}_{h}h'; allr=ev[col]
            gated=allr.where(ev.active,0.0)
            obs=comp(gated); p=gate_mc(allr,ev.active,obs)
            strategies[f'{side}_{h}h']={'active_days':int(ev.active.sum()),'all_days_total':comp(allr),'gated_total':obs,
                'gated_sharpe':sharpe(gated),'gated_maxdd':maxdd(gated),'gated_hit':float((allr[ev.active]>0).mean()),
                'gated_mean_active':float(allr[ev.active].mean()),'random_gate_one_sided_p':p}

    # Stress primary continuation/reversal: remove best 1/3/5 gated days.
    stress={}
    for side in ('net_2h','reversal_net_2h'):
        ar=ev.loc[ev.active,side].dropna().sort_values(ascending=False)
        stress[side]={}
        for k in (0,1,3,5):
            z=ar.iloc[k:] if k else ar
            stress[side][f'remove_best_{k}']=comp(z)

    report={'n_days':int(len(ev)),'active_days':int(ev.active.sum()),'stats':stats,'strategies':strategies,'stress':stress,
            'notes':{'primary_entry':'02:00 UTC top2 relative leaders vs bottom2 relative laggards','cost':COST,
                     'interpretation':'Continuation and reversal are both reported; selecting the better sign after this retrospective event study would be hypothesis-generation only.'}}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,default=str))

    lines=['# DISP-1 event study','',f"Valid days: {len(ev)}; frozen STATE-RISK active days: {int(ev.active.sum())}",'',
           '## Does STATE-RISK forecast intraday/24h dispersion?','',
           '| Target | Active mean | Inactive mean | Difference | Perm p | Pred rho | rho p |','|---|---:|---:|---:|---:|---:|---:|']
    for col in ('disp1','disp2','disp4','disp24','max_abs_rel_excursion','cont_ic_2h'):
        z=stats[col]; lines.append(f"| {col} | {z['active_mean']:.4%} | {z['inactive_mean']:.4%} | {z['active_minus_inactive']:.4%} | {z['perm_two_sided_p']:.4g} | {z['pred_spearman']:.3f} | {z['pred_spearman_p']:.4g} |")
    lines += ['', '## Direction-free monetization probes','',
              '| Rule | All-days total | Gated total | Gated Sharpe | Max DD | Hit | Mean active | Random-gate p |','|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'net_1h':'1h continuation','net_2h':'2h continuation PRIMARY','net_4h':'4h continuation',
            'reversal_net_1h':'1h reversal','reversal_net_2h':'2h reversal PRIMARY mirror','reversal_net_4h':'4h reversal'}
    for key in ('net_1h','net_2h','net_4h','reversal_net_1h','reversal_net_2h','reversal_net_4h'):
        z=strategies[key]; lines.append(f"| {labels[key]} | {z['all_days_total']:.2%} | {z['gated_total']:.2%} | {z['gated_sharpe']:.2f} | {z['gated_maxdd']:.2%} | {z['gated_hit']:.1%} | {z['gated_mean_active']:.3%} | {z['random_gate_one_sided_p']:.4g} |")
    lines += ['', '## Primary stress', '```json',json.dumps(stress,indent=2),'```','',
              'Interpretation rule: this is a retrospective mechanism study. We do not promote whichever sign/horizon looks best into an edge without a new frozen prospective test.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
