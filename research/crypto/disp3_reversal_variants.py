#!/usr/bin/env python3
"""DISP-3: predeclared reversal-mechanism variants.

Frozen before viewing DISP-3 outcomes (2026-08-26):
- Reuse the frozen one-year STATE-RISK active gate from COMBO-1.
- Primary entry remains 02:00 UTC, exit next 00:00 UTC.
- Reversal only: long early relative losers, short early relative winners.
- Compare economically motivated signal definitions without threshold optimization:
  A) raw equal-weight-relative 2h move;
  B) 60-day rolling BTC-beta residual 2h move;
  C) volatility-normalized raw relative move (asset-specific trailing 60d 2h-relative SD).
- Compare top1/bottom1 versus top2/bottom2.
- Minimum-divergence filter is fixed as current 2h cross-sectional dispersion above
  its trailing 60-day median. A trailing 75th percentile version is secondary only.
- Costs reported at 10 bp and 20 bp total long-short round trip, plus actual funding.
- No parameter selection based on DISP-3 outcomes.

Retrospective mechanism research only; positive variants require a new frozen
prospective/OOS validation before being called an edge.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

import disp1_event_study as d1

SYMS=d1.SYMS
COSTS=[10,20]
OUT=Path(os.environ.get('DISP3_OUT','disp3_results'))
COMBO=Path(os.environ.get('COMBO_DAILY','combo1_results/daily_results.csv'))
RNG=np.random.default_rng(20260826)


def comp(x):
    x=pd.Series(x,dtype=float).fillna(0)
    return float((1+x).prod()-1)

def sharpe(x):
    x=pd.Series(x,dtype=float).dropna(); sd=x.std(ddof=1)
    return float(x.mean()/sd*np.sqrt(365)) if len(x)>1 and sd>0 else np.nan

def maxdd(x):
    x=pd.Series(x,dtype=float).fillna(0); eq=(1+x).cumprod(); dd=eq/eq.cummax()-1
    return float(dd.min()) if len(x) else np.nan

def random_gate_p(allr,gate,obs,nperm=10000):
    df=pd.DataFrame({'r':allr,'g':gate}).dropna(); n=int(df.g.sum())
    if n<2 or n>=len(df): return np.nan
    arr=df.r.values; sims=np.empty(nperm)
    for i in range(nperm):
        idx=RNG.choice(len(arr),size=n,replace=False); sims[i]=np.prod(1+arr[idx])-1
    return float((1+np.sum(sims>=obs))/(nperm+1))

def portfolio_reversal(signal, entry_px, exit_px, funding, entry_ts, exit_ts, nside=2):
    s=pd.Series(signal,dtype=float).dropna()
    if len(s)<len(SYMS): return None
    longs=list(s.nsmallest(nside).index); shorts=list(s.nlargest(nside).index)
    lr=np.mean([exit_px[x]/entry_px[x]-1 for x in longs])
    sr=np.mean([exit_px[x]/entry_px[x]-1 for x in shorts])
    gross=float(lr-sr)
    fad=0.0
    w=1.0/nside
    for x in longs:
        fs=funding.get(x,pd.Series(dtype=float)); fad -= w*float(fs[(fs.index>entry_ts)&(fs.index<=exit_ts)].sum())
    for x in shorts:
        fs=funding.get(x,pd.Series(dtype=float)); fad += w*float(fs[(fs.index>entry_ts)&(fs.index<=exit_ts)].sum())
    return {'gross_funding':gross+fad,'longs':','.join(longs),'shorts':','.join(shorts)}


def main():
    OUT.mkdir(exist_ok=True); d1.CACHE.mkdir(exist_ok=True)
    sig=pd.read_csv(COMBO,parse_dates=['date']).set_index('date').sort_index()
    if sig.index.tz is None: sig.index=sig.index.tz_localize('UTC')
    start=sig.index.min()-pd.Timedelta(days=75); end=sig.index.max()+pd.Timedelta(days=2)
    hourly={}; funding={}
    print('Loading hourly/funding data for DISP-3...',flush=True)
    for s in SYMS:
        print(' ',s,flush=True); hourly[s]=d1.load_hourly(s,start,end); funding[s]=d1.load_funding(s,start,end)

    # Build daily 2h and 24h returns for trailing beta/vol estimates.
    all_days=pd.date_range(start.floor('D'),end.floor('D')-pd.Timedelta(days=1),freq='D',tz='UTC')
    r2=pd.DataFrame(index=all_days,columns=SYMS,dtype=float)
    r24=pd.DataFrame(index=all_days,columns=SYMS,dtype=float)
    for day in all_days:
        for s in SYMS:
            h=hourly[s]
            try:
                o=float(h.loc[day,'open']); c2=float(h.loc[day+pd.Timedelta(hours=1),'close']); nx=float(h.loc[day+pd.Timedelta(days=1),'open'])
                r2.loc[day,s]=c2/o-1; r24.loc[day,s]=nx/o-1
            except Exception: pass
    rel2=r2.sub(r2.mean(axis=1),axis=0)
    disp2=rel2.std(axis=1,ddof=1)

    rows=[]
    for day,srow in sig.iterrows():
        day=day.floor('D'); entry_ts=day+pd.Timedelta(hours=2); exit_ts=day+pd.Timedelta(days=1)
        if day not in r2.index or r2.loc[day].isna().any() or day not in r24.index or r24.loc[day].isna().any(): continue
        # strict trailing 60 calendar observations, excluding current day.
        hist=r24.loc[(r24.index<day)&(r24.index>=day-pd.Timedelta(days=75))].dropna()
        hist2=rel2.loc[(rel2.index<day)&(rel2.index>=day-pd.Timedelta(days=75))].dropna()
        histdisp=disp2.loc[(disp2.index<day)&(disp2.index>=day-pd.Timedelta(days=75))].dropna()
        hist=hist.tail(60); hist2=hist2.tail(60); histdisp=histdisp.tail(60)
        if len(hist)<30 or len(hist2)<30 or len(histdisp)<30: continue
        btc=hist['BTCUSDT']; vb=float(btc.var(ddof=1))
        if not np.isfinite(vb) or vb<=0: continue
        betas={s:(1.0 if s=='BTCUSDT' else float(hist[s].cov(btc)/vb)) for s in SYMS}
        raw=rel2.loc[day].astype(float)
        btc2=float(r2.loc[day,'BTCUSDT'])
        resid=pd.Series({s:float(r2.loc[day,s])-betas[s]*btc2 for s in SYMS})
        vols=hist2.std(ddof=1).replace(0,np.nan)
        zraw=(raw/vols).replace([np.inf,-np.inf],np.nan)
        curdisp=float(disp2.loc[day]); med=float(histdisp.median()); q75=float(histdisp.quantile(.75))
        # Entry/exit prices.
        ep={}; xp={}
        ok=True
        for s in SYMS:
            try:
                ep[s]=float(hourly[s].loc[day+pd.Timedelta(hours=1),'close'])
                xp[s]=float(hourly[s].loc[exit_ts,'open'])
            except Exception: ok=False; break
        if not ok: continue
        signals={'raw':raw,'beta_resid':resid,'volnorm':zraw}
        rec={'date':day,'active':bool(srow.get('active',False)),'ood':bool(srow.get('ood',False)),
             'state_pred_disp':float(srow.get('state_pred_disp',np.nan)),'entry_disp':curdisp,
             'div_med':bool(curdisp>med),'div_q75':bool(curdisp>q75)}
        for nm,sg in signals.items():
            for nside in (1,2):
                tr=portfolio_reversal(sg,ep,xp,funding,entry_ts,exit_ts,nside)
                if tr:
                    rec[f'{nm}_top{nside}_grossfund']=tr['gross_funding']
                    rec[f'{nm}_top{nside}_longs']=tr['longs']; rec[f'{nm}_top{nside}_shorts']=tr['shorts']
        rows.append(rec)
    ev=pd.DataFrame(rows).set_index('date').sort_index(); ev.to_csv(OUT/'daily_variants.csv')

    variants=[]
    for sig_nm in ('raw','beta_resid','volnorm'):
        for nside in (1,2):
            base=f'{sig_nm}_top{nside}'; variants.append((base,None))
            variants.append((base,'div_med'))
            variants.append((base,'div_q75'))
    results={}
    for base,filt in variants:
        col=f'{base}_grossfund'; valid=ev[col].notna(); gate=ev.active.astype(bool)&valid
        if filt: gate=gate&ev[filt].astype(bool)
        results_key=base if filt is None else f'{base}_{filt}'
        results[results_key]={}
        for bp in COSTS:
            allr=ev[col]-bp/10000
            gr=allr.where(gate,0.0)
            obs=comp(gr)
            ar=allr[gate].dropna()
            q=ev.loc[gate].copy(); q['r']=ar
            quarters={str(k):{'n':int(len(z)),'total':comp(z.r)} for k,z in q.groupby(q.index.to_period('Q'))}
            results[results_key][str(bp)]={'active_days':int(gate.sum()),'total':obs,'mean_active':float(ar.mean()) if len(ar) else np.nan,
                'hit':float((ar>0).mean()) if len(ar) else np.nan,'sharpe':sharpe(gr),'maxdd':maxdd(gr),
                'random_gate_p':random_gate_p(allr,gate,obs),'quarters':quarters}
            # concentration stress
            srt=ar.sort_values(ascending=False)
            results[results_key][str(bp)]['stress']={}
            for k in (0,1,3,5):
                z=srt.iloc[k:] if k else srt
                results[results_key][str(bp)]['stress'][f'remove_best_{k}']=comp(z)

    # ranking at 10/20 bp is descriptive only; no winner is promoted automatically.
    ranking={}
    for bp in COSTS:
        ranking[str(bp)]=sorted([{'variant':k,**{x:v[str(bp)][x] for x in ('active_days','total','mean_active','hit','sharpe','maxdd','random_gate_p')}} for k,v in results.items()],key=lambda z:z['total'],reverse=True)
    payload={'results':results,'ranking_descriptive_only':ranking,
             'spec':{'entry':'02:00 UTC','exit':'next 00:00 UTC','beta_window':'trailing 60 valid daily returns, min30','volnorm':'raw EW-relative 2h return / trailing 60d own relative-return SD','div_med':'current 2h CS dispersion > trailing 60d median','div_q75':'secondary current > trailing 60d q75','cost_bp':COSTS}}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,default=str))
    lines=['# DISP-3 reversal variants','',f'Valid study days: {len(ev)}','',
           '## Descriptive ranking at 10 bp (NOT model selection)','',
           '| Variant | Active | Total | Sharpe | Hit | Random-gate p | Remove best 3 |','|---|---:|---:|---:|---:|---:|---:|']
    for z in ranking['10']:
        st=results[z['variant']]['10']['stress']['remove_best_3']
        lines.append(f"| {z['variant']} | {z['active_days']} | {z['total']:.2%} | {z['sharpe']:.2f} | {z['hit']:.1%} | {z['random_gate_p']:.4f} | {st:.2%} |")
    lines += ['','## Descriptive ranking at 20 bp (NOT model selection)','',
              '| Variant | Active | Total | Sharpe | Hit | Random-gate p | Remove best 3 |','|---|---:|---:|---:|---:|---:|---:|']
    for z in ranking['20']:
        st=results[z['variant']]['20']['stress']['remove_best_3']
        lines.append(f"| {z['variant']} | {z['active_days']} | {z['total']:.2%} | {z['sharpe']:.2f} | {z['hit']:.1%} | {z['random_gate_p']:.4f} | {st:.2%} |")
    lines += ['','No variant is an edge on this retrospective sample. Any apparent winner must be frozen and validated prospectively/OOS.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
