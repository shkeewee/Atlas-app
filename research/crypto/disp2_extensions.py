#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

IN=Path(os.environ.get('DISP1_DAILY','disp1_results/event_days.csv'))
OUT=Path(os.environ.get('DISP2_OUT','disp2_results'))
RNG=np.random.default_rng(20260826)
COSTS=[0,5,10,15,20,25,30,40]


def comp(x):
    x=pd.Series(x,dtype=float).fillna(0)
    return float((1+x).prod()-1)

def sharpe(x):
    x=pd.Series(x,dtype=float).dropna(); sd=x.std(ddof=1)
    return float(x.mean()/sd*np.sqrt(365)) if len(x)>1 and sd>0 else np.nan

def maxdd(x):
    x=pd.Series(x,dtype=float).fillna(0); eq=(1+x).cumprod(); dd=eq/eq.cummax()-1
    return float(dd.min()) if len(x) else np.nan

def mc_gate(allr,gate,obs,nperm=10000):
    df=pd.DataFrame({'r':allr,'g':gate}).dropna(); n=int(df.g.sum())
    if n<2 or n>=len(df): return np.nan
    arr=df.r.values; sims=[]
    for _ in range(nperm):
        idx=RNG.choice(len(arr),size=n,replace=False); sims.append(np.prod(1+arr[idx])-1)
    return float((1+np.sum(np.asarray(sims)>=obs))/(nperm+1))


def main():
    OUT.mkdir(exist_ok=True)
    ev=pd.read_csv(IN,parse_dates=['date']).set_index('date').sort_index()
    gate=ev.active.astype(bool)
    results={'cost_sensitivity':{},'quintiles':{},'excursion_budget':{},'stress':{}}

    # Cost math for all fixed entry horizons, continuation + reversal.
    for h in (1,2,4):
        cont=ev[f'gross_{h}h']+ev[f'funding_adj_{h}h']
        rev=-ev[f'gross_{h}h']-ev[f'funding_adj_{h}h']
        for name,base in [('continuation',cont),('reversal',rev)]:
            key=f'{name}_{h}h'; results['cost_sensitivity'][key]={}
            for bp in COSTS:
                r=(base-bp/10000).where(gate,0.0)
                results['cost_sensitivity'][key][str(bp)]={
                    'total':comp(r),'mean_active':float((base[gate]-bp/10000).mean()),
                    'hit':float(((base[gate]-bp/10000)>0).mean()),'sharpe':sharpe(r),'maxdd':maxdd(r)}

    # Breakeven friction in bp from arithmetic mean active-day edge.
    for h in (1,2,4):
        base=-ev[f'gross_{h}h']-ev[f'funding_adj_{h}h']
        results['cost_sensitivity'][f'reversal_{h}h']['mean_breakeven_bp']=float(base[gate].mean()*10000)

    # Does stronger predicted dispersion monotonically improve the reversal payoff?
    # Fixed quintiles across all valid days, not optimized thresholds.
    q=pd.qcut(ev.state_pred_disp.rank(method='first'),5,labels=False)+1
    ev['pred_q']=q
    for h in (1,2,4):
        base=-ev[f'gross_{h}h']-ev[f'funding_adj_{h}h']
        results['quintiles'][f'reversal_{h}h']={}
        for qi,z in ev.groupby('pred_q'):
            b=base.loc[z.index]
            results['quintiles'][f'reversal_{h}h'][str(int(qi))]={
                'n':int(len(z)),'gross_funding_mean':float(b.mean()),
                'net20_mean':float((b-.002).mean()),'net20_total':comp(b-.002)}
        rho,p=spearmanr(ev.state_pred_disp,base,nan_policy='omit')
        results['quintiles'][f'reversal_{h}h']['spearman']={'rho':float(rho),'p':float(p)}

    # Convexity/option-proxy budget: how much pathwise relative excursion does STATE-RISK create?
    # This does NOT assume tradability; it measures the upper-bound raw movement available to monetize.
    for col in ['disp1','disp2','disp4','disp24','max_abs_rel_excursion']:
        a=ev.loc[gate,col]; b=ev.loc[~gate,col]
        results['excursion_budget'][col]={'active_mean':float(a.mean()),'inactive_mean':float(b.mean()),
            'ratio':float(a.mean()/b.mean()) if b.mean()!=0 else np.nan}

    # Stress the economically best-looking predeclared family: reversal at 20bp and 10bp, all horizons.
    for h in (1,2,4):
        base=-ev[f'gross_{h}h']-ev[f'funding_adj_{h}h']
        for bp in (10,20):
            ar=(base[gate]-bp/10000).dropna().sort_values(ascending=False)
            key=f'reversal_{h}h_{bp}bp'; results['stress'][key]={}
            for k in (0,1,3,5):
                z=ar.iloc[k:] if k else ar
                results['stress'][key][f'remove_best_{k}']=comp(z)
            gated=(base-bp/10000).where(gate,0)
            results['stress'][key]['random_gate_p']=mc_gate(base-bp/10000,gate,comp(gated))

    # Cross-sectional 'straddle budget': realized dispersion vs simple estimated friction hurdle.
    # For a hypothetical perfect direction-free capture of one unit of dispersion, report how often 24h dispersion exceeds costs.
    for bp in COSTS[1:]:
        hurdle=bp/10000
        results.setdefault('perfect_capture_hurdle',{})[str(bp)]={
            'active_fraction_disp24_gt_cost':float((ev.loc[gate,'disp24']>hurdle).mean()),
            'inactive_fraction_disp24_gt_cost':float((ev.loc[~gate,'disp24']>hurdle).mean())}

    (OUT/'summary.json').write_text(json.dumps(results,indent=2))
    lines=['# DISP-2 extensions','','## Reversal cost sensitivity (gated total return)','',
           '| Cost bp | 1h | 2h | 4h |','|---:|---:|---:|---:|']
    for bp in COSTS:
        vals=[results['cost_sensitivity'][f'reversal_{h}h'][str(bp)]['total'] for h in (1,2,4)]
        lines.append(f'| {bp} | {vals[0]:.2%} | {vals[1]:.2%} | {vals[2]:.2%} |')
    lines += ['', 'Mean-active breakeven friction: ' + ', '.join([f"{h}h {results['cost_sensitivity'][f'reversal_{h}h']['mean_breakeven_bp']:.1f} bp" for h in (1,2,4)]),'',
              '## Stress at 10/20 bp','```json',json.dumps(results['stress'],indent=2),'```','',
              '## Predicted-dispersion quintiles','```json',json.dumps(results['quintiles'],indent=2),'```','',
              '## Excursion budget','```json',json.dumps(results['excursion_budget'],indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)); print('\n'.join(lines))

if __name__=='__main__': main()
