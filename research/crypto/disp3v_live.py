#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd

# Import v2 first: it patches robust ingestion onto combo1_backtest.c.
import combo1_backtest_v2 as v2
import disp1_event_study as d1

TARGET = pd.Timestamp(os.environ.get('DISP3V_TARGET', pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')), tz='UTC')
OUT = Path(os.environ.get('DISP3V_OUT','disp3v_live_results'))
COMBO_OUT = Path(os.environ.get('COMBO_OUT','combo_live_results'))

# Frozen historical start/model logic; extend only evaluation endpoint to target date.
c=v2.c
c.TEST_END=TARGET
c.DOWNLOAD_END=TARGET+pd.Timedelta(days=2)
c.OUT=COMBO_OUT
c.CACHE=Path(os.environ.get('COMBO_CACHE','.combo_live_cache'))

# Run the exact frozen expanding STATE-RISK model through TARGET.
c.main()
res=pd.read_csv(COMBO_OUT/'daily_results.csv',parse_dates=['date']).set_index('date').sort_index()
if res.index.tz is None: res.index=res.index.tz_localize('UTC')
row=res.loc[TARGET]

gate={'date':str(TARGET.date()),'active':bool(row.active),'high_disp_state':bool(row.high_disp_state),'ood':bool(row.ood),
      'gmm_k':int(row.gmm_k),'state':int(row.state),'state_pred_disp':float(row.state_pred_disp),'train_mean_disp':float(row.train_mean_disp),
      'ood_score':float(row.ood_score)}

payload={'gate':gate,'signal':None,'methodology':{
    'universe':d1.SYMS,'entry':'02:00 UTC','exit':'next 00:00 UTC','cost_bp':20,
    'signal':'2h return minus equal-weight basket, divided by asset trailing 60 valid-day SD of 2h relative returns; long bottom2, short top2',
    'state_model':'exact frozen COMBO-1 STATE-RISK expanding GMM/OOD logic'}}

if gate['active']:
    start=TARGET-pd.Timedelta(days=80); end=TARGET+pd.Timedelta(days=1)
    hourly={s:d1.load_hourly(s,start,end) for s in d1.SYMS}
    days=pd.date_range(start.floor('D'),TARGET,freq='D',tz='UTC')
    r2=pd.DataFrame(index=days,columns=d1.SYMS,dtype=float)
    for day in days:
        for s in d1.SYMS:
            h=hourly[s]
            try:
                o=float(h.loc[day,'open']); c2=float(h.loc[day+pd.Timedelta(hours=1),'close']); r2.loc[day,s]=c2/o-1
            except Exception: pass
    rel=r2.sub(r2.mean(axis=1),axis=0)
    hist=rel.loc[rel.index<TARGET].dropna().tail(60)
    cur=rel.loc[TARGET].dropna()
    if len(hist)<30 or len(cur)!=len(d1.SYMS):
        payload['signal']={'status':'MISSING_DATA','history_days':int(len(hist)),'current_assets':int(len(cur))}
    else:
        sd=hist.std(ddof=1).replace(0,np.nan); z=(cur/sd).replace([np.inf,-np.inf],np.nan)
        if z.isna().any():
            payload['signal']={'status':'MISSING_DATA','reason':'nonfinite z score'}
        else:
            longs=list(z.nsmallest(2).index); shorts=list(z.nlargest(2).index)
            payload['signal']={'status':'PAPER_SIGNAL','longs':longs,'shorts':shorts,
                'z_scores':{s:float(z[s]) for s in d1.SYMS},'relative_2h':{s:float(cur[s]) for s in d1.SYMS},
                'trailing_sd':{s:float(sd[s]) for s in d1.SYMS},'history_days':int(len(hist))}
else:
    payload['signal']={'status':'NO_TRADE'}

OUT.mkdir(exist_ok=True)
(OUT/'signal.json').write_text(json.dumps(payload,indent=2,default=str))
print(json.dumps(payload,indent=2,default=str))
