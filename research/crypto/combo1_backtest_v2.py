#!/usr/bin/env python3
"""Robust archive-ingestion wrapper for combo1_backtest.py.

Adds: automatic CSV header detection, timestamp-unit inference, daily Binance
Vision fallback for the current partial month, a live Binance REST fallback for
the final forward daily open, and a safe guard for empty benchmark dates.
Model logic remains unchanged.
"""
from __future__ import annotations
import io, zipfile, inspect, textwrap
import numpy as np
import pandas as pd
import combo1_backtest as c


def _to_dt(x):
    num=pd.to_numeric(x,errors="coerce")
    med=num.dropna().median() if num.notna().any() else np.nan
    if np.isfinite(med):
        if med>1e17: unit="ns"
        elif med>1e14: unit="us"
        elif med>1e11: unit="ms"
        else: unit="s"
        return pd.to_datetime(num,unit=unit,utc=True,errors="coerce")
    return pd.to_datetime(x,utc=True,errors="coerce")


def cached_zip_csv_auto(url: str, key: str):
    p=c.CACHE/key
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=p.read_bytes() if p.exists() else c.get_bytes(url)
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
        header_tokens=("open_time","create_time","funding_time","fundingrate","funding_rate","sum_open_interest","symbol")
        if any(any(tok in v for tok in header_tokens) for v in first):
            cols=[str(v).strip() for v in df.iloc[0].tolist()]
            df=df.iloc[1:].reset_index(drop=True); df.columns=cols
        return df
    except Exception:
        return None

c.cached_zip_csv=cached_zip_csv_auto
_orig_load_monthly=c.load_monthly


def _rest_daily_klines(symbol: str):
    try:
        start=int((c.TEST_END + pd.Timedelta(days=1)).timestamp()*1000)
        r=c.SESSION.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol":symbol,"interval":"1d","startTime":start,"limit":2},
            timeout=30,
        )
        r.raise_for_status()
        rows=r.json()
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def load_monthly_plus_daily(symbol: str, kind: str, interval: str|None=None):
    parts=[]
    try:
        parts.append(_orig_load_monthly(symbol,kind,interval))
    except Exception:
        pass
    start=max(c.DOWNLOAD_START.floor('D'), c.DOWNLOAD_END.floor('D')-pd.Timedelta(days=45))
    for d in pd.date_range(start,c.DOWNLOAD_END.floor('D'),freq='D',tz='UTC'):
        ds=d.strftime('%Y-%m-%d')
        if kind=='klines':
            fn=f"{symbol}-{interval}-{ds}.zip"; url=f"{c.BASE}/daily/klines/{symbol}/{interval}/{fn}"
        elif kind=='premiumIndexKlines':
            fn=f"{symbol}-{interval}-{ds}.zip"; url=f"{c.BASE}/daily/premiumIndexKlines/{symbol}/{interval}/{fn}"
        elif kind=='fundingRate':
            fn=f"{symbol}-fundingRate-{ds}.zip"; url=f"{c.BASE}/daily/fundingRate/{symbol}/{fn}"
        else: raise ValueError(kind)
        x=cached_zip_csv_auto(url,f"daily/{kind}/{symbol}/{fn}")
        if x is not None and len(x): parts.append(x)
    if kind=='klines' and interval=='1d':
        x=_rest_daily_klines(symbol)
        if x is not None and len(x): parts.append(x)
    if not parts: raise RuntimeError(f"No {kind} data for {symbol}")
    return pd.concat(parts,ignore_index=True,sort=False)

c.load_monthly=load_monthly_plus_daily


def parse_klines(df):
    low=[str(x).lower() for x in df.columns]
    if 'open_time' in low:
        t=c.find_col(df,['open_time']); op=c.find_col(df,['open']); hi=c.find_col(df,['high']); lo=c.find_col(df,['low']); cl=c.find_col(df,['close'])
        vol=c.find_col(df,['quote_volume','quote asset volume','quote_asset_volume'])
        if vol is None: vol=c.find_col(df,['volume'])
    else:
        cols=list(df.columns)
        if len(cols)<8: raise RuntimeError(f"Unexpected kline schema {cols}")
        t,op,hi,lo,cl,vol=cols[0],cols[1],cols[2],cols[3],cols[4],cols[7]
    out=pd.DataFrame({'ts':_to_dt(df[t]),'open':pd.to_numeric(df[op],errors='coerce'),'high':pd.to_numeric(df[hi],errors='coerce'),
                      'low':pd.to_numeric(df[lo],errors='coerce'),'close':pd.to_numeric(df[cl],errors='coerce'),'quote_vol':pd.to_numeric(df[vol],errors='coerce')})
    return out.dropna(subset=['ts','open','close']).drop_duplicates('ts').sort_values('ts')

c.parse_klines=parse_klines
c.parse_premium=lambda df: parse_klines(df)[['ts','close']].rename(columns={'close':'premium'})


def parse_funding(df):
    tc=c.find_col(df,['fundingTime','funding_time','calc_time','time'])
    rc=c.find_col(df,['fundingRate','funding_rate','lastFundingRate','last_funding_rate'])
    if tc is None or rc is None:
        cols=list(df.columns)
        if len(cols)>=3: tc,rc=cols[0],cols[-1]
        elif len(cols)>=2: tc,rc=cols[0],cols[1]
        else: raise RuntimeError(f"Unexpected funding schema {cols}")
    return pd.DataFrame({'ts':_to_dt(df[tc]),'funding':pd.to_numeric(df[rc],errors='coerce')}).dropna().drop_duplicates('ts').sort_values('ts')

c.parse_funding=parse_funding


def parse_metrics(df):
    tc=c.find_col(df,['create_time','timestamp','time']); oi=c.find_col(df,['sum_open_interest']); tr=c.find_col(df,['sum_taker_long_short_vol_ratio','taker_long_short_vol_ratio'])
    if oi is None or tr is None: raise RuntimeError(f"Metrics schema lacks OI/taker ratio: {list(df.columns)}")
    if tc is None: tc=df.columns[0]
    out=pd.DataFrame({'ts':_to_dt(df[tc]),'oi':pd.to_numeric(df[oi],errors='coerce'),'taker_ratio':pd.to_numeric(df[tr],errors='coerce')})
    return out.dropna(subset=['ts','oi','taker_ratio']).drop_duplicates('ts').sort_values('ts')

c.parse_metrics=parse_metrics

# Patch only the non-core momentum benchmark. If a benchmark date has no valid
# P/fwd_ret rows, record NaN rather than aborting the completed COMBO-1 run.
_src=inspect.getsource(c.main)
_old='''    for t in res.index:\n        cp=panel[panel.date==t].dropna(subset=["P","fwd_ret"])\n        lg=cp.loc[cp.P.idxmax()]; sh=cp.loc[cp.P.idxmin()]\n        r=float(lg.fwd_ret-sh.fwd_ret-lg.funding_next+sh.funding_next-COST)\n        mom.append(r)\n'''
_new='''    for t in res.index:\n        cp=panel[panel.date==t].dropna(subset=["P","fwd_ret"])\n        if cp.empty:\n            mom.append(np.nan)\n            continue\n        lg=cp.loc[cp.P.idxmax()]; sh=cp.loc[cp.P.idxmin()]\n        r=float(lg.fwd_ret-sh.fwd_ret-lg.funding_next+sh.funding_next-COST)\n        mom.append(r)\n'''
if _old not in _src:
    raise RuntimeError("Expected momentum benchmark block not found; refusing unsafe patch")
_ns=dict(c.__dict__)
exec(textwrap.dedent(_src.replace(_old,_new)),_ns)
c.main=_ns['main']

if __name__=='__main__':
    c.main()
