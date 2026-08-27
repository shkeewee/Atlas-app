# Crypto Prospective/OOS Ledger

Immutable outcome ledger. Do not alter frozen signals after outcomes. Corrections must be appended with an explicit reason.

## 2026-08-27 — CROWD-1 Snapshot 1 official +1d checkpoint

Frozen reference: Binance USDⓈ-M mark-price 5m bar OPEN at 2026-08-25 20:35 UTC.
Official +1d checkpoint: matching mark-price 5m bar OPEN at 2026-08-26 20:35 UTC.

Checkpoint mark opens:
BTC 78315.96044928; ETH 2465.02199225; BNB 698.84000000; SOL 96.48536752; XRP 1.37238111; DOGE 0.08482098; HYPE 80.18601667; ZEC 782.39082649; LINK 11.25116425; ADA 0.20474225.

Forward returns from frozen reference:
BTC -0.304550%; ETH +0.775618%; BNB +0.522098%; SOL -0.540802%; XRP -5.385653%; DOGE -2.605374%; HYPE -0.493874%; ZEC +0.169585%; LINK -1.071272%; ADA -3.559939%.

EW10 benchmark: -1.249416%.
Combined frozen-score vs +1d return Spearman rho = +0.103030 (two-sided p=0.776998); Kendall tau = +0.111111 (p=0.727490). No cross-sectional rank evidence at this checkpoint.

Frozen original positive basket ETH+SOL: raw +0.117408%; costed simple-long at 20bp round trip -0.082592%; comparison BNB+DOGE raw -1.041638%; raw positive-minus-comparison +1.159046pp. Hypothetical true long-short spread after 40bp total round-trip cost: +0.759046%. Positive basket raw excess vs EW10 +1.366824pp; after 20bp long cost +1.166824pp. Classification: relative hit vs comparison and EW benchmark, but not positive absolute P&L after frozen simple-long cost.

Frozen expanded positive basket ZEC+ADA: raw -1.695177%; costed simple-long at 20bp round trip -1.895177%; comparison HYPE+LINK raw -0.782573%; raw positive-minus-comparison -0.912604pp. Hypothetical true long-short spread after 40bp total round-trip cost: -1.312604%. Positive basket raw excess vs EW10 -0.445761pp; after 20bp long cost -0.645761pp. Classification: miss.

Do not replace these numbers with interim observations. +3d remains due 2026-08-28 20:35 UTC and +7d remains due 2026-09-01 20:35 UTC.

## 2026-08-27 — Eligibility refresh

CoinGecko top-20 non-stablecoin screen remains compatible with the Aug26 eligible cohort. No new admission. Current Binance USDⓈ-M 24h futures quote volume for previously failed liquid-perp candidates: TRXUSDT $53.2606M, XMRUSDT $33.1129M, XLMUSDT $45.2635M, all below the fixed $100M gate. Figure Heloc, Rain, WhiteBIT Coin and LEO still lack a qualifying matching Binance USDⓈ-M perpetual in the frozen framework.

## 2026-08-27 — ECON-1 Aug26 state outcome (+1d diagnostic)

Frozen Aug26 00:00 states: BTC C2; ETH C2; BNB C1; DOGE C1; SOL C0; XRP C0. Frozen TRAIN +1d state forecasts: C0 -0.211784%; C1 -0.093153%; C2 +0.131468%; C3 +0.223963% excess return.

Futures daily-open returns from 2026-08-26 00:00 to 2026-08-27 00:00: BTC +0.616390%; ETH +2.618646%; BNB +1.881036%; SOL +5.675228%; XRP -0.830078%; DOGE +2.287582%. Six-asset EW return +2.041467%.

Realized excess returns: BTC -1.425077%; ETH +0.577179%; BNB -0.160431%; SOL +3.633761%; XRP -2.871545%; DOGE +0.246115%.

Single-date cross-sectional Spearman between frozen state forecast and realized excess = 0.000000; Pearson = -0.164782. Sign hit = 3/6 = 50%. No C3 asset was present, so the previously defined C3-vs-C0 trade had no valid trade on this date.

Updated descriptive validation aggregates using the prior frozen Aug15-Aug25 summary plus this date: sign hit 43/72 = 59.72%; mean daily cross-sectional IC falls from +0.15627 over 11 dates to +0.14325 over 12 dates. Updated state means (descriptive): C0 n=17 mean -0.875458%; C1 n=21 mean -0.015101%; C2 n=25 mean +0.771360%; C3 n=9 mean -0.453700%. ECON-1 remains diagnostic only and fails robustness criteria; no trading promotion.
