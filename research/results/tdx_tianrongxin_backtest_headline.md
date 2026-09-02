# Tianrongxin-style all-A breakout backtest headline

- Data source: TongdaXin official Shanghai/Shenzhen/Beijing complete daily package (`hsjday.zip`).
- Signal period: 2023-09-01 through 2026-08-31; data read through 2026-09-02.
- A-share files scanned: 5,931; symbols with usable history: 5,587.
- Usable symbols by board: SH main 1,719; SZ main 1,538; STAR 603; ChiNext 1,449; BSE 278.
- Modeled round trip: 3 bp buy commission, 3 bp sell commission, 5 bp sell stamp duty, and 5 bp slippage per side.
- Win definition: net return above zero after modeled costs, using a T+1-aware 3% stop, 5% target, or fifth holding-session close.

## Headline results

| Variant | Universe | Trades | Win rate | Mean net return | Profit factor |
|---|---|---:|---:|---:|---:|
| Naive 10-day amount breakout | All A | 34,946 | 44.62% | -0.332% | 0.822 |
| Naive 10-day amount breakout | SH/SZ mainboard | 25,826 | 44.54% | -0.316% | 0.837 |
| Confirmed 10-day breakout, next-open entry | All A | 14,403 | 47.66% | -0.106% | 0.931 |
| Confirmed 10-day breakout, next-open entry | SH/SZ mainboard | 10,725 | 47.37% | -0.107% | 0.931 |
| Confirmed breakout plus low-volume retest | All A | 4,151 | 49.07% | +0.034% | 1.025 |
| Confirmed breakout plus low-volume retest | SH/SZ mainboard | 3,061 | 48.55% | +0.013% | 1.007 |

## Market-regime split

Bull regime is defined as Shanghai Composite close > MA60 > MA120.

| Variant | Universe | Regime | Trades | Win rate | Mean net return | Profit factor |
|---|---|---|---:|---:|---:|---:|
| Confirmed breakout, next-open entry | All A | Bear | 6,463 | 42.53% | -0.416% | 0.732 |
| Confirmed breakout, next-open entry | All A | Bull | 7,067 | 52.33% | +0.257% | 1.185 |
| Confirmed breakout, next-open entry | SH/SZ mainboard | Bear | 4,921 | 42.74% | -0.389% | 0.742 |
| Confirmed breakout, next-open entry | SH/SZ mainboard | Bull | 5,161 | 52.06% | +0.220% | 1.164 |
| Breakout plus low-volume retest | All A | Bear | 1,802 | 43.56% | -0.278% | 0.803 |
| Breakout plus low-volume retest | All A | Bull | 2,074 | 54.19% | +0.352% | 1.279 |
| Breakout plus low-volume retest | SH/SZ mainboard | Bear | 1,371 | 43.91% | -0.248% | 0.817 |
| Breakout plus low-volume retest | SH/SZ mainboard | Bull | 1,469 | 53.37% | +0.296% | 1.255 |

## Tested primary signal and execution

- Close at least 0.3% above the prior 10-session highest high, but no more than 4% above it.
- Signal-day amount at least 1.5 times the prior 20-session median; volume at least 1.3 times the prior 20-session mean.
- Signal-day return at least 1.5%; close in the top 30% of the bar; close > MA20 > MA60; not excessively extended.
- Median amount over the prior 20 sessions at least RMB 50 million; at least 120 observed listing sessions.
- Buy at the next open only when the gap is not a failed breakout or an excessive chase; skip one-price and zero-volume bars.
- Retest variant waits up to three sessions for a lower-volume retest near the broken resistance and enters the following open.

## Interpretation and limitations

A volume-confirmed breakout alone was not a positive standalone edge over this sample. The meaningful improvement came from combining a low-volume retest with a bull market regime. Results use current-vintage raw daily bars, lack historical ST labels, may contain survivorship bias, and use conservative daily-bar execution assumptions.