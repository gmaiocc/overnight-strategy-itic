# OVERNIGHT-STRATEGY-ITIC

Automated trading system that exploits the overnight return anomaly in S&P 500 stocks, developed within the ISCTE Trading and Investment Club (ITIC). The strategy buys at market close (20:55 WET) and sells at market open (14:30 WET), capturing the systematic overnight premium documented by Lou et al. (2019).

## Project Structure

```
OVERNIGHT-STRATEGY-ITIC/
├── backtest/
│   ├── Baseline vs. S&P500/          # Baseline strategy vs SPY Buy & Hold
│   ├── Baseline vs. MA200/           # MA200 trend filter evaluation
│   ├── Baseline vs. MA50 LongShort/  # MA50 regime long/short evaluation
│   ├── Baseline vs. VIX/             # VIX volatility filter evaluation
│   ├── Baseline vs. ADR/             # ADR high-volatility filter evaluation
│   └── Sensitivity Stocks/           # Portfolio size & momentum window tests
│
├── baseline strategy/                # Baseline: long-only TOP-10
│   ├── src/
│   │   ├── main.py                   # Main execution loop (buy/sell)
│   │   ├── signal_generator.py       # Generates daily TOP-10 signals
│   │   └── SaxoOrderExecutor.py      # Saxo Bank API connector
│   └── signals/
│
├── final strategy/                   # Final: MA50 Long/Short regime
│   ├── src/
│   │   ├── main.py                   # Execution loop with MA50 regime logic
│   │   ├── SignalGenerator.py        # Generates signals with regime direction
│   │   └── SaxoOrderExecutor.py      # Saxo Bank API connector
│   └── signals/
│
├── notebooks/
│   ├── signal_generator.ipynb        # Interactive signal exploration
│   └── signals/
│
├── .env                              # SAXO_ACCESS_TOKEN & SAXO_BASE_URL
├── .gitignore
├── requirements.txt
└── README.md
```

## How It Works

### Signal Generation
1. Scrapes the current S&P 500 constituent list
2. Downloads 1 year of daily OHLCV data via yfinance
3. Applies a liquidity filter ($10M minimum average daily dollar volume)
4. Calculates 126-day cumulative overnight momentum for each stock
5. Ranks stocks by momentum and selects the TOP-10 (or BOTTOM-10) depending on regime
6. Saves the daily signal file to `signals/`

### MA50 Regime Logic (Final Strategy)
The final strategy conditions trade direction on the broad market regime:
- **SPY > 50-Day MA** → Bull regime → **LONG** the TOP-10 overnight momentum stocks
- **SPY ≤ 50-Day MA** → Bear regime → **SHORT** the BOTTOM-10 overnight momentum stocks

Only one leg is active at any given time.

### Trade Execution
1. **20:30 WET** — Signal generator runs and produces the daily signal file
2. **20:55 WET** — `main.py` loads signals, resolves Saxo UICs, and places orders at the closing auction
3. **14:30 WET (next day)** — `main.py` places exit orders at the opening auction
4. P&L is recorded after execution

### Baseline vs Final Strategy

| | Baseline Strategy | Final Strategy |
|---|---|---|
| Signal | TOP-10 by 126d overnight momentum | Same ranking logic |
| Direction | Long only | Long (SPY > MA50) / Short (SPY ≤ MA50) |
| Position Sizing | Equal weight (capital / 10) | Equal weight (capital / 10) |

### Backtest Results (Mar 2024 – Mar 2026)

| Metric | Baseline | MA50 Long/Short |
|--------|----------|-----------------|
| Ann. Return | +53.60% | +55.02% |
| Volatility | 26.33% | 21.64% |
| Sharpe Ratio | 2.04 | 2.54 |
| Sortino Ratio | 2.38 | 3.52 |
| Max Drawdown | -24.86% | -10.62% |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
SAXO_ACCESS_TOKEN=your_token_here
SAXO_BASE_URL=https://gateway.saxobank.com/sim/openapi
```

Note: Saxo Bank simulation tokens expire every 24 hours and must be refreshed daily.

## Environment

- **Universe:** S&P 500 constituents
- **Broker:** Saxo Bank (Simulation OpenAPI)
- **Timezone:** Europe/London (WET)
- **Language:** Python 3.10+

## References

- Cooper, M.J., Cliff, M.T., & Gulen, H. (2008). "Return differences between trading and non-trading hours: Like night and day."
- Lou, D., Polk, C., & Skouras, S. (2019). "A tug of war: Overnight versus intraday expected returns." *Journal of Financial Economics*, 134(2), 192-213.