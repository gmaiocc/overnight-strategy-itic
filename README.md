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
│   ├── Final Strategy + RM/          # MA50 L/S with Risk Management backtest
│   └── Sensitivity Analysis/         # Portfolio size & momentum window tests
│
├── baseline strategy/                # Pure execution — no risk management
│   ├── src/
│   │   ├── main.py                   # Main execution loop (buy/sell)
│   │   ├── signal_generator.py       # Generates daily TOP-10 signals
│   │   └── SaxoOrderExecutor.py      # Saxo Bank API connector
│   └── signals/                      # Auto-generated daily signal files
│
├── final strategy/                   # Full strategy with risk management
│   ├── src/
│   │   ├── main.py                   # Execution loop with RM validation
│   │   ├── signal_generator.py       # Generates daily TOP-10 signals
│   │   ├── SaxoOrderExecutor.py      # Saxo Bank API connector
│   │   └── Risk_Management.py        # Position sizing, circuit breaker, filters
│   └── signals/                      # Auto-generated daily signal files
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
5. Ranks and selects the TOP-10 stocks → saves to `signals/`

### Trade Execution
1. **20:30 WET** — `signal_generator.py` runs and produces the daily signal file
2. **20:55 WET** — `main.py` loads signals, resolves Saxo UICs, and places BUY orders at the closing auction
3. **14:30 WET (next day)** — `main.py` places SELL orders at the opening auction
4. P&L is recorded after execution

### Baseline vs Final Strategy

| | Baseline Strategy | Final Strategy |
|---|---|---|
| Signal | TOP-10 by 126d overnight momentum | Same |
| Direction | Long only | Long (SPY > MA50) / Short (SPY < MA50) |
| Risk Management | None | Full RM layer |
| Position Sizing | Equal weight (capital / 10) | Max 10% of capital per stock |
| Earnings Filter | No | Skips stocks with earnings next day |
| Circuit Breaker | No | Halts trading 24h if daily loss ≥ 2% |
| Market Cap Filter | No | Excludes stocks < $2B market cap |

## Risk Management Rules (Final Strategy)

| Rule | Value |
|------|-------|
| Max position size | 10% of capital per stock |
| Max simultaneous positions | 10 |
| Min market cap | $2B |
| Min avg daily volume | $10M |
| Daily loss limit | -2% of capital → 24h trading halt |
| Earnings filter | Skip stocks reporting earnings next day |

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