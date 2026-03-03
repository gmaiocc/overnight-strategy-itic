# Overnight Strategy

Automated trading strategy that exploits the overnight return anomaly in S&P 500 stocks. Buy at market close (21:00 WET), sell at market open (14:30 WET).

## Project Structure

```
overnight-strategy/
├── src/
│   ├── signal_generator.py     # Generates daily top-10 stock signals
│   ├── automation.py           # Main execution loop (buy/sell)
│   ├── SaxoOrderExecutor.py    # Saxo Bank API connector
│   └── Risk_Management.py      # Position sizing, stop loss, trade validation
├── signals/
│   └── signals_YYYYMMDD_HHMM.json   # Daily signal files (auto-generated)
└── README.md
```

## How It Works

1. **20:30** — Run `signal_generator.py` to rank S&P 500 stocks by 126-day cumulative overnight return and save top 10 to `signals/`
2. **20:55** — Run `automation.py` — loads signals, validates each stock against risk rules, resolves UICs
3. **21:00** — Buy orders placed at closing auction via Saxo Bank API
4. **14:30 (next day)** — Sell orders placed at opening auction
5. **14:35** — P&L recorded to `risk_state.json`

## Risk Rules

| Rule | Value |
|------|-------|
| Max position size | 10% of capital per stock |
| Max simultaneous positions | 10 |
| Min market cap | $2B |
| Min avg daily volume | $10M |
| Daily loss limit | -2% of capital (halts trading for 24h) |
| Earnings check | Skip stocks with earnings next day |

## Setup

```bash
pip install pandas yfinance requests
```

Update `ACCESS_TOKEN` in `SaxoOrderExecutor.py` and `automation.py` daily (Saxo sim tokens expire every 24h).

## Environment

- Exchange: S&P 500
- Broker: Saxo Bank (sim environment)
- Timezone: Europe/London (WET)
EOF