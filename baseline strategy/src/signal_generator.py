import json
import os
import pandas as pd
import yfinance as yf
import requests
from datetime import date, datetime

# Get S&P 500 tickers
headers = {"User-Agent": "Mozilla/5.0"}
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(requests.get(url, headers=headers).content)
tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
print(f"Found {len(tickers)} tickers.")

# Fetch OHLCV
PERIOD = '1y'

data = yf.download(tickers, period=PERIOD, auto_adjust=True, group_by='ticker', progress=False)

raw = {}
for ticker in tickers:
    try:
        df = data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = 'date'
        df = df.dropna(how='all')
        if not df.empty:
            raw[ticker] = df
    except Exception as e:
        print(f'Error {ticker}: {e}')

print(f'Downloaded: {len(raw)} tickers')

# Clean data
def clean(df):
    df = df[~df.index.duplicated(keep='last')]
    df = df.dropna(subset=['open', 'close'])
    df = df[(df['open'] > 0) & (df['close'] > 0) & (df['volume'] > 0)]
    return df

raw = {ticker: clean(df) for ticker, df in raw.items()}
print('Cleaning done.')

# Liquidity filter
def passes_liquidity(df, min_dv=10_000_000):
    avg_dv = (df['close'] * df['volume']).tail(20).mean()
    return avg_dv >= min_dv

universe = {t: df for t, df in raw.items() if passes_liquidity(df)}
print(f'Universe after liquidity filter: {len(universe)} tickers')

# Signal generator
signals = []
for ticker, df in universe.items():
    if len(df) < 130:
        continue
    df = df.sort_index()  # ensure chronological order before shift
    df['overnight_return'] = (df['open'] / df['close'].shift(1)) - 1
    cum_overnight = (1 + df['overnight_return']).tail(126).prod() - 1
    signals.append({'ticker': ticker, 'cum_overnight_126d': cum_overnight})

if not signals:
    print("ERROR: No stocks passed the filters. Check your data.")
    exit(1)

signals_df = pd.DataFrame(signals).sort_values('cum_overnight_126d', ascending=False)

top10 = signals_df.head(10)['ticker'].tolist()
print("\nTop 10 stocks to buy today:")
print(top10)

# Save JSON
os.makedirs('signals', exist_ok=True)
filename = f"signals/signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

output = {
    "date": str(date.today()),
    "stocks": top10
}
with open(filename, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{filename} saved.")
