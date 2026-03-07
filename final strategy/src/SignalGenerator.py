import json
import math
import pandas as pd
import yfinance as yf
import requests
from datetime import date, datetime
from pathlib import Path

PERIOD = "1y"
MOMENTUM_WINDOW = 126
MIN_DV = 10_000_000
MA50_WINDOW = 50
LONG_N = 10
SHORT_N = 10
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "signals"


def get_sp500_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_spy_regime() -> tuple[str, float, float]:
    spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy.index.name = "date"
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()

    if len(spy) < MA50_WINDOW:
        print(f"WARNING: Only {len(spy)} SPY bars available (need {MA50_WINDOW}) "
              "— defaulting to LONG regime.")
        last_close = float(spy["Close"].iloc[-1]) if not spy.empty else float("nan")
        return "LONG", last_close, float("nan")

    last_close = float(spy["Close"].iloc[-1])
    ma50 = float(spy["Close"].rolling(MA50_WINDOW).mean().iloc[-1])
    regime = "LONG" if last_close > ma50 else "SHORT"

    print(f"SPY close: ${last_close:.2f} | MA50: ${ma50:.2f} → Regime: [{regime}]")
    return regime, last_close, ma50


def download_and_clean(tickers: list[str]) -> dict:
    
    print(f"Downloading {len(tickers)} tickers (period={PERIOD})...")
    data = yf.download(tickers, period=PERIOD, auto_adjust=True, group_by="ticker", progress=True)

    raw = {}
    for ticker in tickers:
        try:
            df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna(subset=["open", "close"])
            df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] > 0)]
            if not df.empty:
                raw[ticker] = df
        except Exception as e:
            print(f"[{ticker}] Data error: {e}")

    print(f"Downloaded & cleaned: {len(raw)}/{len(tickers)} tickers")
    return raw


def apply_liquidity_filter(raw: dict) -> dict:

    universe = {
        ticker: df for ticker, df in raw.items()
        if (df["close"] * df["volume"]).tail(20).mean() >= MIN_DV
    }
    print(f"Universe after liquidity filter (≥${MIN_DV/1e6:.0f}M ADV): "
          f"{len(universe)}/{len(raw)} tickers")
    return universe


def compute_momentum_signals(universe: dict) -> pd.DataFrame:
    
    signals = []
    for ticker, df in universe.items():
        if len(df) < MOMENTUM_WINDOW + 5:
            continue
        df = df.sort_index()
        df["overnight_return"] = (df["open"] / df["close"].shift(1)) - 1
        df = df.dropna(subset=["overnight_return"])
        if len(df) < MOMENTUM_WINDOW:
            continue
        cum_overnight = (1 + df["overnight_return"]).tail(MOMENTUM_WINDOW).prod() - 1
        signals.append({"ticker": ticker, "cum_overnight_126d": cum_overnight})

    if not signals:
        raise RuntimeError(
            "No stocks survived all filters — cannot generate signals. "
            "Check data availability and filter parameters."
        )

    df_signals = (
        pd.DataFrame(signals)
        .sort_values("cum_overnight_126d", ascending=False)
        .reset_index(drop=True)
    )
    print(f"Signals computed for {len(df_signals)} stocks.")
    return df_signals


def run_signal_generation() -> dict:

    print("=" * 60)
    print("SIGNAL GENERATION — MA50 Long/Short Overnight Strategy")
    print("=" * 60)

    # Step 1 — Universe
    print("\n[1/5] Fetching S&P 500 constituents...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers loaded.")

    # Step 2 — Regime
    print("\n[2/5] Determining market regime...")
    regime, spy_close, ma50 = fetch_spy_regime()

    # Step 3 — Data
    print("\n[3/5] Downloading OHLCV data...")
    raw = download_and_clean(tickers)

    # Step 4 — Liquidity
    print("\n[4/5] Applying liquidity filter...")
    universe = apply_liquidity_filter(raw)

    # Step 5 — Momentum signals
    print("\n[5/5] Computing overnight momentum signals...")
    signals_df = compute_momentum_signals(universe)

    # Step 6 — Select stocks based on regime
    if regime == "LONG":
        selected = signals_df.head(LONG_N)["ticker"].tolist()
        action = "BUY"
        label = f"TOP-{LONG_N} momentum winners (LONG)"
    else:
        selected = (
            signals_df.sort_values("cum_overnight_126d", ascending=True).head(SHORT_N)["ticker"].tolist()
        )
        action = "SELL_SHORT"
        label = f"BOTTOM-{SHORT_N} momentum losers (SHORT)"

    print(f"\n  Regime [{regime}] → {label}:")
    for i, t in enumerate(selected, 1):
        row = signals_df[signals_df["ticker"] == t].iloc[0]
        print(f"    {i:>2}. {t:<8}  cum_overnight_126d = {row['cum_overnight_126d']:+.4f}")

    # Step 7 — Persist
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    output = {
        "date": str(date.today()),
        "regime": regime,
        "spy_close": round(spy_close, 4),

        "ma50": round(ma50, 4) if not math.isnan(ma50) else None,
        "action": action,
        "stocks": selected,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSignals saved → {filename}")
    print("=" * 60)
    return output


if __name__ == "__main__":
    run_signal_generation()