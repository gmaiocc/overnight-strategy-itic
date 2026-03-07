"""
signal_generator.py — Overnight Momentum Signal Generation
===========================================================
Implements the MA50 Long/Short Regime strategy as backtested:

  SPY > MA50  →  LONG  regime: select TOP-10  by 126d overnight momentum (BUY)
  SPY ≤ MA50  →  SHORT regime: select BOTTOM-10 by 126d overnight momentum (SELL_SHORT)

Only one leg is active per day. The regime is determined once at signal
generation time and locked into the signals JSON for the execution engine.

Overnight return formula (Lou et al., 2019):
    r_ov,t = (Open_t / Close_{t-1}) - 1

Signal: cumulative product over a 126-day rolling window.

References:
    Lou, D., Polk, C., & Skouras, S. (2019). A tug of war: Overnight versus
    intraday expected returns. Journal of Financial Economics, 134(2), 192-213.
"""

import json
import os
import pandas as pd
import yfinance as yf
import requests
from datetime import date, datetime
from pathlib import Path


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PERIOD          = "1y"          # Historical window for OHLCV download
MOMENTUM_WINDOW = 126           # Trading days (~6 months) — per Lou et al. (2019)
MIN_DV          = 10_000_000    # $10M minimum 20-day average daily dollar volume
MA50_WINDOW     = 50            # Days for SPY moving average regime filter
LONG_N          = 10            # Number of long positions (bull regime)
SHORT_N         = 10            # Number of short positions (bear regime)
OUTPUT_DIR      = Path(__file__).resolve().parent.parent / "signals"  # [*] absolute path, not CWD-relative
# ─────────────────────────────────────────────────────────────────────────────


def get_sp500_tickers() -> list[str]:
    """
    Retrieve the current S&P 500 constituent list from Wikipedia.

    Returns:
        List of ticker strings with dots replaced by hyphens (e.g. 'BRK-B').
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_spy_regime() -> tuple[str, float, float]:
    """
    Determine the current market regime by comparing SPY's last close
    against its 50-day simple moving average.

    Returns:
        Tuple of (regime, spy_close, ma50) where regime is 'LONG' or 'SHORT'.
        Defaults to 'LONG' if insufficient data is available.
    """
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
    """
    Bulk-download OHLCV data via yfinance and return a dict of
    cleaned DataFrames keyed by ticker.

    Cleaning steps:
        - Deduplicate index (keep last)
        - Drop rows with missing open or close
        - Remove rows with non-positive prices or zero volume
    """
    print(f"Downloading {len(tickers)} tickers (period={PERIOD})...")
    data = yf.download(
        tickers, period=PERIOD, auto_adjust=True,
        group_by="ticker", progress=True
    )

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
    """
    Retain only stocks with a 20-day average daily dollar volume
    (Close × Volume) at or above MIN_DV.

    This ensures the portfolio can absorb institutional-sized orders
    without significant market impact or slippage (see Section 3.1).
    """
    universe = {
        ticker: df for ticker, df in raw.items()
        if (df["close"] * df["volume"]).tail(20).mean() >= MIN_DV
    }
    print(f"Universe after liquidity filter (≥${MIN_DV/1e6:.0f}M ADV): "
          f"{len(universe)}/{len(raw)} tickers")
    return universe


def compute_momentum_signals(universe: dict) -> pd.DataFrame:
    """
    Compute 126-day cumulative overnight momentum for each eligible stock.

    Formula (Lou et al., 2019):
        r_ov,t  = (Open_t / Close_{t-1}) - 1
        Momentum = prod(1 + r_ov) over last 126 days - 1

    Returns:
        DataFrame sorted descending by cum_overnight_126d.
        Top rows = momentum winners (long candidates).
        Bottom rows = momentum losers (short candidates).
    """
    signals = []
    for ticker, df in universe.items():
        if len(df) < MOMENTUM_WINDOW + 5:
            continue
        df = df.sort_index()
        df["overnight_return"] = (df["open"] / df["close"].shift(1)) - 1
        df = df.dropna(subset=["overnight_return"])               # [+] drop NaN from first row (no Close_{t-1})
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
    """
    Execute the full signal generation pipeline and persist results.

    Pipeline:
        1. Fetch S&P 500 constituents
        2. Determine market regime (SPY vs MA50)
        3. Download & clean OHLCV data
        4. Apply liquidity filter
        5. Compute 126-day overnight momentum
        6. Select TOP-10 (LONG) or BOTTOM-10 (SHORT) based on regime
        7. Serialise to signals/signals_YYYYMMDD_HHMM.json

    Returns:
        Dict with keys: date, regime, spy_close, ma50, action, stocks.
    """
    print("=" * 60)
    print("  SIGNAL GENERATION — MA50 Long/Short Overnight Strategy")
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
        # Bull regime: long top-N overnight momentum winners
        # Thesis: institutional selling at close depresses prices →
        #         retail buying at open drives them up
        selected = signals_df.head(LONG_N)["ticker"].tolist()
        action = "BUY"
        label = f"TOP-{LONG_N} momentum winners (LONG)"
    else:
        # Bear regime: short bottom-N overnight momentum losers
        # Thesis: overnight discount on weakest stocks in bear regime
        selected = signals_df.tail(SHORT_N)["ticker"].tolist()
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
        "date":      str(date.today()),
        "regime":    regime,
        "spy_close": round(spy_close, 4),
        "ma50":      round(ma50, 4) if not pd.isna(ma50) else None,
        "action":    action,
        "stocks":    selected,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSignals saved → {filename}")
    print("=" * 60)
    return output


if __name__ == "__main__":
    run_signal_generation()