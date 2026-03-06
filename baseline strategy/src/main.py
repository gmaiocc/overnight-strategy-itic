import json
import time
import zoneinfo
import subprocess
from datetime import datetime
from pathlib import Path

import yfinance as yf

from SaxoOrderExecutor import SaxoOrderExecutor, ACCESS_TOKEN, BASE_URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = PROJECT_ROOT / "status.json"
PNL_HISTORY_FILE = PROJECT_ROOT / "pnl_history.json"


def now_london() -> datetime:
    return datetime.now(zoneinfo.ZoneInfo("Europe/London"))


def write_status(data: dict):
    data["timestamp"] = datetime.now().isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def append_pnl_history(date: str, pnl: float, capital_start: float, capital_end: float):
    history = []
    if PNL_HISTORY_FILE.exists():
        with open(PNL_HISTORY_FILE) as f:
            history = json.load(f)
    history.append({
        "date": date,
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round((pnl / capital_start) * 100, 4),
        "capital_start": round(capital_start, 2),
        "capital_end": round(capital_end, 2),
    })
    with open(PNL_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_latest_signals() -> list:
    signal_files = sorted((PROJECT_ROOT / "signals").glob("signals_*.json"), reverse=True)
    if not signal_files:
        print("No signal files found!")
        return []
    with open(signal_files[0]) as f:
        data = json.load(f)
    tickers = data.get("stocks", [])
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    return tickers


def should_run_signals() -> bool:
    now = now_london()
    return now.hour == 20 and 30 <= now.minute < 35


def should_buy() -> bool:
    now = now_london()
    return now.hour == 20 and now.minute >= 55


def should_sell() -> bool:
    now = now_london()
    return now.hour == 14 and 30 <= now.minute < 35


if __name__ == "__main__":
    print("=== Overnight Strategy (Baseline) — Starting ===")

    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if not executor.connect():
        print("Could not connect to Saxo Bank API — aborting.")
        exit(1)

    CAPITAL = executor.get_balance()["data"]["TotalValue"]
    print(f"Account capital: ${CAPITAL:,.2f}")

    # Position sizing: equal weight across 10 positions (10% each)
    NUM_POSITIONS = 10
    position_usd = CAPITAL / NUM_POSITIONS

    tickers = []
    bought = False
    sold = False
    buy_prices = {}
    buy_quantities = {}
    signals_generated = False

    write_status({"phase": "WAITING", "capital": CAPITAL, "bought": False,
                  "sold": False, "positions": [], "pnl_today": 0, "signals": []})

    while True:

        # 20:30 — generate signals
        if should_run_signals() and not signals_generated:
            print("Running signal_generator.py...")
            subprocess.run(["python3", str(Path(__file__).parent / "signal_generator.py")], check=True)
            tickers = load_latest_signals()
            signals_generated = True
            write_status({"phase": "SIGNALS_READY", "capital": CAPITAL, "bought": False,
                          "sold": False, "positions": [], "pnl_today": 0, "signals": tickers})

        # 20:55 — buy
        if should_buy() and not bought and tickers:
            print("=== BUY WINDOW ===")
            successful = []

            for t in tickers:
                try:
                    price = float(yf.Ticker(t).history(period="1d")["Close"].iloc[-1])
                    quantity = max(1, int(position_usd / price))

                    uic = executor.get_uic(t)
                    if uic is None:
                        continue

                    print(f"  [{t}] price ~${price:.2f} -> {quantity} shares")
                    result = executor.execute_order({
                        "action": "BUY", "symbol": t, "uic": uic,
                        "quantity": quantity, "asset_type": "Stock", "order_type": "Market"
                    })
                    print(f"  BUY {t}: {result}")

                    if result.get("success"):
                        buy_prices[t] = uic
                        buy_quantities[t] = quantity
                        successful.append(t)

                except Exception as e:
                    print(f"  Error processing {t}: {e}")

                time.sleep(0.5)

            bought = True
            write_status({"phase": "HOLDING", "capital": CAPITAL, "bought": True,
                          "sold": False, "positions": successful, "pnl_today": 0, "signals": tickers})

        # 14:30 — sell
        if should_sell() and not sold and buy_prices:
            print("=== SELL WINDOW ===")
            for symbol, uic in buy_prices.items():
                quantity = buy_quantities.get(symbol, 1)
                result = executor.execute_order({
                    "action": "SELL", "symbol": symbol, "uic": uic,
                    "quantity": quantity, "asset_type": "Stock", "order_type": "Market"
                })
                print(f"  SELL {symbol} x{quantity}: {result}")
                time.sleep(0.5)

            CAPITAL_END = executor.get_balance()["data"]["TotalValue"]
            daily_pnl = CAPITAL_END - CAPITAL
            append_pnl_history(str(now_london().date()), daily_pnl, CAPITAL, CAPITAL_END)
            print(f"  Day P&L: ${daily_pnl:,.2f}")

            sold = True
            write_status({"phase": "DONE", "capital": CAPITAL_END, "bought": True,
                          "sold": True, "positions": [], "pnl_today": round(daily_pnl, 2), "signals": tickers})
            break

        # Update live P&L while holding
        if bought and not sold:
            live_capital = executor.get_balance()["data"]["TotalValue"]
            write_status({"phase": "HOLDING", "capital": CAPITAL, "bought": True, "sold": False,
                          "positions": list(buy_prices.keys()),
                          "pnl_today": round(live_capital - CAPITAL, 2), "signals": tickers})

        time.sleep(30)