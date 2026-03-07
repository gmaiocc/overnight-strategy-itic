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


def load_latest_signals() -> dict | None:
    signal_files = sorted((PROJECT_ROOT / "signals").glob("signals_*.json"), reverse=True)
    if not signal_files:
        print("No signal files found!")
        return None
    with open(signal_files[0]) as f:
        data = json.load(f)
    print(f"Loaded signals: {data}")
    return data


def get_balance_safe(executor: "SaxoOrderExecutor") -> float | None:
    result = executor.get_balance()
    if not result.get("success"):
        print(f"ERROR: Could not fetch balance — {result.get('error')}")
        return None
    return result["data"]["TotalValue"]


def should_run_signals() -> bool:
    now = now_london()
    return now.hour == 20 and 30 <= now.minute < 35


def should_open() -> bool:
    now = now_london()
    return now.hour == 20 and now.minute >= 55


def should_close() -> bool:
    now = now_london()
    return now.hour == 14 and 30 <= now.minute < 35


if __name__ == "__main__":

    print("=== Overnight Strategy (Long/Short Regime) — Starting ===")

    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if not executor.connect():
        print("Could not connect to Saxo Bank API — aborting.")
        exit(1)

    CAPITAL = get_balance_safe(executor)
    if CAPITAL is None:
        print("Could not fetch initial capital — aborting.")
        exit(1)
    print(f"Account capital: ${CAPITAL:,.2f}")

    NUM_POSITIONS = 10
    position_usd  = CAPITAL / NUM_POSITIONS

    tickers  = []
    regime = None
    action = None

    positions_uic = {}
    positions_qty = {}

    opened = False
    closed = False
    signals_generated = False

    write_status({
        "phase": "WAITING",
        "capital": CAPITAL,
        "positions": [],
        "signals": [],
    })

    while True:

        # 20:30 generate signals
        if should_run_signals() and not signals_generated:

            print("Running signal_generator.py...")
            subprocess.run(["python3", str(Path(__file__).parent / "signal_generator.py")], check=True)

            data = load_latest_signals()
            if not data:
                time.sleep(10)
                continue

            tickers = data.get("stocks", [])
            regime = data.get("regime")
            action = data.get("action")

            print(f"Regime: {regime} | Action: {action} | Stocks: {tickers}")

            signals_generated = True
            write_status({
                "phase":   "SIGNALS_READY",
                "capital": CAPITAL,
                "signals": tickers,
                "regime":  regime,
            })

        #  20:55 open positions
        if should_open() and not opened and tickers:

            # Safety guard: action must be resolved before placing orders
            if action not in ("BUY", "SELL_SHORT"):
                print(f"ERROR: Unexpected action value '{action}' — skipping open.")
            else:
                print(f"=== OPEN WINDOW | Regime: {regime} | Action: {action} ===")
                successful = []

                for t in tickers:
                    try:
                        price = float(yf.Ticker(t).history(period="1d")["Close"].iloc[-1])
                        quantity = max(1, int(position_usd / price))
                        uic = executor.get_uic(t)

                        if uic is None:
                            continue

                        print(f"{action} {t} | price ${price:.2f} | qty {quantity}")
                        result = executor.execute_order({
                            "action": action,
                            "symbol": t,
                            "uic": uic,
                            "quantity": quantity,
                            "asset_type": "Stock",
                            "order_type": "Market",
                        })
                        print(f"{result}")

                        if result.get("success"):
                            positions_uic[t] = uic
                            positions_qty[t] = quantity
                            successful.append(t)

                    except Exception as e:
                        print(f"Error processing {t}: {e}")

                    time.sleep(0.5)

            opened = True
            write_status({
                "phase": "HOLDING",
                "capital": CAPITAL,
                "positions": successful if action in ("BUY", "SELL_SHORT") else [],
                "regime": regime,
            })

        # 14:30 close positions
        if should_close() and not closed and positions_uic:

            if action not in ("BUY", "SELL_SHORT"):
                print(f"ERROR: Cannot close — action is '{action}'. Manual intervention required.")
            else:
                # Long positions are closed with SELL; short positions with BUY
                close_action = "SELL" if action == "BUY" else "BUY"
                print(f"=== CLOSE WINDOW | Closing with: {close_action} ===")

                for symbol, uic in positions_uic.items():
                    quantity = positions_qty.get(symbol, 1)
                    result = executor.execute_order({
                        "action": close_action,
                        "symbol": symbol,
                        "uic": uic,
                        "quantity": quantity,
                        "asset_type": "Stock",
                        "order_type": "Market",
                    })
                    print(f"{close_action} {symbol} x{quantity}: {result}")
                    time.sleep(0.5)

                CAPITAL_END = get_balance_safe(executor)
                if CAPITAL_END is None:
                    print("WARNING: Could not fetch end-of-day balance — P&L not recorded.")
                else:
                    daily_pnl = CAPITAL_END - CAPITAL
                    append_pnl_history(str(now_london().date()), daily_pnl, CAPITAL, CAPITAL_END)
                    print(f"Day P&L: ${daily_pnl:,.2f}")

            closed = True
            write_status({
                "phase": "DONE",
                "capital": CAPITAL_END if CAPITAL_END else CAPITAL,
                "pnl_today": round(daily_pnl, 2) if CAPITAL_END else None,
                "signals": tickers,
                "regime": regime,
            })
            break

        # live P&L update while holding
        if opened and not closed:
            live_capital = get_balance_safe(executor)
            if live_capital is not None:
                write_status({
                    "phase": "HOLDING",
                    "capital": CAPITAL,
                    "positions": list(positions_uic.keys()),
                    "pnl_today": round(live_capital - CAPITAL, 2),
                    "signals": tickers,
                    "regime": regime,
                })

        time.sleep(30)