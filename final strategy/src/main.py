import json
import time
import zoneinfo
import subprocess
from datetime import datetime
from pathlib import Path

import yfinance as yf                                              # [+] needed for price/qty calc

from SaxoOrderExecutor import SaxoOrderExecutor, ACCESS_TOKEN, BASE_URL
from Risk_Management import validate_trade, update_daily_pnl, calculate_position_size, check_daily_loss_limit  # [+] check_daily_loss_limit

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


def load_latest_signals() -> dict:                                # [*] returns full dict, not just list
    signal_files = sorted((PROJECT_ROOT / "signals").glob("signals_*.json"), reverse=True)
    if not signal_files:
        print("No signal files found!")
        return {}
    with open(signal_files[0]) as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('stocks', []))} tickers | "
          f"Regime: {data.get('regime')} | Action: {data.get('action')}")
    return data


def should_run_signals() -> bool:
    # Window 20:30–20:49 WET — wide enough for ~500-ticker download before buy at 20:55
    now = now_london()
    return now.hour == 20 and 30 <= now.minute < 50

def should_buy() -> bool:
    now = now_london()
    return now.hour == 20 and now.minute >= 55

def should_sell() -> bool:
    # Window 14:25–14:44 WET — NYSE opening auction ~14:30 WET (paper Section 3.4)
    now = now_london()
    return now.hour == 14 and 25 <= now.minute < 45


if __name__ == "__main__":
    print("=== Overnight Strategy — Starting ===")

    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if not executor.connect():
        print("Could not connect to Saxo Bank API — aborting.")
        exit(1)

    CAPITAL = executor.get_balance()["data"]["TotalValue"]
    print(f"Account capital: ${CAPITAL:,.2f}")

    # Paper Section 3.3 — circuit breaker: if yesterday triggered a halt, abort today  [+]
    if check_daily_loss_limit(CAPITAL):
        print("CIRCUIT BREAKER ACTIVE: daily loss limit was hit — no trading today.")
        write_status({"phase": "HALTED", "capital": CAPITAL, "bought": False,
                      "sold": False, "positions": [], "pnl_today": 0, "signals": []})
        exit(0)

    tickers = []
    regime = None                                                  # [+] 'LONG' or 'SHORT'
    bought = False
    sold = False
    buy_uics = {}                                                  # [*] renamed: ticker -> UIC
    buy_quantities = {}                                            # [+] ticker -> quantity
    signals_generated = False

    write_status({"phase": "WAITING", "capital": CAPITAL, "bought": False,
                  "sold": False, "positions": [], "pnl_today": 0, "signals": []})

    while True:

        # 20:30 — gerar sinais
        if should_run_signals() and not signals_generated:
            print("Running signal_generator.py...")
            subprocess.run(["python3", str(Path(__file__).parent / "signal_generator.py")], check=True)
            sig_data = load_latest_signals()                       # [*] full dict now
            if not sig_data or not sig_data.get("stocks"):        # [+] only lock if valid; retry otherwise
                print("WARNING: Empty signal data — will retry next loop.")
            else:
                tickers  = sig_data.get("stocks", [])
                regime   = sig_data.get("regime")                 # [+] 'LONG' or 'SHORT'
                signals_generated = True
            write_status({"phase": "SIGNALS_READY", "capital": CAPITAL, "bought": False,
                          "sold": False, "positions": [], "pnl_today": 0,
                          "regime": regime, "signals": tickers})

        # 20:55 — abrir posições
        if should_buy() and not bought and tickers and regime:     # [+] regime guard: None → skip
            print(f"=== {'BUY' if regime == 'LONG' else 'SHORT'} WINDOW [regime={regime}] ===")
            open_action  = "BUY"  if regime == "LONG" else "SELL"  # [+] regime-based action
            position_usd = calculate_position_size(CAPITAL, num_positions=len(tickers))  # [+]
            successful = []

            for t in tickers:
                try:
                    price    = float(yf.Ticker(t).history(period="1d")["Close"].iloc[-1])  # [+]
                    quantity = max(1, int(position_usd / price))                            # [+]
                except Exception:
                    quantity = 1

                if not validate_trade(t, quantity, CAPITAL):
                    continue
                uic = executor.get_uic(t)
                if uic is None:
                    continue
                result = executor.execute_order({"action": open_action, "symbol": t, "uic": uic,  # [*]
                                                  "quantity": quantity, "asset_type": "Stock",
                                                  "order_type": "Market"})
                print(f"{open_action} {t} x{quantity}: {result}")
                if result.get("success"):
                    buy_uics[t]       = uic       # [*] was buy_prices[t] = uic
                    buy_quantities[t] = quantity  # [+]
                    successful.append(t)
                time.sleep(0.5)

            bought = True
            write_status({"phase": "HOLDING", "capital": CAPITAL, "bought": True,
                          "sold": False, "positions": successful,
                          "pnl_today": 0, "regime": regime, "signals": tickers})

        # 14:30 — fechar posições
        if should_sell() and not sold and buy_uics:
            print(f"=== CLOSE WINDOW [regime={regime}] ===")
            close_action = "SELL" if regime == "LONG" else "BUY"   # [+] SELL longs / cover shorts

            for symbol, uic in buy_uics.items():                   # [*] was buy_prices
                quantity = buy_quantities.get(symbol, 1)           # [+]
                result = executor.execute_order({"action": close_action, "symbol": symbol,  # [*]
                                                  "uic": uic, "quantity": quantity,
                                                  "asset_type": "Stock", "order_type": "Market"})
                print(f"CLOSE {symbol} x{quantity}: {result}")
                time.sleep(0.5)

            CAPITAL_END = executor.get_balance()["data"]["TotalValue"]
            daily_pnl = CAPITAL_END - CAPITAL
            update_daily_pnl(daily_pnl)
            append_pnl_history(str(now_london().date()), daily_pnl, CAPITAL, CAPITAL_END)
            print(f"Day P&L: ${daily_pnl:,.2f} | Regime: {regime}")

            write_status({"phase": "DONE", "capital": CAPITAL_END, "bought": True,
                          "sold": True, "positions": [], "pnl_today": round(daily_pnl, 2),
                          "regime": regime, "signals": tickers})
            sold = True   # [+] explicit flag before break — prevents double-sell on unexpected re-entry
            break

        # atualizar P&L em tempo real enquanto está em holding
        if bought and not sold:
            live_capital = executor.get_balance()["data"]["TotalValue"]
            live_pnl = live_capital - CAPITAL
            # Paper Section 3.3 — real-time circuit breaker: halt if loss > 2% of capital  [+]
            if check_daily_loss_limit(CAPITAL):
                print(f"CIRCUIT BREAKER TRIGGERED (live P&L: ${live_pnl:,.2f}) — closing all positions NOW.")
                close_action = "SELL" if regime == "LONG" else "BUY"
                for symbol, uic in buy_uics.items():
                    executor.execute_order({"action": close_action, "symbol": symbol, "uic": uic,
                                            "quantity": buy_quantities.get(symbol, 1),
                                            "asset_type": "Stock", "order_type": "Market"})
                    time.sleep(0.3)
                update_daily_pnl(live_pnl)
                write_status({"phase": "HALTED", "capital": live_capital, "bought": True, "sold": True,
                              "positions": [], "pnl_today": round(live_pnl, 2),
                              "regime": regime, "signals": tickers})
                break
            write_status({"phase": "HOLDING", "capital": CAPITAL, "bought": True, "sold": False,
                          "positions": list(buy_uics.keys()),
                          "pnl_today": round(live_pnl, 2),
                          "regime": regime, "signals": tickers})

        time.sleep(30)