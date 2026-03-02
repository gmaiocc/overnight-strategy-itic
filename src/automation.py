import time
import json
import zoneinfo
from datetime import datetime
from typing import List
from pathlib import Path

from SaxoOrderExecutor import SaxoOrderExecutor
from Risk_Management import validate_trade, update_daily_pnl

ACCESS_TOKEN = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjY3NEM0MjFEMzZEMUE1OUNFNjFBRTIzMjMyOTVFRTAyRTc3MDMzNTkifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiY0dsMk8xVGUxdmdOaW18b1BxR0phdz09IiwiY2lkIjoiY0dsMk8xVGUxdmdOaW18b1BxR0phdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMmIwZTlhMWUyZjI5NDRjMDhhMzI5NWIyOTFjNmZlNzEiLCJkZ2kiOiI4NCIsImV4cCI6IjE3NzI0OTg5NjAiLCJvYWwiOiIxRiIsImlpZCI6ImY0Y2U4MTI4MTJlNzRmOTM2OGE2MDhkZTZmMDBkZTAwIn0.Up7Gw1suvrh2pk7riDz6XjiIOvBruJR6T2IoyA9fTidN9mLBLFma2RJIyDitFlo6pWWS0keRqj-lxgyFTbpESQ"
BASE_URL = "https://gateway.saxobank.com/sim/openapi"


def load_tickers_from_json() -> List[str]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    signal_files = sorted((project_root / 'signals').glob('signals_*.json'), reverse=True)
    if not signal_files:
        print("No signal files found!")
        return []
    json_path = signal_files[0]
    print(f"Loading signals from: {json_path.name}")
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    tickers = data['stocks']
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    return tickers


def should_buy() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 20 and now.minute >= 55


def should_sell() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 14 and 30 <= now.minute < 35


def buy_orders(executor: SaxoOrderExecutor, tickers: List[str], capital: float, quantity: int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        if not validate_trade(t, quantity, capital):
            print(f"[{t}] Skipping — failed risk check")
            continue
        uic = executor.get_uic(t)
        if uic is None:
            print(f"[{t}] Skipping — could not resolve UIC")
            continue
        orders.append({
            "action": "BUY",
            "symbol": t,
            "uic": uic,
            "quantity": quantity,
            "asset_type": "Stock",
            "order_type": "Market",
        })
    return orders


def sell_orders(executor: SaxoOrderExecutor, tickers: List[str], quantity: int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        uic = executor.get_uic(t)
        if uic is None:
            print(f"[{t}] Skipping — could not resolve UIC")
            continue
        orders.append({
            "action": "SELL",
            "symbol": t,
            "uic": uic,
            "quantity": quantity,
            "asset_type": "Stock",
            "order_type": "Market",
        })
    return orders


if __name__ == "__main__":
    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if not executor.connect():
        print("Could not connect to Saxo Bank API — aborting.")
        exit(1)

    # Buscar capital real da conta
    balance_data = executor.get_balance()
    CAPITAL = balance_data['data']['TotalValue']
    print(f"Account capital: ${CAPITAL:,.2f}")

    tickers = load_tickers_from_json()
    bought = False
    sold = False
    buy_prices = {}  # guardar preços de compra para calcular P&L

    while True:
        if should_buy() and not bought:
            orders = buy_orders(executor, tickers, CAPITAL)
            for order in orders:
                result = executor.execute_order(order)
                print(f"BUY {order['symbol']}: {result}")
                if result.get('success'):
                    buy_prices[order['symbol']] = order  # guardar para P&L
                time.sleep(0.5)
            bought = True

        if should_sell() and not sold:
            # Guardar posições antes de vender para calcular P&L
            positions = executor.get_positions()

            orders = sell_orders(executor, list(buy_prices.keys()))
            for order in orders:
                result = executor.execute_order(order)
                print(f"SELL {order['symbol']}: {result}")
                time.sleep(0.5)

            # Calcular P&L do dia
            final_balance = executor.get_balance()
            final_capital = final_balance['data']['TotalValue']
            daily_pnl = final_capital - CAPITAL
            update_daily_pnl(daily_pnl)
            print(f"Day P&L: ${daily_pnl:,.2f}")

            sold = True
            break

        time.sleep(30)