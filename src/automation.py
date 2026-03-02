import time
import json
import zoneinfo
from datetime import datetime
from typing import List
from pathlib import Path

from SaxoOrderExecutor import SaxoOrderExecutor

ACCESS_TOKEN = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiY0o5TEVxd0lySmoxZEpBbmNKa1hRZz09IiwiY2lkIjoiY0o5TEVxd0lySmoxZEpBbmNKa1hRZz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMzA4NDU5YTJjMmY0NGY0OGI5M2VlMTEyZmMwMTFjZTgiLCJkZ2kiOiI4NCIsImV4cCI6IjE3NTY5MDYwNDIiLCJvYWwiOiIxRiIsImlpZCI6ImMwNzk5NmY5ZGUxNjRjZDJmMTQ0MDhkZGU3ODAwNDMzIn0.BNSVBMcMQbTy_hWeiU_DsIdiJQyEhkeuv7kfArevmBPtQXYJoiBzy3pLlvM6jJ-6X8vN3BfVzAwhBX_TZe9g1g"
BASE_URL = "https://gateway.saxobank.com/sim/openapi"


def load_tickers_from_json(filename: str = 'signals/signals_20260227_1030.json') -> List[str]:
    script_path = Path(__file__).resolve()       # src/automation.py
    project_root = script_path.parent.parent     # go up two levels to project root
    json_path = project_root / 'signals' / 'signals_20260227_1030.json'
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    tickers = data['stocks']
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    return data.get("stocks", [])


def should_buy() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 20 and now.minute >= 55  # FIX: was hour==21 AND hour==20 (impossible)


def should_sell() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 14 and 30 <= now.minute < 35


def buy_orders(executor: SaxoOrderExecutor, tickers: List[str], quantity: int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        uic = executor.get_uic(t)  # FIX: resolve UIC per ticker instead of hardcoding 46959
        if uic is None:
            print(f"[{t}] Skipping — could not resolve UIC")
            continue
        orders.append({
            "action": "BUY",
            "symbol": t,
            "uic": uic,
            "quantity": quantity,  # depends risk management
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
            "quantity": quantity,  # depends risk management
            "asset_type": "Stock",
            "order_type": "Market",
        })
    return orders


if __name__ == "__main__":
    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if not executor.connect():
        print("Could not connect to Saxo Bank API — aborting.")
        exit(1)

    tickers = load_tickers_from_json()
    bought = False
    sold = False

    while True:
        if should_buy() and not bought:
            orders = buy_orders(executor, tickers)
            for order in orders:
                result = executor.execute_order(order)
                print(f"BUY {order['symbol']}: {result}")
            bought = True

        if should_sell() and not sold:
            orders = sell_orders(executor, tickers)
            for order in orders:
                result = executor.execute_order(order)
                print(f"SELL {order['symbol']}: {result}")
            sold = True
            break

        time.sleep(30)