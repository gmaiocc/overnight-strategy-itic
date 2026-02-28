import argparse
import datetime
import logging
from typing import List
import zoneinfo
from datetime import datetime
import os
import json
from pathlib import Path

london_time = datetime.now(zoneinfo.ZoneInfo("Europe/London"))

def load_tickers_from_json(filename: str = 'signals/signals_20260227_1030.json') -> List[str]:
    script_path = Path(__file__).resolve()  # src/automation.py
    project_root = script_path.parent.parent  # go up two levels to project root
    signals_dir = project_root
    json_path = signals_dir / 'signals' / 'signals_20260227_1030.json'
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)  # Use json.load(), not json_path.load()
    tickers = data['stocks']
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    return data.get("stocks", [])

def should_buy() -> bool:
    return london_time.hour == 21 and london_time.hour == 20 and london_time.minute >=55

def should_sell() -> bool:
    return london_time.hour == 14 and 30<=london_time.minute <35

#def teste() -> bool:
    return london_time.hour == 21 and london_time.hour == 19 and london_time.minute >=55

def buy_orders(tickers: List[str] , int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        orders.append({
            "action": "BUY",
            "symbol": t,
            "uic": "46959",
            "quantity": 1, # depends risk management
            "asset_type": "Stock",
            "order_type": "Market",
        })
    return orders
def sell_orders(tickers: List[str] , int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        orders.append({
            "action": "SELL",
            "symbol": t,
            "uic": "46959",
            "quantity": 1, # depends risk management
            "asset_type": "Stock",
            "order_type": "Market",
        })
    return orders

if __name__ == "__main__":

    d=load_tickers_from_json()
    orders = sell_orders(d, 1)
    print(orders)


