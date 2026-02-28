import argparse
import datetime
import json
import logging
from typing import List
import zoneinfo
from datetime import datetime


london_time = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
#acessToken=SaxobankAPI.get_access_token()
print(london_time)

def load_tickers_from_json(filename='signals/signals_20260227_1030.json') -> List[str]:
    with open(filename) as json_file:
        data = json.load(json_file)
    tickers = data['stocks']
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    return data.get("stocks", [])

#def build_orders(tickers: List[str], quantity: int = 1) -> List[dict]:
 #   if london_time = 21:
        #abre ordem
 #   if time = 9:
        #fecha ordem

def build_orders(tickers: List[str], quantity: int = 1) -> List[dict]:
    orders = []
    for t in tickers:
        orders.append({
            "ticker": t,
            "side": "buy",
            "side": "buy",
            "qty": quantity,
            "type": "market",
            "time": london_time,
        })
    return orders




if __name__ == "__main__":

    load_tickers_from_json()
