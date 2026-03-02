import os
import time
import json
import glob
import zoneinfo
from datetime import datetime
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from SaxoOrderExecutor import SaxoOrderExecutor

load_dotenv()

ACCESS_TOKEN = os.getenv("SAXO_ACCESS_TOKEN")
BASE_URL = os.getenv("SAXO_BASE_URL", "https://gateway.saxobank.com/sim/openapi")


def load_tickers_from_json() -> List[str]:
    """Carrega automaticamente o ficheiro de sinais mais recente da pasta signals/."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    signals_dir = project_root / 'signals'

    # Procura todos os ficheiros de sinais e ordena pelo mais recente
    pattern = str(signals_dir / 'signals_*.json')
    files = sorted(glob.glob(pattern), reverse=True)

    if not files:
        raise FileNotFoundError(f"Nenhum ficheiro de sinais encontrado em: {signals_dir}")

    latest = files[0]
    print(f"A usar ficheiro de sinais: {latest}")

    with open(latest, 'r') as f:
        data = json.load(f)

    tickers = data.get('stocks', [])
    signal_date = data.get('date', 'desconhecida')
    print(f"Data dos sinais: {signal_date} | Tickers carregados ({len(tickers)}): {tickers}")
    return tickers


def should_buy() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 20 and now.minute >= 55


def should_sell() -> bool:
    now = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
    return now.hour == 14 and 30 <= now.minute < 35


def buy_orders(executor: SaxoOrderExecutor, tickers: List[str], quantity: int = 1) -> List[dict]:
    orders = []
    for t in tickers:
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
    if not ACCESS_TOKEN:
        print("ERRO: SAXO_ACCESS_TOKEN não definido no ficheiro .env — aborting.")
        exit(1)

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