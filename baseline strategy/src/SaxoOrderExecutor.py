import os
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ACCESS_TOKEN = os.getenv("SAXO_ACCESS_TOKEN")
BASE_URL     = os.getenv("SAXO_BASE_URL")

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class SaxoOrderExecutor:
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        })
        self.account_key = None

    def connect(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/port/v1/accounts/me")
            response.raise_for_status()
            accounts = response.json().get('Data', [])
            if not accounts:
                return False
            self.account_key = accounts[0]['AccountKey']
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_uic(self, symbol: str, asset_type: str = "Stock") -> Optional[int]:
        try:
            response = self.session.get(
                f"{self.base_url}/ref/v1/instruments",
                params={"Keywords": symbol, "AssetTypes": asset_type}
            )
            response.raise_for_status()
            data = response.json().get("Data", [])
            if not data:
                print(f"[{symbol}] No UIC found.")
                return None
            for instrument in data:
                if instrument.get("Symbol") == symbol:
                    return instrument["Identifier"]
            return data[0]["Identifier"]
        except Exception as e:
            print(f"[{symbol}] Error resolving UIC: {e}")
            return None

    def execute_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.account_key:
            return {"success": False, "error": "Account not connected"}

        required = ["action", "symbol", "uic", "quantity", "asset_type"]
        missing = [field for field in required if field not in order_params]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}

        action = order_params["action"].upper()
        if action not in ["BUY", "SELL"]:
            return {"success": False, "error": "Action must be BUY or SELL"}

        order = {
            "AccountKey": self.account_key,
            "Amount": int(order_params["quantity"]),
            "AssetType": order_params["asset_type"],
            "BuySell": "Buy" if action == "BUY" else "Sell",
            "OrderType": order_params.get("order_type", "Market"),
            "Uic": int(order_params["uic"]),
            "ManualOrder": True,
            "OrderDuration": {"DurationType": "DayOrder"}
        }

        if order["OrderType"] == "Limit" and "limit_price" in order_params:
            order["OrderPrice"] = float(order_params["limit_price"])

        # Retry logic
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.post(f"{self.base_url}/trade/v2/orders", json=order)
                if response.status_code in [200, 201]:
                    response_data = response.json()
                    return {
                        "success": True,
                        "order_id": response_data.get('OrderId'),
                        "status": "Executed",
                        "timestamp": datetime.now().isoformat()
                    }
                print(f"[{order_params['symbol']}] Attempt {attempt}/{MAX_RETRIES} failed: HTTP {response.status_code}")
            except Exception as e:
                print(f"[{order_params['symbol']}] Attempt {attempt}/{MAX_RETRIES} error: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        return {"success": False, "error": f"Failed after {MAX_RETRIES} attempts"}

    def get_positions(self) -> Dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/port/v1/positions/me",
                params={"ClientKey": self.account_key}
            )
            response.raise_for_status()
            positions = response.json().get('Data', [])
            return {"success": True, "positions": positions, "count": len(positions)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_balance(self) -> Dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/port/v1/balances/me",
                params={"ClientKey": self.account_key}
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    if not ACCESS_TOKEN or not BASE_URL:
        raise EnvironmentError("SAXO_ACCESS_TOKEN or SAXO_BASE_URL not found in .env")

    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if executor.connect():
        print(f"Connected! AccountKey: {executor.account_key}")

        uic = executor.get_uic("AAPL", asset_type="Stock")
        print(f"AAPL UIC: {uic}")

        result = executor.execute_order({
            "action": "BUY",
            "symbol": "AAPL",
            "uic": uic,
            "quantity": 1,
            "asset_type": "Stock",
            "order_type": "Market"
        })
        print(f"Full result: {result}")