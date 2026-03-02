import os
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("SAXO_ACCESS_TOKEN")
BASE_URL = os.getenv("SAXO_BASE_URL", "https://gateway.saxobank.com/sim/openapi")


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
            return data[0]["Identifier"]  # fallback to first result
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

        try:
            response = self.session.post(f"{self.base_url}/trade/v2/orders", json=order)
            if response.status_code not in [200, 201]:
                return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}

            response_data = response.json()
            return {
                "success": True,
                "order_id": response_data.get('OrderId'),
                "status": "Executed",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
    if not ACCESS_TOKEN:
        print("ERRO: SAXO_ACCESS_TOKEN não definido no .env")
        exit(1)

    executor = SaxoOrderExecutor(BASE_URL, ACCESS_TOKEN)
    if executor.connect():
        result = executor.execute_order({
            "action": "BUY",
            "symbol": "TQQQ",
            "uic": 46959,
            "quantity": 1,
            "asset_type": "CfdOnEtf",
            "order_type": "Market"
        })
        print(f"Order Success: {result.get('success')}")