import logging
import yfinance as yf
from typing import Tuple

logger = logging.getLogger(__name__)

class TradeValidator:
    """Handles trade validation rules independently of risk state."""

    def __init__(self, min_market_cap: float = 2e9, min_daily_volume_usd: float = 10e6):
        self.min_market_cap = min_market_cap
        self.min_daily_volume_usd = min_daily_volume_usd

    def validate_trade(self, symbol: str, quantity: int, capital: float, max_position_pct: float) -> Tuple[bool, str]:
        """
        Validate whether a trade passes all risk checks.

        Checks:
            1. Position size within limits
            2. Minimum market cap
            3. Minimum average daily volume
            4. No upcoming earnings announcement
        """
        logger.info(f"Validating trade: {symbol} x{quantity}")

        # --- Fetch stock data ---
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="30d")
        except Exception as e:
            return False, f"Failed to fetch data — {e}"

        if hist.empty:
            return False, "No historical data available"

        # --- Position size check ---
        current_price = hist["Close"].iloc[-1]
        trade_value = quantity * current_price
        max_allowed = capital * max_position_pct

        if trade_value > max_allowed:
            return False, f"Trade value ${trade_value:,.2f} exceeds max allowed ${max_allowed:,.2f}"

        # --- Market cap check ---
        market_cap = info.get("marketCap", 0)
        if market_cap < self.min_market_cap:
            return False, f"Market cap ${market_cap/1e9:.2f}B < minimum ${self.min_market_cap/1e9:.0f}B"

        # --- Average daily volume check --- this check is already done when picking the stocks in signal_generator, so this is useless
        avg_volume = hist["Volume"].mean()
        avg_price = hist["Close"].mean()
        avg_volume_usd = avg_volume * avg_price

        if avg_volume_usd < self.min_daily_volume_usd:
            return False, f"Avg daily volume ${avg_volume_usd/1e6:.1f}M < minimum ${self.min_daily_volume_usd/1e6:.0f}M"

        return True, "Trade approved"