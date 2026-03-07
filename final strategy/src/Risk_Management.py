import json
import logging
import os
from datetime import datetime, timedelta, date
from pathlib import Path
from Validations import TradeValidator

_validator = TradeValidator() # create instance of new class TradeValidator

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("risk_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# RULES SET-UP
MAX_POSITION_PCT = 0.10  # Max 10% of capital per stock
MAX_POSITIONS = 10  # Max simultaneous positions
MIN_MARKET_CAP = 2e9  # $2B minimum market cap
MIN_DAILY_VOLUME_USD = 10e6  # $10M average daily volume
MAX_BID_ASK_SPREAD_PCT = 0.001  # 0.1% max spread
MAX_DAILY_LOSS_PCT = 0.02  # -2% of capital triggers halt
HALT_DURATION_HOURS = 24

# File for persistence across script runs
STATE_FILE = Path("risk_state.json")

def _load_state() -> dict:
    """Load persistent risk state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "daily_pnl": 0.0,
        "trading_halted_until": None,
        "date": str(date.today())
    }


def _save_state(state: dict):
    """Save risk state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _reset_daily_state_if_new_day(state: dict) -> dict:
    """Reset daily P&L if it's a new trading day."""
    today = str(date.today())
    if state.get("date") != today:
        logger.info("New trading day detected. Resetting daily P&L.")
        state["daily_pnl"] = 0.0
        state["date"] = today
        # Don't reset halt — it persists across days if within 24h
    return state


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def calculate_position_size(capital: float, num_positions: int = None) -> float:
    """
    Calculate the dollar amount to allocate per position.

    Args:
        capital: Total portfolio capital in USD
        num_positions: Number of planned positions (defaults to MAX_POSITIONS)

    Returns:
        Dollar amount per position
    """
    if num_positions is None:
        num_positions = MAX_POSITIONS

    num_positions = min(num_positions, MAX_POSITIONS)  # Cap at max
    position_size = capital * MAX_POSITION_PCT

    logger.info(f"Position size: ${position_size:,.2f} ({MAX_POSITION_PCT * 100}% of ${capital:,.2f})")
    return position_size

def calculate_position_lot_size(capital: float, price: float) -> int:
    """
    Calculate the number of shares to buy for a position based on capital and price.

    Args:
        capital: Total portfolio capital in USD
        price: Current price of the stock
    Returns:
        Number of shares to buy (rounded down to nearest whole share)
    """
    position_size = calculate_position_size(capital)
    lot_size = int(position_size // price)  # Round down to whole shares
    logger.info(f"Calculated lot size: {lot_size} shares at ${price:.2f} for position size ${position_size:,.2f}")
    return lot_size

def check_daily_loss_limit(capital: float = None) -> bool:
    """
    Check whether the daily loss limit has been breached.
    If breached, trading is halted for HALT_DURATION_HOURS.

    Args:
        capital: Current total capital (used to update halt threshold if provided)

    Returns:
        True = trading is ALLOWED, False = trading is HALTED
    """
    state = _load_state()
    state = _reset_daily_state_if_new_day(state)

    # Check if currently halted
    if state.get("trading_halted_until"):
        halt_until = datetime.fromisoformat(state["trading_halted_until"])
        if datetime.now() < halt_until:
            logger.warning(f"Trading HALTED until {halt_until}. No orders will be placed.")
            return False
        else:
            logger.info("Trading halt period expired. Resuming trading.")
            state["trading_halted_until"] = None

    # Check if daily loss limit is breached
    if capital and capital > 0:
        loss_pct = abs(state["daily_pnl"]) / capital if state["daily_pnl"] < 0 else 0
        if loss_pct >= MAX_DAILY_LOSS_PCT:
            halt_until = datetime.now() + timedelta(hours=HALT_DURATION_HOURS)
            state["trading_halted_until"] = halt_until.isoformat()
            logger.error(
                f"DAILY LOSS LIMIT BREACHED: {loss_pct * 100:.2f}% loss. "
                f"Trading halted until {halt_until}."
            )
            _save_state(state)
            return False

    _save_state(state)
    return True


def update_daily_pnl(pnl_change: float):
    """
    Call this after each trade closes to update the running daily P&L.

    Args:
        pnl_change: Dollar P&L of the completed trade (negative = loss)
    """
    state = _load_state()
    state = _reset_daily_state_if_new_day(state)
    state["daily_pnl"] += pnl_change
    _save_state(state)
    logger.info(f"Daily P&L updated: ${state['daily_pnl']:,.2f}")

def validate_trade(symbol: str, quantity: int, capital: float) -> bool:
    """Wrapper for backward compatibility"""
    is_valid, message = _validator.validate_trade(symbol, quantity, capital, MAX_POSITION_PCT)
    if not is_valid:
        logger.warning(f"[{symbol}] REJECTED: {message}")
    return is_valid

# ─────────────────────────────────────────────
# DASHBOARD / REPORTING
# ─────────────────────────────────────────────

def get_risk_summary(capital: float) -> dict:
    """
    Return a summary of current risk state for monitoring/dashboard use.
    """
    state = _load_state()
    state = _reset_daily_state_if_new_day(state)

    halted = False
    halt_until = None
    if state.get("trading_halted_until"):
        halt_until = datetime.fromisoformat(state["trading_halted_until"])
        halted = datetime.now() < halt_until

    daily_loss_pct = (state["daily_pnl"] / capital * 100) if capital else 0

    summary = {
        "date": state["date"],
        "daily_pnl_usd": round(state["daily_pnl"], 2),
        "daily_pnl_pct": round(daily_loss_pct, 4),
        "trading_halted": halted,
        "halted_until": str(halt_until) if halt_until else None,
        "loss_limit_pct": MAX_DAILY_LOSS_PCT * 100,
        "max_position_pct": MAX_POSITION_PCT * 100,
        "max_positions": MAX_POSITIONS,
    }

    logger.info(f"Risk Summary: {summary}")
    return summary


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

#if __name__ == "__main__":
 #   TOTAL_CAPITAL = 100_000  # Example: $100,000 portfolio

    # How much to put in each position
#    size = calculate_position_size(TOTAL_CAPITAL, num_positions=10)
#    print(f"\nPosition size per stock: ${size:,.2f}")

    # Check if we can trade today
#    can_trade = check_daily_loss_limit(TOTAL_CAPITAL)
#    print(f"Trading allowed: {can_trade}")

    # Validate a specific trade (e.g. buy 50 shares of AAPL)
#    approved = validate_trade("AAPL", 50, TOTAL_CAPITAL)
#    print(f"Trade approved: {approved}")

    # Simulate a loss and update P&L
#    update_daily_pnl(-500)  # Lost $500

    # Print risk dashboard
#    summary = get_risk_summary(TOTAL_CAPITAL)
#    print(f"\nRisk Summary:\n{json.dumps(summary, indent=2)}")