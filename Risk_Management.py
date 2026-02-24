import json
import logging
import os
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np

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
    """
    Validate whether a trade passes all risk checks.

    Checks:
        1. Trading not halted (daily loss limit)
        2. Position size within limits
        3. Minimum market cap
        4. Minimum average daily volume
        5. No upcoming earnings announcement

    Args:
        symbol: Stock ticker (e.g. 'AAPL')
        quantity: Number of shares to buy
        capital: Total portfolio capital in USD

    Returns:
        True if trade is allowed, False otherwise
    """
    logger.info(f"Validating trade: {symbol} x{quantity}")

    # --- 1. Check if trading is halted ---
    if not check_daily_loss_limit(capital):
        logger.warning(f"[{symbol}] REJECTED: Trading is halted.")
        return False

    # --- Fetch stock data ---
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="30d")
    except Exception as e:
        logger.error(f"[{symbol}] REJECTED: Failed to fetch data — {e}")
        return False

    if hist.empty:
        logger.warning(f"[{symbol}] REJECTED: No historical data available.")
        return False

    # --- 2. Position size check ---
    current_price = hist["Close"].iloc[-1]
    trade_value = quantity * current_price
    max_allowed = capital * MAX_POSITION_PCT

    if trade_value > max_allowed:
        logger.warning(
            f"[{symbol}] REJECTED: Trade value ${trade_value:,.2f} exceeds "
            f"max allowed ${max_allowed:,.2f} ({MAX_POSITION_PCT * 100}% of capital)."
        )
        return False

    # --- 3. Market cap check ---
    market_cap = info.get("marketCap", 0)
    if market_cap < MIN_MARKET_CAP:
        logger.warning(
            f"[{symbol}] REJECTED: Market cap ${market_cap / 1e9:.2f}B < "
            f"minimum ${MIN_MARKET_CAP / 1e9:.0f}B."
        )
        return False

    # --- 4. Average daily volume check (in USD) ---
    avg_volume = hist["Volume"].mean()
    avg_price = hist["Close"].mean()
    avg_volume_usd = avg_volume * avg_price

    if avg_volume_usd < MIN_DAILY_VOLUME_USD:
        logger.warning(
            f"[{symbol}] REJECTED: Avg daily volume ${avg_volume_usd / 1e6:.1f}M < "
            f"minimum ${MIN_DAILY_VOLUME_USD / 1e6:.0f}M."
        )
        return False

    # --- 5. Earnings announcement check ---
    if _has_upcoming_earnings(ticker):
        logger.warning(f"[{symbol}] REJECTED: Earnings announcement tomorrow — gap risk.")
        return False

    logger.info(f"[{symbol}] APPROVED: All risk checks passed.")
    return True


def _has_upcoming_earnings(ticker: yf.Ticker) -> bool:
    """
    Check if the stock has an earnings announcement tomorrow.
    Returns True if earnings are upcoming (risky), False if clear.
    """
    try:
        calendar = ticker.calendar
        if calendar is None or calendar.empty:
            return False

        # calendar columns are dates, rows are metrics
        # Earnings Date is typically in columns
        if "Earnings Date" in calendar.index:
            earnings_dates = calendar.loc["Earnings Date"]
        elif hasattr(calendar, "columns"):
            # Some versions return it differently
            earnings_dates = pd.Series(calendar.columns)
        else:
            return False

        tomorrow = date.today() + timedelta(days=1)
        for ed in pd.to_datetime(earnings_dates, errors="coerce"):
            if pd.notna(ed) and ed.date() == tomorrow:
                return True

    except Exception as e:
        logger.debug(f"Could not fetch earnings calendar: {e}")

    return False


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