"""
Backtest: Full Live-Strategy Simulation
========================================
True-to-implementation backtest — mirrors the complete live stack exactly.

  signal_generator.py:
    - S&P 500 universe
    - 20-day ADV ≥ $10M (liquidity filter)
    - 126-day cumulative overnight momentum: r_ov = Open_t / Close_{t-1} - 1
    - MA50 regime: SPY > MA50 → LONG TOP-10 | SPY ≤ MA50 → SHORT BOTTOM-10

  Risk_Management.py + Validations.py (TradeValidator):
    - Position size: 10% of capital per stock (MAX_POSITION_PCT)
      → uninvested capital stays in cash (partial deployment drag)
    - Market cap ≥ $2B  (TradeValidator.validate_trade)
    - Earnings filter: skip overnight hold if stock has earnings next trading day
      (_has_upcoming_earnings logic from Risk_Management / paper §4.3.3)
    - Circuit breaker: daily portfolio loss > 2% → skip next overnight session
      (check_daily_loss_limit, MAX_DAILY_LOSS_PCT)

  main.py:
    - Entry at NYSE close auction T (overnight hold)
    - Exit at NYSE open auction T+1
    - Transaction cost applied per deployed leg (5bps per side, configurable)

Benchmark: Baseline long-only TOP-10 (no risk management, 0% TC).
Evaluation period: March 2024 – March 2026.
"""

import datetime
import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
START_DATE          = "2023-01-01"          # Warmup start (before eval period)
END_DATE            = "2026-03-06"
EVAL_START          = datetime.datetime(2024, 3, 1)

# Signal — mirrors signal_generator.py exactly
MOMENTUM_WINDOW     = 126                   # Lou et al. (2019)
MA50_WINDOW         = 50
LONG_N              = 10                    # TOP-N in bull regime
SHORT_N             = 10                    # BOTTOM-N in bear regime
MIN_DOLLAR_VOL      = 10_000_000            # $10M 20-day ADV

# Risk — mirrors Risk_Management.py + Validations.TradeValidator exactly
MAX_POSITION_PCT    = 0.10                  # 10% of capital per stock
MAX_DAILY_LOSS      = 0.02                  # 2% → 24h circuit breaker
MIN_MARKET_CAP      = 2_000_000_000         # $2B  (TradeValidator)

# Execution — mirrors main.py
TRANSACTION_COST    = 0.0005                # 5bps per side (entry + exit)

# Backtest setup
CAPITAL_INIT        = 10_000.0
RISK_FREE_RATE      = 0.0
USE_EARNINGS_FILTER = True                  # mirrors _has_upcoming_earnings
USE_MARKET_CAP_FILTER = True                # mirrors TradeValidator market cap check
QUICK_MODE          = False                 # True = 80 tickers (dev/test only)
EARNINGS_CACHE      = Path("earnings_cache.json")
MCAP_CACHE          = Path("mcap_cache.json")
# ───────────────────────────────────────────────────────────────────────────────


# ─── DATA FETCHING ─────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    for t in tables:
        if "Symbol" in t.columns:
            return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    return []


def fetch_spy_ma50(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY with MA50 — mirrors signal_generator.fetch_spy_regime()."""
    print("  Downloading SPY (MA50 regime filter)...")
    spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy.index.name = "date"
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    spy["MA50"] = spy["Close"].rolling(MA50_WINDOW).mean()
    above = (spy["Close"] > spy["MA50"]).sum()
    below = (spy["Close"] <= spy["MA50"]).sum()
    print(f"  SPY regime: {above}d above MA50 | {below}d below MA50")
    return spy[["Close", "MA50"]]


def fetch_market_caps(tickers: list) -> dict:
    """
    Download current market cap per ticker (static proxy).
    Mirrors: Validations.TradeValidator — market cap >= $2B check.
    Note: S&P 500 inclusion requires ~$15B+ so this filter almost never fires,
    but is included for live-code fidelity and to handle delisted/replaced stocks.
    """
    if MCAP_CACHE.exists():
        print("  Loading market caps from cache...")
        with open(MCAP_CACHE) as f:
            return {k: int(v or 0) for k, v in json.load(f).items()}

    if not USE_MARKET_CAP_FILTER:
        return {t: int(MIN_MARKET_CAP) + 1 for t in tickers}

    caps = {}
    print(f"  Downloading market caps ({len(tickers)} tickers)...")
    for i, t in enumerate(tickers):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(tickers)}", end="\r")
        try:
            caps[t] = yf.Ticker(t).info.get("marketCap") or 0
        except Exception:
            caps[t] = 0

    with open(MCAP_CACHE, "w") as f:
        json.dump(caps, f)
    n_pass = sum(1 for v in caps.values() if v >= MIN_MARKET_CAP)
    print(f"\n  Market cap filter: {n_pass}/{len(caps)} pass ≥$2B")
    return caps


def fetch_earnings_calendar(tickers: list) -> dict:
    """
    Pre-download earnings announcement dates for all tickers.
    Returns: ticker → set of Timestamps (announcement dates).

    Mirrors: Risk_Management._has_upcoming_earnings (paper §4.3.3).
    Logic: if earnings_date matches all_dates[i+1] (next trading day),
    skip the overnight hold on day i — avoid earnings gap risk.
    """
    if EARNINGS_CACHE.exists():
        print("  Loading earnings calendar from cache...")
        with open(EARNINGS_CACHE) as f:
            raw = json.load(f)
        return {t: set(pd.to_datetime(dates).normalize()) for t, dates in raw.items()}

    print(f"  Downloading earnings dates ({len(tickers)} tickers, ~5 min)...")
    cache = {}
    for i, t in enumerate(tickers):
        if i % 50 == 0:
            print(f"    {i}/{len(tickers)}", end="\r")
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is not None and not ed.empty:
                dates = [str(d.date()) for d in pd.to_datetime(ed.index).normalize()]
                cache[t] = sorted(set(dates))
            else:
                cache[t] = []
        except Exception:
            cache[t] = []

    with open(EARNINGS_CACHE, "w") as f:
        json.dump(cache, f)
    print(f"\n  Earnings cache saved → {EARNINGS_CACHE}")
    return {t: set(pd.to_datetime(dates).normalize()) for t, dates in cache.items()}


# ─── DATA PREPARATION ──────────────────────────────────────────────────────────

def build_clean_data(raw_data: pd.DataFrame, tickers: list) -> dict:
    """
    Clean OHLCV and pre-compute momentum.
    Mirrors signal_generator.download_and_clean() + compute_momentum_signals():
      - Drop zero/null open, close, volume rows
      - overnight return: r_ov = Open_t / Close_{t-1} - 1  (Lou et al. 2019)
      - 126-day cumulative momentum via rolling product
      - Forward overnight (realized return — backtest only, not look-ahead in live)
    """
    clean = {}
    for t in tickers:
        try:
            df = raw_data[t].copy()
            df = df.dropna(subset=["Open", "Close"])
            df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["Volume"] > 0)]
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")].sort_index()

            # Overnight return (Lou et al. 2019 formula)
            df["Past_Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1

            # 126-day cumulative momentum signal — mirrors compute_momentum_signals()
            df["Momentum"] = (
                (1 + df["Past_Overnight"])
                .rolling(MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW)
                .apply(np.prod, raw=True) - 1
            )

            # Forward overnight return (backtest only — realized P&L)
            df["Forward_Overnight"] = (df["Open"].shift(-1) / df["Close"]) - 1
            df["DollarVol"]         = df["Close"] * df["Volume"]

            if len(df) > MOMENTUM_WINDOW + 5:
                clean[t] = df
        except Exception:
            continue

    print(f"  Built clean data: {len(clean)}/{len(tickers)} tickers")
    return clean


# ─── STRATEGY RUNNERS ──────────────────────────────────────────────────────────

def run_baseline(clean_data: dict) -> dict:
    """
    Baseline: long-only TOP-10, equal weight (100% deployed).
    No risk management, no transaction costs.
    Equivalent to the paper's Section 4.2 baseline.
    """
    all_dates = sorted({d for df in clean_data.values() for d in df.index
                        if isinstance(d, pd.Timestamp)})

    capital = CAPITAL_INIT
    equity, daily_ret, dates_out = [], [], []

    for i in range(MOMENTUM_WINDOW + 5, len(all_dates) - 1):
        date = all_dates[i]
        candidates = []

        for t, df in clean_data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Momentum"]) or pd.isna(row["Forward_Overnight"]):
                continue
            if df.loc[:date, "DollarVol"].tail(20).mean() < MIN_DOLLAR_VOL:
                continue
            candidates.append({"momentum": row["Momentum"], "return": row["Forward_Overnight"]})

        candidates.sort(key=lambda x: x["momentum"], reverse=True)
        top = candidates[:LONG_N]

        if not top:
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            continue

        ret = np.mean([x["return"] for x in top])  # Equal weight, 100% deployed
        capital *= (1 + ret)
        equity.append(capital); daily_ret.append(ret); dates_out.append(date)

    return {"dates": [d.to_pydatetime() for d in dates_out],
            "equity": equity, "rets": daily_ret}


def run_full_strategy(
    clean_data:   dict,
    spy_df:       pd.DataFrame,
    market_caps:  dict,
    earnings_cal: dict,
) -> dict:
    """
    Full live-strategy replication.

    Adds to the baseline:
      1. MA50 regime direction (LONG TOP-10 or SHORT BOTTOM-10)
      2. Position sizing: MAX_POSITION_PCT (10%) per stock
         → if N < 10 stocks pass filters, (1 - N*10%) stays in cash
      3. Market cap filter ≥ $2B   (Validations.TradeValidator)
      4. Earnings filter            (Risk_Management._has_upcoming_earnings)
      5. Circuit breaker: loss > 2% → skip next session (check_daily_loss_limit)
      6. Transaction costs: 5bps per side on deployed capital
    """
    all_dates = sorted({d for df in clean_data.values() for d in df.index
                        if isinstance(d, pd.Timestamp)})

    capital = CAPITAL_INIT
    equity, daily_ret, dates_out = [], [], []
    stat = {"long": 0, "short": 0, "halted": 0, "cash": 0}
    circuit_breaker = False         # mirrors check_daily_loss_limit halt flag

    for i in range(MOMENTUM_WINDOW + 5, len(all_dates) - 1):
        date    = all_dates[i]
        date_t1 = pd.Timestamp(all_dates[i + 1])   # Next trading day

        # ── Circuit breaker (Risk_Management.check_daily_loss_limit) ───────
        # Paper §3.3: "suspending all trading activities for 24 hours"
        # In overnight strategy, 24h = 1 overnight session
        if circuit_breaker:
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            stat["halted"] += 1
            circuit_breaker = False
            continue

        # ── MA50 regime (signal_generator.fetch_spy_regime) ─────────────
        spy_past = spy_df[spy_df.index <= date]
        if spy_past.empty or pd.isna(spy_past["MA50"].iloc[-1]):
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            stat["cash"] += 1
            continue

        spy_row    = spy_past.iloc[-1]
        above_ma50 = bool(spy_row["Close"] > spy_row["MA50"])

        # ── Candidate pool — all Validations/Risk filters applied ──────────
        candidates = []
        for t, df in clean_data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Momentum"]) or pd.isna(row["Forward_Overnight"]):
                continue

            # ADV filter — mirrors signal_generator.apply_liquidity_filter
            if df.loc[:date, "DollarVol"].tail(20).mean() < MIN_DOLLAR_VOL:
                continue

            # Market cap filter — mirrors Validations.TradeValidator
            if USE_MARKET_CAP_FILTER and market_caps.get(t, 0) < MIN_MARKET_CAP:
                continue

            # Earnings filter — mirrors Risk_Management._has_upcoming_earnings
            # Paper §3.3: "rejects any stock scheduled to announce earnings on
            # the following trading day"
            if USE_EARNINGS_FILTER:
                if date_t1 in earnings_cal.get(t, set()):
                    continue    # Skip: earnings announced tomorrow — avoid gap risk

            candidates.append({
                "ticker":   t,
                "momentum": row["Momentum"],
                "return":   row["Forward_Overnight"],
            })

        if len(candidates) < max(LONG_N, SHORT_N):
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            stat["cash"] += 1
            continue

        candidates.sort(key=lambda x: x["momentum"], reverse=True)

        # ── Regime direction ───────────────────────────────────────────────
        if above_ma50:
            selected  = candidates[:LONG_N]
            direction = +1          # LONG: overnight premium in bull regime
            stat["long"] += 1
        else:
            selected  = candidates[-SHORT_N:]
            direction = -1          # SHORT: overnight discount in bear regime
            stat["short"] += 1

        # ── Position sizing — mirrors Risk_Management.calculate_position_size
        # Each stock allocated MAX_POSITION_PCT (10%) of capital.
        # If < 10 stocks survive filters: partial deployment, rest in cash.
        n        = len(selected)
        deployed = min(n * MAX_POSITION_PCT, 1.0)   # cap: cannot deploy > 100%

        # Gross portfolio return = sum of weighted returns
        gross = MAX_POSITION_PCT * direction * sum(s["return"] for s in selected)

        # Transaction cost on deployed capital only (entry + exit)
        # mirrors main.py execute_order calls at open/close auction
        tc = 2 * TRANSACTION_COST * deployed

        net_ret = gross - tc

        # ── Circuit breaker trigger — mirrors check_daily_loss_limit ───────
        # Paper §3.3: "If cumulative daily loss exceeds 2% of total capital,
        # the algorithm triggers a hard halt"
        if net_ret < -MAX_DAILY_LOSS:
            circuit_breaker = True
            print(f"  [{date.date()}] CIRCUIT BREAKER triggered "
                  f"({net_ret*100:+.2f}%) — halting next session.")

        capital *= (1 + net_ret)
        equity.append(capital); daily_ret.append(net_ret); dates_out.append(date)

    total = sum(stat.values())
    if total > 0:
        print(f"  Regime breakdown — "
              f"long: {stat['long']}d ({stat['long']/total*100:.0f}%) | "
              f"short: {stat['short']}d ({stat['short']/total*100:.0f}%) | "
              f"halted: {stat['halted']}d | cash/warmup: {stat['cash']}d")

    return {"dates": [d.to_pydatetime() for d in dates_out],
            "equity": equity, "rets": daily_ret}


# ─── METRICS ───────────────────────────────────────────────────────────────────

def clip_to_eval(result: dict) -> dict:
    """Clip result to EVAL_START, rebasing equity to CAPITAL_INIT."""
    idx = [i for i, d in enumerate(result["dates"]) if d >= EVAL_START]
    if not idx:
        return result
    i0   = idx[0]
    rets = result["rets"][i0:]
    eq   = [CAPITAL_INIT]
    for r in rets[1:]:
        eq.append(eq[-1] * (1 + r))
    return {"dates": result["dates"][i0:], "equity": eq, "rets": rets}


def compute_metrics(equity: list, daily_rets: list, label: str,
                    dates: list = None) -> dict:
    total_ret = (equity[-1] / equity[0]) - 1
    n_days    = len(daily_rets)
    years     = ((dates[-1] - dates[0]).days / 365.25
                 if dates and len(dates) >= 2 else n_days / 252)
    ann_ret   = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    vol       = np.std(daily_rets) * np.sqrt(252) if daily_rets else 0
    sharpe    = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    neg       = [r for r in daily_rets if r < 0]
    down_vol  = np.std(neg) * np.sqrt(252) if neg else 0
    sortino   = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0
    peak = equity[0]; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd
    win_rate = sum(1 for r in daily_rets if r > 0) / n_days if n_days > 0 else 0
    return {"label": label, "total_ret": total_ret, "ann_ret": ann_ret,
            "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "win_rate": win_rate, "final_cap": equity[-1]}


def print_metrics(m: dict):
    print(f"\n{'─'*58}\n  {m['label']}\n{'─'*58}")
    print(f"  Final capital:       ${m['final_cap']:>10,.2f}")
    print(f"  Total return:        {m['total_ret']*100:>+9.2f}%")
    print(f"  Annualised return:   {m['ann_ret']*100:>+9.2f}%")
    print(f"  Volatility (ann):    {m['vol']*100:>9.2f}%")
    print(f"  Sharpe Ratio:        {m['sharpe']:>9.2f}")
    print(f"  Sortino Ratio:       {m['sortino']:>9.2f}")
    print(f"  Max Drawdown:        {m['max_dd']*100:>9.2f}%")
    print(f"  Win Rate:            {m['win_rate']*100:>9.1f}%")


# ─── PLOTTING ──────────────────────────────────────────────────────────────────

COLOR_BASELINE = "#2CA02C"
COLOR_FULL     = "#1F77B4"


def _rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    })


def _clip_plot(res: dict, x0, x1):
    pts = [(d, e) for d, e in zip(res["dates"], res["equity"]) if x0 <= d <= x1]
    if not pts:
        return [], []
    ds, es = zip(*pts)
    return list(ds), list(es)


def plot_equity(baseline: dict, full: dict):
    _rcparams()
    start_date = baseline["dates"][0]
    end_date   = baseline["dates"][-1]
    start_6m   = end_date - datetime.timedelta(days=182)

    def _panel(ax, x0, x1, title, show_ylabel, show_legend):
        ds_b, es_b = _clip_plot(baseline, x0, x1)
        ds_f, es_f = _clip_plot(full,     x0, x1)
        if ds_b:
            ax.plot(ds_b, es_b, label="Baseline (Long-Only TOP-10)",
                    color=COLOR_BASELINE, linewidth=1.8)
        if ds_f:
            ax.plot(ds_f, es_f,
                    label="Full Strategy (MA50 L/S + 10% sizing + earnings + CB)",
                    color=COLOR_FULL, linewidth=1.8, linestyle="--")
        ax.axhline(CAPITAL_INIT, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=6)
        if show_ylabel:
            ax.set_ylabel("Portfolio Value (USD)", labelpad=6)
        if show_legend:
            ax.legend(loc="upper left", handlelength=3.0)
        ax.grid(True, axis="y")
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)

    fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.12)

    _panel(ax1a, start_date, end_date, "Panel A — Full Period (Mar 2024 – Mar 2026)",
           show_ylabel=True, show_legend=True)
    ax1a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    all_vals = [v for v in baseline["equity"] + full["equity"] if not np.isnan(v)]
    ax1a.set_ylim(min(all_vals) * 0.97, max(all_vals) * 1.03)

    _panel(ax1b, start_6m, end_date, "Panel A — Last 6 Months (Zoom)",
           show_ylabel=False, show_legend=False)
    ax1b.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    zoom_vals = ([e for d, e in zip(baseline["dates"], baseline["equity"]) if d >= start_6m] +
                 [e for d, e in zip(full["dates"],     full["equity"])     if d >= start_6m])
    zoom_vals = [v for v in zoom_vals if not np.isnan(v)]
    ax1b.set_ylim(min(zoom_vals) * 0.98, max(zoom_vals) * 1.02)
    ax1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Overnight Effect — Baseline vs. Full Live-Strategy Simulation",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.93,
        f"MA50 regime | 10% sizing/stock (cash drag) | earnings filter | "
        f"2% circuit breaker | {TRANSACTION_COST * 1e4:.0f}bps TC per side",
        ha="center", fontsize=9, color="#555555", style="italic",
    )

    plt.savefig("bt_fig1_equity.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: bt_fig1_equity.png")
    plt.close()


def plot_drawdown(baseline: dict, full: dict):
    _rcparams()
    start_date = baseline["dates"][0]
    end_date   = baseline["dates"][-1]

    def calc_dd(equity):
        peak = equity[0]; out = []
        for v in equity:
            if v > peak: peak = v
            out.append((v - peak) / peak * 100)
        return out

    dd_b = calc_dd(baseline["equity"])
    dd_f = calc_dd(full["equity"])

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.14)

    ax.fill_between(baseline["dates"], dd_b, 0, alpha=0.20, color=COLOR_BASELINE, zorder=1)
    ax.fill_between(full["dates"],     dd_f, 0, alpha=0.20, color=COLOR_FULL,     zorder=0)
    ax.plot(baseline["dates"], dd_b, color=COLOR_BASELINE, linewidth=1.6,
            label="Baseline (Long-Only TOP-10)")
    ax.plot(full["dates"],     dd_f, color=COLOR_FULL,     linewidth=1.6,
            linestyle="--", label="Full Strategy (MA50 L/S + Risk Management)")

    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
    ax.axhline(-MAX_DAILY_LOSS * 100, color="#DD4444", linewidth=0.8,
               linestyle=":", alpha=0.6, label=f"Circuit breaker level ({MAX_DAILY_LOSS*100:.0f}%/day)")
    ax.set_ylabel("Drawdown (%)", labelpad=6)
    ax.set_title("Panel B — Drawdown from Peak (Mar 2024 – Mar 2026)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y")
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)

    plt.savefig("bt_fig2_drawdown.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: bt_fig2_drawdown.png")
    plt.close()


def plot_table(m_b: dict, m_f: dict):
    _rcparams()
    col_headers = ["Metric", "Baseline (Long-Only)", "Full Strategy (L/S + Risk)"]
    rows = [
        ["Final Value",   f"${m_b['final_cap']:,.0f}",      f"${m_f['final_cap']:,.0f}"],
        ["Total Return",  f"{m_b['total_ret']*100:+.2f}%",  f"{m_f['total_ret']*100:+.2f}%"],
        ["Ann. Return",   f"{m_b['ann_ret']*100:+.2f}%",    f"{m_f['ann_ret']*100:+.2f}%"],
        ["Volatility",    f"{m_b['vol']*100:.2f}%",         f"{m_f['vol']*100:.2f}%"],
        ["Sharpe Ratio",  f"{m_b['sharpe']:.2f}",           f"{m_f['sharpe']:.2f}"],
        ["Sortino Ratio", f"{m_b['sortino']:.2f}",          f"{m_f['sortino']:.2f}"],
        ["Max Drawdown",  f"{m_b['max_dd']*100:.2f}%",      f"{m_f['max_dd']*100:.2f}%"],
        ["Win Rate",      f"{m_b['win_rate']*100:.1f}%",    f"{m_f['win_rate']*100:.1f}%"],
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   cellLoc="center", loc="center", bbox=[0.02, 0.02, 0.96, 0.90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    col_widths = [0.32, 0.32, 0.34]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#B00000")
            cell.set_text_props(color="white", fontweight="bold",
                                ha="left" if c == 0 else "center", fontsize=9.5)
            cell.PAD = 0.06
        else:
            cell.set_facecolor("#F7F9FC" if r % 2 == 0 else "#FFFFFF")
            cell.set_text_props(ha="left" if c == 0 else "center", fontsize=10)
            if c == 0:
                cell.PAD = 0.06

    ax.set_title("Panel C — Performance Summary",
                 loc="left", fontsize=10, fontweight="bold", pad=8)

    plt.savefig("bt_fig3_table.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: bt_fig3_table.png")
    plt.close()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  BACKTEST: Full Live-Strategy Simulation")
    print(f"  Risk: {MAX_POSITION_PCT*100:.0f}%/stock | CB: {MAX_DAILY_LOSS*100:.0f}% | "
          f"TC: {TRANSACTION_COST*1e4:.0f}bps | Earnings: {USE_EARNINGS_FILTER}")
    print("=" * 62)

    # [1] Universe
    print("\n[1/5] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
    print(f"  {len(tickers)} tickers loaded.")

    # [2] OHLCV + SPY
    print("\n[2/5] Downloading OHLCV + SPY data...")
    raw_data = yf.download(
        tickers, start=START_DATE, end=END_DATE,
        group_by="ticker", auto_adjust=True, progress=True,
    )
    spy_df = fetch_spy_ma50(START_DATE, END_DATE)

    # [3] Risk management pre-computation
    print("\n[3/5] Pre-loading risk management data...")
    market_caps  = fetch_market_caps(tickers)
    earnings_cal = fetch_earnings_calendar(tickers) if USE_EARNINGS_FILTER else {}

    # [4] Clean data
    print("\n[4/5] Building clean data...")
    clean_data = build_clean_data(raw_data, tickers)
    del raw_data    # free memory

    # [5] Backtests
    print("\n[5/5] Running backtests...")
    print("  → Baseline (TOP-10 long only, no risk management)...")
    baseline = clip_to_eval(run_baseline(clean_data))

    print("  → Full Strategy (MA50 L/S + all risk management)...")
    full = clip_to_eval(run_full_strategy(clean_data, spy_df, market_caps, earnings_cal))

    # Metrics
    m_b = compute_metrics(baseline["equity"], baseline["rets"],
                          "Baseline (Long-Only TOP-10)", baseline["dates"])
    m_f = compute_metrics(full["equity"], full["rets"],
                          "Full Strategy (MA50 L/S + Risk Management)", full["dates"])

    print_metrics(m_b)
    print_metrics(m_f)

    print(f"\n{'=' * 62}")
    print(f"  Sharpe improvement:    {m_f['sharpe']   - m_b['sharpe']:+.2f}")
    print(f"  Ann. return delta:     {(m_f['ann_ret']  - m_b['ann_ret'])*100:+.2f}%")
    print(f"  Max DD improvement:    {(m_f['max_dd']   - m_b['max_dd'])*100:+.2f}%")
    print(f"  Volatility reduction:  {(m_f['vol']      - m_b['vol'])*100:+.2f}%")
    print(f"{'=' * 62}\n")

    # Plots
    plot_equity(baseline, full)
    plot_drawdown(baseline, full)
    plot_table(m_b, m_f)