"""
Backtest: Day & Night Strategy — Baseline vs SPY Buy & Hold
Overnight momentum strategy (TOP10 / 126d) compared against a passive
SPY buy-and-hold benchmark over the same period.

PERIOD = "3y" downloads 3 years of data.
Momentum warm-up consumes ~131 trading days.
Effective evaluation starts March 2024 → March 2026 (~2 years).
"""

import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_DATE       = "2023-01-01"  # enough history for 126-day warm-up before Mar 2024
END_DATE         = "2026-03-06"  # today
TOP_N            = 10
MOMENTUM_WINDOW  = 126
RISK_FREE_RATE   = 0.0
CAPITAL_INIT     = 10_000.0
MIN_DOLLAR_VOL   = 10_000_000
TRANSACTION_COST = 0.0
QUICK_MODE       = False
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    for t in tables:
        if "Symbol" in t.columns:
            return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    return []


def fetch_spy(start: str = "2021-01-01", end: str = "2026-03-06") -> pd.DataFrame:
    print("  Downloading SPY...")
    spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy.index.name = "date"
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    return spy


def build_clean_data(raw_data: pd.DataFrame, tickers: list) -> dict:
    clean = {}
    for t in tickers:
        try:
            df = raw_data[t].copy()
            df = df.dropna(subset=["Open", "Close"])
            df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["Volume"] > 0)]
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")].sort_index()

            df["Past_Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1
            df["Momentum"] = (
                (1 + df["Past_Overnight"])
                .rolling(MOMENTUM_WINDOW)
                .apply(np.prod, raw=True) - 1
            )
            df["Forward_Overnight"] = (df["Open"].shift(-1) / df["Close"]) - 1
            df["DollarVol"] = df["Close"] * df["Volume"]

            if len(df) > MOMENTUM_WINDOW + 5:
                clean[t] = df
        except Exception:
            continue
    return clean


def run_baseline(clean_data: dict) -> dict:
    """Long TOP_N overnight momentum stocks, equal weight, no filters."""
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

    capital   = CAPITAL_INIT
    equity    = []
    daily_ret = []
    dates_out = []

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
            candidates.append({
                "momentum": row["Momentum"],
                "return":   row["Forward_Overnight"],
            })

        candidates.sort(key=lambda x: x["momentum"], reverse=True)
        top = candidates[:TOP_N]

        if not top:
            equity.append(capital)
            daily_ret.append(0.0)
            dates_out.append(date)
            continue

        ret = np.mean([x["return"] for x in top]) - 2 * TRANSACTION_COST
        capital *= (1 + ret)
        equity.append(capital)
        daily_ret.append(ret)
        dates_out.append(date)

    return {
        "dates":  [d.to_pydatetime() for d in dates_out],
        "equity": equity,
        "rets":   daily_ret,
    }


def build_spy_bh(spy_df: pd.DataFrame, strategy_dates: list) -> dict:
    """SPY buy-and-hold aligned to the same dates as the strategy."""
    start = strategy_dates[0]
    end   = strategy_dates[-1]

    spy_window = spy_df[(spy_df.index >= pd.Timestamp(start)) &
                        (spy_df.index <= pd.Timestamp(end))].copy()

    if spy_window.empty:
        return {"dates": strategy_dates, "equity": [CAPITAL_INIT] * len(strategy_dates)}

    spy_window["equity"] = CAPITAL_INIT * (spy_window["Close"] / spy_window["Close"].iloc[0])

    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in strategy_dates])
    equity_aligned = spy_window["equity"].reindex(idx, method="ffill").fillna(method="bfill")

    return {
        "dates":  strategy_dates,
        "equity": equity_aligned.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity: list, daily_rets: list, label: str,
                    dates: list = None) -> dict:
    total_ret = (equity[-1] / equity[0]) - 1
    n_days    = len(daily_rets)
    # Use actual calendar days if dates provided — avoids warm-up distortion
    if dates and len(dates) >= 2:
        years = (dates[-1] - dates[0]).days / 365.25
    else:
        years = n_days / 252
    ann_ret   = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    vol       = np.std(daily_rets) * np.sqrt(252) if daily_rets else 0
    sharpe    = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0

    neg      = [r for r in daily_rets if r < 0]
    down_vol = np.std(neg) * np.sqrt(252) if neg else 0
    sortino  = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

    peak   = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd

    win_rate = sum(1 for r in daily_rets if r > 0) / n_days if n_days > 0 else 0

    return {
        "label": label, "total_ret": total_ret, "ann_ret": ann_ret,
        "vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "win_rate": win_rate, "final_cap": equity[-1],
    }


def print_metrics(m: dict):
    print(f"\n{'─'*56}\n  {m['label']}\n{'─'*56}")
    print(f"  Final capital:       ${m['final_cap']:>10,.2f}")
    print(f"  Total return:        {m['total_ret']*100:>+9.2f}%")
    print(f"  Annualised return:   {m['ann_ret']*100:>+9.2f}%")
    print(f"  Volatility (ann):    {m['vol']*100:>9.2f}%")
    print(f"  Sharpe Ratio:        {m['sharpe']:>9.2f}")
    print(f"  Sortino Ratio:       {m['sortino']:>9.2f}")
    print(f"  Max Drawdown:        {m['max_dd']*100:>9.2f}%")
    print(f"  Win Rate:            {m['win_rate']*100:>9.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _rcparams():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#CCCCCC",
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.6,
    })


def plot_equity(baseline: dict, spy_bh: dict):
    """Figure 1 — dual equity curves (full period + 6m zoom)."""
    _rcparams()
    COLOR_STRAT = "#2CA02C"
    COLOR_SPY   = "#222222"

    end_date   = baseline["dates"][-1]
    start_date = baseline["dates"][0]
    start_6m   = end_date - datetime.timedelta(days=182)

    def clip(series_dict, x_start, x_end):
        pts = [(d, e) for d, e in zip(series_dict["dates"], series_dict["equity"])
               if x_start <= d <= x_end]
        if not pts: return [], []
        ds, es = zip(*pts)
        return list(ds), list(es)

    def _plot_window(ax, x_start, x_end, title_str, show_ylabel, show_legend):
        ds_s, es_s = clip(baseline, x_start, x_end)
        ds_b, es_b = clip(spy_bh,   x_start, x_end)
        if ds_s:
            ax.plot(ds_s, es_s, label="Overnight Strategy (Baseline)",
                    color=COLOR_STRAT, linewidth=1.8, linestyle="-")
        if ds_b:
            ax.plot(ds_b, es_b, label="SPY Buy & Hold",
                    color=COLOR_SPY, linewidth=1.2, linestyle="--")
        ax.axhline(CAPITAL_INIT, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_title(title_str, loc="left", fontsize=10, fontweight="bold", pad=6)
        if show_ylabel:
            ax.set_ylabel("Portfolio Value (USD)", labelpad=6)
        if show_legend:
            ax.legend(loc="upper left", handlelength=3.0)
        ax.grid(True, axis="y")
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)

    fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)

    _plot_window(ax1a, start_date, end_date,
                 "Panel A — Full Period (Mar 2024 – Mar 2026)",
                 show_ylabel=True, show_legend=True)
    ax1a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    full_vals = [e for e in baseline["equity"]] + [e for e in spy_bh["equity"]]
    full_vals = [v for v in full_vals if v is not None and not np.isnan(v)]
    ax1a.set_ylim(min(full_vals) * 0.97, max(full_vals) * 1.03)

    _plot_window(ax1b, start_6m, end_date,
                 "Panel A — Last 6 Months (Zoom)",
                 show_ylabel=False, show_legend=False)
    ax1b.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    zoom_vals  = [e for d, e in zip(baseline["dates"], baseline["equity"]) if d >= start_6m]
    zoom_vals += [e for d, e in zip(spy_bh["dates"],   spy_bh["equity"])   if d >= start_6m]
    zoom_vals  = [v for v in zoom_vals if v is not None and not np.isnan(v)]
    ax1b.set_ylim(min(zoom_vals) * 0.98, max(zoom_vals) * 1.02)
    ax1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Overnight Effect — Baseline Strategy vs. SPY Buy & Hold",
                 fontsize=13, fontweight="bold")

    plt.savefig("baseline_fig1_equity.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: baseline_fig1_equity.png")
    plt.close()


def plot_drawdown(baseline: dict, spy_bh: dict):
    """Figure 2 — drawdown from peak."""
    _rcparams()
    COLOR_STRAT_DD = "#D62728"
    COLOR_SPY_DD   = "#555555"

    start_date = baseline["dates"][0]
    end_date   = baseline["dates"][-1]

    def calc_dd(equity):
        peak, out = equity[0], []
        for v in equity:
            if v > peak: peak = v
            out.append((v - peak) / peak * 100)
        return out

    dd_s = calc_dd(baseline["equity"])
    dd_b = calc_dd(spy_bh["equity"])

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.14)

    ax.fill_between(baseline["dates"], dd_s, 0,
                    alpha=0.25, color=COLOR_STRAT_DD, zorder=1)
    ax.fill_between(spy_bh["dates"], dd_b, 0,
                    alpha=0.18, color=COLOR_SPY_DD, zorder=0)
    ax.plot(baseline["dates"], dd_s,
            color=COLOR_STRAT_DD, linewidth=1.6, label="Overnight Strategy (Baseline)")
    ax.plot(spy_bh["dates"], dd_b,
            color=COLOR_SPY_DD, linewidth=1.2, linestyle="--", label="SPY Buy & Hold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_ylabel("Drawdown (%)", labelpad=6)
    ax.set_title("Panel B — Drawdown from Peak (Mar 2024 – Mar 2026)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y")
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)

    plt.savefig("baseline_fig2_drawdown.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: baseline_fig2_drawdown.png")
    plt.close()


def plot_table(m_baseline: dict, m_spy: dict):
    """Figure 3 — performance summary table."""
    _rcparams()

    col_headers = ["Metric", "Overnight Strategy", "SPY Buy & Hold"]
    rows = [
        ["Final Value",   f"${m_baseline['final_cap']:,.0f}",     f"${m_spy['final_cap']:,.0f}"],
        ["Total Return",  f"{m_baseline['total_ret']*100:+.2f}%", f"{m_spy['total_ret']*100:+.2f}%"],
        ["Ann. Return",   f"{m_baseline['ann_ret']*100:+.2f}%",   f"{m_spy['ann_ret']*100:+.2f}%"],
        ["Volatility",    f"{m_baseline['vol']*100:.2f}%",        f"{m_spy['vol']*100:.2f}%"],
        ["Sharpe Ratio",  f"{m_baseline['sharpe']:.2f}",          f"{m_spy['sharpe']:.2f}"],
        ["Sortino Ratio", f"{m_baseline['sortino']:.2f}",         f"{m_spy['sortino']:.2f}"],
        ["Max Drawdown",  f"{m_baseline['max_dd']*100:.2f}%",     f"{m_spy['max_dd']*100:.2f}%"],
        ["Win Rate",      f"{m_baseline['win_rate']*100:.1f}%",   "N/A"],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    col_widths = [0.32, 0.32, 0.32]
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

    plt.savefig("baseline_fig3_table.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: baseline_fig3_table.png")
    plt.close()


def plot_comparison(baseline: dict, spy_bh: dict,
                    m_baseline: dict, m_spy: dict):
    """Saves all 3 figures separately."""
    plot_equity(baseline, spy_bh)
    plot_drawdown(baseline, spy_bh)
    plot_table(m_baseline, m_spy)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  BACKTEST: Overnight Baseline vs SPY Buy & Hold")
    print("="*62)

    print("\n[1/4] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
    print(f"  {len(tickers)} tickers loaded.")

    print("\n[2/4] Downloading OHLCV + SPY data...")
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by="ticker",
                           auto_adjust=True, progress=True, threads=True)
    spy_df = fetch_spy(start="2021-01-01", end=END_DATE)

    print("\n[3/4] Building clean data...")
    clean_data = build_clean_data(raw_data, tickers)
    print(f"  {len(clean_data)} tickers with valid data.")

    print("\n[4/4] Running backtest...")
    baseline = run_baseline(clean_data)
    spy_bh   = build_spy_bh(spy_df, baseline["dates"])

    # ── Clip to Mar 2024 — clean 2-year evaluation window ────────────────────
    EVAL_START = datetime.datetime(2024, 3, 1)
    eval_idx = [i for i, d in enumerate(baseline["dates"]) if d >= EVAL_START]
    if eval_idx:
        i0 = eval_idx[0]
        rets_clipped = baseline["rets"][i0:]
        eq = [CAPITAL_INIT]
        for r in rets_clipped[1:]:
            eq.append(eq[-1] * (1 + r))
        baseline = {
            "dates":  baseline["dates"][i0:],
            "equity": eq,
            "rets":   rets_clipped,
        }
        spy_bh = build_spy_bh(spy_df, baseline["dates"])

    # SPY daily returns aligned to strategy dates
    spy_rets = [b / a - 1 for a, b in zip(spy_bh["equity"][:-1], spy_bh["equity"][1:])]
    spy_rets = [0.0] + spy_rets

    m_baseline = compute_metrics(baseline["equity"], baseline["rets"],
                                 "Overnight Strategy (Baseline)", baseline["dates"])
    m_spy      = compute_metrics(spy_bh["equity"],   spy_rets,
                                 "SPY Buy & Hold", spy_bh["dates"])

    print_metrics(m_baseline)
    print_metrics(m_spy)

    print(f"\n{'='*62}")
    excess_sharpe = m_baseline["sharpe"] - m_spy["sharpe"]
    excess_ret    = m_baseline["ann_ret"] - m_spy["ann_ret"]
    print(f"  Excess Sharpe vs SPY:   {excess_sharpe:+.2f}")
    print(f"  Excess Ann. Return:     {excess_ret*100:+.2f}%")
    print(f"  Max DD reduction:       {(m_baseline['max_dd'] - m_spy['max_dd'])*100:+.2f}%")
    print(f"{'='*62}\n")

    plot_comparison(baseline, spy_bh, m_baseline, m_spy)