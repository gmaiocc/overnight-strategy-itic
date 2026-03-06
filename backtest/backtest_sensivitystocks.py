"""
Backtest: Day & Night Strategy — Sensitivity Analysis
Tests all combinations of portfolio size (TOP-5, TOP-10, TOP-20)
and momentum window (63d, 126d, 252d) over Mar 2024 – Mar 2026.
Same config, style and clip logic as backtest_baseline_vs_spy.py.
"""

import datetime
import itertools
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_DATE       = "2023-01-01"   # enough history for 126-day warm-up + Mar 2024 start
END_DATE         = "2026-03-06"
RISK_FREE_RATE   = 0.0
CAPITAL_INIT     = 10_000.0
MIN_DOLLAR_VOL   = 10_000_000
TRANSACTION_COST = 0.0
QUICK_MODE       = False
EVAL_START       = datetime.datetime(2024, 3, 1)

# Sensitivity grid
TOP_N_VALUES        = [5, 10]
MOMENTUM_WINDOWS    = [63, 126]
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


def build_clean_data(raw_data: pd.DataFrame, tickers: list,
                     momentum_window: int) -> dict:
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
                .rolling(momentum_window)
                .apply(np.prod, raw=True) - 1
            )
            df["Forward_Overnight"] = (df["Open"].shift(-1) / df["Close"]) - 1
            df["DollarVol"] = df["Close"] * df["Volume"]

            if len(df) > momentum_window + 5:
                clean[t] = df
        except Exception:
            continue
    return clean


def run_strategy(clean_data: dict, top_n: int,
                 momentum_window: int) -> dict:
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

    capital   = CAPITAL_INIT
    equity    = []
    daily_ret = []
    dates_out = []

    for i in range(momentum_window + 5, len(all_dates) - 1):
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
        top = candidates[:top_n]

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


def clip_to_eval(result: dict) -> dict:
    """Clip to EVAL_START and rebuild equity from $10,000."""
    eval_idx = [i for i, d in enumerate(result["dates"]) if d >= EVAL_START]
    if not eval_idx:
        return result
    i0 = eval_idx[0]
    rets_clipped = result["rets"][i0:]
    eq = [CAPITAL_INIT]
    for r in rets_clipped[1:]:
        eq.append(eq[-1] * (1 + r))
    return {
        "dates":  result["dates"][i0:],
        "equity": eq,
        "rets":   rets_clipped,
    }


def compute_metrics(equity: list, daily_rets: list,
                    label: str, dates: list = None) -> dict:
    total_ret = (equity[-1] / equity[0]) - 1
    n_days    = len(daily_rets)
    if dates and len(dates) >= 2:
        years = (dates[-1] - dates[0]).days / 365.25
    else:
        years = n_days / 252
    ann_ret  = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    vol      = np.std(daily_rets) * np.sqrt(252) if daily_rets else 0
    sharpe   = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0

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


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette: 3 portfolio sizes × 3 momentum windows = 9 combinations
# Use hue for portfolio size, linestyle for momentum window
COLORS = {
    5:  "#2166AC",   # blue
    10: "#2CA02C",   # green  (baseline)
}
LINESTYLES = {
    63:  (0, (3, 1)),  # densely dashed
    126: "-",          # solid  (baseline)
}
LINEWIDTHS = {
    63:  1.2,
    126: 2.0,
}


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
        "legend.fontsize":   8.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#CCCCCC",
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.6,
    })


def _legend_elements():
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], color=COLORS[5],  linewidth=1.5, label="TOP-5  (blue)"),
        Line2D([0], [0], color=COLORS[10], linewidth=2.0, label="TOP-10 (green)"),
        Line2D([0], [0], color="#888888",  linewidth=1.5,
               linestyle=LINESTYLES[63],  label="63d momentum"),
        Line2D([0], [0], color="#888888",  linewidth=2.0,
               linestyle=LINESTYLES[126], label="126d momentum  ★ baseline"),
    ]


def _calc_dd(equity):
    peak, out = equity[0], []
    for v in equity:
        if v > peak: peak = v
        out.append((v - peak) / peak * 100)
    return out


def plot_sens_equity(results: dict):
    """Figure 1 — equity curves for all 4 combinations."""
    _rcparams()
    ref = results[(10, 126)]
    start_date = ref["dates"][0]
    end_date   = ref["dates"][-1]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13)

    for (top_n, mw), res in sorted(results.items()):
        is_baseline = (top_n == 10 and mw == 126)
        ax.plot(res["dates"], res["equity"],
                color=COLORS[top_n],
                linestyle=LINESTYLES[mw],
                linewidth=LINEWIDTHS[mw] + (0.6 if is_baseline else 0),
                alpha=0.95 if is_baseline else 0.75,
                zorder=10 if is_baseline else 5)

    ax.axhline(CAPITAL_INIT, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_ylabel("Portfolio Value (USD)", labelpad=6)
    ax.set_title("Panel A — Equity Curves (Mar 2024 – Mar 2026)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, axis="y")
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=_legend_elements(), loc="upper left", ncol=2, handlelength=3.0)

    fig.suptitle("Sensitivity Analysis — Portfolio Size × Momentum Window",
                 fontsize=13, fontweight="bold")
    
    plt.savefig("sensitivity_fig1_equity.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: sensitivity_fig1_equity.png")
    plt.close()


def plot_sens_drawdown(results: dict):
    """Figure 2 — drawdown for all 4 combinations."""
    _rcparams()
    ref = results[(10, 126)]
    start_date = ref["dates"][0]
    end_date   = ref["dates"][-1]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.14)

    for (top_n, mw), res in sorted(results.items()):
        is_baseline = (top_n == 10 and mw == 126)
        dd = _calc_dd(res["equity"])
        ax.plot(res["dates"], dd,
                color=COLORS[top_n],
                linestyle=LINESTYLES[mw],
                linewidth=LINEWIDTHS[mw] + (0.6 if is_baseline else 0),
                alpha=0.95 if is_baseline else 0.70,
                zorder=10 if is_baseline else 5)

    dd_base = _calc_dd(results[(10, 126)]["equity"])
    ax.fill_between(results[(10, 126)]["dates"], dd_base, 0,
                    alpha=0.12, color=COLORS[10], zorder=0)

    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_ylabel("Drawdown (%)", labelpad=6)
    ax.set_title("Panel B — Drawdown from Peak (Mar 2024 – Mar 2026)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, axis="y")
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=_legend_elements(), loc="lower left", ncol=2, handlelength=3.0)

    plt.savefig("sensitivity_fig2_drawdown.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: sensitivity_fig2_drawdown.png")
    plt.close()


def plot_sens_table(metrics: dict):
    """Figure 3 — performance summary table."""
    _rcparams()

    col_headers = ["Scenario", "Final Value", "Ann. Return",
                   "Volatility", "Sharpe", "Sortino", "Max DD", "Win Rate"]
    rows = []
    for (top_n, mw) in sorted(metrics.keys()):
        m = metrics[(top_n, mw)]
        is_baseline = (top_n == 10 and mw == 126)
        label = f"TOP-{top_n} / {mw}d{'  ★' if is_baseline else ''}"
        rows.append([
            label,
            f"${m['final_cap']:,.0f}",
            f"{m['ann_ret']*100:+.2f}%",
            f"{m['vol']*100:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['max_dd']*100:.2f}%",
            f"{m['win_rate']*100:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    col_widths = [0.18, 0.11, 0.11, 0.10, 0.09, 0.09, 0.10, 0.10]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#B00000")
            cell.set_text_props(color="white", fontweight="bold",
                                ha="left" if c == 0 else "center", fontsize=9)
            cell.PAD = 0.05
        else:
            scenario_key = sorted(metrics.keys())[r - 1]
            is_base_row  = (scenario_key == (10, 126))
            if is_base_row:
                cell.set_facecolor("#E8F5E9")
            else:
                cell.set_facecolor("#F7F9FC" if r % 2 == 0 else "#FFFFFF")
            cell.set_text_props(ha="left" if c == 0 else "center", fontsize=9.5)
            if c == 0:
                cell.PAD = 0.05

    ax.set_title("Panel C — Performance Summary by Configuration",
                 loc="left", fontsize=10, fontweight="bold", pad=8)

    plt.savefig("sensitivity_fig3_table.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: sensitivity_fig3_table.png")
    plt.close()


def plot_sensitivity(results: dict, metrics: dict):
    """Saves all 3 figures separately."""
    plot_sens_equity(results)
    plot_sens_drawdown(results)
    plot_sens_table(metrics)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  SENSITIVITY ANALYSIS: Portfolio Size × Momentum Window")
    print("="*62)

    print("\n[1/3] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
    print(f"  {len(tickers)} tickers loaded.")

    print("\n[2/3] Downloading OHLCV data (2022-01-01 to 2026-03-06)...")
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE,
                           group_by="ticker", auto_adjust=True,
                           progress=True, threads=True)

    print("\n[3/3] Running all combinations...")
    results = {}
    metrics = {}

    for top_n, mw in itertools.product(TOP_N_VALUES, MOMENTUM_WINDOWS):
        label = f"TOP-{top_n} / {mw}d"
        print(f"  → {label}...")

        clean = build_clean_data(raw_data, tickers, mw)
        raw_result = run_strategy(clean, top_n, mw)
        clipped    = clip_to_eval(raw_result)
        m          = compute_metrics(clipped["equity"], clipped["rets"],
                                     label, clipped["dates"])

        results[(top_n, mw)] = clipped
        metrics[(top_n, mw)] = m

        print(f"     Final: ${m['final_cap']:,.0f} | "
              f"Ann: {m['ann_ret']*100:+.1f}% | "
              f"Sharpe: {m['sharpe']:.2f} | "
              f"MaxDD: {m['max_dd']*100:.1f}%")

    print("\n" + "="*62)
    print("  RESULTS SUMMARY")
    print("="*62)
    for (top_n, mw) in sorted(metrics.keys()):
        m = metrics[(top_n, mw)]
        star = " ★" if (top_n == 10 and mw == 126) else ""
        print(f"  TOP-{top_n:2d} / {mw:3d}d{star:3s} | "
              f"Ann: {m['ann_ret']*100:+6.2f}% | "
              f"Sharpe: {m['sharpe']:.2f} | "
              f"MaxDD: {m['max_dd']*100:6.2f}%")

    plot_sensitivity(results, metrics)