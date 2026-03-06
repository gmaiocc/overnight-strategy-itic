"""
Backtest: Day & Night Strategy — Baseline vs MA50 Long/Short Regime
Compares the unfiltered baseline (TOP-10 long / 126d) against a regime-based
long/short extension conditioned on SPY vs its 50-day moving average:

  SPY > MA50  →  LONG  TOP-5  stocks by 126d overnight momentum
  SPY < MA50  →  SHORT BOTTOM-5 stocks by 126d overnight momentum

Only one leg is active at a time — the regime determines direction.

Evaluation period: March 2024 – March 2026 (~2 years).
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
START_DATE       = "2023-01-01"
END_DATE         = "2026-03-06"
MOMENTUM_WINDOW  = 126
LONG_N           = 5
SHORT_N          = 5
MA50_WINDOW      = 50
RISK_FREE_RATE   = 0.0
CAPITAL_INIT     = 10_000.0
MIN_DOLLAR_VOL   = 10_000_000
TRANSACTION_COST = 0.0
QUICK_MODE       = False
EVAL_START       = datetime.datetime(2024, 3, 1)
# ─────────────────────────────────────────────────────────────────────────────


def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    for t in tables:
        if "Symbol" in t.columns:
            return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    return []


def fetch_spy_ma50(start: str, end: str) -> pd.DataFrame:
    print("  Downloading SPY (MA50)...")
    spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy.index.name = "date"
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    spy["MA50"] = spy["Close"].rolling(MA50_WINDOW).mean()
    above = (spy["Close"] > spy["MA50"]).sum()
    below = (spy["Close"] <= spy["MA50"]).sum()
    print(f"  SPY above MA50: {above}d | below: {below}d")
    return spy[["Close", "MA50"]]


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
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

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
        top = candidates[:10]

        if not top:
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            continue

        ret = np.mean([x["return"] for x in top]) - 2 * TRANSACTION_COST
        capital *= (1 + ret)
        equity.append(capital); daily_ret.append(ret); dates_out.append(date)

    return {"dates": [d.to_pydatetime() for d in dates_out], "equity": equity, "rets": daily_ret}


def run_longshort(clean_data: dict, spy_df: pd.DataFrame) -> dict:
    """
    Regime-based:
      SPY > MA50  -> LONG  TOP-5  by momentum  (overnight premium)
      SPY <= MA50 -> SHORT BOTTOM-5 by momentum (overnight discount/reversal)
    """
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

    capital = CAPITAL_INIT
    equity, daily_ret, dates_out = [], [], []
    stat = {"long": 0, "short": 0, "cash": 0}

    for i in range(MOMENTUM_WINDOW + 5, len(all_dates) - 1):
        date = all_dates[i]

        # Determine regime
        spy_past = spy_df[spy_df.index <= date]
        if spy_past.empty or pd.isna(spy_past["MA50"].iloc[-1]):
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            stat["cash"] += 1
            continue

        spy_row    = spy_past.iloc[-1]
        above_ma50 = spy_row["Close"] > spy_row["MA50"]

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

        if len(candidates) < max(LONG_N, SHORT_N):
            equity.append(capital); daily_ret.append(0.0); dates_out.append(date)
            stat["cash"] += 1
            continue

        candidates.sort(key=lambda x: x["momentum"], reverse=True)

        if above_ma50:
            # LONG top-5: profit from overnight premium in bull regime
            leg = candidates[:LONG_N]
            ret = np.mean([x["return"] for x in leg])
            stat["long"] += 1
        else:
            # SHORT bottom-5: profit from overnight discount/reversal in bear regime
            leg = candidates[-SHORT_N:]
            ret = np.mean([-x["return"] for x in leg])
            stat["short"] += 1

        combined = ret - 2 * TRANSACTION_COST
        capital *= (1 + combined)
        equity.append(capital); daily_ret.append(combined); dates_out.append(date)

    total = sum(stat.values())
    if total > 0:
        print(f"  long={stat['long']}d ({stat['long']/total*100:.0f}%) | "
              f"short={stat['short']}d ({stat['short']/total*100:.0f}%) | "
              f"cash={stat['cash']}d ({stat['cash']/total*100:.0f}%)")

    return {"dates": [d.to_pydatetime() for d in dates_out], "equity": equity, "rets": daily_ret}


def clip_to_eval(result: dict) -> dict:
    eval_idx = [i for i, d in enumerate(result["dates"]) if d >= EVAL_START]
    if not eval_idx:
        return result
    i0 = eval_idx[0]
    rets_clipped = result["rets"][i0:]
    eq = [CAPITAL_INIT]
    for r in rets_clipped[1:]:
        eq.append(eq[-1] * (1 + r))
    return {"dates": result["dates"][i0:], "equity": eq, "rets": rets_clipped}


def compute_metrics(equity, daily_rets, label, dates=None):
    total_ret = (equity[-1] / equity[0]) - 1
    n_days = len(daily_rets)
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
    peak = equity[0]; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd
    win_rate = sum(1 for r in daily_rets if r > 0) / n_days if n_days > 0 else 0
    return {"label": label, "total_ret": total_ret, "ann_ret": ann_ret,
            "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "win_rate": win_rate, "final_cap": equity[-1]}


def print_metrics(m):
    print(f"\n{'─'*56}\n  {m['label']}\n{'─'*56}")
    print(f"  Final capital:       ${m['final_cap']:>10,.2f}")
    print(f"  Total return:        {m['total_ret']*100:>+9.2f}%")
    print(f"  Annualised return:   {m['ann_ret']*100:>+9.2f}%")
    print(f"  Volatility (ann):    {m['vol']*100:>9.2f}%")
    print(f"  Sharpe Ratio:        {m['sharpe']:>9.2f}")
    print(f"  Sortino Ratio:       {m['sortino']:>9.2f}")
    print(f"  Max Drawdown:        {m['max_dd']*100:>9.2f}%")
    print(f"  Win Rate:            {m['win_rate']*100:>9.1f}%")


COLOR_BASELINE = "#2CA02C"
COLOR_LS       = "#1F77B4"
COLOR_DD_BASE  = "#D62728"
COLOR_DD_LS    = "#1F77B4"


def _rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    })


def plot_equity(baseline, ls):
    _rcparams()
    start_date = baseline["dates"][0]
    end_date   = baseline["dates"][-1]
    start_6m   = end_date - datetime.timedelta(days=182)

    def clip(res, x0, x1):
        pts = [(d, e) for d, e in zip(res["dates"], res["equity"]) if x0 <= d <= x1]
        if not pts: return [], []
        ds, es = zip(*pts); return list(ds), list(es)

    def _panel(ax, x0, x1, title, show_ylabel, show_legend):
        ds_b, es_b = clip(baseline, x0, x1)
        ds_l, es_l = clip(ls, x0, x1)
        if ds_b:
            ax.plot(ds_b, es_b, label="Baseline (Long-Only TOP-10)",
                    color=COLOR_BASELINE, linewidth=1.8)
        if ds_l:
            ax.plot(ds_l, es_l, label="L/S Regime (TOP-5↑ SPY>MA50 / BOTTOM-5↓ SPY<MA50)",
                    color=COLOR_LS, linewidth=1.8, linestyle="--")
        ax.axhline(CAPITAL_INIT, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=6)
        if show_ylabel: ax.set_ylabel("Portfolio Value (USD)", labelpad=6)
        if show_legend: ax.legend(loc="upper left", handlelength=3.0)
        ax.grid(True, axis="y")
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)

    fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)

    _panel(ax1a, start_date, end_date, "Panel A — Full Period (Mar 2024 – Mar 2026)",
           show_ylabel=True, show_legend=True)
    ax1a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    full_vals = [v for v in baseline["equity"] + ls["equity"] if not np.isnan(v)]
    ax1a.set_ylim(min(full_vals) * 0.97, max(full_vals) * 1.03)

    _panel(ax1b, start_6m, end_date, "Panel A — Last 6 Months (Zoom)",
           show_ylabel=False, show_legend=False)
    ax1b.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    zoom_vals  = [e for d, e in zip(baseline["dates"], baseline["equity"]) if d >= start_6m]
    zoom_vals += [e for d, e in zip(ls["dates"], ls["equity"]) if d >= start_6m]
    zoom_vals  = [v for v in zoom_vals if not np.isnan(v)]
    ax1b.set_ylim(min(zoom_vals) * 0.98, max(zoom_vals) * 1.02)
    ax1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Overnight Effect — Baseline vs. MA50 Long/Short Regime",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
             f"SPY > MA50 → Long TOP-{LONG_N}  |  SPY ≤ MA50 → Short BOTTOM-{SHORT_N}  |  126-day momentum",
             ha="center", fontsize=9, color="#555555", style="italic")

    plt.savefig("ls_fig1_equity.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: ls_fig1_equity.png")
    plt.close()


def plot_drawdown(baseline, ls):
    _rcparams()
    start_date = baseline["dates"][0]
    end_date   = baseline["dates"][-1]

    def calc_dd(equity):
        peak, out = equity[0], []
        for v in equity:
            if v > peak: peak = v
            out.append((v - peak) / peak * 100)
        return out

    dd_b = calc_dd(baseline["equity"])
    dd_l = calc_dd(ls["equity"])

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.14)

    ax.fill_between(baseline["dates"], dd_b, 0, alpha=0.20, color=COLOR_DD_BASE, zorder=1)
    ax.fill_between(ls["dates"],       dd_l, 0, alpha=0.20, color=COLOR_DD_LS,   zorder=0)
    ax.plot(baseline["dates"], dd_b, color=COLOR_DD_BASE, linewidth=1.6,
            label="Baseline (Long-Only TOP-10)")
    ax.plot(ls["dates"],       dd_l, color=COLOR_DD_LS,   linewidth=1.6,
            linestyle="--", label="L/S Regime (TOP-5 / BOTTOM-5)")

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

    plt.savefig("ls_fig2_drawdown.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: ls_fig2_drawdown.png")
    plt.close()


def plot_table(m_b, m_l):
    _rcparams()
    col_headers = ["Metric", "Baseline (Long-Only)", "L/S Regime (MA50)"]
    rows = [
        ["Final Value",   f"${m_b['final_cap']:,.0f}",     f"${m_l['final_cap']:,.0f}"],
        ["Total Return",  f"{m_b['total_ret']*100:+.2f}%", f"{m_l['total_ret']*100:+.2f}%"],
        ["Ann. Return",   f"{m_b['ann_ret']*100:+.2f}%",   f"{m_l['ann_ret']*100:+.2f}%"],
        ["Volatility",    f"{m_b['vol']*100:.2f}%",        f"{m_l['vol']*100:.2f}%"],
        ["Sharpe Ratio",  f"{m_b['sharpe']:.2f}",          f"{m_l['sharpe']:.2f}"],
        ["Sortino Ratio", f"{m_b['sortino']:.2f}",         f"{m_l['sortino']:.2f}"],
        ["Max Drawdown",  f"{m_b['max_dd']*100:.2f}%",     f"{m_l['max_dd']*100:.2f}%"],
        ["Win Rate",      f"{m_b['win_rate']*100:.1f}%",   f"{m_l['win_rate']*100:.1f}%"],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   cellLoc="center", loc="center", bbox=[0.02, 0.02, 0.96, 0.90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    col_widths = [0.32, 0.32, 0.32]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#CCCCCC"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#B00000")
            cell.set_text_props(color="white", fontweight="bold",
                                ha="left" if c == 0 else "center", fontsize=9.5)
            cell.PAD = 0.06
        else:
            cell.set_facecolor("#F7F9FC" if r % 2 == 0 else "#FFFFFF")
            cell.set_text_props(ha="left" if c == 0 else "center", fontsize=10)
            if c == 0: cell.PAD = 0.06

    ax.set_title("Panel C — Performance Summary",
                 loc="left", fontsize=10, fontweight="bold", pad=8)

    plt.savefig("ls_fig3_table.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: ls_fig3_table.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*62)
    print("  BACKTEST: Baseline vs MA50 Long/Short Regime")
    print("="*62)

    print("\n[1/3] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if QUICK_MODE: tickers = tickers[:80]
    print(f"  {len(tickers)} tickers loaded.")

    print("\n[2/3] Downloading OHLCV + SPY data...")
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE,
                           group_by="ticker", auto_adjust=True,
                           progress=True, threads=True)
    spy_df = fetch_spy_ma50(START_DATE, END_DATE)

    print("\n[3/3] Building clean data & running backtests...")
    clean_data = build_clean_data(raw_data, tickers)
    print(f"  {len(clean_data)} tickers with valid data.")

    print("  → Baseline (TOP-10 long only)...")
    baseline_raw = run_baseline(clean_data)
    baseline     = clip_to_eval(baseline_raw)

    print("  → MA50 Long/Short Regime...")
    ls_raw = run_longshort(clean_data, spy_df)
    ls     = clip_to_eval(ls_raw)

    m_baseline = compute_metrics(baseline["equity"], baseline["rets"],
                                 "Baseline (Long-Only TOP-10)", baseline["dates"])
    m_ls       = compute_metrics(ls["equity"], ls["rets"],
                                 "MA50 Long/Short Regime", ls["dates"])

    print_metrics(m_baseline)
    print_metrics(m_ls)

    print(f"\n{'='*62}")
    print(f"  Sharpe improvement:   {m_ls['sharpe'] - m_baseline['sharpe']:+.2f}")
    print(f"  Ann. return delta:    {(m_ls['ann_ret'] - m_baseline['ann_ret'])*100:+.2f}%")
    print(f"  Max DD improvement:   {(m_ls['max_dd'] - m_baseline['max_dd'])*100:+.2f}%")
    print(f"{'='*62}\n")

    plot_equity(baseline, ls)
    plot_drawdown(baseline, ls)
    plot_table(m_baseline, m_ls)