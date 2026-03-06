"""
Backtest: Day & Night Strategy — TOP10 / 126d
Master comparison — ALL scenarios tested during development:

  0. Baseline                      — long only, no filters
  1. + Market Filter (MA200)       — cash when SPY below 200-day MA
  2. + VIX Spike Filter            — cash when VIX rises >20% in a day
  3. + Vol Sizing (ADR-inverse)    — overweight low-ADR stocks (ADR<5% → 5/ADR x weight)
  4. Long/Short Regime (MA50)      — long top10 when SPY>MA50, short bottom10 when SPY<MA50

All scenarios use TOP10 / 126d momentum / 0.05% transaction cost per side.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PERIOD           = "2y"
TOP_N            = 10
MOMENTUM_WINDOW  = 126
RISK_FREE_RATE   = 0.0
CAPITAL_INIT     = 10_000.0
MIN_DOLLAR_VOL   = 10_000_000
TRANSACTION_COST = 0       # 0.05% per side
ADR_WINDOW       = 20
ADR_THRESHOLD    = 5.0          # below this → inverse vol weight
VIX_SPIKE_PCT    = 0.20         # VIX daily rise threshold
QUICK_MODE       = False
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url, headers=headers).content)
    for t in tables:
        if "Symbol" in t.columns:
            return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    return []


def fetch_spy_ma(period: str = "5y") -> tuple:
    """Returns (above_ma50, above_ma200) boolean Series."""
    print("  Downloading SPY (MA50 + MA200)...")
    spy = yf.Ticker("SPY").history(period=period, auto_adjust=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy.index.name = "date"
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    spy["MA50"]  = spy["Close"].rolling(50).mean()
    spy["MA200"] = spy["Close"].rolling(200).mean()
    above_ma50  = (spy["Close"] > spy["MA50"]).rename("above_ma50")
    above_ma200 = (spy["Close"] > spy["MA200"]).rename("above_ma200")
    print(f"  SPY above MA50:  {above_ma50.sum()}d | below: {(~above_ma50).sum()}d")
    print(f"  SPY above MA200: {above_ma200.sum()}d | below: {(~above_ma200).sum()}d")
    return above_ma50, above_ma200


def fetch_vix_spike(period: str = "5y") -> pd.Series:
    """Returns boolean Series: True = VIX rose >20% that day."""
    print("  Downloading VIX spike filter...")
    vix = yf.Ticker("^VIX").history(period=period, auto_adjust=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    vix.index.name = "date"
    vix = vix[~vix.index.duplicated(keep="last")].sort_index()
    vix["spike"] = vix["Close"].pct_change() > VIX_SPIKE_PCT
    return vix["spike"].rename("vix_spike")


def build_clean_data(raw_data: pd.DataFrame, tickers: list) -> dict:
    clean = {}
    for t in tickers:
        try:
            df = raw_data[t].copy()
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["Volume"] > 0)]
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")].sort_index()

            # Momentum signal (no lookahead)
            df["Past_Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1
            df["Momentum"] = (
                (1 + df["Past_Overnight"])
                .rolling(MOMENTUM_WINDOW)
                .apply(np.prod, raw=True) - 1
            )

            # Forward return: buy at close T, sell at open T+1
            df["Forward_Overnight"] = (df["Open"].shift(-1) / df["Close"]) - 1
            df["DollarVol"] = df["Close"] * df["Volume"]

            # ADR: 20-day average daily range (%) for vol-sizing scenario
            df["ADR"] = ((df["High"] - df["Low"]) / df["Close"] * 100).rolling(ADR_WINDOW).mean()

            if len(df) > MOMENTUM_WINDOW + 5:
                clean[t] = df
        except Exception:
            continue
    return clean


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime(date, spy_above_ma50, spy_above_ma200):
    """Returns (above_ma50, above_ma200) for a given date."""
    def lookup(series):
        past = series[series.index <= date]
        return past.iloc[-1] if not past.empty else True
    return lookup(spy_above_ma50), lookup(spy_above_ma200)


def vol_weight(adr: float) -> float:
    """Inverse-ADR weight multiplier. ADR<5% → 5/ADR, else 1."""
    if pd.isna(adr) or adr <= 0:
        return 1.0
    return ADR_THRESHOLD / adr if adr < ADR_THRESHOLD else 1.0


def run_backtest(clean_data: dict,
                 spy_above_ma50: pd.Series,
                 spy_above_ma200: pd.Series,
                 vix_spike: pd.Series,
                 scenario: str) -> dict:
    """
    scenario options:
      "baseline"    — long top10, no filters
      "ma200"       — cash when SPY < MA200
      "vix"         — cash when VIX spike >20%, else long top10
      "vol_sizing"  — MA200 filter + inverse-ADR position sizing
      "long_short"  — long top10 when SPY>MA50, short bottom10 when SPY<MA50
    """
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

    capital    = CAPITAL_INIT
    equity     = []
    daily_ret  = []
    dates_out  = []
    stat       = {"long": 0, "short": 0, "cash": 0}

    for i in range(MOMENTUM_WINDOW + 5, len(all_dates) - 1):
        date = all_dates[i]
        above50, above200 = get_regime(date, spy_above_ma50, spy_above_ma200)

        # ── Cash conditions ───────────────────────────────────────────────────
        in_cash = False

        if scenario == "ma200" and not above200:
            in_cash = True
        elif scenario == "vol_sizing" and not above200:
            in_cash = True
        elif scenario == "vix":
            past_vix = vix_spike[vix_spike.index <= date]
            if not past_vix.empty and past_vix.iloc[-1]:
                in_cash = True

        if in_cash:
            equity.append(capital)
            daily_ret.append(0.0)
            dates_out.append(date)
            stat["cash"] += 1
            continue

        # ── Build candidate list ──────────────────────────────────────────────
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
                "adr":      row["ADR"] if not pd.isna(row["ADR"]) else ADR_THRESHOLD,
            })

        if not candidates:
            equity.append(capital)
            daily_ret.append(0.0)
            dates_out.append(date)
            continue

        candidates.sort(key=lambda x: x["momentum"], reverse=True)

        # ── Execute trade ─────────────────────────────────────────────────────
        if scenario == "long_short" and not above50:
            # SHORT bottom10: profit when worst-momentum stocks fall overnight
            bottom = candidates[-TOP_N:]
            port_ret = np.mean([-x["return"] for x in bottom])
            stat["short"] += 1

        elif scenario == "vol_sizing":
            # LONG top10 with inverse-ADR weights, normalised
            top = candidates[:TOP_N]
            raw_w = [vol_weight(s["adr"]) for s in top]
            total_w = sum(raw_w)
            weights = [w / total_w for w in raw_w]
            port_ret = sum(w * s["return"] for w, s in zip(weights, top))
            stat["long"] += 1

        else:
            # LONG top10 equal weight (baseline / ma200 / vix / long_short long side)
            top = candidates[:TOP_N]
            port_ret = np.mean([x["return"] for x in top])
            stat["long"] += 1

        ret = port_ret - 2 * TRANSACTION_COST
        capital *= (1 + ret)
        equity.append(capital)
        daily_ret.append(ret)
        dates_out.append(date)

    # Print stats
    total = sum(stat.values())
    if total > 0:
        parts = []
        if stat["long"]:  parts.append(f"long={stat['long']}d ({stat['long']/total*100:.0f}%)")
        if stat["short"]: parts.append(f"short={stat['short']}d ({stat['short']/total*100:.0f}%)")
        if stat["cash"]:  parts.append(f"cash={stat['cash']}d ({stat['cash']/total*100:.0f}%)")
        print(f"  {' | '.join(parts)}")

    return {
        "dates":  [d.to_pydatetime() for d in dates_out],
        "equity": equity,
        "rets":   daily_ret,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity: list, daily_rets: list, label: str) -> dict:
    total_ret = (equity[-1] / equity[0]) - 1
    n_days    = len(daily_rets)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    vol       = np.std(daily_rets) * np.sqrt(252)
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

    win_rate = sum(1 for r in daily_rets if r > 0) / n_days

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

def shade_regimes(ax, spy_above_ma50, start_date, end_date):
    for is_above, color, alpha in [(True, "#2ECC71", 0.05), (False, "#E74C3C", 0.07)]:
        days = spy_above_ma50[
            (spy_above_ma50 == is_above) &
            (spy_above_ma50.index >= pd.Timestamp(start_date)) &
            (spy_above_ma50.index <= pd.Timestamp(end_date))
        ].index
        if len(days) == 0:
            continue
        groups, s, prev = [], days[0], days[0]
        for d in days[1:]:
            if (d - prev).days > 3:
                groups.append((s, prev))
                s = d
            prev = d
        groups.append((s, prev))
        for s, e in groups:
            ax.axvspan(s, e + pd.Timedelta(days=1), alpha=alpha, color=color, zorder=0)


def plot_all(results: list, spy_above_ma50: pd.Series):
    # ── Academic palette: colorblind-safe, print-friendly ─────────────────────
    COLORS  = ["#222222", "#2166AC", "#D6604D", "#4DAC26", "#8073AC"]
    STYLES  = ["-",       "-",       "--",       "-.",       ":"]
    LWIDTHS = [1.2,       1.8,       1.8,        1.8,        1.8]

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
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#CCCCCC",
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.6,
    })

    import datetime
    end_date   = results[0][1]["dates"][-1]
    start_date = results[0][1]["dates"][0]
    start_12m  = end_date - datetime.timedelta(days=365)
    start_6m   = end_date - datetime.timedelta(days=182)

    # ── Layout: row0 = two side-by-side panels, row1 = drawdown, row2 = table ─
    fig = plt.figure(figsize=(14, 15))
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 1.4],
                                hspace=0.32, left=0.07, right=0.97,
                                top=0.93, bottom=0.04)

    # Split top row into two columns
    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.10)
    ax1a = fig.add_subplot(gs_top[0])   # Last 12 months
    ax1b = fig.add_subplot(gs_top[1])   # Last 6 months
    ax2  = fig.add_subplot(gs_outer[1])
    ax3  = fig.add_subplot(gs_outer[2])

    def _plot_equity_window(ax, x_start, x_end, title_suffix, show_legend, show_ylabel):
        """Plot equity curves clipped to [x_start, x_end]."""
        shade_regimes(ax, spy_above_ma50, x_start, x_end)
        for (label, res, m), color, ls, lw in zip(results, COLORS, STYLES, LWIDTHS):
            short = label.split(". ", 1)[-1]
            # Filter dates to window
            pts = [(d, e) for d, e in zip(res["dates"], res["equity"])
                   if x_start <= d <= x_end]
            if not pts:
                continue
            ds, es = zip(*pts)
            ax.plot(ds, es, label=short, color=color, linewidth=lw, linestyle=ls)

        ax.set_ylabel("Portfolio Value (USD)", labelpad=6) if show_ylabel else None
        ax.set_title(f"Panel A{title_suffix} — Cumulative Value", loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        if show_legend:
            ax.legend(loc="upper left", ncol=1, handlelength=3.0, fontsize=8)
        ax.grid(True, axis="y")
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)

    _plot_equity_window(ax1a, start_12m, end_date,
                        " (Last 12 Months)", show_legend=True,  show_ylabel=True)
    _plot_equity_window(ax1b, start_6m,  end_date,
                        " (Last 6 Months)",  show_legend=False, show_ylabel=False)

    # Share y-axis scale between the two panels for fair comparison
    all_vals = []
    for label, res, m in results:
        all_vals += [e for d, e in zip(res["dates"], res["equity"]) if d >= start_12m]
    y_lo = min(all_vals) * 0.98
    y_hi = max(all_vals) * 1.02
    ax1a.set_ylim(y_lo, y_hi)
    ax1b.set_ylim(y_lo, y_hi)
    ax1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel B: Drawdown (full period) ───────────────────────────────────────
    def calc_dd(equity):
        peak, out = equity[0], []
        for v in equity:
            if v > peak: peak = v
            out.append((v - peak) / peak * 100)
        return out

    shade_regimes(ax2, spy_above_ma50, start_date, end_date)
    for (label, res, m), color, ls, lw in zip(results, COLORS, STYLES, LWIDTHS):
        short = label.split(". ", 1)[-1]
        ax2.plot(res["dates"], calc_dd(res["equity"]),
                 label=short, color=color, linewidth=lw - 0.2, linestyle=ls)

    ax2.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
    ax2.set_ylabel("Drawdown (%)", labelpad=6)
    ax2.set_title("Panel B — Drawdown from Peak (Full Period)", loc="left",
                  fontsize=10, fontweight="bold", pad=6)
    ax2.grid(True, axis="y")
    ax2.set_xlim(start_date, end_date)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Panel C: Performance table ────────────────────────────────────────────
    ax3.axis("off")

    col_headers = ["Strategy", "Final Value", "Ann. Return",
                   "Volatility", "Sharpe", "Sortino", "Max DD", "Win Rate"]

    best_sharpe = max(r[2]["sharpe"] for r in results)
    row_data    = []
    for label, res, m in results:
        is_best = abs(m["sharpe"] - best_sharpe) < 0.01
        short   = label.split(". ", 1)[-1]
        row_data.append({
            "cells": [
                short,
                f"${m['final_cap']:,.0f}",
                f"{m['ann_ret']*100:+.2f}%",
                f"{m['vol']*100:.2f}%",
                f"{m['sharpe']:.2f}",
                f"{m['sortino']:.2f}",
                f"{m['max_dd']*100:.2f}%",
                f"{m['win_rate']*100:.1f}%",
            ],
            "best": is_best,
        })

    tbl = ax3.table(
        cellText=[r["cells"] for r in row_data],
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.05, 1, 0.88],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    col_widths = [0.28, 0.10, 0.11, 0.10, 0.09, 0.09, 0.09, 0.10]

    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)

        if r == 0:
            cell.set_facecolor("#B00000")
            cell.set_text_props(
                color="white", fontweight="bold",
                ha="left" if c == 0 else "center",
                fontsize=8.5
            )
            cell.PAD = 0.05
        else:
            d = row_data[r - 1]
            if d["best"]:
                cell.set_facecolor("#EAF4EA")
            elif r % 2 == 0:
                cell.set_facecolor("#F7F9FC")
            else:
                cell.set_facecolor("#FFFFFF")

            cell.set_text_props(
                ha="left" if c == 0 else "center",
                fontsize=9
            )
            if c == 0:
                cell.PAD = 0.05

    ax3.set_title(
        "Panel C — Performance Summary  "
        "(highlighted row = highest Sharpe ratio)",
        loc="left", fontsize=10, fontweight="bold", pad=8
    )

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Overnight Effect — Strategy Backtest Comparison",
        fontsize=13, fontweight="bold", y=0.978
    )
    
    plt.savefig("backtest_results.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("\n  Chart saved: backtest_results.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  BACKTEST: Day & Night — Master Scenario Comparison")
    print("="*62)

    print("\n[1/4] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
    print(f"  {len(tickers)} tickers loaded.")

    print("\n[2/4] Downloading OHLCV + SPY + VIX data...")
    raw_data = yf.download(tickers, period=PERIOD, group_by="ticker",
                           auto_adjust=True, progress=True, threads=True)
    spy_above_ma50, spy_above_ma200 = fetch_spy_ma(period="5y")
    vix_spike = fetch_vix_spike(period="5y")

    print("\n[3/4] Building clean data...")
    clean_data = build_clean_data(raw_data, tickers)
    print(f"  {len(clean_data)} tickers with valid data.")

    # ── All scenarios ─────────────────────────────────────────────────────────
    scenarios = [
        ("0. Baseline (long only)",              "baseline"),
        ("1. + Market Filter (MA200)",            "ma200"),
        ("2. + VIX Spike Filter (>20%)",          "vix"),
        ("3. + Vol Sizing (ADR-inverse, MA200)",  "vol_sizing"),
        ("4. Long/Short Regime (MA50)",           "long_short"),
    ]

    print("\n[4/4] Running all scenarios...")
    results = []
    for label, scenario in scenarios:
        print(f"\n  → {label}")
        res     = run_backtest(clean_data, spy_above_ma50, spy_above_ma200, vix_spike, scenario)
        metrics = compute_metrics(res["equity"], res["rets"], label)
        print_metrics(metrics)
        results.append((label, res, metrics))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  RANKING BY SHARPE RATIO")
    print(f"{'='*62}")
    ranked = sorted(results, key=lambda x: x[2]["sharpe"], reverse=True)
    for i, (label, _, m) in enumerate(ranked):
        print(f"  #{i+1}  Sharpe {m['sharpe']:.2f}  |  Ann. {m['ann_ret']*100:+.1f}%"
              f"  |  MaxDD {m['max_dd']*100:.1f}%  →  {label}")
    print(f"{'='*62}\n")

    plot_all(results, spy_above_ma50)