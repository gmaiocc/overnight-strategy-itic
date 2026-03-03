"""
Backtest: Day & Night Strategy — Impacto de Comprar antes de Earnings
Versão corrigida: timezone consistente, redistribuição de capital, earnings históricos robustos.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PERIOD         = "1y"
TOP_N          = 10
RISK_FREE_RATE = 0.0
CAPITAL_INIT   = 10_000.0
MIN_DOLLAR_VOL = 10_000_000
MOMENTUM_WINDOW = 126
QUICK_MODE     = False
# ─────────────────────────────────────────────────────────────────────────────


def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=headers)
    tables = pd.read_html(resp.text)
    for t in tables:
        if "Symbol" in t.columns:
            return t["Symbol"].str.replace(".", "-", regex=False).tolist()
    return []


def fetch_earnings_dates(tickers: list) -> dict:
    """
    Obtém datas de earnings históricas via yfinance.earnings_dates.
    Normaliza tudo para datetime.date sem timezone para comparação segura.
    """
    earnings_map = {}
    for i, t in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Earnings: {i+1}/{len(tickers)}...")
        dates = set()
        try:
            ticker_obj = yf.Ticker(t)

            # Fonte principal: histórico de earnings (cobre o passado)
            hist = ticker_obj.earnings_dates
            if hist is not None and not hist.empty:
                for ed in hist.index:
                    try:
                        # Remove timezone antes de converter para date
                        d = pd.Timestamp(ed).tz_localize(None).date()
                        dates.add(d)
                    except Exception:
                        pass

            # Fonte secundária: próximas datas (calendar)
            cal = ticker_obj.calendar
            if cal is not None and not cal.empty:
                for col in (cal.columns if hasattr(cal, 'columns') else []):
                    try:
                        dates.add(pd.Timestamp(col).tz_localize(None).date())
                    except Exception:
                        pass

        except Exception:
            pass

        earnings_map[t] = dates
        time.sleep(0.05)

    return earnings_map


def build_clean_data(raw_data: pd.DataFrame, tickers: list) -> dict:
    """
    Constrói dicionário de DataFrames limpos, com índice normalizado (sem tz).
    """
    clean = {}
    for t in tickers:
        try:
            df = raw_data[t].copy()
            df = df.dropna(subset=["Open", "Close"])
            df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["Volume"] > 0)]

            # ✅ Normalizar índice — remove timezone e hora
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")]

            df["NextOpen"]      = df["Open"].shift(-1)
            df["Overnight_Ret"] = (df["NextOpen"] / df["Close"]) - 1
            df["Past_Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1
            df["Momentum"] = (
                (1 + df["Past_Overnight"])
                .rolling(MOMENTUM_WINDOW)
                .apply(np.prod, raw=True) - 1
            )
            df["DollarVol"] = df["Close"] * df["Volume"]

            if len(df) > MOMENTUM_WINDOW + 5:
                clean[t] = df
        except Exception:
            continue
    return clean


def run_backtest(clean_data: dict, earnings_map: dict) -> dict:
    # ✅ Usar só datas de mercado reais (união dos índices dos tickers)
    all_dates = sorted(set().union(*[set(df.index) for df in clean_data.values()]))
    all_dates = [d for d in all_dates if isinstance(d, pd.Timestamp)]

    capital_risky = CAPITAL_INIT
    capital_safe  = CAPITAL_INIT

    equity_risky, equity_safe   = [], []
    ret_risky_list, ret_safe_list = [], []
    dates_used = []

    for i in range(MOMENTUM_WINDOW + 5, len(all_dates) - 1):
        current_date = all_dates[i]
        next_date    = all_dates[i + 1]
        next_date_obj = next_date.date()  # ✅ pd.Timestamp.date() é sempre seguro

        # ── Ranking ──────────────────────────────────────────────────────────
        candidates = []
        for t, df in clean_data.items():
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if pd.isna(row["Momentum"]) or pd.isna(row["Overnight_Ret"]):
                continue
            recent_vol = df.loc[:current_date, "DollarVol"].tail(20).mean()
            if recent_vol < MIN_DOLLAR_VOL:
                continue
            candidates.append({
                "ticker":  t,
                "momentum": row["Momentum"],
                "return":   row["Overnight_Ret"],
            })

        candidates.sort(key=lambda x: x["momentum"], reverse=True)
        top_n = candidates[:TOP_N]

        if not top_n:
            equity_risky.append(capital_risky)
            equity_safe.append(capital_safe)
            ret_risky_list.append(0.0)
            ret_safe_list.append(0.0)
            dates_used.append(current_date)
            continue

        # ── Carteira SEM filtro ───────────────────────────────────────────────
        avg_ret_risky = np.mean([x["return"] for x in top_n])
        capital_risky *= (1 + avg_ret_risky)
        ret_risky_list.append(avg_ret_risky)

        # ── Carteira COM filtro ───────────────────────────────────────────────
        # ✅ Redistribui capital pelos stocks SEM earnings (não dilui com 0.0)
        safe_stocks = [
            s for s in top_n
            if next_date_obj not in earnings_map.get(s["ticker"], set())
        ]

        if safe_stocks:
            avg_ret_safe = np.mean([s["return"] for s in safe_stocks])
        else:
            avg_ret_safe = 0.0  # Tudo em cash se todos têm earnings

        capital_safe *= (1 + avg_ret_safe)
        ret_safe_list.append(avg_ret_safe)

        equity_risky.append(capital_risky)
        equity_safe.append(capital_safe)
        dates_used.append(current_date)

    return {
        "dates":        [d.to_pydatetime() for d in dates_used],
        "equity_risky": equity_risky,
        "equity_safe":  equity_safe,
        "ret_risky":    ret_risky_list,
        "ret_safe":     ret_safe_list,
    }


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
    print(f"\n{'─'*38}\n  {m['label']}\n{'─'*38}")
    print(f"  Capital final:      ${m['final_cap']:>10,.2f}")
    print(f"  Retorno total:      {m['total_ret']*100:>+9.2f}%")
    print(f"  Retorno anualizado: {m['ann_ret']*100:>+9.2f}%")
    print(f"  Volatilidade (ann): {m['vol']*100:>9.2f}%")
    print(f"  Sharpe Ratio:       {m['sharpe']:>9.2f}")
    print(f"  Sortino Ratio:      {m['sortino']:>9.2f}")
    print(f"  Max Drawdown:       {m['max_dd']*100:>9.2f}%")
    print(f"  Win Rate:           {m['win_rate']*100:>9.1f}%")


def plot_results(res: dict, m_risky: dict, m_safe: dict):
    dates = res["dates"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Backtest Day & Night Strategy\nImpacto do Filtro de Earnings",
                 fontsize=15, fontweight="bold", y=0.98)

    ax1 = axes[0]
    ax1.plot(dates, res["equity_risky"], label="Sem filtro (c/ earnings)",
             color="#E74C3C", linewidth=1.8)
    ax1.plot(dates, res["equity_safe"], label="Com filtro (sem earnings)",
             color="#2ECC71", linewidth=1.8, linestyle="--")
    ax1.axhline(CAPITAL_INIT, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_title("Curva de Capital", fontweight="bold")
    ax1.set_ylabel("Capital ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax2 = axes[1]
    def calc_dd(equity):
        peak, out = equity[0], []
        for v in equity:
            if v > peak: peak = v
            out.append((v - peak) / peak * 100)
        return out

    ax2.fill_between(dates, calc_dd(res["equity_risky"]), 0,
                     alpha=0.4, color="#E74C3C", label="Sem filtro")
    ax2.fill_between(dates, calc_dd(res["equity_safe"]), 0,
                     alpha=0.4, color="#2ECC71", label="Com filtro")
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax3 = axes[2]
    ax3.axis("off")
    rows = [
        ["Capital Final",       f"${m_risky['final_cap']:,.0f}",         f"${m_safe['final_cap']:,.0f}"],
        ["Retorno Total",       f"{m_risky['total_ret']*100:+.2f}%",     f"{m_safe['total_ret']*100:+.2f}%"],
        ["Retorno Anualizado",  f"{m_risky['ann_ret']*100:+.2f}%",       f"{m_safe['ann_ret']*100:+.2f}%"],
        ["Volatilidade",        f"{m_risky['vol']*100:.2f}%",            f"{m_safe['vol']*100:.2f}%"],
        ["Sharpe Ratio",        f"{m_risky['sharpe']:.2f}",              f"{m_safe['sharpe']:.2f}"],
        ["Sortino Ratio",       f"{m_risky['sortino']:.2f}",             f"{m_safe['sortino']:.2f}"],
        ["Max Drawdown",        f"{m_risky['max_dd']*100:.2f}%",         f"{m_safe['max_dd']*100:.2f}%"],
        ["Win Rate",            f"{m_risky['win_rate']*100:.1f}%",       f"{m_safe['win_rate']*100:.1f}%"],
    ]
    tbl = ax3.table(cellText=rows,
                    colLabels=["Métrica", "Sem Filtro", "Com Filtro"],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F497D")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF3FB")
        cell.set_edgecolor("#CCCCCC")
    ax3.set_title("Resumo de Métricas", fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("backtest_earnings_results.png", dpi=150, bbox_inches="tight")
    print("\n  Gráfico guardado: backtest_earnings_results.png")
    plt.show()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  BACKTEST: Day & Night — Filtro de Earnings")
    print("="*50)

    print("\n[1/4] A obter tickers do S&P 500...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
    print(f"  {len(tickers)} tickers carregados.")

    print("\n[2/4] A descarregar preços OHLCV...")
    raw_data = yf.download(tickers, period=PERIOD, group_by="ticker",
                           auto_adjust=True, progress=True, threads=True)
    clean_data = build_clean_data(raw_data, tickers)
    print(f"  {len(clean_data)} tickers com dados válidos.")

    print("\n[3/4] A obter datas de earnings (yfinance)...")
    earnings_map = fetch_earnings_dates(list(clean_data.keys()))
    tickers_with_earnings = sum(1 for v in earnings_map.values() if v)
    print(f"  Earnings obtidos para {tickers_with_earnings}/{len(clean_data)} tickers.")

    print("\n[4/4] A simular backtest...")
    res = run_backtest(clean_data, earnings_map)

    m_risky = compute_metrics(res["equity_risky"], res["ret_risky"], "SEM FILTRO (compra com earnings)")
    m_safe  = compute_metrics(res["equity_safe"],  res["ret_safe"],  "COM FILTRO (evita earnings)")

    print_metrics(m_risky)
    print_metrics(m_safe)

    print(f"\n{'='*50}\n  VEREDICTO\n{'='*50}")
    if m_safe["sharpe"] > m_risky["sharpe"]:
        print(f"  ✅ FILTRAR EARNINGS VALE A PENA")
        print(f"     Sharpe com filtro é {m_safe['sharpe'] - m_risky['sharpe']:.2f} pts melhor.")
    else:
        print(f"  ❌ FILTRAR EARNINGS NÃO MELHORA")
        print(f"     Sharpe sem filtro é {m_risky['sharpe'] - m_safe['sharpe']:.2f} pts melhor.")

    print(f"\n  Diferença de retorno:  {(m_safe['total_ret'] - m_risky['total_ret'])*100:+.2f}%")
    print(f"  Diferença de drawdown: {(m_safe['max_dd'] - m_risky['max_dd'])*100:+.2f}%")
    print(f"{'='*50}\n")

    plot_results(res, m_risky, m_safe)