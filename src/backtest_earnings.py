"""
Backtest: Day & Night Strategy — Impacto de Comprar antes de Earnings
Compara carteira SEM filtro vs COM filtro de earnings (usando dados reais do yfinance)
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

# ─── CONFIGURAÇÃO ────────────────────────────────────────────────────────────
PERIOD        = "1y"
TOP_N         = 10
RISK_FREE_RATE = 0.0
CAPITAL_INIT  = 10_000.0

# Para teste rápido muda para True (usa só 80 tickers)
QUICK_MODE = False
# ─────────────────────────────────────────────────────────────────────────────


def get_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, headers=headers)
        resp.encoding = "utf-8"
        tables = pd.read_html(resp.text)
        for t in tables:
            if "Symbol" in t.columns:
                tickers = t["Symbol"].str.replace(".", "-", regex=False).tolist()
                return tickers
        print("  ERRO: tabela com coluna 'Symbol' nao encontrada.")
        return []
    except Exception as e:
        print(f"  ERRO ao obter tickers: {e}")
        return []


def fetch_earnings_dates(tickers: list) -> dict:
    """
    Obtém datas reais de earnings via yfinance.
    Retorna dict: {ticker: set(dates_of_earnings)}
    """
    earnings_map = {}
    total = len(tickers)
    for i, t in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Earnings: {i+1}/{total}...")
        try:
            ticker_obj = yf.Ticker(t)
            cal = ticker_obj.calendar
            dates = set()

            if cal is not None and not cal.empty:
                # calendar pode ser DataFrame com colunas = datas ou com index = métricas
                if hasattr(cal, 'columns'):
                    for col in cal.columns:
                        try:
                            d = pd.to_datetime(col).date()
                            dates.add(d)
                        except Exception:
                            pass
                if "Earnings Date" in cal.index:
                    for val in cal.loc["Earnings Date"]:
                        try:
                            d = pd.to_datetime(val).date()
                            dates.add(d)
                        except Exception:
                            pass

            # histórico de earnings
            hist_earnings = ticker_obj.earnings_dates
            if hist_earnings is not None and not hist_earnings.empty:
                for ed in hist_earnings.index:
                    try:
                        dates.add(pd.to_datetime(ed).date())
                    except Exception:
                        pass

            earnings_map[t] = dates
            time.sleep(0.05)   # respeitar rate limit

        except Exception:
            earnings_map[t] = set()

    return earnings_map


def build_clean_data(raw_data, tickers: list) -> dict:
    clean = {}
    for t in tickers:
        try:
            df = raw_data[t].copy()
        except Exception:
            continue
        try:
            df = df.dropna(subset=["Open", "Close"])
            df = df[(df["Open"] > 0) & (df["Close"] > 0)]
            df["NextOpen"] = df["Open"].shift(-1)
            df["Overnight_Ret"] = (df["NextOpen"] / df["Close"]) - 1
            df["Past_Overnight"] = (df["Open"] / df["Close"].shift(1)) - 1
            df["Momentum"] = (
                    (1 + df["Past_Overnight"])
                    .rolling(126)
                    .apply(np.prod, raw=True) - 1
            )
            df["DollarVol"] = df["Close"] * df["Volume"]
            if not df.empty:
                clean[t] = df
        except Exception:
            continue
    return clean


def run_backtest(clean_data: dict, earnings_map: dict, valid_dates) -> dict:
    capital_risky = CAPITAL_INIT
    capital_safe  = CAPITAL_INIT

    equity_risky  = []
    equity_safe   = []
    dates_used    = []

    daily_ret_risky_list = []
    daily_ret_safe_list  = []

    start_idx = 130

    for i in range(start_idx, len(valid_dates) - 1):
        current_date = valid_dates[i]
        next_date    = valid_dates[i + 1]  # dia em que vendemos (Open)

        # ── Ranking ──────────────────────────────────────────────────────────
        candidates = []
        for t, df in clean_data.items():
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if pd.isna(row["Momentum"]) or pd.isna(row["Overnight_Ret"]):
                continue
            # Filtro de liquidez: volume médio diário > $10M (20d)
            recent = df.loc[:current_date].tail(20)
            if recent["DollarVol"].mean() < 10_000_000:
                continue
            candidates.append({
                "ticker":   t,
                "momentum": row["Momentum"],
                "return":   row["Overnight_Ret"],
            })

        candidates.sort(key=lambda x: x["momentum"], reverse=True)
        top10 = candidates[:TOP_N]

        if not top10:
            equity_risky.append(capital_risky)
            equity_safe.append(capital_safe)
            dates_used.append(current_date)
            daily_ret_risky_list.append(0.0)
            daily_ret_safe_list.append(0.0)
            continue

        # ── Carteira SEM filtro ───────────────────────────────────────────────
        avg_ret_risky = np.mean([x["return"] for x in top10])
        capital_risky *= (1 + avg_ret_risky)
        daily_ret_risky_list.append(avg_ret_risky)

        # ── Carteira COM filtro (earnings reais) ──────────────────────────────
        next_date_obj = next_date.date() if hasattr(next_date, "date") else next_date
        safe_returns  = []
        skipped       = 0

        for stock in top10:
            t = stock["ticker"]
            earnings_dates = earnings_map.get(t, set())
            # Evitar se houver earnings amanhã (dia da venda = dia do report)
            has_earnings = next_date_obj in earnings_dates

            if has_earnings:
                skipped += 1
                safe_returns.append(0.0)   # slot fica em cash
            else:
                safe_returns.append(stock["return"])

        avg_ret_safe = np.mean(safe_returns)
        capital_safe *= (1 + avg_ret_safe)
        daily_ret_safe_list.append(avg_ret_safe)

        equity_risky.append(capital_risky)
        equity_safe.append(capital_safe)
        dates_used.append(current_date)

    return {
        "dates":            dates_used,
        "equity_risky":     equity_risky,
        "equity_safe":      equity_safe,
        "ret_risky":        daily_ret_risky_list,
        "ret_safe":         daily_ret_safe_list,
    }


def compute_metrics(equity: list, daily_rets: list, label: str) -> dict:
    total_ret  = (equity[-1] / equity[0]) - 1
    n_days     = len(daily_rets)
    ann_ret    = (1 + total_ret) ** (252 / n_days) - 1
    vol        = np.std(daily_rets) * np.sqrt(252)
    sharpe     = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0

    neg  = [r for r in daily_rets if r < 0]
    down_vol = np.std(neg) * np.sqrt(252) if neg else 0
    sortino  = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

    # Max Drawdown
    peak   = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    win_rate = sum(1 for r in daily_rets if r > 0) / n_days

    return {
        "label":      label,
        "total_ret":  total_ret,
        "ann_ret":    ann_ret,
        "vol":        vol,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "max_dd":     max_dd,
        "win_rate":   win_rate,
        "final_cap":  equity[-1],
    }


def print_metrics(m: dict):
    print(f"\n{'─'*38}")
    print(f"  {m['label']}")
    print(f"{'─'*38}")
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

    # ── 1. Equity Curves ─────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, res["equity_risky"], label="Sem filtro (c/ earnings)",
             color="#E74C3C", linewidth=1.8)
    ax1.plot(dates, res["equity_safe"],  label="Com filtro (sem earnings)",
             color="#2ECC71", linewidth=1.8, linestyle="--")
    ax1.axhline(CAPITAL_INIT, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_title("Curva de Capital", fontweight="bold")
    ax1.set_ylabel("Capital ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    def calc_dd_series(equity):
        peak = equity[0]
        dd_series = []
        for v in equity:
            if v > peak: peak = v
            dd_series.append((v - peak) / peak * 100)
        return dd_series

    dd_risky = calc_dd_series(res["equity_risky"])
    dd_safe  = calc_dd_series(res["equity_safe"])
    ax2.fill_between(dates, dd_risky, 0, alpha=0.4, color="#E74C3C", label="Sem filtro")
    ax2.fill_between(dates, dd_safe,  0, alpha=0.4, color="#2ECC71", label="Com filtro")
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── 3. Tabela de métricas ─────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.axis("off")
    metrics_labels = [
        "Capital Final", "Retorno Total", "Retorno Anualizado",
        "Volatilidade", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown", "Win Rate"
    ]
    risky_vals = [
        f"${m_risky['final_cap']:,.0f}",
        f"{m_risky['total_ret']*100:+.2f}%",
        f"{m_risky['ann_ret']*100:+.2f}%",
        f"{m_risky['vol']*100:.2f}%",
        f"{m_risky['sharpe']:.2f}",
        f"{m_risky['sortino']:.2f}",
        f"{m_risky['max_dd']*100:.2f}%",
        f"{m_risky['win_rate']*100:.1f}%",
    ]
    safe_vals = [
        f"${m_safe['final_cap']:,.0f}",
        f"{m_safe['total_ret']*100:+.2f}%",
        f"{m_safe['ann_ret']*100:+.2f}%",
        f"{m_safe['vol']*100:.2f}%",
        f"{m_safe['sharpe']:.2f}",
        f"{m_safe['sortino']:.2f}",
        f"{m_safe['max_dd']*100:.2f}%",
        f"{m_safe['win_rate']*100:.1f}%",
    ]

    table_data = [[l, r, s] for l, r, s in zip(metrics_labels, risky_vals, safe_vals)]
    tbl = ax3.table(
        cellText=table_data,
        colLabels=["Métrica", "Sem Filtro", "Com Filtro"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
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
    out = "backtest_earnings_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {out}")
    plt.show()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  BACKTEST: Day & Night — Filtro de Earnings")
    print("="*50)

    # 1. Tickers
    print("\n[1/4] A obter tickers do S&P 500...")
    tickers = get_sp500_tickers()
    if QUICK_MODE:
        tickers = tickers[:80]
        print(f"  QUICK MODE: usando {len(tickers)} tickers.")
    else:
        print(f"  {len(tickers)} tickers carregados.")

    # 2. Preços
    print("\n[2/4] A descarregar preços OHLCV...")
    raw_data = yf.download(
        tickers, period=PERIOD, group_by="ticker",
        auto_adjust=True, progress=True, threads=True
    )
    clean_data = build_clean_data(raw_data, tickers)
    print(f"  {len(clean_data)} tickers com dados válidos.")

    # 3. Earnings reais
    print("\n[3/4] A obter datas de earnings (yfinance)...")
    valid_tickers = list(clean_data.keys())
    earnings_map = fetch_earnings_dates(valid_tickers)
    tickers_with_earnings = sum(1 for v in earnings_map.values() if v)
    print(f"  Earnings obtidos para {tickers_with_earnings}/{len(valid_tickers)} tickers.")

    # 4. Backtest
    print("\n[4/4] A simular backtest...")
    valid_dates = raw_data.index
    res = run_backtest(clean_data, earnings_map, valid_dates)

    # ── Métricas ──────────────────────────────────────────────────────────────
    m_risky = compute_metrics(res["equity_risky"], res["ret_risky"], "SEM FILTRO (compra com earnings)")
    m_safe  = compute_metrics(res["equity_safe"],  res["ret_safe"],  "COM FILTRO (evita earnings)")

    print_metrics(m_risky)
    print_metrics(m_safe)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  VEREDICTO")
    print(f"{'='*50}")
    if m_safe["sharpe"] > m_risky["sharpe"]:
        diff = m_safe["sharpe"] - m_risky["sharpe"]
        print(f"  ✅ FILTRAR EARNINGS VALE A PENA")
        print(f"     Sharpe com filtro é {diff:.2f} pts melhor.")
    else:
        diff = m_risky["sharpe"] - m_safe["sharpe"]
        print(f"  ❌ FILTRAR EARNINGS NÃO MELHORA")
        print(f"     Sharpe sem filtro é {diff:.2f} pts melhor.")

    ret_diff = (m_safe["total_ret"] - m_risky["total_ret"]) * 100
    dd_diff  = (m_safe["max_dd"]   - m_risky["max_dd"])    * 100
    print(f"\n  Diferença de retorno:    {ret_diff:+.2f}%")
    print(f"  Diferença de drawdown:   {dd_diff:+.2f}%")
    print(f"{'='*50}\n")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    plot_results(res, m_risky, m_safe)