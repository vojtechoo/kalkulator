"""yfinance fallback data provider (~4 years of annual history)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from data.models import (
    AnalystEstimate,
    AnnualFinancials,
    CompanyData,
    InsiderTrade,
    InstitutionalHolder,
    NewsItem,
    PeerSummary,
)
from data.provider import DataProvider, DataProviderError, is_financial_or_reit

# Curated fallback peers for common sectors when FMP peers endpoint is unavailable.
SECTOR_FALLBACK_PEERS: dict[str, list[str]] = {
    "Technology": ["MSFT", "AAPL", "GOOGL", "ORCL", "ADBE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS"],
    "Consumer Cyclical": ["AMZN", "HD", "MCD", "NKE", "LOW"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Healthcare": ["JNJ", "UNH", "LLY", "ABBV", "MRK"],
    "Industrials": ["UNP", "HON", "CAT", "DE", "UPS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Financial Services": ["JPM", "BRK-B", "V", "MA", "BAC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "O", "SPG"],
    "Basic Materials": ["LIN", "SHW", "FCX", "APD", "ECL"],
    "Utilities": ["NEE", "SO", "DUK", "AEP", "SRE"],
}


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        if pd.isna(val):
            return None
        num = float(val)
        return num if num == num else None
    except (TypeError, ValueError):
        return None


def _df_lookup(df: Optional[pd.DataFrame], col: Any, candidates: tuple[str, ...]) -> Optional[float]:
    """Lookup the first matching row label in a yfinance statement DataFrame."""
    if df is None or df.empty or col not in df.columns:
        return None
    index_lower = {str(idx).strip().lower(): idx for idx in df.index}
    for name in candidates:
        key = name.strip().lower()
        if key in index_lower:
            val = _safe_float(df.loc[index_lower[key], col])
            if val is not None:
                return val
    return None


def _abs_or_none(val: Optional[float]) -> Optional[float]:
    return abs(val) if val is not None else None


class YFinanceDataProvider(DataProvider):
    """Fallback data provider powered by yfinance (typically ~4 years of annual statements)."""

    name = "yfinance (záložní zdroj, ~4 roky historie)"

    def _get_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        src = from_ccy.upper().strip()
        dst = to_ccy.upper().strip()
        if not src or not dst or src == dst:
            return 1.0
        try:
            fx_ticker = yf.Ticker(f"{src}{dst}=X")
            hist = fx_ticker.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                val = _safe_float(hist["Close"].dropna().iloc[-1])
                if val and val > 0:
                    return val
        except Exception:
            pass
        return None

    def get_risk_free_rate(self) -> tuple[Optional[float], str]:
        """Fetch 10-year US Treasury yield via Yahoo Finance (^TNX)."""
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="10d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                last_close = _safe_float(hist["Close"].dropna().iloc[-1])
                if last_close is not None and last_close > 0:
                    # ^TNX is quoted in percentage points (e.g., 4.35 -> 0.0435)
                    return last_close / 100.0, "10Y US Treasury (^TNX, Yahoo Finance)"
        except Exception:
            pass
        return None, ""

    def get_company_data(self, ticker: str) -> CompanyData:
        """Fetch fundamental data for a ticker from yfinance."""
        sym = ticker.strip().upper()
        warnings: list[str] = []

        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}
        except Exception as exc:
            raise DataProviderError(f"Nepodařilo se načíst ticker '{sym}' z yfinance: {exc}") from exc

        current_price = (
            _safe_float(info.get("currentPrice"))
            or _safe_float(info.get("regularMarketPrice"))
            or _safe_float(info.get("previousClose"))
        )
        if current_price is None:
            try:
                hist_5d = tk.history(period="5d")
                if hist_5d is not None and not hist_5d.empty:
                    current_price = _safe_float(hist_5d["Close"].dropna().iloc[-1])
            except Exception:
                pass

        if current_price is None and not info.get("shortName") and not info.get("longName"):
            raise DataProviderError(
                f"Ticker '{sym}' nebyl nalezen nebo neobsahuje tržní cenu. Zkontrolujte prosím symbol."
            )

        inc_df = tk.financials if tk.financials is not None else pd.DataFrame()
        bal_df = tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame()
        cf_df = tk.cashflow if tk.cashflow is not None else pd.DataFrame()

        if inc_df.empty and bal_df.empty and cf_df.empty:
            raise DataProviderError(f"Pro ticker '{sym}' nejsou v yfinance dostupné finanční výkazy.")

        quote_ccy = str(info.get("currency") or "USD").upper()
        reporting_ccy = str(info.get("financialCurrency") or quote_ccy).upper()

        fx_rate: Optional[float] = 1.0
        if reporting_ccy != quote_ccy:
            fx_rate = self._get_fx_rate(reporting_ccy, quote_ccy)
            if fx_rate is not None:
                warnings.append(
                    f"Společnost vykazuje v měně {reporting_ccy}, ale akcie se obchoduje v {quote_ccy}. "
                    f"Výkazy byly přepočteny kurzem 1 {reporting_ccy} = {fx_rate:.4f} {quote_ccy}."
                )
            else:
                warnings.append(
                    f"Pozor: výkazy jsou v měně {reporting_ccy}, cena akcie v {quote_ccy}, "
                    "kurz se nepodařilo získat."
                )

        scale = fx_rate if fx_rate is not None else 1.0

        def _conv(val: Optional[float]) -> Optional[float]:
            return val * scale if val is not None else None

        # Historical yearly average prices (12y) for computing 10-year historical multiples
        yearly_avg_price: dict[int, float] = {}
        try:
            hist_max = tk.history(period="12y", interval="1mo")
            if hist_max is not None and not hist_max.empty and "Close" in hist_max.columns:
                grouped = hist_max["Close"].dropna().groupby(hist_max.index.year).mean()
                yearly_avg_price = {int(yr): float(p) for yr, p in grouped.items() if pd.notna(p)}
        except Exception:
            pass

        # Collect columns indexed by fiscal year
        def _col_map(df: pd.DataFrame) -> dict[int, Any]:
            out: dict[int, Any] = {}
            if df is None or df.empty:
                return out
            for col in df.columns:
                try:
                    yr = pd.Timestamp(col).year
                    if yr not in out:
                        out[yr] = col
                except Exception:
                    continue
            return out

        inc_cols = _col_map(inc_df)
        bal_cols = _col_map(bal_df)
        cf_cols = _col_map(cf_df)
        years = sorted(set(inc_cols) | set(bal_cols) | set(cf_cols))

        annuals: list[AnnualFinancials] = []
        for yr in years:
            ic = inc_cols.get(yr)
            bc = bal_cols.get(yr)
            cc = cf_cols.get(yr)

            revenue = _conv(_df_lookup(inc_df, ic, ("Total Revenue", "Operating Revenue")))
            gross_profit = _conv(_df_lookup(inc_df, ic, ("Gross Profit",)))
            operating_income = _conv(
                _df_lookup(inc_df, ic, ("Operating Income", "EBIT", "Total Operating Income As Reported"))
            )
            interest_expense = _conv(
                _abs_or_none(
                    _df_lookup(
                        inc_df,
                        ic,
                        ("Interest Expense", "Interest Expense Non Operating", "Net Non Operating Interest Income Expense"),
                    )
                )
            )
            pretax = _conv(_df_lookup(inc_df, ic, ("Pretax Income",)))
            tax_exp = _conv(_df_lookup(inc_df, ic, ("Tax Provision",)))
            net_income = _conv(
                _df_lookup(
                    inc_df,
                    ic,
                    ("Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"),
                )
            )
            ebitda = _conv(_df_lookup(inc_df, ic, ("EBITDA", "Normalized EBITDA")))
            rd_expense = _conv(
                _abs_or_none(_df_lookup(inc_df, ic, ("Research And Development",)))
            )
            sga_expense = _conv(
                _abs_or_none(
                    _df_lookup(
                        inc_df,
                        ic,
                        (
                            "Selling General And Administration",
                            "General And Administrative Expense",
                        ),
                    )
                )
            )
            dep_amort = _conv(
                _abs_or_none(
                    _df_lookup(
                        cf_df,
                        cc,
                        (
                            "Depreciation And Amortization",
                            "Depreciation Amortization Depletion",
                            "Depreciation",
                        ),
                    )
                    or _df_lookup(
                        inc_df,
                        ic,
                        ("Reconciled Depreciation", "Depreciation And Amortization In Income Statement"),
                    )
                )
            )
            eps_dil = _conv(_df_lookup(inc_df, ic, ("Diluted EPS", "Basic EPS")))
            shares_dil = _df_lookup(
                inc_df, ic, ("Diluted Average Shares", "Basic Average Shares")
            ) or _df_lookup(bal_df, bc, ("Ordinary Shares Number", "Share Issued"))

            cash_st = _conv(
                _df_lookup(
                    bal_df,
                    bc,
                    (
                        "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents",
                    ),
                )
            )
            accounts_receivable = _conv(
                _df_lookup(bal_df, bc, ("Accounts Receivable", "Receivables"))
            )
            current_assets = _conv(_df_lookup(bal_df, bc, ("Current Assets", "Total Current Assets")))
            ppe_net = _conv(_df_lookup(bal_df, bc, ("Net PPE", "Gross PPE")))
            total_assets = _conv(_df_lookup(bal_df, bc, ("Total Assets",)))
            current_liabilities = _conv(
                _df_lookup(bal_df, bc, ("Current Liabilities", "Total Current Liabilities"))
            )
            total_liabilities = _conv(
                _df_lookup(
                    bal_df,
                    bc,
                    ("Total Liabilities Net Minority Interest", "Total Liabilities"),
                )
            )
            retained_earnings = _conv(_df_lookup(bal_df, bc, ("Retained Earnings",)))
            st_debt = _conv(
                _df_lookup(bal_df, bc, ("Current Debt", "Current Debt And Capital Lease Obligation"))
            )
            lt_debt = _conv(
                _df_lookup(
                    bal_df, bc, ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")
                )
            )
            total_debt = _conv(_df_lookup(bal_df, bc, ("Total Debt",)))
            if total_debt is None and (st_debt is not None or lt_debt is not None):
                total_debt = (st_debt or 0.0) + (lt_debt or 0.0)
            total_equity = _conv(
                _df_lookup(
                    bal_df,
                    bc,
                    (
                        "Stockholders Equity",
                        "Common Stock Equity",
                        "Total Equity Gross Minority Interest",
                    ),
                )
            )

            ocf = _conv(
                _df_lookup(
                    cf_df,
                    cc,
                    ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
                )
            )
            capex = _conv(_abs_or_none(_df_lookup(cf_df, cc, ("Capital Expenditure",))))
            fcf = _conv(_df_lookup(cf_df, cc, ("Free Cash Flow",)))
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf - capex

            sbc = _conv(_abs_or_none(_df_lookup(cf_df, cc, ("Stock Based Compensation",))))
            acq_raw = _df_lookup(
                cf_df, cc, ("Net Business Purchase And Sale", "Purchase Of Business")
            )
            acq = _conv(max(0.0, -acq_raw)) if acq_raw is not None else None
            debt_rep_raw = _df_lookup(
                cf_df, cc, ("Repayment Of Debt", "Long Term Debt Payments")
            )
            debt_rep = _conv(_abs_or_none(debt_rep_raw)) if debt_rep_raw is not None else None
            div_raw = _df_lookup(
                cf_df, cc, ("Cash Dividends Paid", "Common Stock Dividend Paid")
            )
            div_paid = _conv(_abs_or_none(div_raw)) if div_raw is not None else None
            buyback_raw = _df_lookup(
                cf_df, cc, ("Repurchase Of Capital Stock", "Common Stock Payments")
            )
            buybacks = _conv(_abs_or_none(buyback_raw)) if buyback_raw is not None else None

            # Estimate yearly multiples from yearly average stock price & diluted shares
            avg_p = yearly_avg_price.get(yr)
            mcap_yr = (avg_p * shares_dil) if (avg_p and shares_dil and shares_dil > 0) else None
            pe_yr = (mcap_yr / net_income) if (mcap_yr and net_income and net_income > 0) else None
            pfcf_yr = (mcap_yr / fcf) if (mcap_yr and fcf and fcf > 0) else None
            ev_yr = (
                (mcap_yr + (total_debt or 0.0) - (cash_st or 0.0))
                if mcap_yr is not None
                else None
            )
            ev_ebitda_yr = (ev_yr / ebitda) if (ev_yr and ebitda and ebitda > 0) else None

            date_val = ic or bc or cc
            date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d") if date_val is not None else f"{yr}-12-31"

            annuals.append(
                AnnualFinancials(
                    year=yr,
                    date=date_str,
                    currency=quote_ccy if fx_rate is not None else reporting_ccy,
                    revenue=revenue,
                    gross_profit=gross_profit,
                    operating_income=operating_income,
                    interest_expense=interest_expense,
                    income_before_tax=pretax,
                    income_tax_expense=tax_exp,
                    net_income=net_income,
                    ebitda=ebitda,
                    rd_expense=rd_expense,
                    sga_expense=sga_expense,
                    depreciation_amortization=dep_amort,
                    eps_diluted=eps_dil,
                    weighted_avg_shares_diluted=shares_dil,
                    cash_and_short_term=cash_st,
                    accounts_receivable=accounts_receivable,
                    current_assets=current_assets,
                    ppe_net=ppe_net,
                    total_assets=total_assets,
                    current_liabilities=current_liabilities,
                    total_liabilities=total_liabilities,
                    retained_earnings=retained_earnings,
                    short_term_debt=st_debt,
                    long_term_debt=lt_debt,
                    total_debt=total_debt,
                    total_equity=total_equity,
                    operating_cash_flow=ocf,
                    capex=capex,
                    free_cash_flow=fcf,
                    stock_based_compensation=sbc,
                    acquisitions_net=acq,
                    debt_repayment=debt_rep,
                    dividends_paid=div_paid,
                    share_buybacks=buybacks,
                    pe_ratio=pe_yr,
                    pfcf_ratio=pfcf_yr,
                    ev_to_ebitda=ev_ebitda_yr,
                    market_cap_year=mcap_yr,
                    avg_share_price=avg_p,
                )
            )

        # Enrich older fiscal years (back to 10+ years) from official free SEC EDGAR 10-K XBRL data
        provider_label = "yfinance"
        if len(annuals) < 9 and reporting_ccy == "USD":
            try:
                from data.sec_edgar import fetch_sec_historical_annuals

                sec_annuals = fetch_sec_historical_annuals(
                    ticker=sym,
                    yearly_avg_price=yearly_avg_price,
                    max_years=12,
                    currency=quote_ccy if fx_rate is not None else reporting_ccy,
                )
                if sec_annuals:
                    existing_years = {a.year for a in annuals}
                    older_sec = [a for a in sec_annuals if a.year not in existing_years]
                    if older_sec:
                        annuals = sorted(older_sec + annuals, key=lambda x: x.year)[-12:]
                        provider_label = f"yfinance + SEC EDGAR ({len(annuals)} let)"
            except Exception:
                pass

        # Quarterly TTM values
        ttm_fcf = ttm_sbc = ttm_ocf = ttm_capex = None
        try:
            qcf = tk.quarterly_cashflow
            if qcf is not None and not qcf.empty and len(qcf.columns) >= 4:
                cols4 = list(qcf.columns)[:4]
                ocfs = [
                    _df_lookup(qcf, c, ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"))
                    for c in cols4
                ]
                capexs = [_abs_or_none(_df_lookup(qcf, c, ("Capital Expenditure",))) for c in cols4]
                fcfs = [_df_lookup(qcf, c, ("Free Cash Flow",)) for c in cols4]
                sbcs = [_abs_or_none(_df_lookup(qcf, c, ("Stock Based Compensation",))) for c in cols4]
                if all(v is not None for v in ocfs):
                    ttm_ocf = _conv(sum(v for v in ocfs if v is not None))
                if all(v is not None for v in capexs):
                    ttm_capex = _conv(sum(v for v in capexs if v is not None))
                if all(v is not None for v in fcfs):
                    ttm_fcf = _conv(sum(v for v in fcfs if v is not None))
                elif ttm_ocf is not None and ttm_capex is not None:
                    ttm_fcf = ttm_ocf - ttm_capex
                if ttm_capex is None and ttm_ocf is not None and ttm_fcf is not None and ttm_ocf >= ttm_fcf:
                    ttm_capex = ttm_ocf - ttm_fcf
                if all(v is not None for v in sbcs):
                    ttm_sbc = _conv(sum(v for v in sbcs if v is not None))
        except Exception:
            pass

        ttm_rev = _conv(_safe_float(info.get("totalRevenue")))
        ttm_ni = _conv(_safe_float(info.get("netIncomeToCommon")))
        ttm_ebitda = _conv(_safe_float(info.get("ebitda")))

        latest = annuals[-1] if annuals else None
        latest_cash = _conv(_safe_float(info.get("totalCash"))) or (
            latest.cash_and_short_term if latest else None
        )
        latest_debt = _conv(_safe_float(info.get("totalDebt"))) or (
            latest.total_debt if latest else None
        )
        latest_net_debt = (
            (latest_debt - latest_cash)
            if (latest_debt is not None and latest_cash is not None)
            else None
        )

        market_cap = _safe_float(info.get("marketCap"))
        shares_current = (
            (latest.weighted_avg_shares_diluted if latest and latest.weighted_avg_shares_diluted else None)
            or _safe_float(info.get("sharesOutstanding"))
        )
        if shares_current is None and market_cap and current_price and current_price > 0:
            shares_current = market_cap / current_price

        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "")
        peers = [p for p in SECTOR_FALLBACK_PEERS.get(sector, []) if p != sym][:4]

        insider_trades = self._parse_insiders(tk)
        institutional_holders = self._parse_institutional_holders(tk)
        analyst_estimates = self._parse_estimates(tk, scale=scale)
        news = self._parse_news(tk)
        rf_rate, rf_source = self.get_risk_free_rate()

        return CompanyData(
            ticker=sym,
            name=str(info.get("longName") or info.get("shortName") or sym),
            exchange=str(info.get("exchange") or ""),
            sector=sector,
            industry=industry,
            country=str(info.get("country") or ""),
            quote_currency=quote_ccy,
            reporting_currency=reporting_ccy,
            fx_rate_reporting_to_quote=fx_rate,
            current_price=current_price,
            market_cap=market_cap,
            beta=_safe_float(info.get("beta")),
            diluted_shares_current=shares_current,
            ttm_fcf=ttm_fcf,
            ttm_sbc=ttm_sbc,
            ttm_capex=ttm_capex,
            ttm_revenue=ttm_rev,
            ttm_net_income=ttm_ni,
            ttm_ebitda=ttm_ebitda,
            ttm_ebit=latest.operating_income if latest else None,
            ttm_ocf=ttm_ocf,
            latest_cash=latest_cash,
            latest_total_debt=latest_debt,
            latest_net_debt=latest_net_debt,
            risk_free_rate=rf_rate,
            risk_free_source=rf_source,
            annuals=annuals,
            peers=peers,
            insider_trades=insider_trades,
            institutional_holders=institutional_holders,
            analyst_estimates=analyst_estimates,
            news=news,
            provider_name=provider_label,
            is_limited_history=len(annuals) < 6,
            is_financial_or_reit=is_financial_or_reit(sector, industry),
            warnings=warnings,
            description=str(info.get("longBusinessSummary") or ""),
        )

    def _parse_institutional_holders(self, tk: yf.Ticker) -> list[InstitutionalHolder]:
        holders: list[InstitutionalHolder] = []
        try:
            df = tk.institutional_holders
        except Exception:
            return holders
        if df is None or df.empty:
            return holders
        for _, row in df.head(10).iterrows():
            name = str(row.get("Holder") or "").strip()
            if not name:
                continue
            pct = _safe_float(row.get("pctHeld") or row.get("% Out"))
            if pct is not None and pct > 1.0:
                pct = pct / 100.0
            raw_dt = row.get("Date Reported")
            try:
                dt_str = pd.Timestamp(raw_dt).strftime("%Y-%m-%d") if raw_dt is not None else ""
            except Exception:
                dt_str = ""
            holders.append(
                InstitutionalHolder(
                    holder_name=name,
                    pct_held=pct,
                    shares=_safe_float(row.get("Shares")),
                    value=_safe_float(row.get("Value")),
                    date_reported=dt_str,
                )
            )
        return holders

    def _parse_insiders(self, tk: yf.Ticker) -> list[InsiderTrade]:
        trades: list[InsiderTrade] = []
        try:
            df = tk.insider_transactions
        except Exception:
            return trades
        if df is None or df.empty:
            return trades

        cutoff = (datetime.now(timezone.utc) - timedelta(days=730)).date().isoformat()
        for _, row in df.iterrows():
            raw_date = row.get("Start Date") or row.get("Date")
            try:
                t_date = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
            except Exception:
                t_date = ""
            if t_date and t_date < cutoff:
                continue

            text = str(row.get("Text") or row.get("Transaction") or "").strip()
            text_l = text.lower()
            shares = _safe_float(row.get("Shares"))
            value = _safe_float(row.get("Value"))
            price = (value / abs(shares)) if (value and shares and abs(shares) > 0) else None

            # Identify open-market purchase/sale vs awards/option exercises/tax
            excluded_kw = ("option", "exercise", "grant", "award", "gift", "conversion", "tax", "withhold")
            is_excluded = any(k in text_l for k in excluded_kw)
            is_buy = ("purchase" in text_l or "buy" in text_l) and not is_excluded
            is_sell = ("sale" in text_l or "sold" in text_l) and not is_excluded

            direction = "BUY" if is_buy else ("SELL" if is_sell else "OTHER")
            trades.append(
                InsiderTrade(
                    date=t_date,
                    insider_name=str(row.get("Insider") or "Neznámý"),
                    role=str(row.get("Position") or ""),
                    transaction_type=text or "Neznámý typ",
                    shares=abs(shares) if shares is not None else None,
                    price=price,
                    value=abs(value) if value is not None else None,
                    is_open_market=is_buy or is_sell,
                    direction=direction,
                    link=str(row.get("URL")) if row.get("URL") else None,
                )
            )
        return trades

    def _parse_estimates(self, tk: yf.Ticker, scale: float = 1.0) -> list[AnalystEstimate]:
        estimates: list[AnalystEstimate] = []
        current_year = datetime.now(timezone.utc).year
        try:
            rev_df = tk.revenue_estimate
            eps_df = tk.earnings_estimate
        except Exception:
            return estimates

        period_offsets = [("0y", current_year), ("+1y", current_year + 1), ("+2y", current_year + 2)]
        for period_key, yr in period_offsets:
            rev_avg = rev_low = rev_high = eps_avg = eps_low = eps_high = None
            n_rev = n_eps = None
            if rev_df is not None and not rev_df.empty and period_key in rev_df.index:
                r = rev_df.loc[period_key]
                rev_avg = _safe_float(r.get("avg"))
                rev_low = _safe_float(r.get("low"))
                rev_high = _safe_float(r.get("high"))
                n_r = _safe_float(r.get("numberOfAnalysts"))
                n_rev = int(n_r) if n_r is not None else None
            if eps_df is not None and not eps_df.empty and period_key in eps_df.index:
                e = eps_df.loc[period_key]
                eps_avg = _safe_float(e.get("avg"))
                eps_low = _safe_float(e.get("low"))
                eps_high = _safe_float(e.get("high"))
                n_e = _safe_float(e.get("numberOfAnalysts"))
                n_eps = int(n_e) if n_e is not None else None

            if rev_avg is not None or eps_avg is not None:
                estimates.append(
                    AnalystEstimate(
                        year=yr,
                        date=f"{yr}-12-31",
                        estimated_revenue_avg=rev_avg * scale if rev_avg is not None else None,
                        estimated_revenue_low=rev_low * scale if rev_low is not None else None,
                        estimated_revenue_high=rev_high * scale if rev_high is not None else None,
                        estimated_eps_avg=eps_avg * scale if eps_avg is not None else None,
                        estimated_eps_low=eps_low * scale if eps_low is not None else None,
                        estimated_eps_high=eps_high * scale if eps_high is not None else None,
                        num_analysts_revenue=n_rev,
                        num_analysts_eps=n_eps,
                    )
                )
        return estimates

    def _parse_news(self, tk: yf.Ticker) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            raw_news = tk.news or []
        except Exception:
            return items

        for entry in raw_news:
            content = entry.get("content") if isinstance(entry, dict) and isinstance(entry.get("content"), dict) else entry
            if not isinstance(content, dict):
                continue
            title = str(content.get("title") or "").strip()
            if not title:
                continue
            pub_date = ""
            if content.get("pubDate"):
                pub_date = str(content["pubDate"])[:10]
            elif entry.get("providerPublishTime"):
                try:
                    pub_date = datetime.fromtimestamp(
                        int(entry["providerPublishTime"]), tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                except Exception:
                    pub_date = ""

            provider = content.get("provider")
            source = (
                str(provider.get("displayName"))
                if isinstance(provider, dict) and provider.get("displayName")
                else str(content.get("publisher") or entry.get("publisher") or "Yahoo Finance")
            )
            canonical = content.get("canonicalUrl") or content.get("clickThroughUrl")
            url = (
                str(canonical.get("url"))
                if isinstance(canonical, dict) and canonical.get("url")
                else str(content.get("link") or entry.get("link") or "")
            )
            items.append(
                NewsItem(
                    published_date=pub_date,
                    title=title,
                    source=source,
                    url=url,
                    text=str(content.get("summary") or content.get("description") or ""),
                )
            )
        return items

    def get_peer_summary(self, ticker: str) -> Optional[PeerSummary]:
        """Fetch compact valuation and quality metrics for a peer via yfinance."""
        sym = ticker.strip().upper()
        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}
            if not info.get("shortName") and not info.get("longName") and not info.get("marketCap"):
                return None

            mcap = _safe_float(info.get("marketCap"))
            fcf = _safe_float(info.get("freeCashflow"))
            pe = _safe_float(info.get("trailingPE"))
            pfcf = (mcap / fcf) if (mcap and fcf and fcf > 0) else None
            ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
            gross_m = _safe_float(info.get("grossMargins"))
            op_m = _safe_float(info.get("operatingMargins"))

            # Approximate ROIC and 3Y revenue CAGR from annual statements
            roic = rev_cagr = None
            inc = tk.financials
            bal = tk.balance_sheet
            if inc is not None and not inc.empty:
                cols = list(inc.columns)
                revs = [_df_lookup(inc, c, ("Total Revenue", "Operating Revenue")) for c in cols]
                valid_revs = [r for r in revs if r and r > 0]
                if len(valid_revs) >= 4:
                    rev_cagr = (valid_revs[0] / valid_revs[3]) ** (1.0 / 3.0) - 1.0
                elif len(valid_revs) >= 3:
                    rev_cagr = (valid_revs[0] / valid_revs[2]) ** (1.0 / 2.0) - 1.0

                if bal is not None and not bal.empty:
                    ebit = _df_lookup(inc, cols[0], ("Operating Income", "EBIT"))
                    pretax = _df_lookup(inc, cols[0], ("Pretax Income",))
                    tax = _df_lookup(inc, cols[0], ("Tax Provision",))
                    tax_rate = (
                        min(max(tax / pretax, 0.0), 0.40)
                        if (tax is not None and pretax and pretax > 0)
                        else 0.21
                    )
                    bcol = bal.columns[0]
                    debt = _df_lookup(bal, bcol, ("Total Debt",)) or 0.0
                    eq = _df_lookup(bal, bcol, ("Stockholders Equity", "Common Stock Equity")) or 0.0
                    cash = _df_lookup(
                        bal,
                        bcol,
                        ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"),
                    ) or 0.0
                    ic = debt + eq - cash
                    if ebit is not None and ic > 0:
                        roic = (ebit * (1.0 - tax_rate)) / ic

            return PeerSummary(
                ticker=sym,
                name=str(info.get("shortName") or info.get("longName") or sym),
                currency=str(info.get("currency") or "USD").upper(),
                market_cap=mcap,
                pe_ratio=pe if (pe and 0 < pe < 300) else None,
                pfcf_ratio=pfcf if (pfcf and 0 < pfcf < 300) else None,
                ev_to_ebitda=ev_ebitda if (ev_ebitda and 0 < ev_ebitda < 200) else None,
                roic=roic,
                gross_margin=gross_m,
                operating_margin=op_m,
                revenue_cagr_3y=rev_cagr,
            )
        except Exception:
            return None
