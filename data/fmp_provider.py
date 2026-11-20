"""Financial Modeling Prep (FMP) API data provider (primary source, 10+ years history)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from data.models import (
    AnalystEstimate,
    AnnualFinancials,
    CompanyData,
    InsiderTrade,
    NewsItem,
    PeerSummary,
)
from data.provider import DataProvider, DataProviderError, RateLimitError, is_financial_or_reit

FMP_BASE_URL = "https://financialmodelingprep.com"
REQUEST_TIMEOUT = 15


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        num = float(value)
        if num != num:  # NaN check
            return None
        return num
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    num = _safe_float(value)
    return int(round(num)) if num is not None else None


def _abs_or_none(value: Optional[float]) -> Optional[float]:
    return abs(value) if value is not None else None


def _extract_year(record: dict[str, Any]) -> Optional[int]:
    cal = record.get("calendarYear") or record.get("fiscalYear")
    if cal:
        try:
            return int(str(cal)[:4])
        except ValueError:
            pass
    date_str = str(record.get("date", ""))
    if len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None


class FMPDataProvider(DataProvider):
    """Primary data provider powered by the Financial Modeling Prep REST API.

    Uses FMP `/stable/*` endpoints first and transparently falls back to legacy
    `/api/v3/*` and `/api/v4/*` routes when required by key plan compatibility.
    """

    name = "Financial Modeling Prep (FMP)"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json", "User-Agent": "ValueAnalyzer/2.0"})

    def _get_json(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        query = dict(params or {})
        query["apikey"] = self.api_key
        url = f"{FMP_BASE_URL}{path}"
        try:
            resp = self._session.get(url, params=query, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise DataProviderError(f"Chyba připojení k FMP API: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("HTTP 429 – překročen limit FMP API")
        if resp.status_code in (401, 403):
            raise DataProviderError(f"HTTP {resp.status_code} – nedostupný endpoint nebo neplatný klíč")
        if resp.status_code == 404:
            raise DataProviderError("HTTP 404 – endpoint nenalezen")
        if resp.status_code >= 400:
            raise DataProviderError(f"FMP API vrátilo chybu HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataProviderError("Neplatná JSON odpověď z FMP API") from exc

        if isinstance(payload, dict) and payload.get("Error Message"):
            msg = str(payload["Error Message"])
            if "limit" in msg.lower() or "upgrade" in msg.lower():
                raise RateLimitError(msg)
            raise DataProviderError(msg)
        return payload

    def _get_with_fallback(
        self,
        stable_path: str,
        stable_params: dict[str, Any],
        legacy_path: str,
        legacy_params: Optional[dict[str, Any]] = None,
        optional: bool = False,
    ) -> Any:
        """Query /stable endpoint first, then fallback to legacy /api/v3 or /api/v4."""
        try:
            data = self._get_json(stable_path, stable_params)
            if data:
                return data
        except RateLimitError:
            if optional:
                return []
            raise
        except DataProviderError:
            pass

        try:
            return self._get_json(legacy_path, legacy_params or {})
        except RateLimitError:
            if optional:
                return []
            raise
        except DataProviderError:
            if optional:
                return []
            raise

    def _get_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        """Fetch FX rate (1 unit of from_ccy in to_ccy) via FMP forex quote."""
        src = from_ccy.upper().strip()
        dst = to_ccy.upper().strip()
        if not src or not dst or src == dst:
            return 1.0
        pair = f"{src}{dst}"
        payload = self._get_with_fallback(
            "/stable/quote",
            {"symbol": pair},
            f"/api/v3/quote/{pair}",
            optional=True,
        )
        if isinstance(payload, list) and payload:
            rate = _safe_float(payload[0].get("price"))
            if rate and rate > 0:
                return rate
        return None

    def get_risk_free_rate(self) -> tuple[Optional[float], str]:
        """Fetch latest 10-year US Treasury yield from FMP treasury endpoint."""
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=30)
        payload = self._get_with_fallback(
            "/stable/treasury-rates",
            {"from": start.isoformat(), "to": today.isoformat()},
            "/api/v4/treasury",
            {"from": start.isoformat(), "to": today.isoformat()},
            optional=True,
        )
        if isinstance(payload, list):
            for row in payload:
                y10 = _safe_float(row.get("year10"))
                if y10 is not None and y10 > 0:
                    return y10 / 100.0, f"10Y US Treasury (FMP, {row.get('date', 'aktuální')})"
        return None, ""

    def get_company_data(self, ticker: str) -> CompanyData:
        """Fetch complete 10+ year company dataset from FMP API."""
        sym = ticker.strip().upper()
        warnings: list[str] = []

        profile_list = self._get_with_fallback(
            "/stable/profile",
            {"symbol": sym},
            f"/api/v3/profile/{sym}",
        )
        if not isinstance(profile_list, list) or not profile_list:
            raise DataProviderError(f"Ticker '{sym}' nebyl nalezen v FMP.")
        profile = profile_list[0]

        quote_list = self._get_with_fallback(
            "/stable/quote",
            {"symbol": sym},
            f"/api/v3/quote/{sym}",
            optional=True,
        )
        quote = quote_list[0] if isinstance(quote_list, list) and quote_list else {}

        income_rows = self._get_with_fallback(
            "/stable/income-statement",
            {"symbol": sym, "period": "annual", "limit": 12},
            f"/api/v3/income-statement/{sym}",
            {"period": "annual", "limit": 12},
        )
        balance_rows = self._get_with_fallback(
            "/stable/balance-sheet-statement",
            {"symbol": sym, "period": "annual", "limit": 12},
            f"/api/v3/balance-sheet-statement/{sym}",
            {"period": "annual", "limit": 12},
        )
        cashflow_rows = self._get_with_fallback(
            "/stable/cash-flow-statement",
            {"symbol": sym, "period": "annual", "limit": 12},
            f"/api/v3/cash-flow-statement/{sym}",
            {"period": "annual", "limit": 12},
        )

        if not income_rows or not balance_rows or not cashflow_rows:
            raise DataProviderError(f"Chybí finanční výkazy pro ticker '{sym}' v FMP.")

        key_metrics_rows = self._get_with_fallback(
            "/stable/key-metrics",
            {"symbol": sym, "period": "annual", "limit": 12},
            f"/api/v3/key-metrics/{sym}",
            {"period": "annual", "limit": 12},
            optional=True,
        )
        ratios_rows = self._get_with_fallback(
            "/stable/ratios",
            {"symbol": sym, "period": "annual", "limit": 12},
            f"/api/v3/ratios/{sym}",
            {"period": "annual", "limit": 12},
            optional=True,
        )
        quarter_cf_rows = self._get_with_fallback(
            "/stable/cash-flow-statement",
            {"symbol": sym, "period": "quarter", "limit": 4},
            f"/api/v3/cash-flow-statement/{sym}",
            {"period": "quarter", "limit": 4},
            optional=True,
        )
        quarter_is_rows = self._get_with_fallback(
            "/stable/income-statement",
            {"symbol": sym, "period": "quarter", "limit": 4},
            f"/api/v3/income-statement/{sym}",
            {"period": "quarter", "limit": 4},
            optional=True,
        )

        quote_ccy = str(profile.get("currency") or "USD").upper()
        reporting_ccy = str(
            (income_rows[0].get("reportedCurrency") if income_rows else None) or quote_ccy
        ).upper()

        fx_rate: Optional[float] = 1.0
        if reporting_ccy != quote_ccy:
            fx_rate = self._get_fx_rate(reporting_ccy, quote_ccy)
            if fx_rate is not None:
                warnings.append(
                    f"Společnost vykazuje v měně {reporting_ccy}, ale akcie se obchoduje v {quote_ccy}. "
                    f"Výkazy pro ocenění byly přepočteny kurzem 1 {reporting_ccy} = {fx_rate:.4f} {quote_ccy}."
                )
            else:
                warnings.append(
                    f"Pozor: výkazy jsou v měně {reporting_ccy}, zatímco cena akcie je v {quote_ccy}, "
                    "a směnný kurz nebyl dostupný. Výpočty na akcii mohou být zkreslené."
                )

        scale = fx_rate if fx_rate is not None else 1.0

        def _conv(val: Optional[float]) -> Optional[float]:
            return val * scale if val is not None else None

        inc_by_yr = {_extract_year(r): r for r in income_rows if _extract_year(r) is not None}
        bal_by_yr = {_extract_year(r): r for r in balance_rows if _extract_year(r) is not None}
        cf_by_yr = {_extract_year(r): r for r in cashflow_rows if _extract_year(r) is not None}
        km_by_yr = {
            _extract_year(r): r
            for r in (key_metrics_rows if isinstance(key_metrics_rows, list) else [])
            if _extract_year(r) is not None
        }
        rat_by_yr = {
            _extract_year(r): r
            for r in (ratios_rows if isinstance(ratios_rows, list) else [])
            if _extract_year(r) is not None
        }

        common_years = sorted(set(inc_by_yr) | set(bal_by_yr) | set(cf_by_yr))
        annuals: list[AnnualFinancials] = []
        for yr in common_years[-12:]:
            inc = inc_by_yr.get(yr, {})
            bal = bal_by_yr.get(yr, {})
            cf = cf_by_yr.get(yr, {})
            km = km_by_yr.get(yr, {})
            rat = rat_by_yr.get(yr, {})

            ocf = _conv(_safe_float(cf.get("operatingCashFlow")))
            capex_raw = _safe_float(cf.get("capitalExpenditure"))
            capex = _conv(_abs_or_none(capex_raw))
            fcf = _conv(_safe_float(cf.get("freeCashFlow")))
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf - capex

            st_debt = _conv(_safe_float(bal.get("shortTermDebt")))
            lt_debt = _conv(_safe_float(bal.get("longTermDebt")))
            tot_debt = _conv(_safe_float(bal.get("totalDebt")))
            if tot_debt is None and (st_debt is not None or lt_debt is not None):
                tot_debt = (st_debt or 0.0) + (lt_debt or 0.0)

            cash_st = _conv(
                _safe_float(bal.get("cashAndShortTermInvestments"))
                or _safe_float(bal.get("cashAndCashEquivalents"))
            )
            shares_dil = _safe_float(inc.get("weightedAverageShsOutDil")) or _safe_float(
                inc.get("weightedAverageShsOut")
            )

            # Outflows stored as positive magnitudes
            acq_raw = _safe_float(cf.get("acquisitionsNet"))
            acq_outflow = _conv(max(0.0, -acq_raw)) if acq_raw is not None else None
            # Some FMP versions report debtRepayment / dividendsPaid / commonStockRepurchased as negative cash outflows
            debt_rep_raw = _safe_float(cf.get("debtRepayment"))
            debt_rep = _conv(max(0.0, -debt_rep_raw)) if debt_rep_raw is not None else None
            if debt_rep == 0.0 and debt_rep_raw is not None and debt_rep_raw > 0:
                debt_rep = _conv(debt_rep_raw)

            div_raw = _safe_float(cf.get("dividendsPaid"))
            div_paid = _conv(_abs_or_none(div_raw)) if div_raw is not None else None

            buyback_raw = _safe_float(cf.get("commonStockRepurchased"))
            buybacks = _conv(_abs_or_none(buyback_raw)) if buyback_raw is not None else None

            pe_ratio = _safe_float(km.get("peRatio")) or _safe_float(rat.get("priceEarningsRatio"))
            pfcf_ratio = _safe_float(km.get("pfcfRatio")) or _safe_float(
                rat.get("priceToFreeCashFlowsRatio")
            )
            ev_ebitda = _safe_float(km.get("enterpriseValueOverEBITDA")) or _safe_float(
                km.get("evToEBITDA")
            )
            mcap_yr = _safe_float(km.get("marketCap"))
            avg_price = (
                (mcap_yr / shares_dil) if (mcap_yr and shares_dil and shares_dil > 0) else None
            )

            annuals.append(
                AnnualFinancials(
                    year=yr,
                    date=str(inc.get("date") or bal.get("date") or cf.get("date") or f"{yr}-12-31"),
                    currency=quote_ccy if fx_rate is not None else reporting_ccy,
                    revenue=_conv(_safe_float(inc.get("revenue"))),
                    gross_profit=_conv(_safe_float(inc.get("grossProfit"))),
                    operating_income=_conv(_safe_float(inc.get("operatingIncome"))),
                    interest_expense=_conv(_abs_or_none(_safe_float(inc.get("interestExpense")))),
                    income_before_tax=_conv(_safe_float(inc.get("incomeBeforeTax"))),
                    income_tax_expense=_conv(_safe_float(inc.get("incomeTaxExpense"))),
                    net_income=_conv(_safe_float(inc.get("netIncome"))),
                    ebitda=_conv(_safe_float(inc.get("ebitda"))),
                    rd_expense=_conv(_abs_or_none(_safe_float(inc.get("researchAndDevelopmentExpenses")))),
                    sga_expense=_conv(
                        _abs_or_none(
                            _safe_float(inc.get("sellingGeneralAndAdministrativeExpenses"))
                            or _safe_float(inc.get("generalAndAdministrativeExpenses"))
                        )
                    ),
                    depreciation_amortization=_conv(
                        _abs_or_none(
                            _safe_float(cf.get("depreciationAndAmortization"))
                            or _safe_float(inc.get("depreciationAndAmortization"))
                        )
                    ),
                    eps_diluted=_conv(_safe_float(inc.get("epsdiluted") or inc.get("epsDiluted") or inc.get("eps"))),
                    weighted_avg_shares_diluted=shares_dil,
                    cash_and_short_term=cash_st,
                    accounts_receivable=_conv(_safe_float(bal.get("netReceivables"))),
                    current_assets=_conv(_safe_float(bal.get("totalCurrentAssets"))),
                    ppe_net=_conv(_safe_float(bal.get("propertyPlantEquipmentNet"))),
                    total_assets=_conv(_safe_float(bal.get("totalAssets"))),
                    current_liabilities=_conv(_safe_float(bal.get("totalCurrentLiabilities"))),
                    total_liabilities=_conv(_safe_float(bal.get("totalLiabilities"))),
                    retained_earnings=_conv(_safe_float(bal.get("retainedEarnings"))),
                    short_term_debt=st_debt,
                    long_term_debt=lt_debt,
                    total_debt=tot_debt,
                    total_equity=_conv(_safe_float(bal.get("totalStockholdersEquity") or bal.get("totalEquity"))),
                    operating_cash_flow=ocf,
                    capex=capex,
                    free_cash_flow=fcf,
                    stock_based_compensation=_conv(
                        _abs_or_none(_safe_float(cf.get("stockBasedCompensation")))
                    ),
                    acquisitions_net=acq_outflow,
                    debt_repayment=debt_rep,
                    dividends_paid=div_paid,
                    share_buybacks=buybacks,
                    pe_ratio=pe_ratio,
                    pfcf_ratio=pfcf_ratio,
                    ev_to_ebitda=ev_ebitda,
                    market_cap_year=mcap_yr,
                    avg_share_price=avg_price,
                )
            )

        # Compute TTM figures when 4 recent quarters exist
        ttm_fcf = ttm_sbc = ttm_ocf = ttm_capex = None
        if isinstance(quarter_cf_rows, list) and len(quarter_cf_rows) >= 4:
            q4 = quarter_cf_rows[:4]
            fcf_vals = [_safe_float(r.get("freeCashFlow")) for r in q4]
            if all(v is not None for v in fcf_vals):
                ttm_fcf = _conv(sum(v for v in fcf_vals if v is not None))
            sbc_vals = [_safe_float(r.get("stockBasedCompensation")) for r in q4]
            if all(v is not None for v in sbc_vals):
                ttm_sbc = _conv(sum(abs(v) for v in sbc_vals if v is not None))
            ocf_vals = [_safe_float(r.get("operatingCashFlow")) for r in q4]
            if all(v is not None for v in ocf_vals):
                ttm_ocf = _conv(sum(v for v in ocf_vals if v is not None))
            capex_vals = [_abs_or_none(_safe_float(r.get("capitalExpenditure"))) for r in q4]
            if all(v is not None for v in capex_vals):
                ttm_capex = _conv(sum(v for v in capex_vals if v is not None))
            elif ttm_ocf is not None and ttm_fcf is not None and ttm_ocf >= ttm_fcf:
                ttm_capex = ttm_ocf - ttm_fcf

        ttm_rev = ttm_ni = ttm_ebitda = ttm_ebit = None
        if isinstance(quarter_is_rows, list) and len(quarter_is_rows) >= 4:
            q4i = quarter_is_rows[:4]
            for attr_name, key in (
                ("rev", "revenue"),
                ("ni", "netIncome"),
                ("ebitda", "ebitda"),
                ("ebit", "operatingIncome"),
            ):
                vals = [_safe_float(r.get(key)) for r in q4i]
                if all(v is not None for v in vals):
                    total = _conv(sum(v for v in vals if v is not None))
                    if attr_name == "rev":
                        ttm_rev = total
                    elif attr_name == "ni":
                        ttm_ni = total
                    elif attr_name == "ebitda":
                        ttm_ebitda = total
                    elif attr_name == "ebit":
                        ttm_ebit = total

        latest = annuals[-1] if annuals else None
        latest_cash = latest.cash_and_short_term if latest else None
        latest_debt = latest.total_debt if latest else None
        latest_net_debt = (
            (latest_debt - latest_cash)
            if (latest_debt is not None and latest_cash is not None)
            else None
        )

        current_price = _safe_float(quote.get("price")) or _safe_float(profile.get("price"))
        market_cap = _safe_float(quote.get("marketCap")) or _safe_float(profile.get("mktCap")) or _safe_float(profile.get("marketCap"))
        shares_current = (
            (latest.weighted_avg_shares_diluted if latest and latest.weighted_avg_shares_diluted else None)
            or _safe_float(quote.get("sharesOutstanding"))
        )
        if shares_current is None and market_cap and current_price and current_price > 0:
            shares_current = market_cap / current_price

        # Peers
        peers = self._fetch_peers(sym)
        # Insider trades (last 24 months)
        insider_trades = self._fetch_insiders(sym)
        # Analyst estimates
        analyst_estimates = self._fetch_analyst_estimates(sym, scale=scale)
        # Stock news
        news = self._fetch_news(sym)
        # Risk-free rate
        rf_rate, rf_source = self.get_risk_free_rate()

        sector = str(profile.get("sector") or "")
        industry = str(profile.get("industry") or "")
        fin_flag = is_financial_or_reit(sector, industry)

        return CompanyData(
            ticker=sym,
            name=str(profile.get("companyName") or sym),
            exchange=str(profile.get("exchangeShortName") or profile.get("exchange") or ""),
            sector=sector,
            industry=industry,
            country=str(profile.get("country") or ""),
            quote_currency=quote_ccy,
            reporting_currency=reporting_ccy,
            fx_rate_reporting_to_quote=fx_rate,
            current_price=current_price,
            market_cap=market_cap,
            beta=_safe_float(profile.get("beta")),
            diluted_shares_current=shares_current,
            ttm_fcf=ttm_fcf,
            ttm_sbc=ttm_sbc,
            ttm_capex=ttm_capex,
            ttm_revenue=ttm_rev,
            ttm_net_income=ttm_ni,
            ttm_ebitda=ttm_ebitda,
            ttm_ebit=ttm_ebit,
            ttm_ocf=ttm_ocf,
            latest_cash=latest_cash,
            latest_total_debt=latest_debt,
            latest_net_debt=latest_net_debt,
            risk_free_rate=rf_rate,
            risk_free_source=rf_source,
            annuals=annuals,
            peers=peers,
            insider_trades=insider_trades,
            analyst_estimates=analyst_estimates,
            news=news,
            provider_name=self.name,
            is_limited_history=len(annuals) < 6,
            is_financial_or_reit=fin_flag,
            warnings=warnings,
            description=str(profile.get("description") or ""),
        )

    def _fetch_peers(self, sym: str) -> list[str]:
        payload = self._get_with_fallback(
            "/stable/stock-peers",
            {"symbol": sym},
            "/api/v4/stock_peers",
            {"symbol": sym},
            optional=True,
        )
        peers: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "peersList" in item and isinstance(item["peersList"], list):
                        peers.extend(str(p).upper() for p in item["peersList"])
                    elif "symbol" in item:
                        peers.append(str(item["symbol"]).upper())
        return [p for p in dict.fromkeys(peers) if p and p != sym][:5]

    def _fetch_insiders(self, sym: str) -> list[InsiderTrade]:
        payload = self._get_with_fallback(
            "/stable/insider-trading/search",
            {"symbol": sym, "limit": 100},
            "/api/v4/insider-trading",
            {"symbol": sym, "limit": 100},
            optional=True,
        )
        trades: list[InsiderTrade] = []
        if not isinstance(payload, list):
            return trades

        cutoff = (datetime.now(timezone.utc) - timedelta(days=730)).date().isoformat()
        for row in payload:
            t_date = str(row.get("transactionDate") or row.get("filingDate") or "")[:10]
            if t_date and t_date < cutoff:
                continue
            tx_type = str(row.get("transactionType") or "").strip()
            tx_upper = tx_type.upper()
            shares = _safe_float(row.get("securitiesTransacted"))
            price = _safe_float(row.get("price"))
            value = (abs(shares) * price) if (shares is not None and price is not None and price > 0) else None

            is_open = tx_upper.startswith("P-") or tx_upper == "P" or tx_upper.startswith("S-") or tx_upper == "S"
            direction = "OTHER"
            if tx_upper.startswith("P-") or tx_upper == "P":
                direction = "BUY"
            elif tx_upper.startswith("S-") or tx_upper == "S":
                direction = "SELL"

            trades.append(
                InsiderTrade(
                    date=t_date,
                    insider_name=str(row.get("reportingName") or "Neznámý"),
                    role=str(row.get("typeOfOwner") or ""),
                    transaction_type=tx_type,
                    shares=abs(shares) if shares is not None else None,
                    price=price,
                    value=value,
                    is_open_market=is_open,
                    direction=direction,
                    link=row.get("link") or row.get("url"),
                )
            )
        return trades

    def _fetch_analyst_estimates(self, sym: str, scale: float = 1.0) -> list[AnalystEstimate]:
        payload = self._get_with_fallback(
            "/stable/analyst-estimates",
            {"symbol": sym, "period": "annual", "limit": 6},
            f"/api/v3/analyst-estimates/{sym}",
            {"period": "annual", "limit": 6},
            optional=True,
        )
        if not isinstance(payload, list):
            return []

        current_year = datetime.now(timezone.utc).year
        estimates: list[AnalystEstimate] = []
        for row in payload:
            date_str = str(row.get("date") or "")
            if len(date_str) < 4 or not date_str[:4].isdigit():
                continue
            yr = int(date_str[:4])
            if yr < current_year:
                continue

            def _c(val: Optional[float]) -> Optional[float]:
                return val * scale if val is not None else None

            estimates.append(
                AnalystEstimate(
                    year=yr,
                    date=date_str[:10],
                    estimated_revenue_avg=_c(
                        _safe_float(row.get("revenueAvg") or row.get("estimatedRevenueAvg"))
                    ),
                    estimated_revenue_low=_c(
                        _safe_float(row.get("revenueLow") or row.get("estimatedRevenueLow"))
                    ),
                    estimated_revenue_high=_c(
                        _safe_float(row.get("revenueHigh") or row.get("estimatedRevenueHigh"))
                    ),
                    estimated_eps_avg=_c(
                        _safe_float(row.get("epsAvg") or row.get("estimatedEpsAvg"))
                    ),
                    estimated_eps_low=_c(
                        _safe_float(row.get("epsLow") or row.get("estimatedEpsLow"))
                    ),
                    estimated_eps_high=_c(
                        _safe_float(row.get("epsHigh") or row.get("estimatedEpsHigh"))
                    ),
                    num_analysts_revenue=_safe_int(
                        row.get("numAnalystsRevenue") or row.get("numberAnalystEstimatedRevenue")
                    ),
                    num_analysts_eps=_safe_int(
                        row.get("numAnalystsEps") or row.get("numberAnalystsEstimatedEps")
                    ),
                )
            )
        estimates.sort(key=lambda e: e.year)
        return estimates[:3]

    def _fetch_news(self, sym: str) -> list[NewsItem]:
        payload = self._get_with_fallback(
            "/stable/news/stock",
            {"symbols": sym, "limit": 50},
            "/api/v3/stock_news",
            {"tickers": sym, "limit": 50},
            optional=True,
        )
        items: list[NewsItem] = []
        if not isinstance(payload, list):
            return items
        for row in payload:
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            if not title:
                continue
            items.append(
                NewsItem(
                    published_date=str(row.get("publishedDate") or "")[:10],
                    title=title,
                    source=str(row.get("site") or row.get("publisher") or "Neznámý zdroj"),
                    url=url,
                    text=str(row.get("text") or ""),
                )
            )
        return items

    def get_peer_summary(self, ticker: str) -> Optional[PeerSummary]:
        """Fetch compact valuation and quality metrics for a peer company."""
        sym = ticker.strip().upper()
        profile_list = self._get_with_fallback(
            "/stable/profile",
            {"symbol": sym},
            f"/api/v3/profile/{sym}",
            optional=True,
        )
        if not isinstance(profile_list, list) or not profile_list:
            return None
        profile = profile_list[0]

        km_ttm = self._get_with_fallback(
            "/stable/key-metrics-ttm",
            {"symbol": sym},
            f"/api/v3/key-metrics-ttm/{sym}",
            optional=True,
        )
        km = km_ttm[0] if isinstance(km_ttm, list) and km_ttm else {}

        inc_rows = self._get_with_fallback(
            "/stable/income-statement",
            {"symbol": sym, "period": "annual", "limit": 4},
            f"/api/v3/income-statement/{sym}",
            {"period": "annual", "limit": 4},
            optional=True,
        )

        gross_margin = op_margin = rev_cagr_3y = None
        if isinstance(inc_rows, list) and inc_rows:
            latest = inc_rows[0]
            rev = _safe_float(latest.get("revenue"))
            gp = _safe_float(latest.get("grossProfit"))
            ebit = _safe_float(latest.get("operatingIncome"))
            if rev and rev > 0:
                gross_margin = gp / rev if gp is not None else None
                op_margin = ebit / rev if ebit is not None else None
            if len(inc_rows) >= 4:
                rev_old = _safe_float(inc_rows[3].get("revenue"))
                if rev and rev_old and rev > 0 and rev_old > 0:
                    rev_cagr_3y = (rev / rev_old) ** (1.0 / 3.0) - 1.0

        pe = _safe_float(km.get("peRatioTTM"))
        pfcf = _safe_float(km.get("pfcfRatioTTM"))
        ev_ebitda = _safe_float(km.get("enterpriseValueOverEBITDATTM")) or _safe_float(
            km.get("evToEBITDATTM")
        )
        roic = _safe_float(km.get("roicTTM")) or _safe_float(km.get("returnOnInvestedCapitalTTM"))

        return PeerSummary(
            ticker=sym,
            name=str(profile.get("companyName") or sym),
            currency=str(profile.get("currency") or "USD").upper(),
            market_cap=_safe_float(profile.get("mktCap") or profile.get("marketCap")),
            pe_ratio=pe if (pe and 0 < pe < 300) else None,
            pfcf_ratio=pfcf if (pfcf and 0 < pfcf < 300) else None,
            ev_to_ebitda=ev_ebitda if (ev_ebitda and 0 < ev_ebitda < 200) else None,
            roic=roic,
            gross_margin=gross_margin,
            operating_margin=op_margin,
            revenue_cagr_3y=rev_cagr_3y,
        )
