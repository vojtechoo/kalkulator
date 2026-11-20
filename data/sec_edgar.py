"""Free official SEC EDGAR XBRL CompanyFacts enricher (10+ years of audited 10-K annual history).

Used automatically by YFinanceDataProvider to extend ~4-year yfinance history to 10+ years
for US-listed companies without requiring any paid API key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import requests

from data.models import AnnualFinancials

SEC_HEADERS = {
    "User-Agent": "ValueFocusStockAnalyzer/2.0 (valuefocus.analyzer@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
SEC_TIMEOUT = 10

# Process-level cache for SEC ticker -> 10-digit CIK mapping
_CIK_MAP_CACHE: Optional[dict[str, str]] = None


def _get_cik_map() -> dict[str, str]:
    """Download and cache the official SEC ticker-to-CIK lookup table."""
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is not None:
        return _CIK_MAP_CACHE
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=SEC_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}
        payload = resp.json()
        mapping: dict[str, str] = {}
        for entry in payload.values():
            tk = str(entry.get("ticker", "")).strip().upper()
            cik = entry.get("cik_str")
            if tk and cik is not None:
                mapping[tk] = f"{int(cik):010d}"
        _CIK_MAP_CACHE = mapping
        return mapping
    except Exception:
        return {}


def _lookup_cik(ticker: str) -> Optional[str]:
    """Resolve a US ticker (handling dot/dash variants like BRK-B / BRK.B) to a 10-digit SEC CIK."""
    cik_map = _get_cik_map()
    if not cik_map:
        return None
    t = ticker.strip().upper()
    if t in cik_map:
        return cik_map[t]
    alt = t.replace(".", "-") if "." in t else t.replace("-", ".")
    return cik_map.get(alt)


def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except Exception:
        return None


def _fiscal_year_from_end(end_dt: datetime) -> int:
    """Assign fiscal year from period end date (dates in early January belong to prior fiscal year)."""
    return end_dt.year - 1 if (end_dt.month == 1 and end_dt.day <= 15) else end_dt.year


def _extract_annual_duration(
    us_gaap: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str = "USD",
) -> dict[int, tuple[float, str]]:
    """Extract full-year (~330–390 day) 10-K/20-F duration values keyed by fiscal year.

    Returns dict[year, (value, end_date_str)] preferring the first matching concept and latest filing date.
    """
    best_by_year: dict[int, tuple[str, float, str]] = {}  # year -> (filed, val, end_str)

    for priority_idx, concept in enumerate(concepts):
        node = us_gaap.get(concept)
        if not isinstance(node, dict):
            continue
        units = node.get("units", {})
        entries = units.get(unit)
        if not isinstance(entries, list):
            continue

        for e in entries:
            form = str(e.get("form", "")).upper()
            if form not in ("10-K", "10-K/A", "20-F", "20-F/A"):
                continue
            start_s = str(e.get("start", ""))
            end_s = str(e.get("end", ""))
            if not start_s or not end_s:
                continue
            start_dt = _parse_date(start_s)
            end_dt = _parse_date(end_s)
            if start_dt is None or end_dt is None:
                continue
            days = (end_dt - start_dt).days
            if not (330 <= days <= 395):
                continue

            val = e.get("val")
            if val is None:
                continue
            try:
                num_val = float(val)
            except (TypeError, ValueError):
                continue

            fy = _fiscal_year_from_end(end_dt)
            filed = str(e.get("filed", ""))
            # Only fill years not already populated by a higher-priority concept; within same concept take latest filed
            existing = best_by_year.get(fy)
            if existing is None:
                best_by_year[fy] = (f"{99 - priority_idx:02d}_{filed}", num_val, end_s[:10])
            else:
                tag = f"{99 - priority_idx:02d}_{filed}"
                if tag > existing[0]:
                    best_by_year[fy] = (tag, num_val, end_s[:10])

    return {yr: (val, end_s) for yr, (_, val, end_s) in best_by_year.items()}


def _extract_annual_instant(
    us_gaap: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str = "USD",
) -> dict[int, float]:
    """Extract 10-K/20-F balance-sheet instant values keyed by fiscal year."""
    best_by_year: dict[int, tuple[str, float]] = {}

    for priority_idx, concept in enumerate(concepts):
        node = us_gaap.get(concept)
        if not isinstance(node, dict):
            continue
        entries = node.get("units", {}).get(unit)
        if not isinstance(entries, list):
            continue

        for e in entries:
            form = str(e.get("form", "")).upper()
            if form not in ("10-K", "10-K/A", "20-F", "20-F/A"):
                continue
            end_s = str(e.get("end", ""))
            end_dt = _parse_date(end_s)
            if end_dt is None:
                continue
            val = e.get("val")
            if val is None:
                continue
            try:
                num_val = float(val)
            except (TypeError, ValueError):
                continue

            fy = _fiscal_year_from_end(end_dt)
            filed = str(e.get("filed", ""))
            # Prefer higher-priority concept, then later end_date within FY, then latest filing
            tag = f"{99 - priority_idx:02d}_{end_s[:10]}_{filed}"
            existing = best_by_year.get(fy)
            if existing is None or tag > existing[0]:
                best_by_year[fy] = (tag, num_val)

    return {yr: val for yr, (_, val) in best_by_year.items()}


def fetch_sec_historical_annuals(
    ticker: str,
    yearly_avg_price: dict[int, float],
    max_years: int = 12,
    currency: str = "USD",
) -> list[AnnualFinancials]:
    """Fetch up to `max_years` of audited 10-K annual financials from free SEC EDGAR XBRL API.

    Returns an empty list cleanly if the ticker is non-US or SEC EDGAR is unreachable.
    """
    cik = _lookup_cik(ticker)
    if not cik:
        return []

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=SEC_TIMEOUT)
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except Exception:
        return []

    facts = payload.get("facts", {})
    us_gaap = facts.get("us-gaap") or facts.get("ifrs-full") or {}
    if not isinstance(us_gaap, dict) or not us_gaap:
        return []

    rev_d = _extract_annual_duration(
        us_gaap,
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
    )
    if len(rev_d) < 3:
        return []

    gp_d = _extract_annual_duration(us_gaap, ("GrossProfit",))
    cogs_d = _extract_annual_duration(
        us_gaap,
        ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"),
    )
    ebit_d = _extract_annual_duration(us_gaap, ("OperatingIncomeLoss",))
    int_d = _extract_annual_duration(us_gaap, ("InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"))
    pretax_d = _extract_annual_duration(
        us_gaap,
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
    )
    tax_d = _extract_annual_duration(us_gaap, ("IncomeTaxExpenseBenefit",))
    ni_d = _extract_annual_duration(us_gaap, ("NetIncomeLoss", "ProfitLoss"))
    rd_d = _extract_annual_duration(
        us_gaap,
        ("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
    )
    sga_d = _extract_annual_duration(
        us_gaap,
        ("SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
    )
    da_d = _extract_annual_duration(
        us_gaap,
        ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"),
    )
    eps_d = _extract_annual_duration(
        us_gaap, ("EarningsPerShareDiluted", "EarningsPerShareBasic"), unit="USD/shares"
    )
    sh_d = _extract_annual_duration(
        us_gaap,
        ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"),
        unit="shares",
    )

    ocf_d = _extract_annual_duration(us_gaap, ("NetCashProvidedByUsedInOperatingActivities",))
    capex_d = _extract_annual_duration(
        us_gaap, ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
    )
    sbc_d = _extract_annual_duration(
        us_gaap, ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense")
    )
    acq_d = _extract_annual_duration(us_gaap, ("PaymentsToAcquireBusinessesNetOfCashAcquired",))
    debt_rep_d = _extract_annual_duration(us_gaap, ("RepaymentsOfLongTermDebt", "RepaymentsOfDebt"))
    div_d = _extract_annual_duration(us_gaap, ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"))
    bb_d = _extract_annual_duration(us_gaap, ("PaymentsForRepurchaseOfCommonStock",))

    # Balance sheet instant items
    cash_i = _extract_annual_instant(us_gaap, ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"))
    st_inv_i = _extract_annual_instant(us_gaap, ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"))
    ar_i = _extract_annual_instant(us_gaap, ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"))
    ca_i = _extract_annual_instant(us_gaap, ("AssetsCurrent",))
    ppe_i = _extract_annual_instant(us_gaap, ("PropertyPlantAndEquipmentNet",))
    ta_i = _extract_annual_instant(us_gaap, ("Assets",))
    cl_i = _extract_annual_instant(us_gaap, ("LiabilitiesCurrent",))
    tl_i = _extract_annual_instant(us_gaap, ("Liabilities",))
    re_i = _extract_annual_instant(us_gaap, ("RetainedEarningsAccumulatedDeficit",))
    st_debt_i = _extract_annual_instant(us_gaap, ("LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"))
    lt_debt_i = _extract_annual_instant(us_gaap, ("LongTermDebtNoncurrent", "LongTermDebt"))
    eq_i = _extract_annual_instant(
        us_gaap,
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    )

    years = sorted(rev_d.keys())[-max_years:]
    annuals: list[AnnualFinancials] = []

    def _dval(d: dict[int, tuple[float, str]], yr: int) -> Optional[float]:
        item = d.get(yr)
        return item[0] if item is not None else None

    for yr in years:
        rev, end_date = rev_d[yr]
        gp = _dval(gp_d, yr)
        if gp is None:
            cogs = _dval(cogs_d, yr)
            if cogs is not None:
                gp = rev - cogs

        ebit = _dval(ebit_d, yr)
        da_raw = _dval(da_d, yr)
        da = abs(da_raw) if da_raw is not None else None
        ebitda = (ebit + da) if (ebit is not None and da is not None) else None
        ni = _dval(ni_d, yr)
        int_raw = _dval(int_d, yr)
        interest = abs(int_raw) if int_raw is not None else None
        pretax = _dval(pretax_d, yr)
        tax = _dval(tax_d, yr)
        rd_raw = _dval(rd_d, yr)
        rd = abs(rd_raw) if rd_raw is not None else None
        sga_raw = _dval(sga_d, yr)
        sga = abs(sga_raw) if sga_raw is not None else None

        eps = _dval(eps_d, yr)
        shares = _dval(sh_d, yr)

        ocf = _dval(ocf_d, yr)
        capex_raw = _dval(capex_d, yr)
        capex = abs(capex_raw) if capex_raw is not None else None
        fcf = (ocf - capex) if (ocf is not None and capex is not None) else None
        sbc_raw = _dval(sbc_d, yr)
        sbc = abs(sbc_raw) if sbc_raw is not None else None
        acq_raw = _dval(acq_d, yr)
        acq = abs(acq_raw) if acq_raw is not None else None
        drep_raw = _dval(debt_rep_d, yr)
        drep = abs(drep_raw) if drep_raw is not None else None
        div_raw = _dval(div_d, yr)
        div_paid = abs(div_raw) if div_raw is not None else None
        bb_raw = _dval(bb_d, yr)
        buybacks = abs(bb_raw) if bb_raw is not None else None

        cash = cash_i.get(yr)
        st_inv = st_inv_i.get(yr)
        cash_st = (cash or 0.0) + (st_inv or 0.0) if (cash is not None or st_inv is not None) else None
        st_debt = st_debt_i.get(yr)
        lt_debt = lt_debt_i.get(yr)
        tot_debt = (
            (st_debt or 0.0) + (lt_debt or 0.0)
            if (st_debt is not None or lt_debt is not None)
            else None
        )

        avg_p = yearly_avg_price.get(yr)
        mcap_yr = (avg_p * shares) if (avg_p and shares and shares > 0) else None
        pe_yr = (mcap_yr / ni) if (mcap_yr and ni and ni > 0) else None
        pfcf_yr = (mcap_yr / fcf) if (mcap_yr and fcf and fcf > 0) else None
        ev_yr = (mcap_yr + (tot_debt or 0.0) - (cash_st or 0.0)) if mcap_yr is not None else None
        ev_ebitda_yr = (ev_yr / ebitda) if (ev_yr and ebitda and ebitda > 0) else None

        annuals.append(
            AnnualFinancials(
                year=yr,
                date=end_date,
                currency=currency,
                revenue=rev,
                gross_profit=gp,
                operating_income=ebit,
                interest_expense=interest,
                income_before_tax=pretax,
                income_tax_expense=tax,
                net_income=ni,
                ebitda=ebitda,
                rd_expense=rd,
                sga_expense=sga,
                depreciation_amortization=da,
                eps_diluted=eps,
                weighted_avg_shares_diluted=shares,
                cash_and_short_term=cash_st,
                accounts_receivable=ar_i.get(yr),
                current_assets=ca_i.get(yr),
                ppe_net=ppe_i.get(yr),
                total_assets=ta_i.get(yr),
                current_liabilities=cl_i.get(yr),
                total_liabilities=tl_i.get(yr),
                retained_earnings=re_i.get(yr),
                short_term_debt=st_debt,
                long_term_debt=lt_debt,
                total_debt=tot_debt,
                total_equity=eq_i.get(yr),
                operating_cash_flow=ocf,
                capex=capex,
                free_cash_flow=fcf,
                stock_based_compensation=sbc,
                acquisitions_net=acq,
                debt_repayment=drep,
                dividends_paid=div_paid,
                share_buybacks=buybacks,
                pe_ratio=pe_yr,
                pfcf_ratio=pfcf_yr,
                ev_to_ebitda=ev_ebitda_yr,
                market_cap_year=mcap_yr,
                avg_share_price=avg_p,
            )
        )

    return annuals
