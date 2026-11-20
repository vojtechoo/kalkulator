"""Capital allocation analysis and open-market insider activity filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data.models import CompanyData, InsiderTrade

ALLOCATION_COLUMNS: list[tuple[str, str]] = [
    ("capex", "CapEx"),
    ("rd", "R&D"),
    ("acquisitions", "Akvizice"),
    ("debt_repayment", "Splácení dluhu"),
    ("dividends", "Dividendy"),
    ("buybacks", "Zpětné odkupy"),
]

# Roles qualifying as key decision-makers (CEO, CFO, Directors / Executive Officers)
KEY_INSIDER_ROLE_KEYWORDS = (
    "ceo",
    "chief executive",
    "cfo",
    "chief financial",
    "director",
    "chairman",
    "president",
    "officer",
    "coo",
    "chief operating",
    "founder",
    "evp",
    "svp",
    "vp",
    "board",
)

# Non-open-market codes/keywords to strictly exclude
EXCLUDED_TX_KEYWORDS = (
    "option",
    "exercise",
    "grant",
    "award",
    "m-exempt",
    "a-award",
    "f-inkind",
    "tax",
    "withhold",
    "gift",
    "g-gift",
    "conversion",
    "c-conversion",
    "disposition to issuer",
)


@dataclass
class CapitalAllocationReport:
    """Annual and cumulative capital allocation breakdown."""

    annual_df: pd.DataFrame  # Absolute amounts and % shares of (OCF + R&D) / OCF
    cumulative_totals: dict[str, float]
    cumulative_shares_pct: dict[str, float]
    total_ocf_plus_rd: float
    total_ocf: float
    years_count: int


@dataclass
class InsiderActivityReport:
    """Filtered open-market insider trading summary with cluster-buy & steady-selling signals."""

    filtered_trades: list[InsiderTrade]
    total_buy_value: float
    total_sell_value: float
    net_value: float
    buy_count: int
    sell_count: int
    cluster_buying_detected: bool
    cluster_buying_detail_cz: str
    consistent_monthly_selling: bool
    consistent_selling_detail_cz: str
    excluded_count: int
    notes_cz: list[str] = field(default_factory=list)


def analyze_capital_allocation(company: CompanyData) -> CapitalAllocationReport:
    """Compute annual and cumulative share of operating cash flow deployed across 6 buckets.

    Note: Because R&D is expensed on the income statement BEFORE Operating Cash Flow (OCF) under
    GAAP/IFRS, the share denominator uses `OCF + R&D` (total pre-R&D cash generation) when R&D > 0,
    while also preserving raw OCF so percentages are economically consistent and never distort R&D.
    """
    rows = []
    cum_totals = {key: 0.0 for key, _ in ALLOCATION_COLUMNS}
    total_ocf = 0.0
    total_ocf_plus_rd = 0.0

    for a in company.annuals[-10:]:
        ocf = a.operating_cash_flow or 0.0
        capex = max(a.capex or 0.0, 0.0)
        rd = max(a.rd_expense or 0.0, 0.0)
        acq = max(a.acquisitions_net or 0.0, 0.0)
        debt_rep = max(a.debt_repayment or 0.0, 0.0)
        div = max(a.dividends_paid or 0.0, 0.0)
        bb = max(a.share_buybacks or 0.0, 0.0)

        denom = ocf + rd
        total_deployed = capex + rd + acq + debt_rep + div + bb
        # For stacked % bar relative to cash generated (or total deployed if OCF+R&D <= 0)
        base = denom if denom > 0 else (total_deployed if total_deployed > 0 else None)

        row = {
            "year": a.year,
            "ocf": ocf,
            "ocf_plus_rd": denom,
            "capex": capex,
            "rd": rd,
            "acquisitions": acq,
            "debt_repayment": debt_rep,
            "dividends": div,
            "buybacks": bb,
            "total_deployed": total_deployed,
        }
        for key, _ in ALLOCATION_COLUMNS:
            row[f"{key}_pct"] = (row[key] / base) if (base and base > 0) else 0.0
            cum_totals[key] += row[key]

        total_ocf += ocf
        total_ocf_plus_rd += denom
        rows.append(row)

    df = pd.DataFrame(rows)
    cum_deployed = sum(cum_totals.values())
    cum_base = total_ocf_plus_rd if total_ocf_plus_rd > 0 else cum_deployed
    cum_shares = {
        key: (val / cum_base) if cum_base > 0 else 0.0 for key, val in cum_totals.items()
    }

    return CapitalAllocationReport(
        annual_df=df,
        cumulative_totals=cum_totals,
        cumulative_shares_pct=cum_shares,
        total_ocf_plus_rd=total_ocf_plus_rd,
        total_ocf=total_ocf,
        years_count=len(rows),
    )


def _is_key_insider(role: str) -> bool:
    """Return True if insider role matches CEO/CFO/Director/Executive Officer (or empty when provider omits role)."""
    r = (role or "").strip().lower()
    if not r:
        # Some fallback feeds omit the officer title on Form 4 summaries; keep only if clearly open-market
        return True
    if "10 percent owner" in r and not any(k in r for k in KEY_INSIDER_ROLE_KEYWORDS):
        return False
    return any(k in r for k in KEY_INSIDER_ROLE_KEYWORDS)


def _is_strictly_open_market(trade: InsiderTrade) -> bool:
    """Strictly include only open-market buys/sells; exclude options, grants, gifts, and tax withholding."""
    tx = (trade.transaction_type or "").strip().lower()
    if any(bad in tx for bad in EXCLUDED_TX_KEYWORDS):
        return False
    if not trade.is_open_market:
        return False
    if trade.direction not in ("BUY", "SELL"):
        return False
    if trade.price is not None and trade.price <= 0:
        return False
    return True


def analyze_insider_activity(company: CompanyData) -> InsiderActivityReport:
    """Filter insider trades to open-market CEO/CFO/Director trades and detect cluster buying / steady selling."""
    all_trades = company.insider_trades
    filtered: list[InsiderTrade] = []
    excluded_count = 0

    for t in all_trades:
        if _is_strictly_open_market(t) and _is_key_insider(t.role):
            filtered.append(t)
        else:
            excluded_count += 1

    filtered.sort(key=lambda x: x.date, reverse=True)

    buys = [t for t in filtered if t.direction == "BUY"]
    sells = [t for t in filtered if t.direction == "SELL"]
    total_buy = sum(t.value for t in buys if t.value is not None)
    total_sell = sum(t.value for t in sells if t.value is not None)
    net_val = total_buy - total_sell

    # Cluster buying detection: >= 3 distinct insiders buying within any 30-day window
    cluster_detected = False
    cluster_detail = "Nákupní shluk (cluster buying) nebyl v posledních 24 měsících zaznamenán."
    parsed_buys: list[tuple[datetime, str]] = []
    for b in buys:
        try:
            dt = datetime.strptime(b.date[:10], "%Y-%m-%d")
            parsed_buys.append((dt, b.insider_name.strip().lower()))
        except ValueError:
            continue
    parsed_buys.sort(key=lambda x: x[0])

    for i, (start_dt, _) in enumerate(parsed_buys):
        window_end = start_dt + timedelta(days=30)
        distinct_buyers = {
            name for dt, name in parsed_buys[i:] if dt <= window_end
        }
        if len(distinct_buyers) >= 3:
            cluster_detected = True
            cluster_detail = (
                f"Detekován silný signál CLUSTER BUYING: {len(distinct_buyers)} různí insideři "
                f"nakupovali na otevřeném trhu během 30 dnů (od {start_dt.strftime('%d.%m.%Y')})."
            )
            break

    # Consistent monthly selling: sales in >= 6 distinct months over the last 12 months with minimal buys
    consistent_selling = False
    selling_detail = "Nebylo zjištěno systematické měsíční rozprodávání ze strany vedení."
    if sells:
        cutoff_12m = datetime.now() - timedelta(days=365)
        sell_months: set[str] = set()
        recent_buy_count = 0
        for s in sells:
            try:
                dt = datetime.strptime(s.date[:10], "%Y-%m-%d")
                if dt >= cutoff_12m:
                    sell_months.add(dt.strftime("%Y-%m"))
            except ValueError:
                continue
        for b in buys:
            try:
                dt = datetime.strptime(b.date[:10], "%Y-%m-%d")
                if dt >= cutoff_12m:
                    recent_buy_count += 1
            except ValueError:
                continue

        if len(sell_months) >= 6 and recent_buy_count == 0:
            consistent_selling = True
            selling_detail = (
                f"Varování: vedení prodávalo na otevřeném trhu v {len(sell_months)} různých měsících "
                "za posledních 12 měsíců bez jediného nákupu (pravděpodobně pravidelné plánované prodeje / 10b5-1)."
            )

    notes: list[str] = []
    if not all_trades:
        notes.append("Zdroj dat nevrátil pro tento ticker žádné záznamy o insider transakcích.")
    elif not filtered:
        notes.append(
            f"Všech {excluded_count} nalezených záznamů bylo odfiltrováno (šlo o uplatnění opcí, "
            "udělení akcií, daňové srážky nebo jiné neotevřené transakce)."
        )

    return InsiderActivityReport(
        filtered_trades=filtered,
        total_buy_value=total_buy,
        total_sell_value=total_sell,
        net_value=net_val,
        buy_count=len(buys),
        sell_count=len(sells),
        cluster_buying_detected=cluster_detected,
        cluster_buying_detail_cz=cluster_detail,
        consistent_monthly_selling=consistent_selling,
        consistent_selling_detail_cz=selling_detail,
        excluded_count=excluded_count,
        notes_cz=notes,
    )
