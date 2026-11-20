"""Historical valuation bands (P/E, P/FCF, EV/EBITDA) and peer comparison builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.models import CompanyData, PeerSummary


@dataclass
class ValuationBand:
    """Historical distribution and current percentile for a valuation multiple."""

    metric_key: str
    label_cz: str
    current: Optional[float]
    min_val: Optional[float]
    avg_val: Optional[float]
    median_val: Optional[float]
    max_val: Optional[float]
    percentile: Optional[float]  # 0..100
    years_count: int
    note_cz: str = ""


def _compute_band(
    metric_key: str,
    label_cz: str,
    historical_values: list[Optional[float]],
    current_value: Optional[float],
    upper_cap: float = 200.0,
) -> ValuationBand:
    """Compute min/avg/median/max and current percentile, excluding negative and extreme outliers."""
    clean = [
        float(v)
        for v in historical_values
        if v is not None and np.isfinite(v) and 0.0 < v <= upper_cap
    ]
    curr_clean = (
        float(current_value)
        if (current_value is not None and np.isfinite(current_value) and 0.0 < current_value <= upper_cap)
        else None
    )

    if not clean:
        note = "Nedostatek kladných historických hodnot (firma byla ztrátová nebo chybí násobky)."
        return ValuationBand(
            metric_key=metric_key,
            label_cz=label_cz,
            current=curr_clean,
            min_val=None,
            avg_val=None,
            median_val=None,
            max_val=None,
            percentile=None,
            years_count=0,
            note_cz=note,
        )

    arr = np.array(clean, dtype=float)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    avg_v = float(np.mean(arr))
    med_v = float(np.median(arr))

    percentile: Optional[float] = None
    if curr_clean is not None:
        # Mid-rank percentile clamped to [0, 100]
        below = float(np.sum(arr < curr_clean))
        equal = float(np.sum(arr == curr_clean))
        percentile = 100.0 * (below + 0.5 * equal) / len(arr)

    return ValuationBand(
        metric_key=metric_key,
        label_cz=label_cz,
        current=curr_clean,
        min_val=min_v,
        avg_val=avg_v,
        median_val=med_v,
        max_val=max_v,
        percentile=percentile,
        years_count=len(clean),
    )


def compute_current_multiples(company: CompanyData) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute current P/E, P/FCF, and EV/EBITDA from live market cap and TTM/latest financials."""
    mcap = company.market_cap
    latest = company.annuals[-1] if company.annuals else None

    ni = company.ttm_net_income if company.ttm_net_income is not None else (latest.net_income if latest else None)
    fcf = company.ttm_fcf if company.ttm_fcf is not None else (latest.free_cash_flow if latest else None)
    ebitda = company.ttm_ebitda if company.ttm_ebitda is not None else (latest.ebitda if latest else None)

    pe = (mcap / ni) if (mcap and ni and ni > 0) else None
    pfcf = (mcap / fcf) if (mcap and fcf and fcf > 0) else None

    net_debt = company.latest_net_debt if company.latest_net_debt is not None else 0.0
    ev = (mcap + net_debt) if mcap is not None else None
    ev_ebitda = (ev / ebitda) if (ev and ev > 0 and ebitda and ebitda > 0) else None

    return pe, pfcf, ev_ebitda


def compute_valuation_bands(company: CompanyData) -> list[ValuationBand]:
    """Compute 5–10 year historical valuation bands for P/E, P/FCF, and EV/EBITDA."""
    curr_pe, curr_pfcf, curr_ev_ebitda = compute_current_multiples(company)

    pe_hist = [a.pe_ratio for a in company.annuals[-10:]]
    pfcf_hist = [a.pfcf_ratio for a in company.annuals[-10:]]
    ev_hist = [a.ev_to_ebitda for a in company.annuals[-10:]]

    return [
        _compute_band("pe", "P/E", pe_hist, curr_pe, upper_cap=200.0),
        _compute_band("pfcf", "P/FCF", pfcf_hist, curr_pfcf, upper_cap=200.0),
        _compute_band("ev_ebitda", "EV/EBITDA", ev_hist, curr_ev_ebitda, upper_cap=120.0),
    ]


def build_target_peer_summary(
    company: CompanyData,
    latest_roic: Optional[float],
    rev_cagr_3y: Optional[float],
) -> PeerSummary:
    """Create a PeerSummary row for the analyzed target company."""
    curr_pe, curr_pfcf, curr_ev_ebitda = compute_current_multiples(company)
    latest = company.annuals[-1] if company.annuals else None
    gm = (
        (latest.gross_profit / latest.revenue)
        if (latest and latest.gross_profit is not None and latest.revenue and latest.revenue > 0)
        else None
    )
    om = (
        (latest.operating_income / latest.revenue)
        if (latest and latest.operating_income is not None and latest.revenue and latest.revenue > 0)
        else None
    )
    return PeerSummary(
        ticker=company.ticker,
        name=company.name,
        currency=company.quote_currency,
        market_cap=company.market_cap,
        pe_ratio=curr_pe if (curr_pe and 0 < curr_pe < 300) else None,
        pfcf_ratio=curr_pfcf if (curr_pfcf and 0 < curr_pfcf < 300) else None,
        ev_to_ebitda=curr_ev_ebitda if (curr_ev_ebitda and 0 < curr_ev_ebitda < 200) else None,
        roic=latest_roic,
        gross_margin=gm,
        operating_margin=om,
        revenue_cagr_3y=rev_cagr_3y,
        is_target=True,
    )


def build_peer_comparison_df(
    target: PeerSummary,
    peers: list[PeerSummary],
) -> pd.DataFrame:
    """Assemble side-by-side comparison DataFrame with target company, peers, and peer median."""
    rows = [target] + [p for p in peers if p.ticker != target.ticker]
    records = [
        {
            "ticker": r.ticker,
            "name": r.name,
            "market_cap": r.market_cap,
            "pe_ratio": r.pe_ratio,
            "pfcf_ratio": r.pfcf_ratio,
            "ev_to_ebitda": r.ev_to_ebitda,
            "roic": r.roic,
            "gross_margin": r.gross_margin,
            "operating_margin": r.operating_margin,
            "revenue_cagr_3y": r.revenue_cagr_3y,
            "is_target": r.is_target,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)

    peer_only = df[~df["is_target"]]
    if not peer_only.empty:
        median_row = {
            "ticker": "MEDIÁN",
            "name": "Medián konkurence",
            "market_cap": float(peer_only["market_cap"].median()) if not peer_only["market_cap"].dropna().empty else None,
            "pe_ratio": float(peer_only["pe_ratio"].median()) if not peer_only["pe_ratio"].dropna().empty else None,
            "pfcf_ratio": float(peer_only["pfcf_ratio"].median()) if not peer_only["pfcf_ratio"].dropna().empty else None,
            "ev_to_ebitda": float(peer_only["ev_to_ebitda"].median()) if not peer_only["ev_to_ebitda"].dropna().empty else None,
            "roic": float(peer_only["roic"].median()) if not peer_only["roic"].dropna().empty else None,
            "gross_margin": float(peer_only["gross_margin"].median()) if not peer_only["gross_margin"].dropna().empty else None,
            "operating_margin": float(peer_only["operating_margin"].median()) if not peer_only["operating_margin"].dropna().empty else None,
            "revenue_cagr_3y": float(peer_only["revenue_cagr_3y"].median()) if not peer_only["revenue_cagr_3y"].dropna().empty else None,
            "is_target": False,
        }
        df = pd.concat([df, pd.DataFrame([median_row])], ignore_index=True)

    return df
