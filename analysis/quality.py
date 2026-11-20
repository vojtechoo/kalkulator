"""10-year Quality X-Ray: ROIC vs. WACC (CAPM), margins trajectory, and Czech moat verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from analysis.formatting import fmt_percent, fmt_pp
from data.models import CompanyData

TrendDirection = Literal["rising", "stable", "falling", "na"]


@dataclass
class WACCComponents:
    """Detailed CAPM WACC inputs and calculated cost of capital."""

    risk_free_rate: Optional[float]
    risk_free_source: str
    equity_risk_premium: float
    beta: Optional[float]
    cost_of_equity: Optional[float]
    cost_of_debt_pre_tax: Optional[float]
    effective_tax_rate: Optional[float]
    cost_of_debt_after_tax: Optional[float]
    equity_weight: Optional[float]
    debt_weight: Optional[float]
    wacc: Optional[float]
    note_cz: str = ""


@dataclass
class QualityXRayReport:
    """Complete 10-year ROIC, WACC, margins, and moat evidence report."""

    roic_df: pd.DataFrame
    margins_df: pd.DataFrame
    wacc_components: WACCComponents
    avg_roic: Optional[float]
    latest_roic: Optional[float]
    latest_spread: Optional[float]
    avg_spread: Optional[float]
    years_below_wacc: int
    total_roic_years: int
    gross_margin_trend: TrendDirection
    gross_margin_trend_label_cz: str
    avg_gross_margin: Optional[float]
    avg_operating_margin: Optional[float]
    avg_fcf_margin: Optional[float]
    moat_verdict_title_cz: str
    moat_verdict_cz: str


def compute_effective_tax_rate(
    tax_expense: Optional[float],
    pretax_income: Optional[float],
) -> tuple[Optional[float], bool]:
    """Compute effective tax rate, clamping distorted one-off ratios into [0 %, 40 %].

    Returns:
        (effective_tax_rate, was_clamped)
    """
    if tax_expense is None or pretax_income is None or pretax_income <= 0:
        return None, False
    raw = tax_expense / pretax_income
    if raw < 0.0:
        return 0.0, True
    if raw > 0.40:
        return 0.40, True
    return float(raw), False


def compute_wacc_capm(
    company: CompanyData,
    equity_risk_premium: float = 0.05,
    fallback_rf: Optional[float] = None,
) -> WACCComponents:
    """Compute WACC using CAPM.

    Formulas:
        Ke = Rf + Beta * ERP
        Kd = Interest Expense / Average Total Debt
        WACC = (E / (E + D)) * Ke + (D / (E + D)) * Kd * (1 - Tax Rate)
    """
    rf = company.risk_free_rate if company.risk_free_rate is not None else fallback_rf
    rf_source = company.risk_free_source or ("Ruční / záložní odhad" if fallback_rf is not None else "Nedostupné")
    beta = company.beta

    latest = company.annuals[-1] if company.annuals else None
    prev = company.annuals[-2] if len(company.annuals) >= 2 else None

    tax_rate = None
    if latest is not None:
        tax_rate, _ = compute_effective_tax_rate(latest.income_tax_expense, latest.income_before_tax)
    if tax_rate is None:
        # Use median valid historical tax rate if latest year had negative pretax income
        valid_rates = []
        for a in company.annuals:
            tr, _ = compute_effective_tax_rate(a.income_tax_expense, a.income_before_tax)
            if tr is not None:
                valid_rates.append(tr)
        tax_rate = float(np.median(valid_rates)) if valid_rates else 0.21

    ke = (rf + beta * equity_risk_premium) if (rf is not None and beta is not None) else None

    debt_now = company.latest_total_debt if company.latest_total_debt is not None else (
        latest.total_debt if latest else None
    )
    debt_prev = prev.total_debt if prev else None
    avg_debt = (
        (debt_now + debt_prev) / 2.0
        if (debt_now is not None and debt_prev is not None)
        else debt_now
    )
    interest = latest.interest_expense if latest else None

    if debt_now is not None and debt_now <= 0:
        kd_pre = 0.0
    elif interest is not None and avg_debt and avg_debt > 0:
        # Guard against extreme distortion when debt shrank sharply mid-year
        kd_pre = min(max(interest / avg_debt, 0.0), 0.20)
    elif rf is not None and debt_now and debt_now > 0:
        kd_pre = rf + 0.015  # Transparent fallback note below
    else:
        kd_pre = None

    kd_after = (kd_pre * (1.0 - tax_rate)) if (kd_pre is not None and tax_rate is not None) else None

    mcap = company.market_cap
    e_val = mcap if (mcap is not None and mcap > 0) else None
    d_val = max(debt_now, 0.0) if debt_now is not None else 0.0

    wacc = e_weight = d_weight = None
    note = ""
    if e_val is not None and ke is not None:
        total_cap = e_val + d_val
        e_weight = e_val / total_cap
        d_weight = d_val / total_cap
        if d_val == 0 or kd_after is None:
            wacc = ke if d_val == 0 else None
        else:
            wacc = e_weight * ke + d_weight * kd_after
    else:
        missing = []
        if rf is None:
            missing.append("bezriziková sazba (10Y Treasury)")
        if beta is None:
            missing.append("beta")
        if e_val is None:
            missing.append("tržní kapitalizace")
        note = "Chybí vstupy pro výpočet WACC: " + ", ".join(missing) + "."

    return WACCComponents(
        risk_free_rate=rf,
        risk_free_source=rf_source,
        equity_risk_premium=equity_risk_premium,
        beta=beta,
        cost_of_equity=ke,
        cost_of_debt_pre_tax=kd_pre,
        effective_tax_rate=tax_rate,
        cost_of_debt_after_tax=kd_after,
        equity_weight=e_weight,
        debt_weight=d_weight,
        wacc=wacc,
        note_cz=note,
    )


def _compute_margin_trend(values: pd.Series) -> tuple[TrendDirection, str]:
    """Determine margin trend direction using linear slope in percentage points per year."""
    clean = values.dropna()
    if len(clean) < 3:
        return "na", "Nedostatek let pro určení trendu"
    x = np.arange(len(clean), dtype=float)
    slope = float(np.polyfit(x, clean.to_numpy(dtype=float), 1)[0])
    # Threshold: +/- 0.30 percentage points per year
    if slope >= 0.003:
        return "rising", "Rostoucí (silná cenová síla)"
    if slope <= -0.003:
        return "falling", "Klesající (tlak na marže)"
    return "stable", "Stabilní (udržitelná cenová síla)"


def run_quality_xray(
    company: CompanyData,
    equity_risk_premium: float = 0.05,
) -> QualityXRayReport:
    """Compute 10-year ROIC vs WACC, margin trajectories, and a data-backed Czech moat verdict."""
    wacc_comp = compute_wacc_capm(company, equity_risk_premium=equity_risk_premium)
    wacc_val = wacc_comp.wacc

    roic_rows = []
    margin_rows = []
    prev_ic: Optional[float] = None

    for a in company.annuals:
        tax_rate, _ = compute_effective_tax_rate(a.income_tax_expense, a.income_before_tax)
        eff_tax = tax_rate if tax_rate is not None else (wacc_comp.effective_tax_rate or 0.21)
        nopat = (a.operating_income * (1.0 - eff_tax)) if a.operating_income is not None else None

        # Invested Capital = Total Debt + Total Equity - Cash & Short-Term Investments
        ic_end: Optional[float] = None
        if a.total_equity is not None:
            ic_end = (a.total_debt or 0.0) + a.total_equity - (a.cash_and_short_term or 0.0)

        ic_avg = ((ic_end + prev_ic) / 2.0) if (ic_end is not None and prev_ic is not None and prev_ic > 0 and ic_end > 0) else ic_end
        roic = (nopat / ic_avg) if (nopat is not None and ic_avg is not None and ic_avg > 0) else None
        spread = (roic - wacc_val) if (roic is not None and wacc_val is not None) else None

        roic_rows.append(
            {
                "year": a.year,
                "nopat": nopat,
                "invested_capital": ic_avg,
                "roic": roic,
                "wacc": wacc_val,
                "spread": spread,
                "below_wacc": (roic < wacc_val) if (roic is not None and wacc_val is not None) else False,
            }
        )
        if ic_end is not None and ic_end > 0:
            prev_ic = ic_end

        rev = a.revenue
        margin_rows.append(
            {
                "year": a.year,
                "gross_margin": (a.gross_profit / rev) if (a.gross_profit is not None and rev and rev > 0) else None,
                "operating_margin": (a.operating_income / rev) if (a.operating_income is not None and rev and rev > 0) else None,
                "fcf_margin": (a.free_cash_flow / rev) if (a.free_cash_flow is not None and rev and rev > 0) else None,
            }
        )

    roic_df = pd.DataFrame(roic_rows)
    margins_df = pd.DataFrame(margin_rows)

    valid_roic = roic_df["roic"].dropna() if not roic_df.empty else pd.Series(dtype=float)
    avg_roic = float(valid_roic.mean()) if not valid_roic.empty else None
    latest_roic = float(valid_roic.iloc[-1]) if not valid_roic.empty else None
    latest_spread = (
        float(roic_df["spread"].dropna().iloc[-1])
        if (not roic_df.empty and not roic_df["spread"].dropna().empty)
        else None
    )
    avg_spread = (
        float(roic_df["spread"].dropna().mean())
        if (not roic_df.empty and not roic_df["spread"].dropna().empty)
        else None
    )
    years_below = int(roic_df["below_wacc"].sum()) if not roic_df.empty else 0
    total_roic_years = int(len(valid_roic))

    gm_series = margins_df["gross_margin"] if not margins_df.empty else pd.Series(dtype=float)
    om_series = margins_df["operating_margin"] if not margins_df.empty else pd.Series(dtype=float)
    fm_series = margins_df["fcf_margin"] if not margins_df.empty else pd.Series(dtype=float)

    gm_trend, gm_trend_label = _compute_margin_trend(gm_series)
    avg_gm = float(gm_series.dropna().mean()) if not gm_series.dropna().empty else None
    avg_om = float(om_series.dropna().mean()) if not om_series.dropna().empty else None
    avg_fm = float(fm_series.dropna().mean()) if not fm_series.dropna().empty else None

    # Data-driven moat verdict in Czech
    if avg_roic is None:
        moat_title = "Nedostatečné podklady pro posouzení ekonomického příkopu"
        moat_text = (
            "ROIC nebylo možné spočítat (záporný investovaný kapitál nebo chybějící data). "
            "Zkontrolujte rozvahu — záporný investovaný kapitál může vznikat u extrémně kapitálově nenáročných firem "
            "nebo po velkých zpětných odkupech, ale i v důsledku akumulovaných ztrát."
        )
    elif (
        avg_roic >= 0.15
        and (avg_spread is None or avg_spread >= 0.05)
        and years_below == 0
        and gm_trend in ("rising", "stable")
    ):
        moat_title = "Silné důkazy ekonomického příkopu (Wide Moat)"
        moat_text = (
            f"Průměrné ROIC dosahuje {fmt_percent(avg_roic)} se spreadem {fmt_pp(avg_spread)} nad WACC "
            f"a firma nepoklesla pod WACC ani v jednom z {total_roic_years} sledovaných let. "
            f"Hrubá marže ({fmt_percent(avg_gm)}) je {gm_trend_label.lower()}, což potvrzuje trvalou cenovou sílu."
        )
    elif avg_roic >= 0.10 and (avg_spread is None or avg_spread > 0.01) and years_below <= max(1, total_roic_years // 4):
        moat_title = "Umírněná konkurenční výhoda (Narrow Moat)"
        moat_text = (
            f"Firma v průměru tvoří hodnotu nad náklady kapitálu (ROIC {fmt_percent(avg_roic)}, "
            f"spread {fmt_pp(avg_spread)}). Hrubá marže ({fmt_percent(avg_gm)}) vykazuje trend: {gm_trend_label.lower()}."
        )
    else:
        moat_title = "Slabý nebo nepřítomný ekonomický příkop (No Moat)"
        moat_text = (
            f"Průměrné ROIC ({fmt_percent(avg_roic)}) je nízké nebo nestabilní "
            f"(spread nad WACC {fmt_pp(avg_spread)}, pod WACC v {years_below} z {total_roic_years} let). "
            f"Dlouhodobý růst takové firmy bez zlepšení návratnosti kapitálu nevytváří nadprůměrnou hodnotu."
        )

    return QualityXRayReport(
        roic_df=roic_df,
        margins_df=margins_df,
        wacc_components=wacc_comp,
        avg_roic=avg_roic,
        latest_roic=latest_roic,
        latest_spread=latest_spread,
        avg_spread=avg_spread,
        years_below_wacc=years_below,
        total_roic_years=total_roic_years,
        gross_margin_trend=gm_trend,
        gross_margin_trend_label_cz=gm_trend_label,
        avg_gross_margin=avg_gm,
        avg_operating_margin=avg_om,
        avg_fcf_margin=avg_fm,
        moat_verdict_title_cz=moat_title,
        moat_verdict_cz=moat_text,
    )
