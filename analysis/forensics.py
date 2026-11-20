"""Quantitative forensic & compounding models not available on Yahoo Finance:
- Piotroski F-Score (0-9)
- Altman Z-Score (bankruptcy / financial distress)
- Beneish M-Score (earnings manipulation detection)
- Reinvestment Engine (ROIIC x Reinvestment Rate -> Intrinsic Compounding Rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from analysis.formatting import fmt_number, fmt_percent
from analysis.quality import compute_effective_tax_rate
from data.models import AnnualFinancials, CompanyData

StatusLevel = Literal["green", "amber", "red", "na"]


@dataclass
class PiotroskiCriterion:
    """Single binary test within the Piotroski F-Score."""

    name_cz: str
    passed: Optional[bool]
    detail_cz: str


@dataclass
class PiotroskiResult:
    """Piotroski F-Score (0-9) assessment."""

    score: Optional[int]
    max_evaluated: int
    status: StatusLevel
    label_cz: str
    summary_cz: str
    criteria: list[PiotroskiCriterion] = field(default_factory=list)


@dataclass
class AltmanResult:
    """Altman Z-Score financial distress model."""

    z_score: Optional[float]
    status: StatusLevel
    zone_cz: str
    summary_cz: str
    components: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class BeneishResult:
    """Beneish M-Score earnings manipulation detector (cutoff = -1.78)."""

    m_score: Optional[float]
    observed_ratios: int
    status: StatusLevel
    label_cz: str
    summary_cz: str
    ratios: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class CompoundingEngineResult:
    """Incremental ROIC (ROIIC) and Reinvestment Rate -> natural compounding power."""

    roiic: Optional[float]
    reinvestment_rate: Optional[float]
    intrinsic_compounding_rate: Optional[float]
    span_years: int
    delta_nopat: Optional[float]
    delta_invested_capital: Optional[float]
    status: StatusLevel
    summary_cz: str


@dataclass
class ForensicsReport:
    """Combined quantitative forensic and reinvestment intelligence report."""

    piotroski: PiotroskiResult
    altman: AltmanResult
    beneish: BeneishResult
    compounding: CompoundingEngineResult


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_piotroski_f_score(company: CompanyData) -> PiotroskiResult:
    """Compute Piotroski F-Score (0–9) comparing the latest fiscal year vs. prior fiscal year."""
    if len(company.annuals) < 2:
        return PiotroskiResult(
            score=None,
            max_evaluated=0,
            status="na",
            label_cz="N/A",
            summary_cz="Pro výpočet Piotroski F-Score jsou potřeba alespoň 2 účetní roky.",
        )

    curr: AnnualFinancials = company.annuals[-1]
    prev: AnnualFinancials = company.annuals[-2]

    criteria: list[PiotroskiCriterion] = []

    # 1. Positive ROA (Net Income / Beginning Total Assets > 0)
    roa_curr = _safe_div(curr.net_income, prev.total_assets or curr.total_assets)
    p1 = (roa_curr > 0) if roa_curr is not None else None
    criteria.append(PiotroskiCriterion("1. Kladné ROA", p1, f"ROA = {fmt_percent(roa_curr, 1)}"))

    # 2. Positive Operating Cash Flow
    p2 = (curr.operating_cash_flow > 0) if curr.operating_cash_flow is not None else None
    criteria.append(
        PiotroskiCriterion(
            "2. Kladné provozní CF (OCF)",
            p2,
            "OCF > 0" if p2 else ("OCF ≤ 0" if p2 is False else "N/A"),
        )
    )

    # 3. Improving ROA (ROA_t > ROA_{t-1})
    prev_assets_base = company.annuals[-3].total_assets if len(company.annuals) >= 3 else prev.total_assets
    roa_prev = _safe_div(prev.net_income, prev_assets_base)
    p3 = (roa_curr > roa_prev) if (roa_curr is not None and roa_prev is not None) else None
    criteria.append(
        PiotroskiCriterion(
            "3. Rostoucí ROA meziročně",
            p3,
            f"{fmt_percent(roa_prev, 1)} → {fmt_percent(roa_curr, 1)}",
        )
    )

    # 4. Quality of Earnings (OCF > Net Income)
    p4 = (
        curr.operating_cash_flow > curr.net_income
        if (curr.operating_cash_flow is not None and curr.net_income is not None)
        else None
    )
    criteria.append(
        PiotroskiCriterion(
            "4. Hotovostní krytí zisku (OCF > NI)",
            p4,
            "Hotovost převyšuje účetní zisk" if p4 else ("Účetní zisk > OCF" if p4 is False else "N/A"),
        )
    )

    # 5. Lower Long-Term Debt / Assets ratio
    lev_curr = _safe_div(curr.long_term_debt or 0.0, curr.total_assets)
    lev_prev = _safe_div(prev.long_term_debt or 0.0, prev.total_assets)
    p5 = (lev_curr <= lev_prev + 1e-6) if (lev_curr is not None and lev_prev is not None) else None
    criteria.append(
        PiotroskiCriterion(
            "5. Klesající dlouhodobé zadlužení",
            p5,
            f"DL dluh/Aktiva: {fmt_percent(lev_prev, 1)} → {fmt_percent(lev_curr, 1)}",
        )
    )

    # 6. Higher Current Ratio (Current Assets / Current Liabilities)
    cr_curr = _safe_div(curr.current_assets, curr.current_liabilities)
    cr_prev = _safe_div(prev.current_assets, prev.current_liabilities)
    p6 = (cr_curr > cr_prev) if (cr_curr is not None and cr_prev is not None) else None
    criteria.append(
        PiotroskiCriterion(
            "6. Rostoucí běžná likvidita",
            p6,
            f"{fmt_number(cr_prev, 2)}x → {fmt_number(cr_curr, 2)}x",
        )
    )

    # 7. No New Share Issuance (Diluted shares <= prior year + 0.5% tolerance)
    sh_curr = curr.weighted_avg_shares_diluted
    sh_prev = prev.weighted_avg_shares_diluted
    p7 = (sh_curr <= sh_prev * 1.005) if (sh_curr is not None and sh_prev is not None and sh_prev > 0) else None
    criteria.append(
        PiotroskiCriterion(
            "7. Žádné ředění novými akciemi",
            p7,
            f"Změna akcií: {fmt_percent((sh_curr - sh_prev) / sh_prev, 1, signed=True)}"
            if (sh_curr and sh_prev)
            else "N/A",
        )
    )

    # 8. Higher Gross Margin
    gm_curr = _safe_div(curr.gross_profit, curr.revenue)
    gm_prev = _safe_div(prev.gross_profit, prev.revenue)
    p8 = (gm_curr > gm_prev) if (gm_curr is not None and gm_prev is not None) else None
    criteria.append(
        PiotroskiCriterion(
            "8. Rostoucí hrubá marže",
            p8,
            f"{fmt_percent(gm_prev, 1)} → {fmt_percent(gm_curr, 1)}",
        )
    )

    # 9. Higher Asset Turnover (Revenue / Beginning Total Assets)
    at_curr = _safe_div(curr.revenue, prev.total_assets or curr.total_assets)
    at_prev = _safe_div(prev.revenue, prev_assets_base)
    p9 = (at_curr > at_prev) if (at_curr is not None and at_prev is not None) else None
    criteria.append(
        PiotroskiCriterion(
            "9. Rostoucí obrat aktiv",
            p9,
            f"{fmt_number(at_prev, 2)}x → {fmt_number(at_curr, 2)}x",
        )
    )

    evaluated = [c.passed for c in criteria if c.passed is not None]
    if len(evaluated) < 5:
        return PiotroskiResult(
            score=None,
            max_evaluated=len(evaluated),
            status="na",
            label_cz="N/A",
            summary_cz="Chybí příliš mnoho rozvahových položek pro spolehlivé F-Score.",
            criteria=criteria,
        )

    raw_score = sum(1 for x in evaluated if x)
    # Normalize to 9-point scale only transparently by displaying raw_score / len(evaluated)
    ratio = raw_score / len(evaluated)
    if ratio >= 7 / 9:
        status: StatusLevel = "green"
        label = f"Silná kondice ({raw_score}/{len(evaluated)})"
        summary = "Fundamentální momentum a kvalita rozvahy se zlepšují – profil vítězné hodnotové akcie."
    elif ratio >= 4 / 9:
        status = "amber"
        label = f"Smíšené signály ({raw_score}/{len(evaluated)})"
        summary = "Průměrné provozní momentum bez jasné akcelerace kvality."
    else:
        status = "red"
        label = f"Slabý fundament ({raw_score}/{len(evaluated)})"
        summary = "Zhoršující se ziskovost, likvidita nebo zadlužení – vysoké riziko hodnotové pasti."

    return PiotroskiResult(
        score=raw_score,
        max_evaluated=len(evaluated),
        status=status,
        label_cz=label,
        summary_cz=summary,
        criteria=criteria,
    )


def compute_altman_z_score(company: CompanyData) -> AltmanResult:
    """Compute Altman Z-Score for non-financial firms.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Cap / Total Liabilities
      X5 = Revenue / Total Assets
    """
    if company.is_financial_or_reit:
        return AltmanResult(
            z_score=None,
            status="na",
            zone_cz="Nevhodné pro finance/REIT",
            summary_cz="Altmanův Z-Score není kalibrován pro banky, pojišťovny a REITy.",
        )

    if not company.annuals:
        return AltmanResult(
            z_score=None,
            status="na",
            zone_cz="N/A",
            summary_cz="Chybí rozvahová data pro výpočet Altman Z-Score.",
        )

    a = company.annuals[-1]
    ta = a.total_assets
    if ta is None or ta <= 0:
        return AltmanResult(
            z_score=None,
            status="na",
            zone_cz="N/A",
            summary_cz="Chybí celková aktiva pro výpočet Altman Z-Score.",
        )

    wc = (
        (a.current_assets - a.current_liabilities)
        if (a.current_assets is not None and a.current_liabilities is not None)
        else None
    )
    x1 = _safe_div(wc, ta)
    x2 = _safe_div(a.retained_earnings, ta)
    x3 = _safe_div(a.operating_income, ta)
    tot_liab = a.total_liabilities
    if tot_liab is None and a.total_equity is not None:
        tot_liab = ta - a.total_equity
    x4 = _safe_div(company.market_cap, tot_liab) if (tot_liab is not None and tot_liab > 0) else None
    x5 = _safe_div(a.revenue, ta)

    comps = {"X1 (Pracovní kapitál/Aktiva)": x1, "X2 (Nerozdělený zisk/Aktiva)": x2, "X3 (EBIT/Aktiva)": x3, "X4 (Tržní kap./Závazky)": x4, "X5 (Tržby/Aktiva)": x5}

    if any(v is None for v in (x1, x2, x3, x4, x5)):
        return AltmanResult(
            z_score=None,
            status="na",
            zone_cz="Neúplná data",
            summary_cz="Chybí některá povinná položka rozvahy (oběžná aktiva, nerozdělený zisk nebo závazky).",
            components=comps,
        )

    assert x1 is not None and x2 is not None and x3 is not None and x4 is not None and x5 is not None
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        status: StatusLevel = "green"
        zone = f"Bezpečná zóna ({fmt_number(z, 2)})"
        summary = "Velmi nízká pravděpodobnost finanční tísně či bankrotu v horizontu 2–3 let."
    elif z >= 1.81:
        status = "amber"
        zone = f"Šedá zóna ({fmt_number(z, 2)})"
        summary = "Zvýšená citlivost rozvahy na recesi nebo pokles marží – sledujte dluhovou zeď."
    else:
        status = "red"
        zone = f"Zóna finanční tísně ({fmt_number(z, 2)})"
        summary = "Z-Score pod 1,81 signalizuje statisticky zvýšené riziko finančního stresu či restrukturalizace."

    return AltmanResult(
        z_score=float(z),
        status=status,
        zone_cz=zone,
        summary_cz=summary,
        components=comps,
    )


def compute_beneish_m_score(company: CompanyData) -> BeneishResult:
    """Compute Beneish M-Score (8-factor earnings manipulation detector).

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
              + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Cutoff: M > -1.78 flags elevated probability of earnings manipulation.
    """
    if company.is_financial_or_reit:
        return BeneishResult(
            m_score=None,
            observed_ratios=0,
            status="na",
            label_cz="Nevhodné pro finance/REIT",
            summary_cz="Beneish M-Score není určen pro finanční instituce a REITy.",
        )

    if len(company.annuals) < 2:
        return BeneishResult(
            m_score=None,
            observed_ratios=0,
            status="na",
            label_cz="N/A",
            summary_cz="Pro Beneish M-Score jsou potřeba alespoň 2 účetní roky.",
        )

    t = company.annuals[-1]
    p = company.annuals[-2]

    def _ratio_growth(curr_num: Optional[float], curr_den: Optional[float], prev_num: Optional[float], prev_den: Optional[float]) -> Optional[float]:
        r_curr = _safe_div(curr_num, curr_den)
        r_prev = _safe_div(prev_num, prev_den)
        if r_curr is None or r_prev is None or r_prev <= 0 or r_curr < 0:
            return None
        return r_curr / r_prev

    # 1. DSRI: Days Sales in Receivables Index
    dsri = _ratio_growth(t.accounts_receivable, t.revenue, p.accounts_receivable, p.revenue)
    # 2. GMI: Gross Margin Index (prior GM / current GM -> >1 means deteriorating margin)
    gm_t = _safe_div(t.gross_profit, t.revenue)
    gm_p = _safe_div(p.gross_profit, p.revenue)
    gmi = (gm_p / gm_t) if (gm_t and gm_p and gm_t > 0 and gm_p > 0) else None
    # 3. AQI: Asset Quality Index
    aq_t = (
        1.0 - ((t.current_assets + t.ppe_net) / t.total_assets)
        if (t.current_assets is not None and t.ppe_net is not None and t.total_assets and t.total_assets > 0)
        else None
    )
    aq_p = (
        1.0 - ((p.current_assets + p.ppe_net) / p.total_assets)
        if (p.current_assets is not None and p.ppe_net is not None and p.total_assets and p.total_assets > 0)
        else None
    )
    aqi = (aq_t / aq_p) if (aq_t is not None and aq_p is not None and aq_p > 0.01 and aq_t >= 0) else None
    # 4. SGI: Sales Growth Index
    sgi = (t.revenue / p.revenue) if (t.revenue and p.revenue and p.revenue > 0) else None
    # 5. DEPI: Depreciation Index (prior dep rate / current dep rate)
    dr_t = (
        t.depreciation_amortization / (t.depreciation_amortization + t.ppe_net)
        if (t.depreciation_amortization and t.ppe_net and (t.depreciation_amortization + t.ppe_net) > 0)
        else None
    )
    dr_p = (
        p.depreciation_amortization / (p.depreciation_amortization + p.ppe_net)
        if (p.depreciation_amortization and p.ppe_net and (p.depreciation_amortization + p.ppe_net) > 0)
        else None
    )
    depi = (dr_p / dr_t) if (dr_t and dr_p and dr_t > 0) else None
    # 6. SGAI: SG&A Expense Index
    sgai = _ratio_growth(t.sga_expense, t.revenue, p.sga_expense, p.revenue)
    # 7. LVGI: Leverage Index
    lev_t = (
        ((t.current_liabilities or 0.0) + (t.long_term_debt or 0.0)) / t.total_assets
        if (t.total_assets and t.total_assets > 0 and (t.current_liabilities is not None or t.long_term_debt is not None))
        else None
    )
    lev_p = (
        ((p.current_liabilities or 0.0) + (p.long_term_debt or 0.0)) / p.total_assets
        if (p.total_assets and p.total_assets > 0 and (p.current_liabilities is not None or p.long_term_debt is not None))
        else None
    )
    lvgi = (lev_t / lev_p) if (lev_t is not None and lev_p is not None and lev_p > 0) else None
    # 8. TATA: Total Accruals to Total Assets
    tata = (
        (t.net_income - t.operating_cash_flow) / t.total_assets
        if (t.net_income is not None and t.operating_cash_flow is not None and t.total_assets and t.total_assets > 0)
        else None
    )

    ratios_map: dict[str, Optional[float]] = {
        "DSRI (Pohledávky/Tržby)": dsri,
        "GMI (Pokles hrubé marže)": gmi,
        "AQI (Kvalita aktiv)": aqi,
        "SGI (Růst tržeb)": sgi,
        "DEPI (Zpomalení odpisů)": depi,
        "SGAI (Režijní náklady)": sgai,
        "LVGI (Růst páky)": lvgi,
        "TATA (Akruály/Aktiva)": tata,
    }
    observed = sum(1 for v in ratios_map.values() if v is not None)
    if observed < 5 or tata is None or sgi is None:
        return BeneishResult(
            m_score=None,
            observed_ratios=observed,
            status="na",
            label_cz=f"Neúplná data ({observed}/8)",
            summary_cz="Chybí klíčové účetní položky (pohledávky, odpisy, PPE) pro spolehlivý Beneish M-Score.",
            ratios=ratios_map,
        )

    # Neutral fallback (1.0 for indices) ONLY for up to 3 missing non-core sub-ratios, transparently reported.
    v_dsri = min(dsri, 4.0) if dsri is not None else 1.0
    v_gmi = min(gmi, 3.0) if gmi is not None else 1.0
    v_aqi = min(aqi, 4.0) if aqi is not None else 1.0
    v_sgi = min(sgi, 4.0) if sgi is not None else 1.0
    v_depi = min(depi, 3.0) if depi is not None else 1.0
    v_sgai = min(sgai, 3.0) if sgai is not None else 1.0
    v_lvgi = min(lvgi, 3.0) if lvgi is not None else 1.0
    v_tata = float(np.clip(tata, -0.5, 0.5))

    m_score = (
        -4.84
        + 0.92 * v_dsri
        + 0.528 * v_gmi
        + 0.404 * v_aqi
        + 0.892 * v_sgi
        + 0.115 * v_depi
        - 0.172 * v_sgai
        + 4.679 * v_tata
        - 0.327 * v_lvgi
    )

    obs_note = f" (vypočteno z {observed}/8 přímo dostupných poměrů)" if observed < 8 else ""
    if m_score > -1.78:
        status: StatusLevel = "red"
        label = f"Riziko manipulace ({fmt_number(m_score, 2)})"
        summary = (
            f"M-Score překračuje hranici −1,78{obs_note}. Zkontrolujte nárůst pohledávek, "
            "pokles marží nebo vysoké akruály – účetní zisky mohou být agresivně vykazovány."
        )
    elif m_score > -2.22:
        status = "amber"
        label = f"Hraniční pásmo ({fmt_number(m_score, 2)})"
        summary = f"M-Score je pod limitem −1,78, ale v ostražité zóně (−2,22 až −1,78){obs_note}."
    else:
        status = "green"
        label = f"Bez známek manipulace ({fmt_number(m_score, 2)})"
        summary = f"M-Score bezpečně pod −2,22{obs_note}. Účetní výkazy nevykazují statistické znaky manipulace zisků."

    return BeneishResult(
        m_score=float(m_score),
        observed_ratios=observed,
        status=status,
        label_cz=label,
        summary_cz=summary,
        ratios=ratios_map,
    )


def compute_compounding_engine(company: CompanyData) -> CompoundingEngineResult:
    """Compute incremental ROIC (ROIIC) and Reinvestment Rate to estimate intrinsic compounding rate.

    Over available multi-year span (preferring 5 years, minimum 3 years):
        ΔNOPAT = NOPAT_end - NOPAT_start
        ΔInvested Capital = IC_end - IC_start
        ROIIC = ΔNOPAT / ΔInvested Capital
        Reinvestment Rate = ΔInvested Capital / Sum(NOPAT_{start+1..end})
        Intrinsic Compounding Rate = ROIIC × Reinvestment Rate (when both > 0)
    """
    annuals = company.annuals
    if len(annuals) < 3:
        return CompoundingEngineResult(
            roiic=None,
            reinvestment_rate=None,
            intrinsic_compounding_rate=None,
            span_years=0,
            delta_nopat=None,
            delta_invested_capital=None,
            status="na",
            summary_cz="Pro výpočet přírůstkové návratnosti (ROIIC) jsou potřeba alespoň 3 roky historie.",
        )

    window = annuals[-6:] if len(annuals) >= 6 else annuals
    start_a = window[0]
    end_a = window[-1]
    span = end_a.year - start_a.year

    def _nopat(a: AnnualFinancials) -> Optional[float]:
        if a.operating_income is None:
            return None
        tr, _ = compute_effective_tax_rate(a.income_tax_expense, a.income_before_tax)
        eff_tax = tr if tr is not None else 0.21
        return a.operating_income * (1.0 - eff_tax)

    def _ic(a: AnnualFinancials) -> Optional[float]:
        if a.total_equity is None:
            return None
        return (a.total_debt or 0.0) + a.total_equity - (a.cash_and_short_term or 0.0)

    n_start, n_end = _nopat(start_a), _nopat(end_a)
    ic_start, ic_end = _ic(start_a), _ic(end_a)

    if n_start is None or n_end is None or ic_start is None or ic_end is None:
        return CompoundingEngineResult(
            roiic=None,
            reinvestment_rate=None,
            intrinsic_compounding_rate=None,
            span_years=span,
            delta_nopat=None,
            delta_invested_capital=None,
            status="na",
            summary_cz="Chybí data pro výpočet NOPAT nebo investovaného kapitálu.",
        )

    delta_nopat = n_end - n_start
    delta_ic = ic_end - ic_start
    cum_nopat = sum(_nopat(a) or 0.0 for a in window[1:])

    # Special case: capital-light compounder growing NOPAT while shrinking/maintaining IC
    if delta_ic <= 0 and delta_nopat > 0:
        nopat_cagr = (
            ((n_end / n_start) ** (1.0 / max(span, 1)) - 1.0)
            if (n_start > 0 and n_end > 0 and span >= 1)
            else None
        )
        return CompoundingEngineResult(
            roiic=None,
            reinvestment_rate=0.0,
            intrinsic_compounding_rate=nopat_cagr,
            span_years=span,
            delta_nopat=delta_nopat,
            delta_invested_capital=delta_ic,
            status="green",
            summary_cz=(
                f"Kapitálově extrémně nenáročný růst: za {span} let vzrostl NOPAT při nulové či záporné "
                f"potřebě nového kapitálu (organické tempo NOPAT {fmt_percent(nopat_cagr, 1)} ročně)."
            ),
        )

    if delta_ic <= 0:
        return CompoundingEngineResult(
            roiic=None,
            reinvestment_rate=None,
            intrinsic_compounding_rate=None,
            span_years=span,
            delta_nopat=delta_nopat,
            delta_invested_capital=delta_ic,
            status="amber",
            summary_cz="Investovaný kapitál i provozní zisk v posledním období klesaly – motor reinvestic neběží.",
        )

    roiic = float(np.clip(delta_nopat / delta_ic, -1.0, 2.0))
    reinv_rate = (
        float(np.clip(delta_ic / cum_nopat, 0.0, 1.5))
        if cum_nopat > 0
        else None
    )
    comp_rate = (
        float(np.clip(roiic * reinv_rate, -0.30, 0.45))
        if (reinv_rate is not None)
        else None
    )

    if comp_rate is not None and comp_rate >= 0.12 and roiic >= 0.15:
        status: StatusLevel = "green"
        summary = (
            f"Silný motor složeného úročení: za {span} let firma reinvestovala {fmt_percent(reinv_rate, 0)} zisku "
            f"s přírůstkovou návratností ROIIC {fmt_percent(roiic, 1)} → přirozené tempo růstu {fmt_percent(comp_rate, 1)} ročně."
        )
    elif comp_rate is not None and comp_rate >= 0.05 and roiic >= 0.08:
        status = "amber"
        summary = (
            f"Střední reinvestiční síla: ROIIC {fmt_percent(roiic, 1)} × reinvestiční míra {fmt_percent(reinv_rate, 0)} "
            f"= přirozený růst {fmt_percent(comp_rate, 1)} ročně."
        )
    else:
        status = "red"
        summary = (
            f"Slabá návratnost nového kapitálu (ROIIC {fmt_percent(roiic, 1)}). "
            "Nově vložený kapitál zatím nevytváří dostatečný dodatečný zisk."
        )

    return CompoundingEngineResult(
        roiic=roiic,
        reinvestment_rate=reinv_rate,
        intrinsic_compounding_rate=comp_rate,
        span_years=span,
        delta_nopat=delta_nopat,
        delta_invested_capital=delta_ic,
        status=status,
        summary_cz=summary,
    )


def run_forensics_analysis(company: CompanyData) -> ForensicsReport:
    """Execute Piotroski F-Score, Altman Z-Score, Beneish M-Score, and ROIIC Compounding Engine."""
    return ForensicsReport(
        piotroski=compute_piotroski_f_score(company),
        altman=compute_altman_z_score(company),
        beneish=compute_beneish_m_score(company),
        compounding=compute_compounding_engine(company),
    )
