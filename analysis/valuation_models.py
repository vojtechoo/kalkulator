"""Complete GuruFocus-style mathematical valuation suite & 5-pillar 10-point scorecard.

Implements every explicit value-investing formula purely from hard financial data:
1. FCF = OCF - CapEx
2. FCF_adj = OCF - CapEx - SBC
3. DCF Enterprise Value = sum(FCF_t / (1+r)^t) + TV / (1+r)^n
4. Terminal Value (Gordon) = FCF_n * (1 + g) / (r - g)
5. Terminal Value (Exit Multiple) = FCF_n * Multiple
6. Intrinsic Value per Share = (V_firmy + Cash - Total Debt) / Diluted Shares
7. DDM = D_1 / (r - g)
8. Graham Formula = EPS * (8.5 + 2*g) * 4.4 / Y
9. EPV = NOPAT / WACC  -> per share = (EPV + Cash - Total Debt) / Diluted Shares
10. Buffett Owner Earnings = Net Income + D&A - Maintenance CapEx - Delta Working Capital
11. Relative P/E = Expected EPS * Fair P/E
12. Relative P/FCF = FCF per share * Fair P/FCF
13. Target Buy Price (Margin of Safety) = V_akcie * (1 - MoS)
14. ROIC = NOPAT / Invested Capital
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

from analysis.crystal_ball import (
    MonteCarloResult,
    _estimate_maintenance_capex,
    _historical_revenue_growth_stats,
)
from analysis.dcf import (
    ReverseDCFInputs,
    compute_dcf_value_per_share,
    compute_growth_optionality_per_share,
    compute_two_stage_fading_dcf_value_per_share,
    select_fcf_base,
)
from analysis.forensics import ForensicsReport
from analysis.formatting import fmt_large_currency, fmt_multiple, fmt_number, fmt_percent, fmt_price
from analysis.quality import QualityXRayReport
from analysis.red_flags import RedFlagsReport
from data.models import AnnualFinancials, CompanyData

GFStatus = Literal["significantly_undervalued", "modestly_undervalued", "fairly_valued", "modestly_overvalued", "significantly_overvalued", "na"]


@dataclass
class FormulaModelRow:
    """Single mathematical valuation model with exact inputs, value, buy price, and formula breakdown."""

    key: str
    name_cz: str
    formula_latex: str
    value_per_share: Optional[float]
    buy_price_mos: Optional[float]
    upside_pct: Optional[float]
    weight: float
    inputs_summary_cz: str
    status_note_cz: str = ""


@dataclass
class CashFlowDecomposition:
    """Explicit FCF, FCF_adj, and Buffett Owner Earnings decomposition from latest annual statements."""

    year: Optional[int]
    ocf: Optional[float]
    capex: Optional[float]
    sbc: Optional[float]
    fcf: Optional[float]
    fcf_adj: Optional[float]
    net_income: Optional[float]
    depreciation_amortization: Optional[float]
    maintenance_capex: Optional[float]
    delta_working_capital: Optional[float]
    owner_earnings: Optional[float]
    owner_earnings_per_share: Optional[float]


@dataclass
class GFPillarCard:
    """GuruFocus-style 10-point summary pillar."""

    key: str
    title_cz: str
    score_10: int  # 0..10
    color: Literal["green", "amber", "red"]
    key_metric_cz: str
    detail_cz: str


@dataclass
class ValuationSuiteReport:
    """Full GuruFocus-style valuation suite, composite fair value, bands, and 5-pillar scorecard."""

    models: list[FormulaModelRow]
    gf_fair_value: Optional[float]
    gf_buy_price: Optional[float]
    margin_of_safety: float
    price_to_gf_value: Optional[float]
    upside_pct: Optional[float]
    gf_status: GFStatus
    gf_status_label_cz: str
    gf_status_color: Literal["green", "amber", "red", "na"]
    epv_enterprise_value: Optional[float]
    epv_per_share: Optional[float]
    epv_share_of_price: Optional[float]
    growth_bet_share_of_price: Optional[float]
    cashflow_decomp: CashFlowDecomposition
    aaa_yield_pct: float
    predictability_stars: float  # 1.0 to 5.0
    predictability_label_cz: str
    gf_pillars: list[GFPillarCard]
    gf_value_history_df: pd.DataFrame  # year, price, fair_value, band_low30, band_low10, band_high10, band_high30
    composite_gf_score: int  # 0..100
    # Growth Optionality & Advanced Scenario breakdown
    default_stage1_growth: float = 0.08
    effective_stage1_growth: float = 0.08
    used_two_stage_fade: bool = False
    core_dcf_gordon_per_share: Optional[float] = None
    core_dcf_exit_per_share: Optional[float] = None
    optionality_per_share: float = 0.0
    optionality_active: bool = False
    optionality_note_cz: str = ""
    graham_weight_pct: float = 10.0


def compute_non_cash_working_capital(a: AnnualFinancials) -> Optional[float]:
    """Compute non-cash operating working capital: (Current Assets - Cash) - (Current Liab - ST Debt)."""
    if a.current_assets is None or a.current_liabilities is None:
        return None
    op_ca = a.current_assets - (a.cash_and_short_term or 0.0)
    op_cl = a.current_liabilities - (a.short_term_debt or 0.0)
    return op_ca - op_cl


def compute_cashflow_decomposition(company: CompanyData) -> CashFlowDecomposition:
    """Compute exact FCF, FCF_adj (FCF - SBC), and Buffett Owner Earnings with Delta Working Capital."""
    if not company.annuals:
        return CashFlowDecomposition(None, None, None, None, None, None, None, None, None, None, None, None)

    latest = company.annuals[-1]
    prev = company.annuals[-2] if len(company.annuals) >= 2 else None

    ocf = latest.operating_cash_flow
    capex = latest.capex
    sbc = latest.stock_based_compensation
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else latest.free_cash_flow
    fcf_adj = (fcf - sbc) if (fcf is not None and sbc is not None) else fcf

    maint_capex = _estimate_maintenance_capex(company.annuals)
    wc_curr = compute_non_cash_working_capital(latest)
    wc_prev = compute_non_cash_working_capital(prev) if prev is not None else None
    delta_wc = (wc_curr - wc_prev) if (wc_curr is not None and wc_prev is not None) else 0.0

    oe: Optional[float] = None
    if latest.net_income is not None:
        da = latest.depreciation_amortization or 0.0
        mc = maint_capex if maint_capex is not None else (capex or 0.0)
        oe = latest.net_income + da - mc - delta_wc

    shares = company.diluted_shares_current or latest.weighted_avg_shares_diluted
    oe_ps = (oe / shares) if (oe is not None and shares and shares > 0) else None

    return CashFlowDecomposition(
        year=latest.year,
        ocf=ocf,
        capex=capex,
        sbc=sbc,
        fcf=fcf,
        fcf_adj=fcf_adj,
        net_income=latest.net_income,
        depreciation_amortization=latest.depreciation_amortization,
        maintenance_capex=maint_capex,
        delta_working_capital=delta_wc if (wc_curr is not None and wc_prev is not None) else None,
        owner_earnings=oe,
        owner_earnings_per_share=oe_ps,
    )


def compute_predictability_stars(company: CompanyData) -> tuple[float, str]:
    """Compute GuruFocus-style Business Predictability Rank (1.0 to 5.0 stars) from revenue & FCF/EBITDA consistency."""
    annuals = company.annuals[-10:]
    if len(annuals) < 3:
        return 2.5, "Omezená historie (nelze plně ověřit)"

    revs = [a.revenue for a in annuals if a.revenue is not None and a.revenue > 0]
    ebitdas = [a.ebitda for a in annuals if a.ebitda is not None]
    margins = [
        a.operating_income / a.revenue
        for a in annuals
        if a.operating_income is not None and a.revenue and a.revenue > 0
    ]

    score = 3.0
    if len(revs) >= 3:
        rev_yoy = [(revs[i] / revs[i - 1]) - 1.0 for i in range(1, len(revs))]
        pos_rev_share = sum(1 for g in rev_yoy if g > 0) / len(rev_yoy)
        rev_vol = float(np.std(rev_yoy))
        score += (pos_rev_share - 0.5) * 2.2
        score -= min(1.2, rev_vol * 4.5)

    if len(ebitdas) >= 3:
        neg_ebitda_share = sum(1 for e in ebitdas if e <= 0) / len(ebitdas)
        score -= neg_ebitda_share * 2.0

    if len(margins) >= 3:
        m_std = float(np.std(margins))
        if m_std < 0.025:
            score += 0.6
        elif m_std > 0.08:
            score -= 0.7

    stars = round(float(np.clip(score, 1.0, 5.0)) * 2.0) / 2.0
    if stars >= 4.5:
        lbl = "Mimořádně předvídatelný byznys"
    elif stars >= 3.5:
        lbl = "Nadprůměrně stabilní poptávka"
    elif stars >= 2.5:
        lbl = "Střední předvídatelnost / cykličnost"
    else:
        lbl = "Vysoká cykličnost nebo nestabilita"
    return stars, lbl


def _build_gf_value_history(
    company: CompanyData,
    fair_pe: float,
    fair_pfcf: float,
    gf_fair_value_now: Optional[float] = None,
    annual_growth: float = 0.08,
    min_projection_year: int = 2032,
) -> pd.DataFrame:
    """Construct GuruFocus-style Fair Value corridor (historical + projected forward through at least 2032)."""
    rows = []
    last_hist_year = 0
    for a in company.annuals[-10:]:
        price_yr = a.avg_share_price
        shares = a.weighted_avg_shares_diluted
        if price_yr is None or price_yr <= 0 or shares is None or shares <= 0:
            continue

        ests: list[float] = []
        if a.eps_diluted is not None and a.eps_diluted > 0:
            ests.append(a.eps_diluted * fair_pe)
        if a.free_cash_flow is not None:
            fcf_adj = a.free_cash_flow - (a.stock_based_compensation or 0.0)
            if fcf_adj > 0:
                ests.append((fcf_adj / shares) * fair_pfcf)

        if not ests:
            continue
        fv_yr = float(np.mean(ests))
        last_hist_year = max(last_hist_year, a.year)
        rows.append(
            {
                "year": str(a.year),
                "price": float(price_yr),
                "fair_value": fv_yr,
                "band_low30": fv_yr * 0.70,
                "band_low10": fv_yr * 0.90,
                "band_high10": fv_yr * 1.10,
                "band_high30": fv_yr * 1.30,
                "is_projected": False,
            }
        )

    current_year = max(datetime.now().year, last_hist_year + 1 if last_hist_year else datetime.now().year)
    anchor_fv = (
        float(gf_fair_value_now)
        if (gf_fair_value_now is not None and gf_fair_value_now > 0)
        else (rows[-1]["fair_value"] if rows else None)
    )

    if anchor_fv is not None and anchor_fv > 0:
        # "Dnes" anchor point (live price + live GF Fair Value)
        rows.append(
            {
                "year": "Dnes",
                "price": float(company.current_price) if (company.current_price and company.current_price > 0) else None,
                "fair_value": anchor_fv,
                "band_low30": anchor_fv * 0.70,
                "band_low10": anchor_fv * 0.90,
                "band_high10": anchor_fv * 1.10,
                "band_high30": anchor_fv * 1.30,
                "is_projected": False,
            }
        )

        # Project GF Fair Value corridor forward through at least 2032 (like real GuruFocus)
        end_proj_year = max(min_projection_year, current_year + 5)
        proj_fv = anchor_fv
        base_g = float(np.clip(annual_growth, -0.05, 0.18))
        for step, yr in enumerate(range(current_year + 1, end_proj_year + 1), start=1):
            # Gentle growth fade over longer projection horizons
            g_step = base_g * max(0.70, 1.0 - 0.04 * (step - 1))
            proj_fv = proj_fv * (1.0 + g_step)
            rows.append(
                {
                    "year": str(yr),
                    "price": None,
                    "fair_value": proj_fv,
                    "band_low30": proj_fv * 0.70,
                    "band_low10": proj_fv * 0.90,
                    "band_high10": proj_fv * 1.10,
                    "band_high30": proj_fv * 1.30,
                    "is_projected": True,
                }
            )

    return pd.DataFrame(rows)


def compute_full_valuation_suite(
    company: CompanyData,
    dcf_params: ReverseDCFInputs,
    quality: QualityXRayReport,
    red_flags: RedFlagsReport,
    forensics: ForensicsReport,
    monte_carlo: MonteCarloResult,
    margin_of_safety: float = 0.25,
    aaa_yield_override_pct: Optional[float] = None,
    graham_weight_pct: float = 10.0,
) -> ValuationSuiteReport:
    """Compute all 8 per-share valuation models, exact formulas, GF Value bands, and 5-pillar 10-point scorecards."""
    price = company.current_price
    shares = company.diluted_shares_current
    cash = company.latest_cash or 0.0
    debt = company.latest_total_debt or 0.0
    net_debt = debt - cash
    ccy = company.quote_currency
    latest = company.annuals[-1] if company.annuals else None

    decomp = compute_cashflow_decomposition(company)
    rev_g_mean, _ = _historical_revenue_growth_stats(company.annuals)
    comp_g = forensics.compounding.intrinsic_compounding_rate
    hard_g = float(np.clip(np.mean([g for g in (rev_g_mean, comp_g) if g is not None]), 0.0, 0.18))

    # AAA corporate bond yield Y for Graham formula: 10Y Treasury + 1.0 p.b. spread unless overridden
    rf_pct = (company.risk_free_rate * 100.0) if company.risk_free_rate is not None else 4.3
    aaa_yield_pct = aaa_yield_override_pct if aaa_yield_override_pct is not None else round(rf_pct + 1.0, 2)

    def _upside(v: Optional[float]) -> Optional[float]:
        if v is None or price is None or price <= 0:
            return None
        return (v / price) - 1.0

    def _buy_p(v: Optional[float]) -> Optional[float]:
        return v * (1.0 - margin_of_safety) if (v is not None and v > 0) else None

    models: list[FormulaModelRow] = []

    # Growth Optionality (AI / Future Projects) per share
    opt_per_share = 0.0
    opt_note = ""
    opt_active = bool(dcf_params.enable_growth_optionality and dcf_params.optionality_fcf_billions > 0)
    if dcf_params.enable_growth_optionality and shares and shares > 0:
        opt_val, opt_err = compute_growth_optionality_per_share(
            fcf_billions=dcf_params.optionality_fcf_billions,
            ramp_years=dcf_params.optionality_ramp_years,
            probability=dcf_params.optionality_probability,
            discount_rate=dcf_params.discount_rate,
            perpetual_growth_rate=dcf_params.perpetual_growth_rate,
            diluted_shares=shares,
        )
        if opt_err:
            opt_note = opt_err
        elif opt_val is not None:
            opt_per_share = opt_val
            if opt_active:
                opt_note = (
                    f"Opce: {fmt_number(dcf_params.optionality_fcf_billions, 1)} mld. FCF za "
                    f"{dcf_params.optionality_ramp_years} let × {int(dcf_params.optionality_probability * 100)} % "
                    f"pravděpodobnost = +{fmt_price(opt_per_share, ccy)}/akcii"
                )

    use_two_stage = dcf_params.stage1_growth_rate is not None
    eff_stage1_g = dcf_params.stage1_growth_rate if dcf_params.stage1_growth_rate is not None else hard_g

    # 1. DCF – Gordon Growth Model
    fcf_0, fcf_lbl = select_fcf_base(
        company,
        dcf_params.fcf_base_type,
        normalize_capex=dcf_params.normalize_capex,
        custom_fcf_0_millions=dcf_params.custom_fcf_0_millions,
    )
    core_dcf_gordon_val: Optional[float] = None
    dcf_gordon_val: Optional[float] = None
    dcf_g_note = ""
    if fcf_0 is not None and fcf_0 > 0 and shares and shares > 0:
        try:
            if use_two_stage:
                v_ps, _, _ = compute_two_stage_fading_dcf_value_per_share(
                    fcf_0=fcf_0,
                    stage1_growth_rate=eff_stage1_g,
                    discount_rate=dcf_params.discount_rate,
                    horizon_years=dcf_params.horizon_years,
                    net_debt=net_debt,
                    diluted_shares=shares,
                    terminal_method="perpetual_growth",
                    perpetual_growth_rate=dcf_params.perpetual_growth_rate,
                )
            else:
                v_ps, _, _ = compute_dcf_value_per_share(
                    fcf_0=fcf_0,
                    growth_rate=hard_g,
                    discount_rate=dcf_params.discount_rate,
                    horizon_years=dcf_params.horizon_years,
                    net_debt=net_debt,
                    diluted_shares=shares,
                    terminal_method="perpetual_growth",
                    perpetual_growth_rate=dcf_params.perpetual_growth_rate,
                )
            if v_ps > 0:
                core_dcf_gordon_val = v_ps
                dcf_gordon_val = v_ps + (opt_per_share if opt_active else 0.0)
        except ValueError as exc:
            dcf_g_note = str(exc)
    else:
        dcf_g_note = "Záporné nebo chybějící FCF₀."

    g_desc = (
        f"1. fáze (1–5 r.) {fmt_percent(eff_stage1_g, 1)} → lineární pokles na {fmt_percent(dcf_params.perpetual_growth_rate, 1)}"
        if use_two_stage
        else f"g = {fmt_percent(hard_g, 1)}"
    )
    opt_suffix = f" | Core {fmt_price(core_dcf_gordon_val, ccy)} + Opce {fmt_price(opt_per_share, ccy)}" if (opt_active and core_dcf_gordon_val) else ""

    models.append(
        FormulaModelRow(
            key="dcf_gordon",
            name_cz="DCF – Gordonův model růstu",
            formula_latex=r"V_{\text{akcie}} = \frac{\sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{FCF_n(1+g_{\infty})}{(r-g_{\infty})(1+r)^n} + \text{Hotovost} - \text{Dluh}}{\text{Zředěné akcie}}",
            value_per_share=dcf_gordon_val,
            buy_price_mos=_buy_p(dcf_gordon_val),
            upside_pct=_upside(dcf_gordon_val),
            weight=0.22,
            inputs_summary_cz=f"{fcf_lbl}, {g_desc}, r = {fmt_percent(dcf_params.discount_rate, 1)}{opt_suffix}",
            status_note_cz=dcf_g_note,
        )
    )

    # 2. DCF – Exit Multiple Model
    core_dcf_exit_val: Optional[float] = None
    dcf_exit_val: Optional[float] = None
    dcf_e_note = ""
    if fcf_0 is not None and fcf_0 > 0 and shares and shares > 0:
        try:
            if use_two_stage:
                v_ps, _, _ = compute_two_stage_fading_dcf_value_per_share(
                    fcf_0=fcf_0,
                    stage1_growth_rate=eff_stage1_g,
                    discount_rate=dcf_params.discount_rate,
                    horizon_years=dcf_params.horizon_years,
                    net_debt=net_debt,
                    diluted_shares=shares,
                    terminal_method="exit_multiple",
                    perpetual_growth_rate=dcf_params.perpetual_growth_rate,
                    exit_multiple=dcf_params.exit_multiple,
                )
            else:
                v_ps, _, _ = compute_dcf_value_per_share(
                    fcf_0=fcf_0,
                    growth_rate=hard_g,
                    discount_rate=dcf_params.discount_rate,
                    horizon_years=dcf_params.horizon_years,
                    net_debt=net_debt,
                    diluted_shares=shares,
                    terminal_method="exit_multiple",
                    exit_multiple=dcf_params.exit_multiple,
                )
            if v_ps > 0:
                core_dcf_exit_val = v_ps
                dcf_exit_val = v_ps + (opt_per_share if opt_active else 0.0)
        except ValueError as exc:
            dcf_e_note = str(exc)
    else:
        dcf_e_note = "Záporné nebo chybějící FCF₀."

    opt_suffix_e = f" | Core {fmt_price(core_dcf_exit_val, ccy)} + Opce {fmt_price(opt_per_share, ccy)}" if (opt_active and core_dcf_exit_val) else ""

    models.append(
        FormulaModelRow(
            key="dcf_exit",
            name_cz="DCF – Exit Multiple",
            formula_latex=r"TV = FCF_n \cdot \text{Násobek}, \quad V_{\text{akcie}} = \frac{PV(FCF) + PV(TV) + \text{Hotovost} - \text{Dluh}}{\text{Zředěné akcie}}",
            value_per_share=dcf_exit_val,
            buy_price_mos=_buy_p(dcf_exit_val),
            upside_pct=_upside(dcf_exit_val),
            weight=0.18,
            inputs_summary_cz=f"{fcf_lbl}, {g_desc}, Exit Multiple = {fmt_multiple(dcf_params.exit_multiple, 0)}, r = {fmt_percent(dcf_params.discount_rate, 1)}{opt_suffix_e}",
            status_note_cz=dcf_e_note,
        )
    )

    # 3. Buffett Owner Earnings (NI + D&A - Maintenance CapEx - Delta Working Capital)
    oe_val: Optional[float] = None
    oe_note = ""
    if decomp.owner_earnings is not None and decomp.owner_earnings > 0 and shares and shares > 0:
        g_oe = float(np.clip(hard_g * 0.85, 0.0, 0.14))
        r_oe = max(dcf_params.discount_rate, 0.06)
        try:
            v_oe, _, _ = compute_dcf_value_per_share(
                fcf_0=decomp.owner_earnings,
                growth_rate=g_oe,
                discount_rate=r_oe,
                horizon_years=5,
                net_debt=net_debt,
                diluted_shares=shares,
                terminal_method="perpetual_growth",
                perpetual_growth_rate=min(0.025, r_oe - 0.02),
            )
            if v_oe > 0:
                oe_val = v_oe
        except ValueError as exc:
            oe_note = str(exc)
    else:
        oe_note = "Owner Earnings jsou záporné nebo chybí složky."

    models.append(
        FormulaModelRow(
            key="owner_earnings",
            name_cz="Buffettovy Owner Earnings",
            formula_latex=r"\text{Owner Earnings} = \text{Čistý zisk} + \text{D\&A} - \text{Udržovací CapEx} - \Delta\text{Pracovní kapitál}",
            value_per_share=oe_val,
            buy_price_mos=_buy_p(oe_val),
            upside_pct=_upside(oe_val),
            weight=0.16,
            inputs_summary_cz=f"OE = {fmt_large_currency(decomp.owner_earnings, ccy)} (ΔWC = {fmt_large_currency(decomp.delta_working_capital, ccy)}), 5Y růst {fmt_percent(hard_g * 0.85, 1)}",
            status_note_cz=oe_note,
        )
    )

    # 4. EPV – Earnings Power Value (Bruce Greenwald)
    epv_ev: Optional[float] = None
    epv_ps: Optional[float] = None
    epv_note = ""
    wacc = quality.wacc_components.wacc
    if company.annuals and shares and shares > 0 and wacc and wacc > 0.03:
        op_margins = [
            a.operating_income / a.revenue
            for a in company.annuals[-7:]
            if a.operating_income is not None and a.revenue and a.revenue > 0
        ]
        latest_rev = company.ttm_revenue or (latest.revenue if latest else None)
        if op_margins and latest_rev and latest_rev > 0:
            norm_m = float(np.median(op_margins))
            eff_tax = quality.wacc_components.effective_tax_rate or 0.21
            norm_nopat = latest_rev * norm_m * (1.0 - eff_tax)
            if norm_nopat > 0:
                epv_ev = norm_nopat / wacc
                epv_eq = epv_ev + cash - debt
                if epv_eq > 0:
                    epv_ps = epv_eq / shares
                else:
                    epv_note = "Čistý dluh převyšuje EPV bez růstu."
            else:
                epv_note = "Normalizovaný NOPAT není kladný."
    else:
        epv_note = "Chybí WACC nebo provozní zisk."

    models.append(
        FormulaModelRow(
            key="epv",
            name_cz="EPV – Earnings Power Value (bez růstu)",
            formula_latex=r"EPV = \frac{NOPAT}{WACC}, \quad V_{\text{akcie}} = \frac{EPV + \text{Hotovost} - \text{Dluh}}{\text{Zředěné akcie}}",
            value_per_share=epv_ps,
            buy_price_mos=_buy_p(epv_ps),
            upside_pct=_upside(epv_ps),
            weight=0.10,
            inputs_summary_cz=f"EPV podniku = {fmt_large_currency(epv_ev, ccy)}, WACC = {fmt_percent(wacc, 2)}",
            status_note_cz=epv_note,
        )
    )

    # 5. Benjamin Graham Formula: V = EPS * (8.5 + 2*g) * 4.4 / Y
    graham_val: Optional[float] = None
    graham_note = ""
    eps_now = (
        (company.ttm_net_income / shares)
        if (company.ttm_net_income is not None and shares and shares > 0)
        else (latest.eps_diluted if latest else None)
    )
    g_pct_graham = float(np.clip(hard_g * 100.0, 0.0, 20.0))
    if eps_now is not None and eps_now > 0 and aaa_yield_pct > 0:
        graham_val = (eps_now * (8.5 + 2.0 * g_pct_graham) * 4.4) / aaa_yield_pct
    else:
        graham_note = "Záporné nebo nedostupné EPS."

    models.append(
        FormulaModelRow(
            key="graham",
            name_cz="Grahamův vzorec",
            formula_latex=r"V_{\text{akcie}} = \frac{EPS \cdot (8{,}5 + 2g) \cdot 4{,}4}{Y}",
            value_per_share=graham_val,
            buy_price_mos=_buy_p(graham_val),
            upside_pct=_upside(graham_val),
            weight=0.10,
            inputs_summary_cz=f"EPS = {fmt_price(eps_now, ccy)}, g = {fmt_number(g_pct_graham, 1)} %, výnos AAA (Y) = {fmt_number(aaa_yield_pct, 2)} %",
            status_note_cz=graham_note,
        )
    )

    # 6. DDM – Dividend Discount Model: V = D_1 / (r - g)
    ddm_val: Optional[float] = None
    ddm_note = ""
    dps_series = [
        (a.year, a.dividends_paid / a.weighted_avg_shares_diluted)
        for a in company.annuals
        if a.dividends_paid and a.dividends_paid > 0 and a.weighted_avg_shares_diluted and a.weighted_avg_shares_diluted > 0
    ]
    if dps_series:
        d0 = dps_series[-1][1]
        if len(dps_series) >= 3:
            span = max(1, dps_series[-1][0] - dps_series[0][0])
            raw_dg = (dps_series[-1][1] / dps_series[0][1]) ** (1.0 / span) - 1.0
        else:
            raw_dg = 0.03
        g_div = float(np.clip(raw_dg, 0.0, min(0.08, dcf_params.discount_rate - 0.015)))
        if dcf_params.discount_rate > g_div + 0.005:
            d1 = d0 * (1.0 + g_div)
            ddm_val = d1 / (dcf_params.discount_rate - g_div)
            ddm_inputs = f"D₁ = {fmt_price(d1, ccy)}, g_div = {fmt_percent(g_div, 1)}, r = {fmt_percent(dcf_params.discount_rate, 1)}"
        else:
            ddm_note = "Požadovaná návratnost r musí převyšovat růst dividendy g."
            ddm_inputs = f"D₀ = {fmt_price(d0, ccy)}"
    else:
        ddm_note = "Společnost nevyplácí dividendu (DDM se neaplikuje)."
        ddm_inputs = "Dividenda = 0"

    # Only include DDM in weighted average when company meaningfully returns capital via dividends (yield >= 1.5%)
    div_yield_now = (dps_series[-1][1] / price) if (dps_series and price and price > 0) else 0.0
    ddm_weight = 0.08 if (ddm_val is not None and div_yield_now >= 0.015) else 0.0

    models.append(
        FormulaModelRow(
            key="ddm",
            name_cz="DDM – Dividendový diskontní model",
            formula_latex=r"V_{\text{akcie}} = \frac{D_1}{r - g}",
            value_per_share=ddm_val,
            buy_price_mos=_buy_p(ddm_val),
            upside_pct=_upside(ddm_val),
            weight=ddm_weight,
            inputs_summary_cz=ddm_inputs,
            status_note_cz=ddm_note if ddm_note else ("Informační (nízký div. výnos < 1,5 %)" if ddm_weight == 0 else ""),
        )
    )

    # 7. Relative Valuation – P/E: V = Expected EPS * Fair P/E
    hist_pe = [a.pe_ratio for a in company.annuals[-10:] if a.pe_ratio is not None and 5.0 <= a.pe_ratio <= 60.0]
    fair_pe = float(np.clip(0.65 * np.median(hist_pe) + 0.35 * 16.0, 9.0, 30.0)) if hist_pe else 16.0
    eps_exp = (eps_now * (1.0 + hard_g)) if (eps_now is not None and eps_now > 0) else None
    rel_pe_val = (eps_exp * fair_pe) if eps_exp is not None else None

    models.append(
        FormulaModelRow(
            key="rel_pe",
            name_cz="Relativní ocenění (P/E)",
            formula_latex=r"V_{\text{akcie}} = EPS_{\text{očekávané}} \cdot \text{Férové } P/E",
            value_per_share=rel_pe_val,
            buy_price_mos=_buy_p(rel_pe_val),
            upside_pct=_upside(rel_pe_val),
            weight=0.12,
            inputs_summary_cz=f"EPS oček. = {fmt_price(eps_exp, ccy)}, Férové P/E = {fmt_multiple(fair_pe, 1)}",
            status_note_cz="" if rel_pe_val else "Záporné nebo chybějící EPS.",
        )
    )

    # 8. Relative Valuation – P/FCF: V = FCF_adj per share * Fair P/FCF
    hist_pfcf = [a.pfcf_ratio for a in company.annuals[-10:] if a.pfcf_ratio is not None and 5.0 <= a.pfcf_ratio <= 60.0]
    fair_pfcf = float(np.clip(0.65 * np.median(hist_pfcf) + 0.35 * 16.0, 9.0, 30.0)) if hist_pfcf else 16.0
    fcf_ps = (fcf_0 / shares) if (fcf_0 is not None and fcf_0 > 0 and shares and shares > 0) else None
    rel_pfcf_val = (fcf_ps * fair_pfcf) if fcf_ps is not None else None

    models.append(
        FormulaModelRow(
            key="rel_pfcf",
            name_cz="Relativní ocenění (P/FCF)",
            formula_latex=r"V_{\text{akcie}} = \text{FCF na akcii} \cdot \text{Férové } P/FCF",
            value_per_share=rel_pfcf_val,
            buy_price_mos=_buy_p(rel_pfcf_val),
            upside_pct=_upside(rel_pfcf_val),
            weight=0.12,
            inputs_summary_cz=f"FCF/akcii = {fmt_price(fcf_ps, ccy)}, Férové P/FCF = {fmt_multiple(fair_pfcf, 1)}",
            status_note_cz="" if rel_pfcf_val else "Záporné nebo chybějící FCF.",
        )
    )

    # Weighted GF-style Fair Value honoring explicit Graham formula weight (0 % .. 100 %)
    g_w_frac = float(np.clip(graham_weight_pct / 100.0, 0.0, 1.0))
    graham_model = next((m for m in models if m.key == "graham"), None)
    if graham_model is not None and g_w_frac == 0.0:
        graham_model.weight = 0.0
        if not graham_model.status_note_cz:
            graham_model.inputs_summary_cz = f"{graham_model.inputs_summary_cz} (váha 0 % – vypnuto z průměru)"

    non_graham = [
        m for m in models
        if m.key != "graham" and m.value_per_share is not None and m.value_per_share > 0 and m.weight > 0
    ]
    graham_valid = (
        graham_model is not None
        and graham_model.value_per_share is not None
        and graham_model.value_per_share > 0
    )

    gf_fv: Optional[float] = None
    if g_w_frac >= 1.0 and graham_valid:
        assert graham_model is not None and graham_model.value_per_share is not None
        gf_fv = graham_model.value_per_share
    elif g_w_frac > 0.0 and graham_valid and non_graham:
        assert graham_model is not None and graham_model.value_per_share is not None
        other_w_sum = sum(m.weight for m in non_graham)
        other_avg = sum((m.value_per_share or 0.0) * m.weight for m in non_graham) / other_w_sum
        gf_fv = g_w_frac * graham_model.value_per_share + (1.0 - g_w_frac) * other_avg
    elif non_graham:
        other_w_sum = sum(m.weight for m in non_graham)
        gf_fv = sum((m.value_per_share or 0.0) * m.weight for m in non_graham) / other_w_sum
    elif graham_valid and g_w_frac > 0.0:
        assert graham_model is not None and graham_model.value_per_share is not None
        gf_fv = graham_model.value_per_share

    gf_buy = _buy_p(gf_fv)
    p_to_gf = (price / gf_fv) if (price and price > 0 and gf_fv and gf_fv > 0) else None
    upside = _upside(gf_fv)

    # GuruFocus-style 5-tier valuation label
    if p_to_gf is None:
        gf_status: GFStatus = "na"
        gf_label = "Nelze spolehlivě ocenit"
        gf_color: Literal["green", "amber", "red", "na"] = "na"
    elif p_to_gf <= 0.75:
        gf_status = "significantly_undervalued"
        gf_label = "Výrazně podhodnoceno"
        gf_color = "green"
    elif p_to_gf <= 0.92:
        gf_status = "modestly_undervalued"
        gf_label = "Mírně podhodnoceno"
        gf_color = "green"
    elif p_to_gf <= 1.10:
        gf_status = "fairly_valued"
        gf_label = "Férově oceněno"
        gf_color = "amber"
    elif p_to_gf <= 1.30:
        gf_status = "modestly_overvalued"
        gf_label = "Mírně nadhodnoceno"
        gf_color = "red"
    else:
        gf_status = "significantly_overvalued"
        gf_label = "Výrazně nadhodnoceno"
        gf_color = "red"

    epv_share = (epv_ps / price) if (epv_ps is not None and price and price > 0) else None
    growth_bet = (1.0 - epv_share) if epv_share is not None else None

    stars, star_lbl = compute_predictability_stars(company)
    gf_hist_df = _build_gf_value_history(
        company,
        fair_pe=fair_pe,
        fair_pfcf=fair_pfcf,
        gf_fair_value_now=gf_fv,
        annual_growth=hard_g,
        min_projection_year=2032,
    )

    # --- GuruFocus 5-Pillar 10-Point Scorecards ---
    def _c10(s: int) -> Literal["green", "amber", "red"]:
        return "green" if s >= 7 else ("amber" if s >= 4 else "red")

    # 1. Financial Strength (0..10)
    ds = red_flags.debt_stress
    fs = 5.0
    if forensics.altman.z_score is not None:
        z = forensics.altman.z_score
        fs = 9.0 if z > 4.0 else (8.0 if z > 2.99 else (5.0 if z >= 1.81 else 2.0))
    if ds.net_debt is not None and ds.net_debt <= 0:
        fs = min(10.0, fs + 1.5)
    elif ds.net_debt_to_ebitda is not None and ds.net_debt_to_ebitda > 3.5:
        fs = max(1.0, fs - 2.5)
    if ds.stressed_interest_coverage is not None and ds.stressed_interest_coverage < 2.5:
        fs = max(1.0, fs - 2.0)
    fs_int = int(round(float(np.clip(fs, 1, 10))))

    # 2. Profitability Rank (0..10)
    prof = 5.0
    if quality.avg_roic is not None:
        prof = float(np.clip(3.0 + quality.avg_roic * 28.0, 1.0, 9.5))
    if quality.avg_operating_margin is not None and quality.avg_operating_margin >= 0.20:
        prof = min(10.0, prof + 1.0)
    if quality.years_below_wacc > 0:
        prof = max(1.0, prof - quality.years_below_wacc * 0.8)
    prof_int = int(round(float(np.clip(prof, 1, 10))))

    # 3. Growth & Compounding Rank (0..10)
    gr = 5.0
    g_ref = forensics.compounding.intrinsic_compounding_rate if forensics.compounding.intrinsic_compounding_rate is not None else rev_g_mean
    gr = float(np.clip(3.5 + g_ref * 35.0, 1.0, 9.5))
    if stars >= 4.0:
        gr = min(10.0, gr + 0.8)
    gr_int = int(round(float(np.clip(gr, 1, 10))))

    # 4. GF Value Rank (0..10)
    if p_to_gf is not None:
        # 0.65 -> 10, 1.00 -> 6, 1.45 -> 1
        gv = float(np.clip(10.0 - (p_to_gf - 0.65) * 11.2, 1.0, 10.0))
        if monte_carlo.available:
            gv = 0.6 * gv + 0.4 * (monte_carlo.prob_beat_hurdle * 10.0)
    else:
        gv = 3.0
    gv_int = int(round(float(np.clip(gv, 1, 10))))

    # 5. Forensic Cleanliness Rank (0..10)
    for_r = (100 - red_flags.red_flag_score) / 10.0
    if forensics.piotroski.score is not None and forensics.piotroski.max_evaluated > 0:
        for_r = 0.5 * for_r + 0.5 * (forensics.piotroski.score / forensics.piotroski.max_evaluated) * 10.0
    if forensics.beneish.status == "red":
        for_r = min(for_r, 3.0)
    for_int = int(round(float(np.clip(for_r, 1, 10))))

    pillars = [
        GFPillarCard(
            "strength",
            "Finanční síla",
            fs_int,
            _c10(fs_int),
            f"Altman Z: {fmt_number(forensics.altman.z_score, 2)}",
            f"Čistý dluh/EBITDA {fmt_multiple(ds.net_debt_to_ebitda)} · Stress krytí {fmt_multiple(ds.stressed_interest_coverage)}",
        ),
        GFPillarCard(
            "profitability",
            "Ziskovost",
            prof_int,
            _c10(prof_int),
            f"ROIC: {fmt_percent(quality.avg_roic, 1)}",
            f"Hrubá m. {fmt_percent(quality.avg_gross_margin, 1)} · Provoz. m. {fmt_percent(quality.avg_operating_margin, 1)}",
        ),
        GFPillarCard(
            "growth",
            "Růst & Compounding",
            gr_int,
            _c10(gr_int),
            f"Compounding: {fmt_percent(g_ref, 1)}/r.",
            f"Předvídatelnost: {'★' * int(stars)}{'☆' * (5 - int(stars))} ({fmt_number(stars, 1)}/5)",
        ),
        GFPillarCard(
            "gf_value",
            "GF Ocenění",
            gv_int,
            _c10(gv_int),
            f"Cena/GF: {fmt_number(p_to_gf, 2)}x" if p_to_gf else "N/A",
            f"Férová cena {fmt_price(gf_fv, ccy)} ({fmt_percent(upside, 0, signed=True)})",
        ),
        GFPillarCard(
            "forensics",
            "Účetní čistota",
            for_int,
            _c10(for_int),
            f"F-Score: {forensics.piotroski.score}/{forensics.piotroski.max_evaluated}"
            if forensics.piotroski.score is not None
            else "F-Score: N/A",
            f"Beneish M: {fmt_number(forensics.beneish.m_score, 2)} · Red Flags {red_flags.red_flag_score}/100",
        ),
    ]

    gf_score_100 = int(round((fs_int * 0.20 + prof_int * 0.25 + gr_int * 0.15 + gv_int * 0.25 + for_int * 0.15) * 10.0))

    return ValuationSuiteReport(
        models=models,
        gf_fair_value=gf_fv,
        gf_buy_price=gf_buy,
        margin_of_safety=margin_of_safety,
        price_to_gf_value=p_to_gf,
        upside_pct=upside,
        gf_status=gf_status,
        gf_status_label_cz=gf_label,
        gf_status_color=gf_color,
        epv_enterprise_value=epv_ev,
        epv_per_share=epv_ps,
        epv_share_of_price=epv_share,
        growth_bet_share_of_price=growth_bet,
        cashflow_decomp=decomp,
        aaa_yield_pct=aaa_yield_pct,
        predictability_stars=stars,
        predictability_label_cz=star_lbl,
        gf_pillars=pillars,
        gf_value_history_df=gf_hist_df,
        composite_gf_score=gf_score_100,
        default_stage1_growth=hard_g,
        effective_stage1_growth=eff_stage1_g,
        used_two_stage_fade=use_two_stage,
        core_dcf_gordon_per_share=core_dcf_gordon_val,
        core_dcf_exit_per_share=core_dcf_exit_val,
        optionality_per_share=opt_per_share if opt_active else 0.0,
        optionality_active=opt_active,
        optionality_note_cz=opt_note,
        graham_weight_pct=float(graham_weight_pct),
    )
