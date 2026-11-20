"""Pure Reverse DCF model, brentq solver, reality check, and sensitivity grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from analysis.formatting import fmt_number, fmt_percent, fmt_price
from data.models import AnalystEstimate, AnnualFinancials, CompanyData

TerminalMethod = Literal["perpetual_growth", "exit_multiple"]
FcfBaseType = Literal["fcf_minus_sbc", "reported_fcf"]
VerdictLevel = Literal["realistic", "ambitious", "extreme", "unavailable"]


@dataclass(frozen=True)
class ReverseDCFInputs:
    """Parameters governing the Reverse DCF, 2-stage DCF, and Growth Optionality valuations."""

    discount_rate: float = 0.12
    horizon_years: int = 10
    terminal_method: TerminalMethod = "perpetual_growth"
    perpetual_growth_rate: float = 0.025
    exit_multiple: float = 15.0
    fcf_base_type: FcfBaseType = "fcf_minus_sbc"
    normalize_capex: bool = False
    custom_fcf_0_millions: float = 0.0
    # 2-Stage Fading DCF (Stage 1: years 1-5, Stage 2: years 6..N linear fade to perpetual_growth_rate)
    stage1_growth_rate: Optional[float] = None
    # Growth Optionality (AI / Autonomous Driving / Future Business Lines)
    enable_growth_optionality: bool = False
    optionality_fcf_billions: float = 0.0
    optionality_ramp_years: int = 5
    optionality_probability: float = 0.50


@dataclass
class CapExDistortionDiagnostics:
    """Diagnostics detecting extraordinary CapEx spikes distorting TTM FCF."""

    current_capex_to_rev: Optional[float]
    hist_4y_avg_capex_to_rev: Optional[float]
    hist_5y_median_capex_to_rev: Optional[float]
    spike_ratio: Optional[float]
    raw_fcf_0: Optional[float]
    normalized_capex: Optional[float]
    normalized_fcf_0: Optional[float]
    should_warn: bool
    warning_cz: str


@dataclass
class HistoricalCAGRs:
    """Historical compound annual growth rates for reality-checking implied growth."""

    revenue_cagr_3y: Optional[float] = None
    revenue_cagr_5y: Optional[float] = None
    revenue_cagr_10y: Optional[float] = None
    fcf_cagr_3y: Optional[float] = None
    fcf_cagr_5y: Optional[float] = None
    fcf_cagr_10y: Optional[float] = None
    consensus_rev_cagr: Optional[float] = None
    consensus_eps_cagr: Optional[float] = None


@dataclass
class ReverseDCFResult:
    """Output of the Reverse DCF solver and reality check."""

    fcf_0: Optional[float]
    fcf_0_label: str
    fcf_0_per_share: Optional[float]
    net_debt: float
    diluted_shares: Optional[float]
    current_price: Optional[float]
    implied_growth: Optional[float]
    implied_pv_fcf: Optional[float]
    implied_pv_tv: Optional[float]
    tv_share_of_ev: Optional[float]
    error_message: Optional[str]
    headline_cz: str
    cagrs: HistoricalCAGRs
    verdict: VerdictLevel
    verdict_label_cz: str
    verdict_explanation_cz: str
    benchmark_growth: Optional[float]


def compute_dcf_value_per_share(
    fcf_0: float,
    growth_rate: float,
    discount_rate: float,
    horizon_years: int,
    net_debt: float,
    diluted_shares: float,
    terminal_method: TerminalMethod = "perpetual_growth",
    perpetual_growth_rate: float = 0.025,
    exit_multiple: float = 15.0,
) -> tuple[float, float, float]:
    """Compute intrinsic equity value per share via a 2-stage FCF DCF model.

    Model:
        FCF_t = FCF_0 * (1 + g)^t  for t = 1..N
        TV = FCF_N * (1 + g_term) / (r - g_term)   [perpetual_growth]
           = FCF_N * exit_multiple                  [exit_multiple]
        Equity Value = PV(FCFs) + PV(TV) - Net Debt
        Value per share = Equity Value / Diluted Shares

    Returns:
        Tuple of (value_per_share, pv_fcfs, pv_tv).

    Raises:
        ValueError: If diluted_shares <= 0, horizon_years < 1, or r <= g_term in Gordon mode.
    """
    if diluted_shares <= 0:
        raise ValueError("Počet zředěných akcií musí být kladný.")
    if horizon_years < 1:
        raise ValueError("Horizont projekce musí být alespoň 1 rok.")
    if terminal_method == "perpetual_growth" and discount_rate <= perpetual_growth_rate:
        raise ValueError(
            "Diskontní sazba (požadovaná návratnost) musí být vyšší než míra trvalého růstu."
        )

    pv_fcfs = 0.0
    fcf_t = fcf_0
    for t in range(1, horizon_years + 1):
        fcf_t = fcf_0 * ((1.0 + growth_rate) ** t)
        pv_fcfs += fcf_t / ((1.0 + discount_rate) ** t)

    if terminal_method == "perpetual_growth":
        terminal_value = fcf_t * (1.0 + perpetual_growth_rate) / (discount_rate - perpetual_growth_rate)
    else:
        terminal_value = fcf_t * exit_multiple

    pv_tv = terminal_value / ((1.0 + discount_rate) ** horizon_years)
    equity_value = pv_fcfs + pv_tv - net_debt
    value_per_share = equity_value / diluted_shares
    return value_per_share, pv_fcfs, pv_tv


def compute_two_stage_fading_dcf_value_per_share(
    fcf_0: float,
    stage1_growth_rate: float,
    discount_rate: float,
    horizon_years: int,
    net_debt: float,
    diluted_shares: float,
    terminal_method: TerminalMethod = "perpetual_growth",
    perpetual_growth_rate: float = 0.025,
    exit_multiple: float = 15.0,
    stage1_years: int = 5,
) -> tuple[float, float, float]:
    """Compute intrinsic value per share via a 2-stage DCF with linear growth fade.

    Model:
        - Years 1..stage1_years (default 1..5): FCF grows at `stage1_growth_rate`.
        - Years (stage1_years+1)..horizon_years (default 6..10): annual growth rate
          linearly interpolates from `stage1_growth_rate` down to `perpetual_growth_rate`.
        - Terminal value at year N per `terminal_method` (Gordon or Exit Multiple).
        - Equity Value = PV(FCFs) + PV(TV) - Net Debt.

    Returns:
        Tuple of (value_per_share, pv_fcfs, pv_tv).
    """
    if diluted_shares <= 0:
        raise ValueError("Počet zředěných akcií musí být kladný.")
    if horizon_years < 1:
        raise ValueError("Horizont projekce musí být alespoň 1 rok.")
    if terminal_method == "perpetual_growth" and discount_rate <= perpetual_growth_rate:
        raise ValueError(
            "Diskontní sazba (požadovaná návratnost) musí být vyšší než míra trvalého růstu."
        )

    pv_fcfs = 0.0
    fcf_t = fcf_0
    fade_span = max(0, horizon_years - stage1_years)

    for t in range(1, horizon_years + 1):
        if t <= stage1_years or fade_span == 0:
            g_t = stage1_growth_rate
        else:
            progress = (t - stage1_years) / float(fade_span)
            g_t = stage1_growth_rate + (perpetual_growth_rate - stage1_growth_rate) * progress
        fcf_t = fcf_t * (1.0 + g_t)
        pv_fcfs += fcf_t / ((1.0 + discount_rate) ** t)

    if terminal_method == "perpetual_growth":
        terminal_value = fcf_t * (1.0 + perpetual_growth_rate) / (discount_rate - perpetual_growth_rate)
    else:
        terminal_value = fcf_t * exit_multiple

    pv_tv = terminal_value / ((1.0 + discount_rate) ** horizon_years)
    equity_value = pv_fcfs + pv_tv - net_debt
    value_per_share = equity_value / diluted_shares
    return value_per_share, pv_fcfs, pv_tv


def compute_growth_optionality_per_share(
    fcf_billions: float,
    ramp_years: int,
    probability: float,
    discount_rate: float,
    perpetual_growth_rate: float,
    diluted_shares: float,
) -> tuple[Optional[float], Optional[str]]:
    """Compute present value per share of future growth projects (Growth Optionality).

    Formula:
        TV_project = (FCF_project * (1 + g)) / (r - g)
        PV_project = TV_project / (1 + r)^ramp_years
        Expected_Value_per_share = (PV_project * probability) / diluted_shares

    Returns:
        Tuple of (optionality_value_per_share, error_message_cz).
    """
    if fcf_billions <= 0.0 or probability <= 0.0:
        return 0.0, None
    if diluted_shares <= 0:
        return None, "Chybí platný počet zředěných akcií pro výpočet růstové opce."
    if discount_rate <= perpetual_growth_rate:
        return (
            None,
            "Diskontní sazba r musí být vyšší než terminální růst g pro výpočet růstové opce.",
        )

    fcf_project = fcf_billions * 1e9
    tv_project = (fcf_project * (1.0 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
    pv_project = tv_project / ((1.0 + discount_rate) ** ramp_years)
    expected_pv = pv_project * probability
    return float(expected_pv / diluted_shares), None


def solve_implied_growth(
    current_price: float,
    fcf_0: float,
    discount_rate: float,
    horizon_years: int,
    net_debt: float,
    diluted_shares: float,
    terminal_method: TerminalMethod = "perpetual_growth",
    perpetual_growth_rate: float = 0.025,
    exit_multiple: float = 15.0,
    g_min: float = -0.85,
    g_max: float = 4.00,
) -> tuple[Optional[float], Optional[str]]:
    """Solve for the annual FCF growth rate g that reproduces `current_price` using scipy.optimize.brentq.

    Returns:
        Tuple of (implied_growth_rate, error_message_cz). Exactly one element is None.
    """
    if current_price <= 0:
        return None, "Aktuální tržní cena akcie není kladná – nelze vypočítat implikovaný růst."
    if diluted_shares <= 0:
        return None, "Chybí platný počet zředěných akcií."
    if fcf_0 <= 0:
        return (
            None,
            "Výchozí peněžní tok (FCF₀) je záporný nebo nulový. Reverse DCF nelze spolehlivě vyřešit – "
            "trh u této firmy oceňuje budoucí obrat k ziskovosti, nikoli prostý růst současného FCF.",
        )
    if terminal_method == "perpetual_growth" and discount_rate <= perpetual_growth_rate:
        return (
            None,
            "Požadovaná návratnost (diskontní sazba) musí být vyšší než míra trvalého růstu. "
            "Zvyšte diskontní sazbu nebo snižte terminální růst v postranním panelu.",
        )

    def objective(g: float) -> float:
        val, _, _ = compute_dcf_value_per_share(
            fcf_0=fcf_0,
            growth_rate=g,
            discount_rate=discount_rate,
            horizon_years=horizon_years,
            net_debt=net_debt,
            diluted_shares=diluted_shares,
            terminal_method=terminal_method,
            perpetual_growth_rate=perpetual_growth_rate,
            exit_multiple=exit_multiple,
        )
        return val - current_price

    f_low = objective(g_min)
    f_high = objective(g_max)

    if f_low > 0 and f_high > 0:
        return (
            None,
            f"I při poklesu FCF o {abs(g_min)*100:.0f} % ročně převyšuje vypočtená hodnota dnešní cenu akcie "
            "(trh buď oceňuje velmi silný propad, nebo má firma mimořádně vysokou čistou hotovost).",
        )
    if f_low < 0 and f_high < 0:
        return (
            None,
            f"Ani při extrémním růstu {g_max*100:.0f} % ročně nedosahuje vnitřní hodnota dnešní tržní ceny "
            "(trh oceňuje extrémní expanzi marží, nové produkty nebo nižší požadovanou návratnost).",
        )

    try:
        solved_g = float(brentq(objective, g_min, g_max, xtol=1e-7, rtol=1e-7, maxiter=200))
        return solved_g, None
    except ValueError as exc:
        return None, f"Numerický řešitel nenašel řešení pro zadané parametry ({exc})."


def compute_cagr(values: list[tuple[int, Optional[float]]], years_back: int) -> Optional[float]:
    """Compute CAGR over `years_back` fiscal years using the latest available positive endpoints."""
    valid = [(yr, val) for yr, val in values if val is not None]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda x: x[0])
    end_yr, end_val = valid[-1]
    if end_val is None or end_val <= 0:
        return None

    target_start_yr = end_yr - years_back
    # Find exact match or closest year within 1 year tolerance (requires >= years_back - 1 span)
    candidates = [item for item in valid[:-1] if abs(item[0] - target_start_yr) <= 1]
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(x[0] - target_start_yr))
    start_yr, start_val = candidates[0]
    span = end_yr - start_yr
    if span < max(2, years_back - 1) or start_val is None or start_val <= 0:
        return None
    return float((end_val / start_val) ** (1.0 / span) - 1.0)


def compute_historical_cagrs(
    annuals: list[AnnualFinancials],
    estimates: list[AnalystEstimate],
    fcf_base_type: FcfBaseType,
) -> HistoricalCAGRs:
    """Calculate 3Y/5Y/10Y historical revenue & FCF CAGRs plus forward analyst consensus CAGRs."""
    rev_series = [(a.year, a.revenue) for a in annuals]
    fcf_series: list[tuple[int, Optional[float]]] = []
    for a in annuals:
        if fcf_base_type == "fcf_minus_sbc" and a.free_cash_flow is not None:
            fcf_series.append((a.year, a.free_cash_flow - (a.stock_based_compensation or 0.0)))
        else:
            fcf_series.append((a.year, a.free_cash_flow))

    # Forward consensus CAGRs relative to latest annual baseline
    cons_rev_cagr: Optional[float] = None
    cons_eps_cagr: Optional[float] = None
    if annuals and estimates:
        latest = annuals[-1]
        future_rev = [e for e in estimates if e.year > latest.year and e.estimated_revenue_avg and e.estimated_revenue_avg > 0]
        if future_rev and latest.revenue and latest.revenue > 0:
            last_e = future_rev[-1]
            span = last_e.year - latest.year
            if span >= 1 and last_e.estimated_revenue_avg is not None:
                cons_rev_cagr = (last_e.estimated_revenue_avg / latest.revenue) ** (1.0 / span) - 1.0

        future_eps = [e for e in estimates if e.year > latest.year and e.estimated_eps_avg and e.estimated_eps_avg > 0]
        if future_eps and latest.eps_diluted and latest.eps_diluted > 0:
            last_e = future_eps[-1]
            span = last_e.year - latest.year
            if span >= 1 and last_e.estimated_eps_avg is not None:
                cons_eps_cagr = (last_e.estimated_eps_avg / latest.eps_diluted) ** (1.0 / span) - 1.0

    return HistoricalCAGRs(
        revenue_cagr_3y=compute_cagr(rev_series, 3),
        revenue_cagr_5y=compute_cagr(rev_series, 5),
        revenue_cagr_10y=compute_cagr(rev_series, 10),
        fcf_cagr_3y=compute_cagr(fcf_series, 3),
        fcf_cagr_5y=compute_cagr(fcf_series, 5),
        fcf_cagr_10y=compute_cagr(fcf_series, 10),
        consensus_rev_cagr=cons_rev_cagr,
        consensus_eps_cagr=cons_eps_cagr,
    )


def evaluate_growth_verdict(
    implied_growth: Optional[float],
    cagrs: HistoricalCAGRs,
) -> tuple[VerdictLevel, str, str, Optional[float]]:
    """Classify implied growth into Realistic / Ambitious / Extreme using explicit thresholds.

    Documented rules:
    1. Benchmark growth = highest available reference among 5Y revenue CAGR, 5Y FCF CAGR,
       and analyst consensus revenue CAGR (falling back to 3Y/10Y CAGRs if 5Y is missing).
       Capped at 20 % when acting as a "realistic" ceiling because compounding FCF above 20–25 %
       for an entire decade is historically rare.
    2. EXTREME (Extrémní):
       - implied_growth > 25 % p.a., OR
       - benchmark exists and implied_growth > benchmark + 6 p.b. (and implied_growth > 12 %).
    3. AMBITIOUS (Ambiciózní):
       - implied_growth > 15 % p.a., OR
       - benchmark exists and implied_growth > benchmark + 2 p.b. (and implied_growth > 8 %).
    4. REALISTIC (Realistické):
       - otherwise (implied growth is at or below historical/consensus pace + 2 p.b. and <= 15 %).
    """
    if implied_growth is None:
        return (
            "unavailable",
            "Nelze vyhodnotit",
            "Implikovaný růst nebylo možné spočítat (záporné FCF nebo řešení mimo rozsah).",
            None,
        )

    primary_refs = [
        v for v in (cagrs.revenue_cagr_5y, cagrs.fcf_cagr_5y, cagrs.consensus_rev_cagr) if v is not None
    ]
    fallback_refs = [
        v
        for v in (
            cagrs.revenue_cagr_3y,
            cagrs.fcf_cagr_3y,
            cagrs.revenue_cagr_10y,
            cagrs.fcf_cagr_10y,
            cagrs.consensus_eps_cagr,
        )
        if v is not None
    ]
    refs = primary_refs if primary_refs else fallback_refs
    benchmark: Optional[float] = max(refs) if refs else None

    if implied_growth > 0.25 or (
        benchmark is not None and implied_growth > min(benchmark, 0.20) + 0.06 and implied_growth > 0.12
    ):
        expl = (
            f"Trh oceňuje velmi agresivní dlouhodobý růst ({fmt_percent(implied_growth)} ročně), "
            + (
                f"který výrazně překračuje referenční tempo firmy ({fmt_percent(benchmark)}). "
                if benchmark is not None
                else "což vyžaduje mimořádnou dlouhodobou expanzi bez prostoru pro chybu. "
            )
            + "Bezpečnostní polštář je nízký."
        )
        return "extreme", "Extrémní", expl, benchmark

    if implied_growth > 0.15 or (
        benchmark is not None and implied_growth > min(benchmark, 0.18) + 0.02 and implied_growth > 0.08
    ):
        expl = (
            f"Očekávaný růst ({fmt_percent(implied_growth)} ročně) je nad historickým či konsenzuálním tempem "
            + (f"({fmt_percent(benchmark)}). " if benchmark is not None else ". ")
            + "Vyžaduje bezchybnou exekuci a udržení silných marží."
        )
        return "ambitious", "Ambiciózní", expl, benchmark

    expl = (
        f"Trhem implikovaný růst ({fmt_percent(implied_growth)} ročně) je dosažitelný a v souladu "
        + (
            f"s historickým/konsenzuálním tempem firmy ({fmt_percent(benchmark)})."
            if benchmark is not None
            else "s dlouhodobě udržitelným růstem kvalitního podniku."
        )
    )
    return "realistic", "Realistické", expl, benchmark


def diagnose_capex_distortion(
    company: CompanyData,
    params: ReverseDCFInputs,
) -> CapExDistortionDiagnostics:
    """Detect whether extraordinary CapEx spikes (>= 1.5x 4Y avg CapEx/Revenue) or negative FCF distort TTM FCF."""
    latest = company.annuals[-1] if company.annuals else None

    ttm_rev = company.ttm_revenue if (company.ttm_revenue and company.ttm_revenue > 0) else (
        latest.revenue if (latest and latest.revenue and latest.revenue > 0) else None
    )
    ttm_ocf = company.ttm_ocf if company.ttm_ocf is not None else (
        latest.operating_cash_flow if latest else None
    )
    ttm_capex = company.ttm_capex
    if ttm_capex is None and company.ttm_ocf is not None and company.ttm_fcf is not None:
        ttm_capex = max(0.0, company.ttm_ocf - company.ttm_fcf)
    if ttm_capex is None and latest is not None:
        ttm_capex = latest.capex

    ttm_sbc = company.ttm_sbc if company.ttm_sbc is not None else (
        latest.stock_based_compensation if latest else None
    )

    current_ratio = (
        (ttm_capex / ttm_rev) if (ttm_capex is not None and ttm_rev and ttm_rev > 0) else None
    )

    # Historical CapEx / Revenue ratios (last 4-5 years)
    all_ratios = [
        a.capex / a.revenue
        for a in company.annuals[-5:]
        if a.capex is not None and a.revenue and a.revenue > 0
    ]
    # Use prior 4 years when 5 years are available so a current-year CapEx surge doesn't mask itself
    baseline_4y = all_ratios[-5:-1] if len(all_ratios) >= 5 else all_ratios[-4:]
    hist_4y_avg = float(np.mean(baseline_4y)) if baseline_4y else None
    hist_5y_med = float(np.median(all_ratios)) if all_ratios else None
    norm_ratio = hist_5y_med if hist_5y_med is not None else hist_4y_avg

    spike_ratio = (
        (current_ratio / hist_4y_avg)
        if (current_ratio is not None and hist_4y_avg is not None and hist_4y_avg > 1e-4)
        else None
    )

    raw_fcf_0, _ = _select_raw_fcf_base(company, params.fcf_base_type)

    norm_capex = (ttm_rev * norm_ratio) if (ttm_rev is not None and norm_ratio is not None) else None
    norm_fcf_0: Optional[float] = None
    if ttm_ocf is not None and norm_capex is not None:
        norm_fcf = ttm_ocf - norm_capex
        if params.fcf_base_type == "fcf_minus_sbc":
            norm_fcf_0 = norm_fcf - (ttm_sbc or 0.0)
        else:
            norm_fcf_0 = norm_fcf

    is_negative_or_zero = raw_fcf_0 is not None and raw_fcf_0 <= 0
    is_capex_spike = spike_ratio is not None and spike_ratio >= 1.50
    should_warn = (
        params.custom_fcf_0_millions == 0.0
        and not params.normalize_capex
        and (is_negative_or_zero or is_capex_spike)
    )

    warning_cz = ""
    if should_warn:
        reasons: list[str] = []
        if is_capex_spike and current_ratio is not None and hist_4y_avg is not None and spike_ratio is not None:
            reasons.append(
                f"aktuální poměr CapEx / Tržby ({fmt_percent(current_ratio, 1)}) je "
                f"**{fmt_number(spike_ratio, 1)}× vyšší** než 4letý průměr firmy ({fmt_percent(hist_4y_avg, 1)})"
            )
        if is_negative_or_zero:
            reasons.append("vypočtené startovní TTM FCF je nulové nebo záporné")
        reason_txt = " a ".join(reasons)
        warning_cz = (
            f"**Upozornění na zkreslené TTM FCF:** {reason_txt.capitalize()}. "
            "Mimořádné jednorázové investice (CapEx) tak mohou výrazně zkreslit DCF ocenění směrem dolů. "
            "Doporučujeme v postranním panelu zapnout **„Normalizovat mimořádný CapEx (dle historického průměru CapEx/Tržby)“** "
            "nebo zadat vlastní startovní FCF₀ v sekci *Pokročilé parametry*."
        )

    return CapExDistortionDiagnostics(
        current_capex_to_rev=current_ratio,
        hist_4y_avg_capex_to_rev=hist_4y_avg,
        hist_5y_median_capex_to_rev=hist_5y_med,
        spike_ratio=spike_ratio,
        raw_fcf_0=raw_fcf_0,
        normalized_capex=norm_capex,
        normalized_fcf_0=norm_fcf_0,
        should_warn=should_warn,
        warning_cz=warning_cz,
    )


def _select_raw_fcf_base(
    company: CompanyData,
    fcf_base_type: FcfBaseType,
) -> tuple[Optional[float], str]:
    """Select raw un-normalized FCF_0 (preferring TTM if complete, otherwise latest fiscal year)."""
    latest = company.annuals[-1] if company.annuals else None

    if company.ttm_fcf is not None:
        if fcf_base_type == "fcf_minus_sbc":
            if company.ttm_sbc is not None:
                return company.ttm_fcf - company.ttm_sbc, "TTM (FCF − SBC)"
            if latest and latest.stock_based_compensation is not None:
                return (
                    company.ttm_fcf - latest.stock_based_compensation,
                    f"TTM FCF − FY{latest.year} SBC",
                )
            return company.ttm_fcf, "TTM FCF (SBC neuvedeno)"
        return company.ttm_fcf, "TTM vykázané FCF"

    if latest is not None and latest.free_cash_flow is not None:
        if fcf_base_type == "fcf_minus_sbc":
            if latest.stock_based_compensation is not None:
                return (
                    latest.free_cash_flow - latest.stock_based_compensation,
                    f"FY{latest.year} (FCF − SBC)",
                )
            return latest.free_cash_flow, f"FY{latest.year} FCF (SBC neuvedeno)"
        return latest.free_cash_flow, f"FY{latest.year} vykázané FCF"

    return None, "Nedostupné"


def select_fcf_base(
    company: CompanyData,
    fcf_base_type: FcfBaseType,
    normalize_capex: bool = False,
    custom_fcf_0_millions: float = 0.0,
) -> tuple[Optional[float], str]:
    """Select FCF_0 supporting custom override, historical CapEx/Revenue normalization, or raw TTM/FY."""
    if custom_fcf_0_millions != 0.0:
        return (
            float(custom_fcf_0_millions) * 1e6,
            f"Vlastní startovní FCF₀ ({fmt_number(custom_fcf_0_millions, 0)} mil.)",
        )

    if normalize_capex:
        diag = diagnose_capex_distortion(
            company,
            ReverseDCFInputs(fcf_base_type=fcf_base_type, normalize_capex=True),
        )
        if diag.normalized_fcf_0 is not None:
            ratio_used = diag.hist_5y_median_capex_to_rev or diag.hist_4y_avg_capex_to_rev
            base_lbl = "FCF_adj" if fcf_base_type == "fcf_minus_sbc" else "FCF"
            return (
                diag.normalized_fcf_0,
                f"Normalizované {base_lbl} (CapEx {fmt_percent(ratio_used, 1)} tržeb)",
            )

    return _select_raw_fcf_base(company, fcf_base_type)


def run_reverse_dcf(company: CompanyData, params: ReverseDCFInputs) -> ReverseDCFResult:
    """Execute the full Reverse DCF workflow for a company."""
    fcf_0, fcf_label = select_fcf_base(
        company,
        params.fcf_base_type,
        normalize_capex=params.normalize_capex,
        custom_fcf_0_millions=params.custom_fcf_0_millions,
    )
    net_debt = company.latest_net_debt if company.latest_net_debt is not None else 0.0
    shares = company.diluted_shares_current
    price = company.current_price
    cagrs = compute_historical_cagrs(company.annuals, company.analyst_estimates, params.fcf_base_type)

    fcf_0_ps = (fcf_0 / shares) if (fcf_0 is not None and shares and shares > 0) else None

    if price is None or price <= 0 or shares is None or shares <= 0 or fcf_0 is None:
        err = (
            "Chybí vstupní data pro Reverse DCF (cena akcie, počet zředěných akcií nebo FCF)."
        )
        return ReverseDCFResult(
            fcf_0=fcf_0,
            fcf_0_label=fcf_label,
            fcf_0_per_share=fcf_0_ps,
            net_debt=net_debt,
            diluted_shares=shares,
            current_price=price,
            implied_growth=None,
            implied_pv_fcf=None,
            implied_pv_tv=None,
            tv_share_of_ev=None,
            error_message=err,
            headline_cz=err,
            cagrs=cagrs,
            verdict="unavailable",
            verdict_label_cz="Nelze vyhodnotit",
            verdict_explanation_cz=err,
            benchmark_growth=None,
        )

    implied_g, err = solve_implied_growth(
        current_price=price,
        fcf_0=fcf_0,
        discount_rate=params.discount_rate,
        horizon_years=params.horizon_years,
        net_debt=net_debt,
        diluted_shares=shares,
        terminal_method=params.terminal_method,
        perpetual_growth_rate=params.perpetual_growth_rate,
        exit_multiple=params.exit_multiple,
    )

    pv_fcf = pv_tv = tv_share = None
    if implied_g is not None:
        _, pv_fcf, pv_tv = compute_dcf_value_per_share(
            fcf_0=fcf_0,
            growth_rate=implied_g,
            discount_rate=params.discount_rate,
            horizon_years=params.horizon_years,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method=params.terminal_method,
            perpetual_growth_rate=params.perpetual_growth_rate,
            exit_multiple=params.exit_multiple,
        )
        total_ev = pv_fcf + pv_tv
        tv_share = (pv_tv / total_ev) if total_ev > 0 else None

        verb = "poroste" if implied_g >= 0 else "bude klesat"
        headline = (
            f"Při dnešní ceně {fmt_price(price, company.quote_currency)} a požadované návratnosti "
            f"{fmt_percent(params.discount_rate)} trh předpokládá, že firma {verb} o "
            f"{fmt_percent(abs(implied_g))} ročně po dobu {params.horizon_years} let."
        )
    else:
        headline = err or "Implikovaný růst nelze pro zadané parametry vyřešit."

    verdict, v_label, v_expl, benchmark = evaluate_growth_verdict(implied_g, cagrs)

    return ReverseDCFResult(
        fcf_0=fcf_0,
        fcf_0_label=fcf_label,
        fcf_0_per_share=fcf_0_ps,
        net_debt=net_debt,
        diluted_shares=shares,
        current_price=price,
        implied_growth=implied_g,
        implied_pv_fcf=pv_fcf,
        implied_pv_tv=pv_tv,
        tv_share_of_ev=tv_share,
        error_message=err,
        headline_cz=headline,
        cagrs=cagrs,
        verdict=verdict,
        verdict_label_cz=v_label,
        verdict_explanation_cz=v_expl,
        benchmark_growth=benchmark,
    )


def build_intrinsic_value_grid(
    fcf_0: float,
    net_debt: float,
    diluted_shares: float,
    params: ReverseDCFInputs,
    center_growth: Optional[float],
    x_axis_mode: Literal["discount_rate", "exit_multiple"] = "discount_rate",
) -> tuple[list[float], list[float], np.ndarray]:
    """Build a 2D sensitivity matrix of intrinsic value per share.

    Y-axis: annual FCF growth rate (7 levels centered near implied growth or 10 %).
    X-axis: discount rate (7 levels) OR exit multiple (7 levels).
    """
    base_g = center_growth if (center_growth is not None and -0.25 <= center_growth <= 0.45) else 0.10
    base_g_rounded = round(base_g * 20.0) / 20.0  # Snap to nearest 5 % or 2.5 %
    growth_rates = [round(base_g_rounded + step * 0.03, 4) for step in range(-3, 4)]

    if x_axis_mode == "discount_rate":
        r0 = params.discount_rate
        x_values = [round(max(0.05, r0 + step * 0.015), 4) for step in range(-3, 4)]
    else:
        m0 = params.exit_multiple
        x_values = [round(max(4.0, m0 + step * 3.0), 2) for step in range(-3, 4)]

    matrix = np.full((len(growth_rates), len(x_values)), np.nan, dtype=float)
    for i, g in enumerate(growth_rates):
        for j, x in enumerate(x_values):
            try:
                if x_axis_mode == "discount_rate":
                    val, _, _ = compute_dcf_value_per_share(
                        fcf_0=fcf_0,
                        growth_rate=g,
                        discount_rate=x,
                        horizon_years=params.horizon_years,
                        net_debt=net_debt,
                        diluted_shares=diluted_shares,
                        terminal_method=params.terminal_method,
                        perpetual_growth_rate=params.perpetual_growth_rate,
                        exit_multiple=params.exit_multiple,
                    )
                else:
                    val, _, _ = compute_dcf_value_per_share(
                        fcf_0=fcf_0,
                        growth_rate=g,
                        discount_rate=params.discount_rate,
                        horizon_years=params.horizon_years,
                        net_debt=net_debt,
                        diluted_shares=diluted_shares,
                        terminal_method="exit_multiple",
                        perpetual_growth_rate=params.perpetual_growth_rate,
                        exit_multiple=x,
                    )
                matrix[i, j] = val
            except ValueError:
                matrix[i, j] = np.nan

    return growth_rates, x_values, matrix


def build_implied_growth_grid(
    current_price: float,
    fcf_0: float,
    net_debt: float,
    diluted_shares: float,
    params: ReverseDCFInputs,
) -> tuple[list[float], list[float], np.ndarray]:
    """Build a 2D sensitivity matrix of implied annual FCF growth rate.

    Y-axis: share price (-30 % to +30 % around current_price in 10 % steps).
    X-axis: discount rate (7 levels around params.discount_rate).
    """
    price_multipliers = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    prices = [round(current_price * m, 2) for m in price_multipliers]
    r0 = params.discount_rate
    discount_rates = [round(max(0.05, r0 + step * 0.015), 4) for step in range(-3, 4)]

    matrix = np.full((len(prices), len(discount_rates)), np.nan, dtype=float)
    for i, p in enumerate(prices):
        for j, r in enumerate(discount_rates):
            g, _ = solve_implied_growth(
                current_price=p,
                fcf_0=fcf_0,
                discount_rate=r,
                horizon_years=params.horizon_years,
                net_debt=net_debt,
                diluted_shares=diluted_shares,
                terminal_method=params.terminal_method,
                perpetual_growth_rate=params.perpetual_growth_rate,
                exit_multiple=params.exit_multiple,
            )
            if g is not None:
                matrix[i, j] = g

    return prices, discount_rates, matrix
