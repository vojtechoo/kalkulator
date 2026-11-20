"""Pure hard-data Crystal Ball engine:
1. 10,000-scenario Monte Carlo simulation (default 3Y mean-reversion horizon, switchable 3/5/10Y)
2. 4-model Intrinsic Value Triangulation (DCF, Buffett Owner Earnings, Greenwald EPV, Mean-Reversion Multiple)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analysis.dcf import ReverseDCFInputs, compute_dcf_value_per_share, select_fcf_base
from analysis.forensics import CompoundingEngineResult
from analysis.quality import QualityXRayReport, compute_effective_tax_rate
from data.models import AnnualFinancials, CompanyData


@dataclass
class ValuationModelEstimate:
    """Single independent quantitative fair-value model output."""

    key: str
    name_cz: str
    value_per_share: Optional[float]
    weight: float
    upside_pct: Optional[float]
    formula_note_cz: str


@dataclass
class TriangulationResult:
    """Weighted intrinsic value consensus from 4 independent hard-data models."""

    models: list[ValuationModelEstimate]
    fair_value_per_share: Optional[float]
    max_buy_price: Optional[float]  # Fair value * (1 - margin_of_safety)
    margin_of_safety_target: float
    upside_to_fair_value: Optional[float]
    gap_to_buy_price: Optional[float]  # (max_buy_price / current_price) - 1
    epv_share_of_price: Optional[float]  # How much of current price is backed by zero-growth EPV
    growth_bet_share_of_price: Optional[float]  # 1 - epv_share_of_price
    valid_models_count: int


@dataclass
class MonteCarloResult:
    """Probabilistic 10,000-scenario forward simulation derived purely from historical hard data."""

    horizon_years: int
    n_simulations: int
    hurdle_rate: float
    current_price: float
    # Simulated terminal price & IRR arrays
    terminal_prices: np.ndarray
    annualized_irrs: np.ndarray
    # Key percentiles
    p10_price: float
    p25_price: float
    p50_price: float
    p75_price: float
    p90_price: float
    p10_irr: float
    p50_irr: float
    p90_irr: float
    # Probabilities
    prob_beat_hurdle: float
    prob_positive_return: float
    prob_loss_gt_20pct: float
    # Risk/reward asymmetry
    upside_downside_ratio: Optional[float]
    # Price needed today for median IRR == hurdle_rate
    price_for_hurdle_irr: Optional[float]
    # Year-by-year fan chart DataFrame (year, p10, p25, p50, p75, p90)
    fan_df: pd.DataFrame
    # Transparently documented calibrated hard-data inputs
    calibrated_rev_growth_mean: float
    calibrated_rev_growth_std: float
    calibrated_fcf_margin_mean: float
    calibrated_fcf_margin_std: float
    calibrated_exit_multiple_median: float
    calibrated_share_drift: float
    calibrated_div_yield: float
    available: bool = True
    error_message_cz: Optional[str] = None
    # 1-Year projected price percentiles (from year_offset == 1)
    p10_price_1y: Optional[float] = None
    p50_price_1y: Optional[float] = None
    p90_price_1y: Optional[float] = None
    expected_return_1y: Optional[float] = None


def _estimate_maintenance_capex(annuals: list[AnnualFinancials]) -> Optional[float]:
    """Estimate maintenance CapEx via Bruce Greenwald's PPE/Revenue intensity method.

    Growth CapEx = max(0, ΔRevenue × median(PPE_net / Revenue))
    Maintenance CapEx = clamp(CapEx - Growth CapEx, 0.35 * CapEx, CapEx)
    Falls back to reported CapEx if PPE_net or revenue growth is unavailable.
    """
    latest = annuals[-1]
    if latest.capex is None:
        return None
    if len(annuals) < 2:
        return latest.capex

    ppe_ratios = [
        a.ppe_net / a.revenue
        for a in annuals
        if a.ppe_net is not None and a.ppe_net > 0 and a.revenue is not None and a.revenue > 0
    ]
    prev = annuals[-2]
    if not ppe_ratios or latest.revenue is None or prev.revenue is None:
        return latest.capex

    med_ppe_ratio = float(np.median(ppe_ratios))
    delta_rev = max(0.0, latest.revenue - prev.revenue)
    growth_capex = delta_rev * med_ppe_ratio
    maint = latest.capex - growth_capex
    return float(np.clip(maint, 0.35 * latest.capex, latest.capex))


def _historical_revenue_growth_stats(annuals: list[AnnualFinancials]) -> tuple[float, float]:
    """Extract historical YoY revenue growth mean & std with conservative Bayesian shrinkage."""
    revs = [a.revenue for a in annuals if a.revenue is not None and a.revenue > 0]
    if len(revs) < 2:
        return 0.04, 0.08
    yoy = [(revs[i] / revs[i - 1]) - 1.0 for i in range(1, len(revs))]
    yoy_clipped = [float(np.clip(g, -0.35, 0.55)) for g in yoy]
    raw_mean = float(np.mean(yoy_clipped))
    raw_std = float(np.std(yoy_clipped, ddof=1)) if len(yoy_clipped) >= 2 else 0.08

    # Bayesian shrinkage toward long-term corporate base rate (6%) to prevent overfitting short booms
    n_obs = len(yoy_clipped)
    shrink_weight = n_obs / (n_obs + 3.0)
    shrunk_mean = shrink_weight * raw_mean + (1.0 - shrink_weight) * 0.06
    calibrated_mean = float(np.clip(shrunk_mean, -0.10, 0.22))
    calibrated_std = float(np.clip(raw_std, 0.04, 0.22))
    return calibrated_mean, calibrated_std


def _historical_real_fcf_margin_stats(
    annuals: list[AnnualFinancials],
    subtract_sbc: bool = True,
) -> tuple[Optional[float], float]:
    """Extract historical Real FCF (or Reported FCF) margin mean & std."""
    margins: list[float] = []
    for a in annuals:
        if a.revenue is None or a.revenue <= 0 or a.free_cash_flow is None:
            continue
        fcf_val = a.free_cash_flow - (a.stock_based_compensation or 0.0) if subtract_sbc else a.free_cash_flow
        margins.append(float(np.clip(fcf_val / a.revenue, -0.40, 0.60)))
    if not margins:
        return None, 0.05
    mean_m = float(np.mean(margins))
    std_m = float(np.std(margins, ddof=1)) if len(margins) >= 2 else max(0.03, abs(mean_m) * 0.20)
    return mean_m, float(np.clip(std_m, 0.02, 0.15))


def _historical_share_drift(annuals: list[AnnualFinancials]) -> float:
    """Compute historical average annual diluted share count change (negative = buybacks)."""
    shares = [a.weighted_avg_shares_diluted for a in annuals if a.weighted_avg_shares_diluted and a.weighted_avg_shares_diluted > 0]
    if len(shares) < 2:
        return 0.0
    changes = [(shares[i] / shares[i - 1]) - 1.0 for i in range(1, len(shares))]
    return float(np.clip(np.median(changes), -0.05, 0.05))


def _historical_median_pfcf(company: CompanyData) -> float:
    """Compute mean-reversion anchor multiple from historical P/FCF blended with market anchor (16x)."""
    hist_pfcf = [
        a.pfcf_ratio
        for a in company.annuals[-10:]
        if a.pfcf_ratio is not None and 4.0 <= a.pfcf_ratio <= 60.0
    ]
    if hist_pfcf:
        med = float(np.median(hist_pfcf))
        # Blend 70% company's own historical median + 30% market mean-reversion anchor (16x)
        return float(np.clip(0.70 * med + 0.30 * 16.0, 8.0, 32.0))
    return 16.0


def run_monte_carlo_crystal_ball(
    company: CompanyData,
    horizon_years: int = 3,
    hurdle_rate: float = 0.12,
    subtract_sbc: bool = True,
    n_simulations: int = 10_000,
) -> MonteCarloResult:
    """Run a 10,000-scenario stochastic Monte Carlo projection purely from historical hard statements.

    Mechanics:
    - Correlated shocks (rho = 0.45) link revenue growth and terminal exit multiple:
      companies that beat revenue expectations retain higher multiples; slowdowns trigger de-rating.
    - Terminal multiple mean-reverts toward the company's own historical median P/FCF blended with 16x.
    - Includes historical dividend yield and diluted share count drift (buybacks/dilution).
    - Deterministic seed per ticker ensures reproducible numbers across reruns.
    """
    price = company.current_price
    shares = company.diluted_shares_current
    latest = company.annuals[-1] if company.annuals else None
    rev_0 = company.ttm_revenue if (company.ttm_revenue and company.ttm_revenue > 0) else (
        latest.revenue if (latest and latest.revenue and latest.revenue > 0) else None
    )

    fcf_m_mean, fcf_m_std = _historical_real_fcf_margin_stats(company.annuals, subtract_sbc=subtract_sbc)

    if price is None or price <= 0 or shares is None or shares <= 0 or rev_0 is None or fcf_m_mean is None or fcf_m_mean <= 0:
        empty_arr = np.array([], dtype=float)
        return MonteCarloResult(
            horizon_years=horizon_years,
            n_simulations=n_simulations,
            hurdle_rate=hurdle_rate,
            current_price=price or 0.0,
            terminal_prices=empty_arr,
            annualized_irrs=empty_arr,
            p10_price=0.0,
            p25_price=0.0,
            p50_price=0.0,
            p75_price=0.0,
            p90_price=0.0,
            p10_irr=0.0,
            p50_irr=0.0,
            p90_irr=0.0,
            prob_beat_hurdle=0.0,
            prob_positive_return=0.0,
            prob_loss_gt_20pct=1.0,
            upside_downside_ratio=None,
            price_for_hurdle_irr=None,
            fan_df=pd.DataFrame(),
            calibrated_rev_growth_mean=0.0,
            calibrated_rev_growth_std=0.0,
            calibrated_fcf_margin_mean=fcf_m_mean or 0.0,
            calibrated_fcf_margin_std=fcf_m_std,
            calibrated_exit_multiple_median=16.0,
            calibrated_share_drift=0.0,
            calibrated_div_yield=0.0,
            available=False,
            error_message_cz=(
                "Monte Carlo simulaci nelze spolehlivě spustit: historická marže skutečného FCF "
                "je záporná nebo chybí tržby / cena akcie."
            ),
        )

    rev_g_mean, rev_g_std = _historical_revenue_growth_stats(company.annuals)
    share_drift = _historical_share_drift(company.annuals)
    exit_mult_med = _historical_median_pfcf(company)

    # Trailing dividend yield from hard cash flow statement
    div_yield = 0.0
    if latest and latest.dividends_paid and company.market_cap and company.market_cap > 0:
        div_yield = float(np.clip(latest.dividends_paid / company.market_cap, 0.0, 0.12))

    # Deterministic seed from ticker so toggling unrelated UI widgets doesn't jitter results
    seed_int = int(hashlib.sha256(f"{company.ticker}_{horizon_years}_{subtract_sbc}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    # Simulate annual revenue paths and correlated multiple/margin shocks
    rev_paths = np.full((n_simulations, horizon_years + 1), rev_0, dtype=float)
    z_common = rng.standard_normal(n_simulations)  # Persistent business-quality regime shock

    for t in range(1, horizon_years + 1):
        # Mean-reversion damping over longer horizons (year 1..N)
        fade = max(0.65, 1.0 - 0.04 * (t - 1))
        g_t = rng.normal(
            loc=rev_g_mean * fade + 0.015 * z_common,
            scale=rev_g_std,
            size=n_simulations,
        )
        g_t = np.clip(g_t, -0.35, 0.45)
        rev_paths[:, t] = rev_paths[:, t - 1] * (1.0 + g_t)

    # Simulated FCF margin at each year (positively correlated with common business shock)
    margin_paths = np.zeros((n_simulations, horizon_years + 1), dtype=float)
    margin_paths[:, 0] = fcf_m_mean
    for t in range(1, horizon_years + 1):
        m_t = rng.normal(
            loc=fcf_m_mean + 0.25 * fcf_m_std * z_common,
            scale=fcf_m_std,
            size=n_simulations,
        )
        margin_paths[:, t] = np.clip(m_t, 0.01, 0.65)

    fcf_paths = rev_paths * margin_paths

    # Share count trajectory
    share_paths = np.array([shares * ((1.0 + share_drift) ** t) for t in range(horizon_years + 1)])

    # Correlated exit multiple distribution (log-normal around historical median anchor)
    # Correlation rho = 0.45 with z_common
    z_mult = 0.45 * z_common + np.sqrt(1.0 - 0.45**2) * rng.standard_normal(n_simulations)
    mult_sigma = 0.22  # ~22% valuation multiple dispersion
    # Transition multiple smoothly from current P/FCF toward mean-reverting anchor by year H
    curr_fcf = rev_0 * fcf_m_mean
    curr_pfcf = float(np.clip((price * shares) / curr_fcf, 6.0, 50.0)) if curr_fcf > 0 else exit_mult_med

    price_paths = np.zeros((n_simulations, horizon_years + 1), dtype=float)
    price_paths[:, 0] = price

    for t in range(1, horizon_years + 1):
        frac = t / float(horizon_years)
        anchor_t = (1.0 - frac) * curr_pfcf + frac * exit_mult_med
        mult_t = np.clip(anchor_t * np.exp(mult_sigma * np.sqrt(frac) * z_mult), 5.0, 55.0)
        eq_val_t = fcf_paths[:, t] * mult_t
        price_paths[:, t] = np.maximum(eq_val_t / share_paths[t], 0.01)

    # Terminal total value includes reinvested cash dividends
    terminal_prices = price_paths[:, horizon_years]
    total_wealth_ratio = (terminal_prices / price) * ((1.0 + div_yield) ** horizon_years)
    annualized_irrs = np.power(np.maximum(total_wealth_ratio, 1e-6), 1.0 / horizon_years) - 1.0

    p10_p, p25_p, p50_p, p75_p, p90_p = np.percentile(terminal_prices, [10, 25, 50, 75, 90])
    p10_irr, p50_irr, p90_irr = np.percentile(annualized_irrs, [10, 50, 90])

    prob_beat = float(np.mean(annualized_irrs >= hurdle_rate))
    prob_pos = float(np.mean(annualized_irrs > 0.0))
    prob_loss20 = float(np.mean(terminal_prices < price * 0.80))

    upside_amt = max(0.0, float(p90_p) - price)
    downside_amt = max(0.0, price - float(p10_p))
    ud_ratio: Optional[float]
    if downside_amt < 0.01 * price:
        ud_ratio = 10.0  # Even the 10th percentile is at or above today's price
    else:
        ud_ratio = float(upside_amt / downside_amt)

    # Price today that delivers hurdle_rate IRR to median terminal price (including dividends)
    price_for_hurdle = float(
        (p50_p * ((1.0 + div_yield) ** horizon_years)) / ((1.0 + hurdle_rate) ** horizon_years)
    )

    fan_rows = []
    for t in range(horizon_years + 1):
        qs = np.percentile(price_paths[:, t], [10, 25, 50, 75, 90])
        fan_rows.append(
            {
                "year_offset": t,
                "label": "Dnes" if t == 0 else f"+{t} r.",
                "p10": float(qs[0]),
                "p25": float(qs[1]),
                "p50": float(qs[2]),
                "p75": float(qs[3]),
                "p90": float(qs[4]),
            }
        )

    p10_1y = float(fan_rows[1]["p10"]) if len(fan_rows) > 1 else None
    p50_1y = float(fan_rows[1]["p50"]) if len(fan_rows) > 1 else None
    p90_1y = float(fan_rows[1]["p90"]) if len(fan_rows) > 1 else None
    exp_ret_1y = ((p50_1y / price) - 1.0 + div_yield) if (p50_1y is not None and price > 0) else None

    return MonteCarloResult(
        horizon_years=horizon_years,
        n_simulations=n_simulations,
        hurdle_rate=hurdle_rate,
        current_price=price,
        terminal_prices=terminal_prices,
        annualized_irrs=annualized_irrs,
        p10_price=float(p10_p),
        p25_price=float(p25_p),
        p50_price=float(p50_p),
        p75_price=float(p75_p),
        p90_price=float(p90_p),
        p10_irr=float(p10_irr),
        p50_irr=float(p50_irr),
        p90_irr=float(p90_irr),
        prob_beat_hurdle=prob_beat,
        prob_positive_return=prob_pos,
        prob_loss_gt_20pct=prob_loss20,
        upside_downside_ratio=ud_ratio,
        price_for_hurdle_irr=price_for_hurdle,
        p10_price_1y=p10_1y,
        p50_price_1y=p50_1y,
        p90_price_1y=p90_1y,
        expected_return_1y=exp_ret_1y,
        fan_df=pd.DataFrame(fan_rows),
        calibrated_rev_growth_mean=rev_g_mean,
        calibrated_rev_growth_std=rev_g_std,
        calibrated_fcf_margin_mean=fcf_m_mean,
        calibrated_fcf_margin_std=fcf_m_std,
        calibrated_exit_multiple_median=exit_mult_med,
        calibrated_share_drift=share_drift,
        calibrated_div_yield=div_yield,
        available=True,
    )


def compute_triangulated_fair_value(
    company: CompanyData,
    dcf_params: ReverseDCFInputs,
    quality: QualityXRayReport,
    compounding: CompoundingEngineResult,
    monte_carlo: MonteCarloResult,
    margin_of_safety_target: float = 0.25,
) -> TriangulationResult:
    """Triangulate intrinsic value per share across 4 independent hard-data valuation models.

    Models & base weights:
    1. Konzervativní DCF (FCF − SBC) [35 %] – growth capped by hard-data revenue/compounding history
    2. Buffett Owner Earnings [25 %] – (NI + D&A − Maintenance CapEx − SBC) capitalized at hurdle rate
    3. Greenwald EPV (nulový růst) [15 %] – Normalized NOPAT / WACC − Net Debt
    4. Mean-Reversion Exit Multiple (3Y/HY) [25 %] – Monte Carlo median terminal value discounted at hurdle rate
    """
    price = company.current_price
    shares = company.diluted_shares_current
    net_debt = company.latest_net_debt if company.latest_net_debt is not None else 0.0
    latest = company.annuals[-1] if company.annuals else None

    models: list[ValuationModelEstimate] = []

    def _upside(val: Optional[float]) -> Optional[float]:
        if val is None or price is None or price <= 0:
            return None
        return (val / price) - 1.0

    # --- Model 1: Hard-data Conservative DCF ---
    fcf_0, fcf_lbl = select_fcf_base(
        company,
        dcf_params.fcf_base_type,
        normalize_capex=dcf_params.normalize_capex,
        custom_fcf_0_millions=dcf_params.custom_fcf_0_millions,
    )
    rev_g_mean, _ = _historical_revenue_growth_stats(company.annuals)
    comp_g = compounding.intrinsic_compounding_rate
    hard_g = float(np.clip(np.mean([g for g in (rev_g_mean, comp_g) if g is not None]), 0.0, 0.18))

    dcf_val: Optional[float] = None
    if fcf_0 is not None and fcf_0 > 0 and shares and shares > 0:
        try:
            v_ps, _, _ = compute_dcf_value_per_share(
                fcf_0=fcf_0,
                growth_rate=hard_g,
                discount_rate=dcf_params.discount_rate,
                horizon_years=dcf_params.horizon_years,
                net_debt=net_debt,
                diluted_shares=shares,
                terminal_method=dcf_params.terminal_method,
                perpetual_growth_rate=dcf_params.perpetual_growth_rate,
                exit_multiple=dcf_params.exit_multiple,
            )
            if v_ps > 0:
                dcf_val = v_ps
        except ValueError:
            dcf_val = None

    models.append(
        ValuationModelEstimate(
            key="dcf",
            name_cz="Konzervativní DCF (tvrdá data)",
            value_per_share=dcf_val,
            weight=0.35,
            upside_pct=_upside(dcf_val),
            formula_note_cz=(
                f"Základna {fcf_lbl}, tempo růstu odvozeno z historie a ROIIC = {hard_g*100:.1f} % p.a., "
                f"diskont {dcf_params.discount_rate*100:.1f} %."
            ),
        )
    )

    # --- Model 2: Buffett Owner Earnings ---
    oe_val: Optional[float] = None
    oe_note = "Chybí čistý zisk, odpisy nebo CapEx."
    if latest and latest.net_income is not None and shares and shares > 0:
        da = latest.depreciation_amortization or 0.0
        maint_capex = _estimate_maintenance_capex(company.annuals) or 0.0
        sbc = latest.stock_based_compensation or 0.0
        owner_earnings = latest.net_income + da - maint_capex - sbc
        if owner_earnings > 0:
            # 2-stage Owner Earnings capitalization: conservative 5Y fade growth then 2.5% perpetual
            g_oe = float(np.clip(hard_g * 0.85, 0.0, 0.14))
            r_oe = max(dcf_params.discount_rate, 0.06)
            try:
                v_oe, _, _ = compute_dcf_value_per_share(
                    fcf_0=owner_earnings,
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
                    oe_note = (
                        f"Owner Earnings = NI + D&A − Maintenance CapEx − SBC; "
                        f"5letý růst {g_oe*100:.1f} %, diskont {r_oe*100:.1f} %."
                    )
            except ValueError:
                pass
        else:
            oe_note = "Owner Earnings jsou záporné (Maintenance CapEx + SBC převyšují zisk + D&A)."

    models.append(
        ValuationModelEstimate(
            key="owner_earnings",
            name_cz="Buffett Owner Earnings",
            value_per_share=oe_val,
            weight=0.25,
            upside_pct=_upside(oe_val),
            formula_note_cz=oe_note,
        )
    )

    # --- Model 3: Bruce Greenwald EPV (Zero-Growth Earnings Power Value) ---
    epv_val: Optional[float] = None
    epv_note = "Chybí provozní marže, tržby nebo WACC."
    wacc = quality.wacc_components.wacc
    if company.annuals and shares and shares > 0 and wacc and wacc > 0.03:
        # Normalized EBIT margin across available years applied to latest revenue
        op_margins = [
            a.operating_income / a.revenue
            for a in company.annuals[-7:]
            if a.operating_income is not None and a.revenue and a.revenue > 0
        ]
        latest_rev = company.ttm_revenue or (latest.revenue if latest else None)
        if op_margins and latest_rev and latest_rev > 0:
            norm_margin = float(np.median(op_margins))
            eff_tax = quality.wacc_components.effective_tax_rate or 0.21
            norm_nopat = latest_rev * norm_margin * (1.0 - eff_tax)
            if norm_nopat > 0:
                epv_ev = norm_nopat / wacc
                epv_eq = epv_ev - net_debt
                if epv_eq > 0:
                    epv_val = epv_eq / shares
                    epv_note = (
                        f"Hodnota bez jakéhokoliv budoucího růstu: normalizovaný NOPAT "
                        f"(medián EBIT marže {norm_margin*100:.1f} %) / WACC ({wacc*100:.1f} %) − Čistý dluh."
                    )

    models.append(
        ValuationModelEstimate(
            key="epv",
            name_cz="Greenwald EPV (hodnota bez růstu)",
            value_per_share=epv_val,
            weight=0.15,
            upside_pct=_upside(epv_val),
            formula_note_cz=epv_note,
        )
    )

    # --- Model 4: Mean-Reversion Exit Multiple (from Monte Carlo median discounted at hurdle rate) ---
    mr_val = monte_carlo.price_for_hurdle_irr if (monte_carlo.available and monte_carlo.price_for_hurdle_irr and monte_carlo.price_for_hurdle_irr > 0) else None
    models.append(
        ValuationModelEstimate(
            key="mean_reversion",
            name_cz=f"Mean-Reversion násobek ({monte_carlo.horizon_years}Y)",
            value_per_share=mr_val,
            weight=0.25,
            upside_pct=_upside(mr_val),
            formula_note_cz=(
                f"Medián Monte Carlo za {monte_carlo.horizon_years} r. při návratu P/FCF k "
                f"{monte_carlo.calibrated_exit_multiple_median:.1f}x, diskontováno požadovaným výnosem "
                f"{monte_carlo.hurdle_rate*100:.1f} %."
                if mr_val is not None
                else "Nedostatek dat pro simulaci návratu násobku k mediánu."
            ),
        )
    )

    valid_models = [m for m in models if m.value_per_share is not None and m.value_per_share > 0]
    fair_value: Optional[float] = None
    if valid_models:
        w_sum = sum(m.weight for m in valid_models)
        fair_value = sum((m.value_per_share or 0.0) * m.weight for m in valid_models) / w_sum

    max_buy = (fair_value * (1.0 - margin_of_safety_target)) if fair_value is not None else None
    upside_fv = _upside(fair_value)
    gap_buy = ((max_buy / price) - 1.0) if (max_buy is not None and price and price > 0) else None

    epv_share = (epv_val / price) if (epv_val is not None and price and price > 0) else None
    growth_bet = (1.0 - epv_share) if epv_share is not None else None

    return TriangulationResult(
        models=models,
        fair_value_per_share=fair_value,
        max_buy_price=max_buy,
        margin_of_safety_target=margin_of_safety_target,
        upside_to_fair_value=upside_fv,
        gap_to_buy_price=gap_buy,
        epv_share_of_price=epv_share,
        growth_bet_share_of_price=growth_bet,
        valid_models_count=len(valid_models),
    )
