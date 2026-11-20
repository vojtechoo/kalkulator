"""Automated Quantitative Decision Engine — synthesizes all hard data into ONE clear verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from analysis.capital_allocation import InsiderActivityReport
from analysis.crystal_ball import MonteCarloResult, TriangulationResult
from analysis.dcf import ReverseDCFResult
from analysis.forensics import ForensicsReport
from analysis.formatting import fmt_number, fmt_percent, fmt_pp, fmt_price
from analysis.quality import QualityXRayReport
from analysis.red_flags import RedFlagsReport
from data.models import CompanyData

VerdictCode = Literal["strong_buy", "buy", "wait", "overvalued", "avoid"]


@dataclass
class PillarScore:
    """Sub-score (0–100) for one of the 4 quantitative decision pillars."""

    name_cz: str
    score: int
    weight_pct: int
    summary_cz: str


@dataclass
class DecisionVerdict:
    """Complete automated investment decision synthesized purely from hard numbers."""

    verdict_code: VerdictCode
    verdict_title_cz: str
    verdict_subtitle_cz: str
    total_score: int  # 0..100
    badge_color: Literal["green", "amber", "red"]
    action_plan_cz: str
    pillars: list[PillarScore] = field(default_factory=list)
    pros_cz: list[str] = field(default_factory=list)
    cons_cz: list[str] = field(default_factory=list)
    veto_triggered: bool = False
    veto_reason_cz: str = ""


def _clamp_score(val: float) -> int:
    return int(round(float(np.clip(val, 0.0, 100.0))))


def run_decision_engine(
    company: CompanyData,
    dcf: ReverseDCFResult,
    red_flags: RedFlagsReport,
    quality: QualityXRayReport,
    forensics: ForensicsReport,
    monte_carlo: MonteCarloResult,
    triangulation: TriangulationResult,
    insiders: InsiderActivityReport,
) -> DecisionVerdict:
    """Synthesize all hard-data quantitative signals into a single actionable verdict and 0-100 score."""
    pros: list[str] = []
    cons: list[str] = []

    # ------------------------------------------------------------------
    # Pillar 1: Valuation & Margin of Safety (35%)
    # ------------------------------------------------------------------
    val_points = 50.0
    if triangulation.upside_to_fair_value is not None:
        up = triangulation.upside_to_fair_value
        # Map -40%..+40% upside linearly to 10..95
        val_points = float(np.clip(52.5 + up * 110.0, 5.0, 98.0))
        if up >= 0.25:
            pros.append(
                f"Výrazná sleva k triangulované férové hodnotě ({fmt_percent(up, 1, signed=True)}, "
                f"férová cena {fmt_price(triangulation.fair_value_per_share, company.quote_currency)})."
            )
        elif up <= -0.15:
            cons.append(
                f"Akcie se obchoduje nad triangulovanou férovou hodnotou ({fmt_percent(up, 1, signed=True)}, "
                f"férová cena {fmt_price(triangulation.fair_value_per_share, company.quote_currency)})."
            )

    if monte_carlo.available:
        # Blend with Monte Carlo probability of beating hurdle rate
        mc_val_score = monte_carlo.prob_beat_hurdle * 100.0
        val_points = 0.55 * val_points + 0.45 * mc_val_score
        if monte_carlo.prob_beat_hurdle >= 0.65:
            pros.append(
                f"Monte Carlo ({monte_carlo.horizon_years}Y): {fmt_percent(monte_carlo.prob_beat_hurdle, 0)} "
                f"scénářů překonává požadovaný výnos {fmt_percent(monte_carlo.hurdle_rate, 0)} "
                f"(medián IRR {fmt_percent(monte_carlo.p50_irr, 1)})."
            )
        elif monte_carlo.prob_beat_hurdle <= 0.30:
            cons.append(
                f"Nízká šance na požadovaný výnos: jen {fmt_percent(monte_carlo.prob_beat_hurdle, 0)} "
                f"simulací překonává {fmt_percent(monte_carlo.hurdle_rate, 0)} p.a. "
                f"(medián IRR {fmt_percent(monte_carlo.p50_irr, 1)})."
            )

    if dcf.verdict == "realistic":
        val_points = min(100.0, val_points + 8.0)
        pros.append(
            f"Reverse DCF: trhem očekávaný růst ({fmt_percent(dcf.implied_growth, 1)}/rok) je realistický."
        )
    elif dcf.verdict == "extreme":
        val_points = max(0.0, val_points - 14.0)
        cons.append(
            f"Reverse DCF varuje: cena vyžaduje extrémní růst FCF {fmt_percent(dcf.implied_growth, 1)} ročně."
        )

    if triangulation.growth_bet_share_of_price is not None and triangulation.growth_bet_share_of_price > 0.65:
        cons.append(
            f"Greenwald EPV: {fmt_percent(triangulation.growth_bet_share_of_price, 0)} dnešní ceny je čistá sázka na budoucí růst."
        )

    p1_score = _clamp_score(val_points)

    # ------------------------------------------------------------------
    # Pillar 2: Business Quality & Compounding Engine (25%)
    # ------------------------------------------------------------------
    qual_points = 50.0
    if quality.avg_spread is not None:
        # Spread of +10 p.b. -> ~90, 0 p.b. -> 48, -5 p.b. -> 20
        qual_points = float(np.clip(48.0 + quality.avg_spread * 420.0, 10.0, 96.0))
        if quality.avg_spread >= 0.06 and quality.years_below_wacc == 0:
            pros.append(
                f"Špičková tvorba hodnoty: průměrné ROIC {fmt_percent(quality.avg_roic, 1)} "
                f"(spread {fmt_pp(quality.avg_spread, 1)} nad WACC, 0 let pod WACC)."
            )
        elif quality.avg_spread < 0:
            cons.append(
                f"Ničení hodnoty: průměrné ROIC ({fmt_percent(quality.avg_roic, 1)}) nedosahuje ani na WACC."
            )

    if forensics.compounding.intrinsic_compounding_rate is not None:
        cr = forensics.compounding.intrinsic_compounding_rate
        if cr >= 0.12:
            qual_points = min(100.0, qual_points + 8.0)
            pros.append(
                f"Silný reinvestiční motor (ROIIC × reinvestice): přirozené složené úročení {fmt_percent(cr, 1)} ročně."
            )
        elif cr < 0.04:
            qual_points = max(0.0, qual_points - 8.0)

    if quality.gross_margin_trend == "rising":
        qual_points = min(100.0, qual_points + 6.0)
    elif quality.gross_margin_trend == "falling":
        qual_points = max(0.0, qual_points - 10.0)
        cons.append("Klesající trend hrubé marže naznačuje oslabování cenové síly.")

    p2_score = _clamp_score(qual_points)

    # ------------------------------------------------------------------
    # Pillar 3: Forensic Safety & Accounting Cleanliness (25%)
    # ------------------------------------------------------------------
    # Start from inverse of Red Flag Score (0..100 -> 100..0)
    for_points = float(100 - red_flags.red_flag_score)
    sub_signals: list[float] = [for_points]

    if forensics.piotroski.score is not None and forensics.piotroski.max_evaluated > 0:
        f_norm = (forensics.piotroski.score / forensics.piotroski.max_evaluated) * 100.0
        sub_signals.append(f_norm)
        if forensics.piotroski.status == "green":
            pros.append(f"Piotroski F-Score {forensics.piotroski.score}/{forensics.piotroski.max_evaluated} potvrzuje silné fundamentální momentum.")
        elif forensics.piotroski.status == "red":
            cons.append(f"Slabé Piotroski F-Score ({forensics.piotroski.score}/{forensics.piotroski.max_evaluated}) – zhoršující se fundament.")

    if forensics.altman.z_score is not None:
        z = forensics.altman.z_score
        z_score_pts = 95.0 if z > 3.5 else (80.0 if z > 2.99 else (50.0 if z >= 1.81 else 15.0))
        sub_signals.append(z_score_pts)
        if forensics.altman.status == "red":
            cons.append(f"Altman Z-Score v zóně finanční tísně ({fmt_number(z, 2)} < 1,81).")

    if forensics.beneish.m_score is not None:
        m = forensics.beneish.m_score
        m_pts = 15.0 if m > -1.78 else (60.0 if m > -2.22 else 95.0)
        sub_signals.append(m_pts)
        if forensics.beneish.status == "red":
            cons.append(f"Beneish M-Score ({fmt_number(m, 2)} > −1,78) signalizuje zvýšené riziko manipulace zisků.")
        elif forensics.beneish.status == "green":
            pros.append(f"Čisté účetnictví dle Beneish M-Score ({fmt_number(m, 2)}).")

    p3_score = _clamp_score(float(np.mean(sub_signals)))

    # ------------------------------------------------------------------
    # Pillar 4: Risk/Reward Asymmetry & Alignment (15%)
    # ------------------------------------------------------------------
    asym_points = 50.0
    if monte_carlo.available:
        # Low loss probability + strong upside/downside ratio
        loss_penalty = monte_carlo.prob_loss_gt_20pct * 100.0
        ud = monte_carlo.upside_downside_ratio or 1.0
        ud_score = float(np.clip(ud * 28.0, 10.0, 100.0))
        asym_points = 0.55 * (100.0 - loss_penalty) + 0.45 * ud_score
        if monte_carlo.prob_loss_gt_20pct >= 0.30:
            cons.append(
                f"Vysoké riziko poklesu: {fmt_percent(monte_carlo.prob_loss_gt_20pct, 0)} "
                f"scénářů končí se ztrátou > 20 % (10. percentil ceny {fmt_price(monte_carlo.p10_price, company.quote_currency)})."
            )
        elif monte_carlo.upside_downside_ratio and monte_carlo.upside_downside_ratio >= 2.5:
            pros.append(
                f"Příznivá asymetrie výnos/riziko ({fmt_number(monte_carlo.upside_downside_ratio, 1)} : 1) "
                f"s nízkým rizikem ztráty > 20 % ({fmt_percent(monte_carlo.prob_loss_gt_20pct, 0)})."
            )

    if insiders.cluster_buying_detected:
        asym_points = min(100.0, asym_points + 15.0)
        pros.append("Silný signál kůže ve hře: detekován nákupní shluk insiderů (Cluster Buying).")
    elif insiders.consistent_monthly_selling:
        asym_points = max(0.0, asym_points - 8.0)

    p4_score = _clamp_score(asym_points)

    # ------------------------------------------------------------------
    # Composite Score & Hard Veto Guards
    # ------------------------------------------------------------------
    total_score = _clamp_score(
        0.35 * p1_score + 0.25 * p2_score + 0.25 * p3_score + 0.15 * p4_score
    )

    veto_triggered = False
    veto_reason = ""
    if forensics.beneish.status == "red" and forensics.altman.status == "red":
        veto_triggered = True
        veto_reason = (
            "Automatické bezpečnostní VETO: současně červený Beneish M-Score (riziko manipulace) "
            "a Altman Z-Score v zóně finanční tísně."
        )
        total_score = min(total_score, 25)
    elif red_flags.red_flag_score >= 75 and not company.is_financial_or_reit:
        veto_triggered = True
        veto_reason = (
            f"Automatické bezpečnostní VETO: kritické Red Flag skóre ({red_flags.red_flag_score}/100)."
        )
        total_score = min(total_score, 29)

    # Determine final verdict tier
    if total_score >= 78 and not veto_triggered:
        v_code: VerdictCode = "strong_buy"
        v_title = "SILNÁ KOUPĚ · Vysoká kvantitativní přesvědčivost"
        v_sub = "Kvalitní byznys s čistým účetnictvím, atraktivní slevou a příznivou asymetrií výnos/riziko."
        color = "green"
    elif total_score >= 64 and not veto_triggered:
        v_code = "buy"
        v_title = "ZAJÍMAVÁ PŘÍLEŽITOST · Koupit / Budovat pozici"
        v_sub = "Tvrdá data hrají ve váš prospěch; očekávaný výnos a kvalita firmy převyšují zjištěná rizika."
        color = "green"
    elif total_score >= 46 and not veto_triggered:
        v_code = "wait"
        v_title = "VYČKAT NA LEPŠÍ CENU · Férově oceněno"
        v_sub = "Podnik má solidní základy, ale dnešní cena se obchoduje poblíž nebo nad férovou hodnotou."
        color = "amber"
    elif total_score >= 30 and not veto_triggered:
        v_code = "overvalued"
        v_title = "PŘEDRAŽENÉ · Nad férovou hodnotou"
        v_sub = "Trh oceňuje příliš optimistický scénář; pravděpodobnost překonání požadovaného výnosu je nízká."
        color = "red"
    else:
        v_code = "avoid"
        v_title = "NEKUPOVAT · Vysoké riziko / Hodnotová past"
        v_sub = veto_reason or "Zásadní slabiny v ocenění, kvalitě kapitálu nebo účetní bezpečnosti."
        color = "red"

    # Concrete Czech action plan based directly on pure Fair Value (no Margin of Safety haircut)
    ccy = company.quote_currency
    if veto_triggered or v_code == "avoid":
        action_plan = (
            "Akční plán: Vyřadit ze seznamu kandidátů (nebo vyčkat na zásadní očištění rozvahy a FCF)."
        )
    elif triangulation.fair_value_per_share is not None and company.current_price is not None:
        fv = triangulation.fair_value_per_share
        if company.current_price <= fv:
            discount = (fv / company.current_price) - 1.0
            action_plan = (
                f"Akční plán: Dnešní cena ({fmt_price(company.current_price, ccy)}) je pod férovou hodnotou "
                f"{fmt_price(fv, ccy)} (potenciál růstu {fmt_percent(discount, 1, signed=True)}). "
                "Vhodné k hlubšímu prověření a vstupu."
            )
        else:
            drop_needed = 1.0 - (fv / company.current_price)
            action_plan = (
                f"Akční plán: Akcie je nad férovou cenou {fmt_price(fv, ccy)} – "
                f"nastavit cenový alert na pokles o {fmt_percent(drop_needed, 1)} "
                f"z dnešní ceny {fmt_price(company.current_price, ccy)}."
            )
    else:
        action_plan = (
            "Akční plán: Kvůli zápornému FCF nebo neúplným výkazům nelze stanovit férovou cenu – postupujte s maximální opatrností."
        )

    pillars = [
        PillarScore("Ocenění vs. férová hodnota", p1_score, 35, "Matematické modely + Monte Carlo IRR + Reverse DCF"),
        PillarScore("Kvalita a motor složeného úročení", p2_score, 25, "ROIC − WACC spread + ROIIC × reinvestice + trend marží"),
        PillarScore("Forenzní bezpečnost", p3_score, 25, "Piotroski F-Score + Altman Z + Beneish M + Red Flags"),
        PillarScore("Asymetrie rizika a kůže ve hře", p4_score, 15, "Poměr Upside/Downside + riziko ztráty > 20 % + Insideři"),
    ]

    return DecisionVerdict(
        verdict_code=v_code,
        verdict_title_cz=v_title,
        verdict_subtitle_cz=v_sub,
        total_score=total_score,
        badge_color=color,
        action_plan_cz=action_plan,
        pillars=pillars,
        pros_cz=pros[:5] if pros else ["Nebyla nalezena výrazná kvantitativní výhoda oproti trhu."],
        cons_cz=cons[:5] if cons else ["Nebyly detekovány závažné kvantitativní red flags."],
        veto_triggered=veto_triggered,
        veto_reason_cz=veto_reason,
    )
