"""Evaluation of a stock through the quantitative checklists of 5 legendary value investors:
1. Warren Buffett (Quality Compounder & Moat)
2. Benjamin Graham (Deep Value & Margin of Safety)
3. Peter Lynch (GARP – Growth at a Reasonable Price)
4. Joel Greenblatt (Magic Formula – High ROIC + High Earnings Yield)
5. Charlie Munger / Terry Smith (Capital-Light Wide Moat & Clean Cash Conversion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from analysis.capital_allocation import InsiderActivityReport
from analysis.forensics import ForensicsReport
from analysis.formatting import fmt_multiple, fmt_number, fmt_percent, fmt_pp, fmt_price
from analysis.quality import QualityXRayReport
from analysis.red_flags import RedFlagsReport
from analysis.valuation import compute_current_multiples
from analysis.valuation_models import ValuationSuiteReport
from data.models import CompanyData

LegendVerdictColor = Literal["green", "amber", "red"]


@dataclass
class InvestorCriterion:
    """Single documented hard-data rule for a legendary investor."""

    label_cz: str
    passed: Optional[bool]
    observed_cz: str


@dataclass
class LegendEvaluation:
    """Complete scorecard for one legendary investor."""

    investor_name: str
    style_cz: str
    passed_count: int
    evaluated_count: int
    score_pct: int  # 0..100
    verdict_cz: str  # "Koupil by", "Sledoval by", "Nekoupil by"
    color: LegendVerdictColor
    summary_cz: str
    criteria: list[InvestorCriterion] = field(default_factory=list)


@dataclass
class LegendsReport:
    """Aggregate consensus across all 5 legendary investor frameworks."""

    investors: list[LegendEvaluation]
    buy_votes: int
    watch_votes: int
    pass_votes: int
    avg_score_pct: int
    consensus_label_cz: str
    consensus_color: LegendVerdictColor


def _finalize_legend(
    name: str,
    style_cz: str,
    criteria: list[InvestorCriterion],
    buy_summary: str,
    watch_summary: str,
    pass_summary: str,
) -> LegendEvaluation:
    evaluated = [c for c in criteria if c.passed is not None]
    passed = sum(1 for c in evaluated if c.passed is True)
    total = len(evaluated)
    pct = int(round((passed / total) * 100.0)) if total > 0 else 0

    if pct >= 75:
        verdict = "Koupil by"
        color: LegendVerdictColor = "green"
        summary = buy_summary
    elif pct >= 50:
        verdict = "Sledoval by"
        color = "amber"
        summary = watch_summary
    else:
        verdict = "Nekoupil by"
        color = "red"
        summary = pass_summary

    return LegendEvaluation(
        investor_name=name,
        style_cz=style_cz,
        passed_count=passed,
        evaluated_count=total,
        score_pct=pct,
        verdict_cz=verdict,
        color=color,
        summary_cz=summary,
        criteria=criteria,
    )


def evaluate_legendary_investors(
    company: CompanyData,
    quality: QualityXRayReport,
    red_flags: RedFlagsReport,
    forensics: ForensicsReport,
    suite: ValuationSuiteReport,
    insiders: InsiderActivityReport,
) -> LegendsReport:
    """Score the stock strictly on hard-data checklists of 5 legendary investors."""
    pe, pfcf, _ = compute_current_multiples(company)
    latest = company.annuals[-1] if company.annuals else None
    price = company.current_price
    ccy = company.quote_currency

    # Helper metrics
    net_debt = company.latest_net_debt
    fcf_latest = suite.cashflow_decomp.fcf_adj or suite.cashflow_decomp.fcf
    nd_to_fcf = (
        (net_debt / fcf_latest)
        if (net_debt is not None and fcf_latest is not None and fcf_latest > 0)
        else (0.0 if (net_debt is not None and net_debt <= 0) else None)
    )
    curr_ratio = (
        (latest.current_assets / latest.current_liabilities)
        if (latest and latest.current_assets and latest.current_liabilities and latest.current_liabilities > 0)
        else None
    )

    # Earnings Yield = EBIT / EV
    ebit = company.ttm_ebit if company.ttm_ebit is not None else (latest.operating_income if latest else None)
    ev = (
        (company.market_cap + (net_debt or 0.0))
        if company.market_cap is not None
        else None
    )
    earnings_yield = (ebit / ev) if (ebit is not None and ev and ev > 0) else None

    # Historical revenue growth (3-5Y)
    revs = [a.revenue for a in company.annuals[-5:] if a.revenue and a.revenue > 0]
    rev_cagr = ((revs[-1] / revs[0]) ** (1.0 / (len(revs) - 1)) - 1.0) if len(revs) >= 2 else None

    # Graham model value
    graham_row = next((m for m in suite.models if m.key == "graham"), None)
    graham_val = graham_row.value_per_share if graham_row else None

    # Average dilution / buyback rate
    sh_changes = red_flags.shares_df["net_change_pct"].dropna() if not red_flags.shares_df.empty else []
    avg_sh_chg = float(sh_changes.tail(5).mean()) if len(sh_changes) > 0 else None

    # SBC / FCF ratio
    sbc_pct_fcf = (
        float(red_flags.real_fcf_df["sbc_pct_fcf"].dropna().tail(3).mean())
        if (not red_flags.real_fcf_df.empty and not red_flags.real_fcf_df["sbc_pct_fcf"].dropna().empty)
        else None
    )
    avg_ocf_ni = (
        float(red_flags.earnings_quality_df["ocf_to_ni"].dropna().tail(5).mean())
        if (not red_flags.earnings_quality_df.empty and not red_flags.earnings_quality_df["ocf_to_ni"].dropna().empty)
        else None
    )

    # ------------------------------------------------------------------
    # 1. Warren Buffett
    # ------------------------------------------------------------------
    buffett_criteria = [
        InvestorCriterion(
            "Vysoké ROIC (≥ 15 %)",
            (quality.avg_roic >= 0.15) if quality.avg_roic is not None else None,
            fmt_percent(quality.avg_roic, 1),
        ),
        InvestorCriterion(
            "Silná hrubá marže (≥ 40 %)",
            (quality.avg_gross_margin >= 0.40) if quality.avg_gross_margin is not None else None,
            fmt_percent(quality.avg_gross_margin, 1),
        ),
        InvestorCriterion(
            "Konzervativní dluh (Čistý dluh / FCF ≤ 4x)",
            (nd_to_fcf <= 4.0) if nd_to_fcf is not None else None,
            "Čistá hotovost" if (net_debt is not None and net_debt <= 0) else fmt_multiple(nd_to_fcf),
        ),
        InvestorCriterion(
            "Kladné Owner Earnings & předvídatelnost (≥ 3,5 ★)",
            (suite.cashflow_decomp.owner_earnings is not None and suite.cashflow_decomp.owner_earnings > 0 and suite.predictability_stars >= 3.5),
            f"{fmt_number(suite.predictability_stars, 1)} ★",
        ),
        InvestorCriterion(
            "Rozumná cena (Cena ≤ 1,10× GF Hodnota)",
            (suite.price_to_gf_value <= 1.10) if suite.price_to_gf_value is not None else None,
            fmt_multiple(suite.price_to_gf_value, 2),
        ),
    ]
    buffett = _finalize_legend(
        "Warren Buffett",
        "Kvalitní byznys s příkopem za rozumnou cenu",
        buffett_criteria,
        "Silný ekonomický příkop, vysoké ROIC a rozumné ocenění – přesně Buffettův styl.",
        "Byznys má kvalitu, ale cena nebo dluh nesplňuje všechna přísná kritéria.",
        "Firma nemá dostatečně hluboký příkop, stabilitu nebo je příliš drahá.",
    )

    # ------------------------------------------------------------------
    # 2. Benjamin Graham
    # ------------------------------------------------------------------
    all_profitable = (
        all((a.net_income or 0) > 0 for a in company.annuals[-5:])
        if len(company.annuals) >= 3
        else None
    )
    graham_criteria = [
        InvestorCriterion(
            "Cena pod Grahamovým vzorcem",
            (price <= graham_val) if (price and graham_val) else None,
            f"Graham: {fmt_price(graham_val, ccy)}",
        ),
        InvestorCriterion(
            "Konzervativní P/E (≤ 18x)",
            (pe <= 18.0) if pe is not None else None,
            fmt_multiple(pe),
        ),
        InvestorCriterion(
            "Finanční pevnost (Altman Z > 2,99 nebo likvidita ≥ 1,5x)",
            (
                (forensics.altman.z_score is not None and forensics.altman.z_score > 2.99)
                or (curr_ratio is not None and curr_ratio >= 1.5)
            )
            if (forensics.altman.z_score is not None or curr_ratio is not None)
            else None,
            f"Z: {fmt_number(forensics.altman.z_score, 2)} | CR: {fmt_multiple(curr_ratio)}",
        ),
        InvestorCriterion(
            "Ziskovost každý rok (posledních 5 let)",
            all_profitable,
            "Ano (bez ztráty)" if all_profitable else ("Ztrátové roky" if all_profitable is False else "N/A"),
        ),
        InvestorCriterion(
            "Žádné ředění akcionářů (≤ +0,5 %/rok)",
            (avg_sh_chg <= 0.005) if avg_sh_chg is not None else None,
            fmt_percent(avg_sh_chg, 1, signed=True),
        ),
    ]
    graham = _finalize_legend(
        "Benjamin Graham",
        "Hluboká hodnota a pevnost rozvahy",
        graham_criteria,
        "Nízké násobky, pevná rozvaha a cena pod Grahamovou formulí.",
        "Rozvaha je solidní, ale tržní násobek P/E je pro Grahama na hraně.",
        "Příliš vysoké násobky P/E nebo slabá rozvahová opora pro defenzivního investora.",
    )

    # ------------------------------------------------------------------
    # 3. Peter Lynch (GARP)
    # ------------------------------------------------------------------
    g_for_peg = forensics.compounding.intrinsic_compounding_rate or rev_cagr
    peg = (
        (pe / (g_for_peg * 100.0))
        if (pe is not None and pe > 0 and g_for_peg is not None and g_for_peg > 0.02)
        else None
    )
    lynch_criteria = [
        InvestorCriterion(
            "Atraktivní PEG poměr (≤ 1,5)",
            (peg <= 1.5) if peg is not None else None,
            fmt_number(peg, 2),
        ),
        InvestorCriterion(
            "Růst tržeb (≥ 8 % ročně)",
            (rev_cagr >= 0.08) if rev_cagr is not None else None,
            fmt_percent(rev_cagr, 1),
        ),
        InvestorCriterion(
            "Nízké zadlužení (Čistý dluh / EBITDA < 2,0x)",
            (
                (net_debt is not None and net_debt <= 0)
                or (red_flags.debt_stress.net_debt_to_ebitda is not None and red_flags.debt_stress.net_debt_to_ebitda < 2.0)
            )
            if (net_debt is not None or red_flags.debt_stress.net_debt_to_ebitda is not None)
            else None,
            "Čistá hotovost" if (net_debt is not None and net_debt <= 0) else fmt_multiple(red_flags.debt_stress.net_debt_to_ebitda),
        ),
        InvestorCriterion(
            "Silná hotovostní konverze (OCF / NI ≥ 0,9x)",
            (avg_ocf_ni >= 0.90) if avg_ocf_ni is not None else None,
            fmt_multiple(avg_ocf_ni, 2),
        ),
        InvestorCriterion(
            "Zpětné odkupy nebo nákupy insiderů",
            ((avg_sh_chg is not None and avg_sh_chg < 0) or insiders.buy_count > 0)
            if (avg_sh_chg is not None or insiders.buy_count > 0)
            else None,
            f"Akcie {fmt_percent(avg_sh_chg, 1, signed=True)} · Insideři {insiders.buy_count} nák.",
        ),
    ]
    lynch = _finalize_legend(
        "Peter Lynch",
        "GARP – Růst za rozumnou cenu",
        lynch_criteria,
        "Rychlý růst za férový PEG a silná hotovost – ideální růstový příběh.",
        "Slušný růst, ale PEG nebo tempo expanze už není mimořádně levné.",
        "Růst je příliš pomalý vzhledem k ceně, nebo je PEG nad komfortní hranicí.",
    )

    # ------------------------------------------------------------------
    # 4. Joel Greenblatt (Magic Formula)
    # ------------------------------------------------------------------
    greenblatt_criteria = [
        InvestorCriterion(
            "Vysoké ROIC (≥ 20 %)",
            (quality.avg_roic >= 0.20) if quality.avg_roic is not None else None,
            fmt_percent(quality.avg_roic, 1),
        ),
        InvestorCriterion(
            "Vysoký Earnings Yield (EBIT / EV ≥ 7 %)",
            (earnings_yield >= 0.07) if earnings_yield is not None else None,
            fmt_percent(earnings_yield, 1),
        ),
        InvestorCriterion(
            "Tvorba hodnoty nad WACC (Spread ≥ +5 p.b.)",
            (quality.avg_spread >= 0.05) if quality.avg_spread is not None else None,
            fmt_pp(quality.avg_spread, 1),
        ),
        InvestorCriterion(
            "Čisté účetnictví (Beneish M-Score < −1,78)",
            (forensics.beneish.m_score < -1.78) if forensics.beneish.m_score is not None else None,
            fmt_number(forensics.beneish.m_score, 2),
        ),
    ]
    greenblatt = _finalize_legend(
        "Joel Greenblatt",
        "Magická formule (vysoké ROIC + vysoký výnos)",
        greenblatt_criteria,
        "Vynikající kombinace návratnosti kapitálu a levného provozního výnosu EBIT/EV.",
        "Vysoká kvalita kapitálu, ale výnosnost EBIT/EV je kvůli vyšší ceně nižší.",
        "Nesplňuje podmínku levného a zároveň vysoce rentabilního kapitálu.",
    )

    # ------------------------------------------------------------------
    # 5. Charlie Munger / Terry Smith
    # ------------------------------------------------------------------
    munger_criteria = [
        InvestorCriterion(
            "0 let pod WACC (ROIC > WACC v každém roce)",
            (quality.years_below_wacc == 0 and quality.total_roic_years > 0)
            if quality.total_roic_years > 0
            else None,
            f"{quality.years_below_wacc} / {quality.total_roic_years} let pod WACC",
        ),
        InvestorCriterion(
            "Nízké ředění přes SBC (SBC < 15 % FCF)",
            (sbc_pct_fcf < 0.15) if sbc_pct_fcf is not None else None,
            fmt_percent(sbc_pct_fcf, 1),
        ),
        InvestorCriterion(
            "Stabilní či rostoucí hrubá marže",
            (quality.gross_margin_trend in ("rising", "stable"))
            if quality.gross_margin_trend != "na"
            else None,
            quality.gross_margin_trend_label_cz.split(" (")[0],
        ),
        InvestorCriterion(
            "Silné Piotroski F-Score (≥ 6/9)",
            (forensics.piotroski.score >= 6) if forensics.piotroski.score is not None else None,
            f"{forensics.piotroski.score}/{forensics.piotroski.max_evaluated}"
            if forensics.piotroski.score is not None
            else "N/A",
        ),
        InvestorCriterion(
            "Vysoká návratnost nových investic (ROIIC ≥ 15 %)",
            (
                (forensics.compounding.roiic is not None and forensics.compounding.roiic >= 0.15)
                or (forensics.compounding.status == "green")
            )
            if forensics.compounding.status != "na"
            else None,
            fmt_percent(forensics.compounding.roiic, 1)
            if forensics.compounding.roiic is not None
            else ("Kapitálově nenáročné" if forensics.compounding.status == "green" else "N/A"),
        ),
    ]
    munger = _finalize_legend(
        "Charlie Munger & Terry Smith",
        "Kvalitní compounder bez účetních triků",
        munger_criteria,
        "Učebnicový dlouhodobý compounder s čistou hotovostí a disciplinovaným kapitálem.",
        "Dobrá kvalita byznysu, ale s dílčí vadou (např. vyšší SBC nebo kolísání marží).",
        "Nevyhovuje nárokům na elitní dlouhodobý compounder.",
    )

    investors = [buffett, graham, lynch, greenblatt, munger]
    buys = sum(1 for inv in investors if inv.color == "green")
    watches = sum(1 for inv in investors if inv.color == "amber")
    passes = sum(1 for inv in investors if inv.color == "red")
    avg_pct = int(round(sum(inv.score_pct for inv in investors) / len(investors)))

    if buys >= 4 or avg_pct >= 75:
        c_lbl = f"Silná shoda legend ({buys}/5 by koupilo)"
        c_col: LegendVerdictColor = "green"
    elif buys + watches >= 3 and avg_pct >= 52:
        c_lbl = f"Smíšený pohled legend ({buys} koupit · {watches} sledovat · {passes} ne)"
        c_col = "amber"
    else:
        c_lbl = f"Většina legend by vynechala ({passes}/5 nekoupit)"
        c_col = "red"

    return LegendsReport(
        investors=investors,
        buy_votes=buys,
        watch_votes=watches,
        pass_votes=passes,
        avg_score_pct=avg_pct,
        consensus_label_cz=c_lbl,
        consensus_color=c_col,
    )
