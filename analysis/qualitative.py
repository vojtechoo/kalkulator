"""Structured non-mathematical (qualitative) value-investing scorecard with hard-data auto-prefill.

Covers all 6 qualitative pillars:
1. Ekonomický příkop (Moat): síťový efekt, náklady na změnu dodavatele, značka, patenty, úspory z rozsahu
2. Cenotvorba (Pricing Power): schopnost zvyšovat ceny bez ztráty zákazníků
3. Management a alokace kapitálu: Skin in the game, reinvestice, akvizice, buybacks, dividendy
4. Předvídatelnost byznysu: stabilita poptávky a odolnost vůči technologické změně
5. Struktura odvětví (Porterových 5 sil): zákazníci, dodavatelé, bariéry vstupu, substituty, rivalita
6. Rizika mimo výkazy: koncentrace zákazníků/dodavatelů, regulace, soudní spory, geopolitika
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from analysis.capital_allocation import InsiderActivityReport
from analysis.forensics import ForensicsReport
from analysis.formatting import fmt_number, fmt_percent, fmt_pp
from analysis.quality import QualityXRayReport
from analysis.valuation_models import ValuationSuiteReport

MOAT_SOURCES_CZ: tuple[str, ...] = (
    "Síťový efekt",
    "Náklady na změnu dodavatele (Switching Costs)",
    "Silná značka",
    "Patenty / licence / regulovaná výhoda",
    "Úspory z rozsahu",
)

OFF_STATEMENT_RISKS_CZ: tuple[str, ...] = (
    "Koncentrace zákazníků nebo dodavatelů",
    "Regulační / antimonopolní riziko",
    "Závažné soudní spory",
    "Geopolitické nebo měnové riziko",
)

PORTER_FORCES_CZ: tuple[tuple[str, str], ...] = (
    ("customers", "Vyjednávací síla zákazníků (nízká = lepší)"),
    ("suppliers", "Vyjednávací síla dodavatelů (nízká = lepší)"),
    ("barriers", "Bariéry vstupu do odvětví (vysoké = lepší)"),
    ("substitutes", "Hrozba substitutů (nízká = lepší)"),
    ("rivalry", "Intenzita rivality v odvětví (nízká = lepší)"),
)


@dataclass
class QualitativeDefaults:
    """Data-derived initial scores (1..5) and Czech hard-data evidence hints for each qualitative pillar."""

    moat_score: int
    moat_hint_cz: str
    pricing_power_score: int
    pricing_hint_cz: str
    management_score: int
    management_hint_cz: str
    predictability_score: int
    predictability_hint_cz: str
    porter_default_score: int
    porter_hint_cz: str


@dataclass
class QualitativeEvaluation:
    """Computed qualitative score (0..100) and breakdown across all 6 non-mathematical criteria."""

    total_score: int  # 0..100
    grade_cz: str
    color: Literal["green", "amber", "red"]
    pillar_scores: dict[str, int] = field(default_factory=dict)  # each 0..100
    selected_moat_sources: list[str] = field(default_factory=list)
    selected_risks: list[str] = field(default_factory=list)
    summary_cz: str = ""


def infer_qualitative_defaults(
    quality: QualityXRayReport,
    forensics: ForensicsReport,
    insiders: InsiderActivityReport,
    suite: ValuationSuiteReport,
) -> QualitativeDefaults:
    """Estimate conservative starting scores (1..5) for qualitative pillars directly from hard-data proxies."""
    # 1. Moat proxy from ROIC - WACC spread & consistency
    if quality.avg_spread is not None and quality.avg_spread >= 0.08 and quality.years_below_wacc == 0:
        moat_s = 5
    elif quality.avg_spread is not None and quality.avg_spread >= 0.04:
        moat_s = 4
    elif quality.avg_spread is not None and quality.avg_spread > 0:
        moat_s = 3
    else:
        moat_s = 2
    moat_hint = (
        f"Tvrdá stopa příkopu: průměrné ROIC {fmt_percent(quality.avg_roic, 1)}, "
        f"spread nad WACC {fmt_pp(quality.avg_spread, 1)}, roky pod WACC: {quality.years_below_wacc}/{quality.total_roic_years}."
    )

    # 2. Pricing power proxy from Gross Margin level & trend
    gm = quality.avg_gross_margin
    if gm is not None and gm >= 0.55 and quality.gross_margin_trend in ("rising", "stable"):
        pp_s = 5
    elif gm is not None and gm >= 0.35 and quality.gross_margin_trend in ("rising", "stable"):
        pp_s = 4
    elif quality.gross_margin_trend == "falling":
        pp_s = 2
    else:
        pp_s = 3
    pricing_hint = (
        f"Tvrdá stopa cenotvorby: průměrná hrubá marže {fmt_percent(gm, 1)}, "
        f"trend: {quality.gross_margin_trend_label_cz}."
    )

    # 3. Management & capital allocation proxy from ROIIC + insiders
    mgmt_s = 3
    if forensics.compounding.status == "green":
        mgmt_s += 1
    elif forensics.compounding.status == "red":
        mgmt_s -= 1
    if insiders.cluster_buying_detected or insiders.net_value > 0:
        mgmt_s = min(5, mgmt_s + 1)
    elif insiders.consistent_monthly_selling:
        mgmt_s = max(1, mgmt_s - 1)
    mgmt_hint = (
        f"Tvrdá stopa alokace: ROIIC {fmt_percent(forensics.compounding.roiic, 1)}, "
        f"open-market insideři: {insiders.buy_count} nákupů / {insiders.sell_count} prodejů."
    )

    # 4. Business predictability proxy from GF-style predictability stars
    pred_s = max(1, min(5, int(round(suite.predictability_stars))))
    pred_hint = (
        f"Tvrdá stopa předvídatelnosti: {fmt_number(suite.predictability_stars, 1)} / 5 ★ "
        f"({suite.predictability_label_cz})."
    )

    # 5. Porter 5 forces default proxy from operating margin resilience
    om = quality.avg_operating_margin
    porter_s = 4 if (om is not None and om >= 0.22) else (3 if (om is not None and om >= 0.12) else 2)
    porter_hint = (
        f"Struktura odvětví (proxy z marží): průměrná provozní marže {fmt_percent(om, 1)} "
        "(dlouhodobě vysoká provozní marže značí příznivé konkurenční prostředí)."
    )

    return QualitativeDefaults(
        moat_score=moat_s,
        moat_hint_cz=moat_hint,
        pricing_power_score=pp_s,
        pricing_hint_cz=pricing_hint,
        management_score=mgmt_s,
        management_hint_cz=mgmt_hint,
        predictability_score=pred_s,
        predictability_hint_cz=pred_hint,
        porter_default_score=porter_s,
        porter_hint_cz=porter_hint,
    )


def evaluate_qualitative_scorecard(
    moat_rating: int,
    selected_moat_sources: list[str],
    pricing_power_rating: int,
    management_rating: int,
    predictability_rating: int,
    porter_ratings: dict[str, int],
    selected_risks: list[str],
) -> QualitativeEvaluation:
    """Aggregate all 6 non-mathematical criteria into a 0–100 qualitative moat & risk score."""
    # Scale 1..5 -> 20..100, with slight bonus for concrete identified moat sources
    moat_pct = min(100, moat_rating * 18 + min(len(selected_moat_sources), 3) * 4)
    pricing_pct = pricing_power_rating * 20
    mgmt_pct = management_rating * 20
    pred_pct = predictability_rating * 20

    porter_vals = list(porter_ratings.values()) if porter_ratings else [3]
    porter_pct = int(round((sum(porter_vals) / len(porter_vals)) * 20.0))

    # Each checked off-statement risk deducts 20 points from the Risk Resilience pillar (100 -> 20)
    risk_resilience_pct = max(15, 100 - len(selected_risks) * 22)

    total = int(
        round(
            0.25 * moat_pct
            + 0.18 * pricing_pct
            + 0.18 * mgmt_pct
            + 0.14 * pred_pct
            + 0.15 * porter_pct
            + 0.10 * risk_resilience_pct
        )
    )

    if total >= 78:
        grade = "Výjimečný byznys (Wide Moat profil)"
        color: Literal["green", "amber", "red"] = "green"
    elif total >= 60:
        grade = "Kvalitní byznys s udržitelnou výhodou"
        color = "green"
    elif total >= 45:
        grade = "Průměrná kvalita / smíšená rizika"
        color = "amber"
    else:
        grade = "Křehký byznys nebo vysoká rizika mimo výkazy"
        color = "red"

    moat_txt = ", ".join(selected_moat_sources) if selected_moat_sources else "nevybrány konkrétní zdroje příkopu"
    risk_txt = ", ".join(selected_risks) if selected_risks else "bez označených kritických rizik mimo výkazy"
    summary = f"Zdroje příkopu: {moat_txt}. Rizika mimo výkazy: {risk_txt}."

    return QualitativeEvaluation(
        total_score=total,
        grade_cz=grade,
        color=color,
        pillar_scores={
            "Ekonomický příkop (Moat)": moat_pct,
            "Cenotvorba (Pricing Power)": pricing_pct,
            "Management a alokace": mgmt_pct,
            "Předvídatelnost byznysu": pred_pct,
            "Porterových 5 sil": porter_pct,
            "Odolnost mimo výkazy": risk_resilience_pct,
        },
        selected_moat_sources=selected_moat_sources,
        selected_risks=selected_risks,
        summary_cz=summary,
    )
