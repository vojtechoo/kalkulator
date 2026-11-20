"""Accounting tricks & red flags analysis (Real FCF, share count/buybacks, earnings quality, debt wall)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from analysis.formatting import fmt_multiple, fmt_number, fmt_percent, fmt_price
from data.models import CompanyData

TrafficLight = Literal["green", "amber", "red", "na"]


@dataclass
class FlagItem:
    """Single traffic-light red flag pillar with status and Czech explanation."""

    key: str
    title_cz: str
    status: TrafficLight
    headline_metric_cz: str
    explanation_cz: str


@dataclass
class DebtStressResult:
    """Debt wall metrics and +2 p.b. refinancing interest-coverage stress test."""

    net_debt: Optional[float]
    total_debt: Optional[float]
    short_term_debt: Optional[float]
    long_term_debt: Optional[float]
    ebitda: Optional[float]
    ebit: Optional[float]
    interest_expense: Optional[float]
    net_debt_to_ebitda: Optional[float]
    interest_coverage: Optional[float]
    effective_interest_rate: Optional[float]
    stressed_interest_expense: Optional[float]
    stressed_interest_coverage: Optional[float]
    short_term_share: Optional[float]


@dataclass
class RedFlagsReport:
    """Complete accounting red flags assessment."""

    flags: list[FlagItem]
    red_flag_score: int  # 0 (clean) to 100 (high risk)
    overall_status: TrafficLight
    overall_label_cz: str
    real_fcf_df: pd.DataFrame
    shares_df: pd.DataFrame
    earnings_quality_df: pd.DataFrame
    debt_stress: DebtStressResult
    buyback_verdict_cz: str
    warnings_cz: list[str] = field(default_factory=list)


def analyze_real_fcf(company: CompanyData) -> tuple[pd.DataFrame, FlagItem]:
    """Analyze reported FCF, stock-based compensation (SBC), and Real FCF (FCF - SBC)."""
    rows = []
    for a in company.annuals:
        fcf = a.free_cash_flow
        sbc = a.stock_based_compensation
        real_fcf = (fcf - (sbc or 0.0)) if fcf is not None else None
        rev = a.revenue
        shares = a.weighted_avg_shares_diluted

        rows.append(
            {
                "year": a.year,
                "fcf": fcf,
                "sbc": sbc,
                "real_fcf": real_fcf,
                "fcf_margin": (fcf / rev) if (fcf is not None and rev and rev > 0) else None,
                "sbc_pct_rev": (sbc / rev) if (sbc is not None and rev and rev > 0) else None,
                "real_fcf_margin": (real_fcf / rev) if (real_fcf is not None and rev and rev > 0) else None,
                "sbc_pct_fcf": (sbc / fcf) if (sbc is not None and fcf and fcf > 0) else None,
                "real_fcf_per_share": (real_fcf / shares) if (real_fcf is not None and shares and shares > 0) else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty or df["fcf"].dropna().empty:
        return df, FlagItem(
            key="real_fcf",
            title_cz="Skutečné FCF (FCF − SBC)",
            status="na",
            headline_metric_cz="N/A",
            explanation_cz="Chybí data o volném peněžním toku (FCF) nebo akciových odměnách (SBC).",
        )

    recent = df.tail(3)
    latest = df.iloc[-1]
    latest_fcf = latest["fcf"]
    latest_sbc = latest["sbc"]
    latest_real = latest["real_fcf"]

    avg_sbc_fcf = recent["sbc_pct_fcf"].dropna().mean() if not recent["sbc_pct_fcf"].dropna().empty else None
    avg_sbc_rev = recent["sbc_pct_rev"].dropna().mean() if not recent["sbc_pct_rev"].dropna().empty else None

    if latest_sbc is None:
        return df, FlagItem(
            key="real_fcf",
            title_cz="Skutečné FCF (FCF − SBC)",
            status="amber",
            headline_metric_cz="SBC neuvedeno",
            explanation_cz="Zdroj dat neuvádí samostatně akciové odměny (SBC); zkontrolujte výroční zprávu.",
        )

    if (latest_fcf is not None and latest_fcf > 0 and latest_real is not None and latest_real <= 0) or (
        avg_sbc_fcf is not None and avg_sbc_fcf > 0.30
    ) or (avg_sbc_rev is not None and avg_sbc_rev > 0.10):
        status: TrafficLight = "red"
        expl = (
            f"Vysoké akciové odměny: SBC ukrajuje v posledních 3 letech průměrně "
            f"{fmt_percent(avg_sbc_fcf)} z FCF ({fmt_percent(avg_sbc_rev)} tržeb). Skutečné FCF je výrazně nižší než vykázané."
        )
    elif (avg_sbc_fcf is not None and avg_sbc_fcf >= 0.15) or (
        avg_sbc_rev is not None and avg_sbc_rev >= 0.05
    ) or (latest_real is not None and latest_real <= 0):
        status = "amber"
        expl = (
            f"Zvýšený podíl SBC ({fmt_percent(avg_sbc_fcf)} z FCF, {fmt_percent(avg_sbc_rev)} tržeb). "
            "Vyplatí se oceňovat firmu na bázi FCF − SBC."
        )
    else:
        status = "green"
        expl = (
            f"Nízký vliv SBC ({fmt_percent(avg_sbc_fcf)} z FCF, {fmt_percent(avg_sbc_rev)} tržeb). "
            "Vykázané FCF věrně odráží skutečnou tvorbu hotovosti pro akcionáře."
        )

    headline = (
        f"SBC = {fmt_percent(avg_sbc_fcf)} FCF"
        if avg_sbc_fcf is not None
        else f"SBC = {fmt_percent(avg_sbc_rev)} tržeb"
    )
    return df, FlagItem(
        key="real_fcf",
        title_cz="Skutečné FCF (FCF − SBC)",
        status=status,
        headline_metric_cz=headline,
        explanation_cz=expl,
    )


def analyze_shares_and_buybacks(company: CompanyData) -> tuple[pd.DataFrame, FlagItem, str]:
    """Analyze diluted share count trend, annual dilution/buyback %, and buyback price efficiency."""
    rows = []
    prev_shares: Optional[float] = None

    for a in company.annuals:
        sh = a.weighted_avg_shares_diluted
        net_change_pct: Optional[float] = None
        shares_retired: Optional[float] = None
        est_buyback_price: Optional[float] = None

        if sh is not None and prev_shares is not None and prev_shares > 0:
            net_change_pct = (sh - prev_shares) / prev_shares
            if sh < prev_shares:
                shares_retired = prev_shares - sh

        if shares_retired and shares_retired > 0 and a.share_buybacks and a.share_buybacks > 0:
            est_buyback_price = a.share_buybacks / shares_retired

        # Evaluate whether that year's buybacks were cheap, fair, or expensive
        buyback_quality = "Bez odkupů"
        if a.share_buybacks and a.share_buybacks > 0:
            pe = a.pe_ratio if (a.pe_ratio and 0 < a.pe_ratio < 200) else None
            pfcf = a.pfcf_ratio if (a.pfcf_ratio and 0 < a.pfcf_ratio < 200) else None
            if pe is not None or pfcf is not None:
                ref_mult = np.nanmean([v for v in (pe, pfcf) if v is not None])
                if ref_mult <= 16.0:
                    buyback_quality = "Levné odkupy"
                elif ref_mult <= 25.0:
                    buyback_quality = "Přiměřené odkupy"
                else:
                    buyback_quality = "Drahé odkupy"
            else:
                buyback_quality = "Odkupy (násobek N/A)"

        rows.append(
            {
                "year": a.year,
                "diluted_shares": sh,
                "net_change_pct": net_change_pct,
                "buyback_spend": a.share_buybacks,
                "shares_retired_net": shares_retired,
                "est_buyback_price": est_buyback_price,
                "avg_share_price": a.avg_share_price,
                "pe_ratio": a.pe_ratio if (a.pe_ratio and 0 < a.pe_ratio < 200) else None,
                "pfcf_ratio": a.pfcf_ratio if (a.pfcf_ratio and 0 < a.pfcf_ratio < 200) else None,
                "buyback_quality": buyback_quality,
            }
        )
        if sh is not None and sh > 0:
            prev_shares = sh

    df = pd.DataFrame(rows)
    valid_changes = df["net_change_pct"].dropna()
    if df.empty or valid_changes.empty:
        return (
            df,
            FlagItem(
                key="shares",
                title_cz="Počet akcií a odkupy",
                status="na",
                headline_metric_cz="N/A",
                explanation_cz="Nedostatek ročních dat o počtu zředěných akcií.",
            ),
            "Nedostatek dat pro vyhodnocení zpětných odkupů.",
        )

    recent_changes = valid_changes.tail(5)
    avg_annual_change = float(recent_changes.mean())  # negative = buybacks, positive = dilution

    # Weighted average valuation in years with significant buybacks
    bb_rows = df[(df["buyback_spend"].fillna(0) > 0)]
    weighted_pe = weighted_pfcf = None
    if not bb_rows.empty:
        pe_sub = bb_rows.dropna(subset=["pe_ratio", "buyback_spend"])
        if not pe_sub.empty and pe_sub["buyback_spend"].sum() > 0:
            weighted_pe = float(
                np.average(pe_sub["pe_ratio"], weights=pe_sub["buyback_spend"])
            )
        pfcf_sub = bb_rows.dropna(subset=["pfcf_ratio", "buyback_spend"])
        if not pfcf_sub.empty and pfcf_sub["buyback_spend"].sum() > 0:
            weighted_pfcf = float(
                np.average(pfcf_sub["pfcf_ratio"], weights=pfcf_sub["buyback_spend"])
            )

    est_prices = bb_rows.dropna(subset=["est_buyback_price"])
    avg_est_price_str = (
        fmt_price(float(est_prices["est_buyback_price"].mean()), company.quote_currency)
        if not est_prices.empty
        else "N/A (odkupy pouze kompenzovaly SBC)"
    )

    if bb_rows.empty:
        buyback_verdict = "Firma v posledních letech neprováděla významné zpětné odkupy akcií."
    else:
        val_parts = []
        if weighted_pe is not None:
            val_parts.append(f"vážené P/E {fmt_multiple(weighted_pe)}")
        if weighted_pfcf is not None:
            val_parts.append(f"vážené P/FCF {fmt_multiple(weighted_pfcf)}")
        val_txt = " a ".join(val_parts) if val_parts else "násobky ocenění nejsou dostupné"

        expensive = (weighted_pe is not None and weighted_pe > 25.0) or (
            weighted_pfcf is not None and weighted_pfcf > 25.0
        )
        cheap = (
            (weighted_pe is not None and weighted_pe <= 16.0)
            or (weighted_pfcf is not None and weighted_pfcf <= 16.0)
        ) and not expensive
        if cheap:
            grade = "Odkupy probíhaly za atraktivní (levné) násobky."
        elif expensive:
            grade = "Odkupy probíhaly za vysoké (drahé) násobky – tvorba hodnoty je sporná."
        else:
            grade = "Odkupy probíhaly za přiměřené tržní ocenění."
        buyback_verdict = (
            f"{grade} Průměrné ocenění v letech odkupů: {val_txt}. "
            f"Odhadovaná průměrná cena čistě stažené akcie: {avg_est_price_str}."
        )

    if avg_annual_change > 0.02:
        status: TrafficLight = "red"
        expl = (
            f"Výrazné ředění akcionářů: počet zředěných akcií rostl v průměru o "
            f"{fmt_percent(avg_annual_change, signed=True)} ročně."
        )
    elif avg_annual_change > 0.005:
        status = "amber"
        expl = (
            f"Mírné ředění akcionářů ({fmt_percent(avg_annual_change, signed=True)} ročně). "
            "Sledujte, zda odkupy stačí pokrývat akciové odměny."
        )
    else:
        expensive_bb = (
            not bb_rows.empty
            and (
                (weighted_pe is not None and weighted_pe > 28.0)
                or (weighted_pfcf is not None and weighted_pfcf > 28.0)
            )
        )
        if expensive_bb:
            status = "amber"
            expl = (
                f"Počet akcií klesá ({fmt_percent(avg_annual_change, signed=True)}/rok), "
                "ale zpětné odkupy probíhaly při drahém ocenění (> 28x)."
            )
        else:
            status = "green"
            expl = (
                f"Příznivý vývoj počtu akcií ({fmt_percent(avg_annual_change, signed=True)}/rok). "
                "Management neředí stávající akcionáře."
            )

    return (
        df,
        FlagItem(
            key="shares",
            title_cz="Počet akcií a zpětné odkupy",
            status=status,
            headline_metric_cz=f"{fmt_percent(avg_annual_change, signed=True)} / rok",
            explanation_cz=expl,
        ),
        buyback_verdict,
    )


def analyze_earnings_quality(company: CompanyData) -> tuple[pd.DataFrame, FlagItem]:
    """Analyze OCF / Net Income conversion, divergence (NI up while OCF flat/down), and Sloan accruals."""
    rows = []
    prev_assets: Optional[float] = None

    for a in company.annuals:
        ocf = a.operating_cash_flow
        ni = a.net_income
        ratio = (ocf / ni) if (ocf is not None and ni is not None and ni > 0) else None

        avg_assets = (
            (a.total_assets + prev_assets) / 2.0
            if (a.total_assets is not None and prev_assets is not None)
            else a.total_assets
        )
        accruals = (
            ((ni - ocf) / avg_assets)
            if (ni is not None and ocf is not None and avg_assets and avg_assets > 0)
            else None
        )
        rows.append(
            {
                "year": a.year,
                "net_income": ni,
                "ocf": ocf,
                "ocf_to_ni": ratio,
                "accruals_ratio": accruals,
            }
        )
        if a.total_assets is not None and a.total_assets > 0:
            prev_assets = a.total_assets

    df = pd.DataFrame(rows)
    if df.empty or df["ocf_to_ni"].dropna().empty:
        return df, FlagItem(
            key="earnings_quality",
            title_cz="Kvalita zisků (OCF / Čistý zisk)",
            status="na",
            headline_metric_cz="N/A",
            explanation_cz="Nelze vyhodnotit poměr OCF / čistý zisk (ztrátová firma nebo chybějící data).",
        )

    recent5 = df.tail(5)
    valid_ratios = recent5["ocf_to_ni"].dropna()
    low_years = int((valid_ratios < 0.80).sum())
    avg_ratio = float(valid_ratios.mean())
    latest_accruals = (
        float(recent5["accruals_ratio"].dropna().iloc[-1])
        if not recent5["accruals_ratio"].dropna().empty
        else None
    )

    # Detect divergence: Net income grew > 5% while OCF stagnated or declined (<= 0%)
    divergence_years = 0
    for idx in range(1, len(recent5)):
        prev_r = recent5.iloc[idx - 1]
        curr_r = recent5.iloc[idx]
        if (
            pd.notna(prev_r["net_income"])
            and pd.notna(curr_r["net_income"])
            and prev_r["net_income"] > 0
            and pd.notna(prev_r["ocf"])
            and pd.notna(curr_r["ocf"])
            and prev_r["ocf"] > 0
        ):
            ni_chg = (curr_r["net_income"] - prev_r["net_income"]) / prev_r["net_income"]
            ocf_chg = (curr_r["ocf"] - prev_r["ocf"]) / prev_r["ocf"]
            if ni_chg > 0.05 and ocf_chg <= 0.0:
                divergence_years += 1

    if low_years >= 2 or (latest_accruals is not None and latest_accruals > 0.10) or divergence_years >= 2:
        status: TrafficLight = "red"
        reasons = []
        if low_years >= 2:
            reasons.append(f"OCF/Čistý zisk < 0,8 v {low_years} z posledních {len(valid_ratios)} let")
        if divergence_years >= 2:
            reasons.append("zisk opakovaně rostl, zatímco provozní cash flow stagnovalo či klesalo")
        if latest_accruals is not None and latest_accruals > 0.10:
            reasons.append(f"vysoký poměr akruálů ({fmt_percent(latest_accruals)})")
        expl = "Varovná kvalita zisků: " + "; ".join(reasons) + "."
    elif (
        low_years == 1
        or divergence_years == 1
        or (latest_accruals is not None and latest_accruals > 0.05)
        or avg_ratio < 0.90
    ):
        status = "amber"
        expl = (
            f"Průměrná konverze OCF/Čistý zisk je {fmt_number(avg_ratio, 2)}x "
            f"(akruály {fmt_percent(latest_accruals)}). Objevily se dílčí odchylky mezi účetním ziskem a hotovostí."
        )
    else:
        status = "green"
        expl = (
            f"Silná konverze účetního zisku do hotovosti (průměr OCF/NI = {fmt_number(avg_ratio, 2)}x, "
            f"akruály {fmt_percent(latest_accruals)})."
        )

    return df, FlagItem(
        key="earnings_quality",
        title_cz="Kvalita zisků a akruály",
        status=status,
        headline_metric_cz=f"OCF/NI {fmt_number(avg_ratio, 2)}x",
        explanation_cz=expl,
    )


def analyze_debt_wall(company: CompanyData) -> tuple[DebtStressResult, FlagItem]:
    """Analyze leverage (Net Debt/EBITDA), interest coverage, maturity split, and +2 p.b. stress test."""
    latest = company.annuals[-1] if company.annuals else None
    total_debt = company.latest_total_debt if company.latest_total_debt is not None else (
        latest.total_debt if latest else None
    )
    cash = company.latest_cash if company.latest_cash is not None else (
        latest.cash_and_short_term if latest else None
    )
    net_debt = (total_debt - cash) if (total_debt is not None and cash is not None) else company.latest_net_debt
    st_debt = latest.short_term_debt if latest else None
    lt_debt = latest.long_term_debt if latest else None
    ebitda = company.ttm_ebitda if company.ttm_ebitda is not None else (latest.ebitda if latest else None)
    ebit = company.ttm_ebit if company.ttm_ebit is not None else (latest.operating_income if latest else None)
    interest = latest.interest_expense if latest else None

    nd_ebitda = (net_debt / ebitda) if (net_debt is not None and ebitda and ebitda > 0) else None
    int_cov = (ebit / interest) if (ebit is not None and interest and interest > 0) else None
    eff_rate = (interest / total_debt) if (interest is not None and total_debt and total_debt > 0) else None

    # Stress test: refinancing cost increases by +2 percentage points across total debt
    stressed_interest: Optional[float] = None
    stressed_cov: Optional[float] = None
    if total_debt is not None and total_debt > 0:
        base_interest = interest if interest is not None else 0.0
        stressed_interest = base_interest + total_debt * 0.02
        if ebit is not None and stressed_interest > 0:
            stressed_cov = ebit / stressed_interest

    st_share = (st_debt / total_debt) if (st_debt is not None and total_debt and total_debt > 0) else None

    stress = DebtStressResult(
        net_debt=net_debt,
        total_debt=total_debt,
        short_term_debt=st_debt,
        long_term_debt=lt_debt,
        ebitda=ebitda,
        ebit=ebit,
        interest_expense=interest,
        net_debt_to_ebitda=nd_ebitda,
        interest_coverage=int_cov,
        effective_interest_rate=eff_rate,
        stressed_interest_expense=stressed_interest,
        stressed_interest_coverage=stressed_cov,
        short_term_share=st_share,
    )

    if total_debt is None and net_debt is None:
        return stress, FlagItem(
            key="debt_wall",
            title_cz="Dluhová zeď a stress test",
            status="na",
            headline_metric_cz="N/A",
            explanation_cz="Chybí rozvahová data o dluhu společnosti.",
        )

    # Net cash position is structurally safe unless EBIT is negative with heavy gross debt
    if net_debt is not None and net_debt <= 0 and (stressed_cov is None or stressed_cov >= 3.5):
        return stress, FlagItem(
            key="debt_wall",
            title_cz="Dluhová zeď a stress test",
            status="green",
            headline_metric_cz="Čistá hotovost",
            explanation_cz=(
                "Firma drží více hotovosti než dluhu (čistý dluh ≤ 0). "
                + (
                    f"Úrokové krytí po stress testu (+2 p.b.): {fmt_multiple(stressed_cov)}."
                    if stressed_cov is not None
                    else "Riziko refinancování je minimální."
                )
            ),
        )

    is_red = (
        (nd_ebitda is not None and nd_ebitda > 3.5)
        or (int_cov is not None and int_cov < 3.0)
        or (stressed_cov is not None and stressed_cov < 2.0)
        or (ebit is not None and ebit <= 0 and total_debt is not None and total_debt > 0)
    )
    is_amber = (
        (nd_ebitda is not None and nd_ebitda >= 2.0)
        or (int_cov is not None and int_cov < 6.0)
        or (stressed_cov is not None and stressed_cov < 3.5)
    )

    if is_red:
        status: TrafficLight = "red"
        expl = (
            f"Vysoké dluhové riziko: Čistý dluh/EBITDA = {fmt_multiple(nd_ebitda)}, "
            f"úrokové krytí = {fmt_multiple(int_cov)} (po +2 p.b. stress testu {fmt_multiple(stressed_cov)})."
        )
    elif is_amber:
        status = "amber"
        expl = (
            f"Střední zadlužení: Čistý dluh/EBITDA = {fmt_multiple(nd_ebitda)}, "
            f"úrokové krytí klesá ve stress testu (+2 p.b.) z {fmt_multiple(int_cov)} na {fmt_multiple(stressed_cov)}."
        )
    else:
        status = "green"
        expl = (
            f"Zdravá rozvaha: Čistý dluh/EBITDA = {fmt_multiple(nd_ebitda)}, "
            f"úrokové krytí {fmt_multiple(int_cov)} (stress test +2 p.b.: {fmt_multiple(stressed_cov)})."
        )

    headline = (
        f"ND/EBITDA {fmt_multiple(nd_ebitda)}"
        if nd_ebitda is not None
        else f"Krytí {fmt_multiple(int_cov)}"
    )
    return stress, FlagItem(
        key="debt_wall",
        title_cz="Dluhová zeď a stress test",
        status=status,
        headline_metric_cz=headline,
        explanation_cz=expl,
    )


def run_red_flags_analysis(company: CompanyData) -> RedFlagsReport:
    """Run all 4 accounting red-flag checks and aggregate an overall 0-100 Red Flag Score."""
    real_fcf_df, f1 = analyze_real_fcf(company)
    shares_df, f2, buyback_verdict = analyze_shares_and_buybacks(company)
    eq_df, f3 = analyze_earnings_quality(company)
    debt_stress, f4 = analyze_debt_wall(company)

    flags = [f1, f2, f3, f4]
    weights = {"green": 0, "amber": 50, "red": 100}
    scored = [weights[f.status] for f in flags if f.status != "na"]
    score = int(round(sum(scored) / len(scored))) if scored else 0

    red_count = sum(1 for f in flags if f.status == "red")
    amber_count = sum(1 for f in flags if f.status == "amber")

    if red_count >= 2 or score >= 60:
        overall_status: TrafficLight = "red"
        overall_label = f"Vysoké riziko ({score}/100)"
    elif red_count == 1 or amber_count >= 2 or score >= 25:
        overall_status = "amber"
        overall_label = f"Zvýšená opatrnost ({score}/100)"
    else:
        overall_status = "green"
        overall_label = f"Čisté účetnictví ({score}/100)"

    return RedFlagsReport(
        flags=flags,
        red_flag_score=score,
        overall_status=overall_status,
        overall_label_cz=overall_label,
        real_fcf_df=real_fcf_df,
        shares_df=shares_df,
        earnings_quality_df=eq_df,
        debt_stress=debt_stress,
        buyback_verdict_cz=buyback_verdict,
    )
