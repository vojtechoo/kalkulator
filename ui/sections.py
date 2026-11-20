"""GuruFocus-inspired UI sections:
- Sticky Header Strip & 5-Box 10-Point Scorecards
- GuruFocus Valuation Gauge + GF Value Band Chart + All 13 Mathematical Valuation Formulas
- Monte Carlo 10,000-Scenario Crystal Ball (3Y default)
- Interactive 6-Pillar Non-Mathematical (Qualitative) Scorecard + Radar Chart
- Collapsible Deep-Dive Expanders (LaTeX formulas, Reverse DCF heatmaps, Forensics F/Z/M, ROIC/WACC, Peers, Insiders)
"""

from __future__ import annotations

import html
from typing import Literal

import pandas as pd
import streamlit as st

from analysis.capital_allocation import CapitalAllocationReport, InsiderActivityReport
from analysis.crystal_ball import MonteCarloResult
from analysis.dcf import (
    CapExDistortionDiagnostics,
    ReverseDCFInputs,
    ReverseDCFResult,
    build_implied_growth_grid,
    build_intrinsic_value_grid,
)
from analysis.decision_engine import DecisionVerdict
from analysis.forensics import ForensicsReport
from analysis.legend_investors import LegendsReport
from analysis.formatting import (
    fmt_large_currency,
    fmt_multiple,
    fmt_number,
    fmt_percent,
    fmt_pp,
    fmt_price,
    fmt_shares,
)
from analysis.qualitative import (
    MOAT_SOURCES_CZ,
    OFF_STATEMENT_RISKS_CZ,
    PORTER_FORCES_CZ,
    QualitativeEvaluation,
    evaluate_qualitative_scorecard,
    infer_qualitative_defaults,
)
from analysis.quality import QualityXRayReport
from analysis.red_flags import RedFlagsReport, TrafficLight
from analysis.valuation import ValuationBand
from analysis.valuation_models import ValuationSuiteReport
from data.models import CompanyData
from ui.charts import (
    make_all_models_valuation_chart,
    make_capital_allocation_annual_chart,
    make_capital_allocation_cumulative_chart,
    make_debt_stress_chart,
    make_earnings_quality_chart,
    make_gf_valuation_gauge,
    make_gf_value_band_chart,
    make_implied_growth_heatmap,
    make_intrinsic_value_heatmap,
    make_margins_chart,
    make_monte_carlo_fan_chart,
    make_monte_carlo_irr_histogram,
    make_qualitative_radar_chart,
    make_real_fcf_chart,
    make_roic_wacc_chart,
    make_shares_chart,
    make_valuation_bands_chart,
)


def _section_header(number: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-num">{html.escape(number)}</div>
            <div class="section-title">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_badge_html(status: TrafficLight, label: str) -> str:
    cls_map = {
        "green": "badge-green",
        "amber": "badge-amber",
        "red": "badge-red",
        "na": "badge-gray",
    }
    icon_map = {"green": "●", "amber": "▲", "red": "■", "na": "○"}
    return f'<span class="badge {cls_map[status]}">{icon_map[status]} {html.escape(label)}</span>'


def render_sticky_summary_strip(
    company: CompanyData,
    verdict: DecisionVerdict,
    suite: ValuationSuiteReport,
    monte_carlo: MonteCarloResult,
) -> None:
    """GuruFocus-style sticky header strip with GF Score, valuation status, Fair Value, Buy Price & IRR."""
    ccy = company.quote_currency
    price_str = fmt_price(company.current_price, ccy)
    fv_str = fmt_price(suite.gf_fair_value, ccy)
    upside_str = fmt_percent(suite.upside_pct, 1, signed=True) if suite.upside_pct is not None else "N/A"
    upside_color = (
        "#4ADE80"
        if (suite.upside_pct is not None and suite.upside_pct >= 0)
        else ("#F87171" if suite.upside_pct is not None else "#CBD5E1")
    )
    irr_str = fmt_percent(monte_carlo.p50_irr, 1, signed=True) if monte_carlo.available else "N/A"
    irr_color = (
        "#4ADE80"
        if (monte_carlo.available and monte_carlo.p50_irr >= monte_carlo.hurdle_rate)
        else ("#FBBF24" if (monte_carlo.available and monte_carlo.p50_irr > 0) else "#F87171")
    )
    gf_badge = _status_badge_html(verdict.badge_color, f"GF {suite.composite_gf_score}/100")

    stars_full = int(suite.predictability_stars)
    stars_str = "★" * stars_full + "☆" * (5 - stars_full)
    sub_parts = [p for p in (company.exchange, company.sector, f"Předvídatelnost {stars_str}") if p]
    subtitle = " · ".join(sub_parts)

    st.markdown(
        f"""
        <div class="sticky-strip">
            <div class="sticky-grid">
                <div>
                    <div class="sticky-title">{html.escape(company.name)} ({html.escape(company.ticker)})</div>
                    <div class="sticky-sub">{html.escape(subtitle)}</div>
                </div>
                <div class="sticky-kpi">
                    <div class="sticky-kpi-label">GF Skóre</div>
                    <div class="sticky-kpi-val">{gf_badge}</div>
                </div>
                <div class="sticky-kpi">
                    <div class="sticky-kpi-label">Cena dnes</div>
                    <div class="sticky-kpi-val">{html.escape(price_str)}</div>
                </div>
                <div class="sticky-kpi">
                    <div class="sticky-kpi-label">GF Férová hodnota</div>
                    <div class="sticky-kpi-val">{html.escape(fv_str)}</div>
                </div>
                <div class="sticky-kpi">
                    <div class="sticky-kpi-label">Rozdíl vs. férová cena</div>
                    <div class="sticky-kpi-val" style="color: {upside_color};">{html.escape(upside_str)}</div>
                </div>
                <div class="sticky-kpi">
                    <div class="sticky-kpi-label">Medián IRR ({monte_carlo.horizon_years}Y)</div>
                    <div class="sticky-kpi-val" style="color: {irr_color};">{html.escape(irr_str)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warnings_and_banners(
    company: CompanyData,
    capex_diag: CapExDistortionDiagnostics,
    dcf_params: ReverseDCFInputs,
) -> None:
    """Transparent Czech notices for financials/REITs, extraordinary CapEx safeguard, and FX conversions."""
    if company.is_financial_or_reit:
        st.warning(
            "**Upozornění pro finanční sektor / REIT:** Společnost působí v sektoru "
            f"*{company.sector or company.industry}*. Standardní FCF, EPV, ROIC, Altman Z-Score a Beneish M-Score "
            "mají u bank, pojišťoven a REITů omezenou platnost (vhodnější je hodnotit P/B, ROE nebo FFO/AFFO)."
        )

    if capex_diag.should_warn:
        st.warning(capex_diag.warning_cz)
    elif dcf_params.custom_fcf_0_millions != 0.0:
        st.info(
            f"**Aktivní vlastní startovní FCF₀:** Pro DCF a Reverse DCF se používá ručně zadaná hodnota "
            f"**{fmt_large_currency(dcf_params.custom_fcf_0_millions * 1e6, company.quote_currency)}**."
        )
    elif dcf_params.normalize_capex and capex_diag.normalized_fcf_0 is not None:
        ratio_used = capex_diag.hist_5y_median_capex_to_rev or capex_diag.hist_4y_avg_capex_to_rev
        st.info(
            f"**Aktivní normalizace mimořádného CapExu:** Normalizovaný CapEx = "
            f"**{fmt_large_currency(capex_diag.normalized_capex, company.quote_currency)}** "
            f"(historický poměr {fmt_percent(ratio_used, 1)} tržeb) → startovní FCF₀ = "
            f"**{fmt_large_currency(capex_diag.normalized_fcf_0, company.quote_currency)}** "
            f"(oproti surovému TTM {fmt_large_currency(capex_diag.raw_fcf_0, company.quote_currency)})."
        )

    for warn in company.warnings:
        st.caption(f"ℹ️ {warn}")


def render_gurufocus_summary_cockpit(
    company: CompanyData,
    verdict: DecisionVerdict,
    suite: ValuationSuiteReport,
    monte_carlo: MonteCarloResult,
    forensics: ForensicsReport,
) -> None:
    """Render the iconic GuruFocus Summary View:
    1. 5-Box 10-Point Scorecards (Financial Strength, Profitability, Growth, GF Value, Forensics)
    2. Verdict Hero + GF Valuation Gauge + GF Value Band Line Chart
    3. All Mathematical Valuation Models Table & Chart + FCF / FCF_adj / Owner Earnings Decomposition
    4. Monte Carlo 10,000-Scenario Crystal Ball (3Y default)
    """
    ccy = company.quote_currency

    # --- 1. GuruFocus 5-Box 10-Point Scorecards ---
    gf_cols = st.columns(5)
    color_hex = {"green": "#22C55E", "amber": "#F59E0B", "red": "#EF4444"}
    for col, p in zip(gf_cols, suite.gf_pillars):
        hex_c = color_hex[p.color]
        with col:
            st.markdown(
                f"""
                <div class="gf-box">
                    <div class="gf-box-top">
                        <span class="gf-box-title">{html.escape(p.title_cz)}</span>
                        <span class="gf-box-score" style="color: {hex_c};">{p.score_10}/10</span>
                    </div>
                    <div class="gf-bar-track">
                        <div class="gf-bar-fill" style="width: {p.score_10 * 10}%; background: {hex_c};"></div>
                    </div>
                    <div class="gf-box-metric">{html.escape(p.key_metric_cz)}</div>
                    <div class="gf-box-detail">{html.escape(p.detail_cz)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # --- 2. Automated Verdict Hero + Pros/Cons ---
    score_color = color_hex[verdict.badge_color]
    st.markdown(
        f"""
        <div class="verdict-hero verdict-hero-{verdict.badge_color}">
            <div style="flex: 1 1 560px;">
                <div class="verdict-kicker">Automatický verdikt (GuruFocus + Věštecká koule · 100 % tvrdá data)</div>
                <div class="verdict-title">{html.escape(verdict.verdict_title_cz)}</div>
                <div class="verdict-sub">{html.escape(verdict.verdict_subtitle_cz)}</div>
                <div class="verdict-action">{html.escape(verdict.action_plan_cz)}</div>
            </div>
            <div class="score-box">
                <div class="score-num" style="color: {score_color};">{verdict.total_score}</div>
                <div class="score-denom">Rozhodovací skóre</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if verdict.veto_triggered:
        st.error(f"🛑 **{verdict.veto_reason_cz}**")

    pc1, pc2 = st.columns(2, gap="medium")
    with pc1:
        pros_li = "".join(f"<li>{html.escape(x)}</li>" for x in verdict.pros_cz)
        st.markdown(
            f"""
            <div class="pros-cons-box" style="border-left: 3px solid #22C55E;">
                <strong style="color: #4ADE80;">🟢 Silné stránky (z tvrdých výkazů)</strong>
                <ul>{pros_li}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pc2:
        cons_li = "".join(f"<li>{html.escape(x)}</li>" for x in verdict.cons_cz)
        st.markdown(
            f"""
            <div class="pros-cons-box" style="border-left: 3px solid #EF4444;">
                <strong style="color: #F87171;">🔴 Rizika a varování (z tvrdých výkazů)</strong>
                <ul>{cons_li}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 3. GuruFocus Valuation Gauge + GF Value Band Chart ---
    _section_header(
        "GuruFocus Ocenění · Všechny matematické modely",
        "Tachometr férové hodnoty, GF Value koridor a kompletní matice oceňovacích vzorců",
    )

    vg_left, vg_right = st.columns([0.92, 1.35], gap="medium")
    with vg_left:
        st.plotly_chart(
            make_gf_valuation_gauge(
                price_to_gf=suite.price_to_gf_value,
                current_price=company.current_price,
                gf_fair_value=suite.gf_fair_value,
                currency=ccy,
                status_label=suite.gf_status_label_cz,
            ),
            use_container_width=True,
        )
        g1, g2, g3 = st.columns(3)
        g1.metric(
            "GF Férová hodnota",
            fmt_price(suite.gf_fair_value, ccy),
            delta=fmt_percent(suite.upside_pct, 1, signed=True) if suite.upside_pct is not None else None,
            help="Čistá vážená vnitřní hodnota z matematických modelů.",
        )
        g2.metric(
            "Cena za 1 rok (odhad)",
            fmt_price(monte_carlo.p50_price_1y, ccy) if monte_carlo.available else "N/A",
            delta=fmt_percent(monte_carlo.expected_return_1y, 1, signed=True)
            if (monte_carlo.available and monte_carlo.expected_return_1y is not None)
            else None,
            help=(
                f"Medián (P50) z 10 000 simulací za 12 měsíců. "
                f"Pásmo P10–P90: {fmt_price(monte_carlo.p10_price_1y, ccy)} – {fmt_price(monte_carlo.p90_price_1y, ccy)}."
                if monte_carlo.available
                else "Nelze spočítat pro záporné historické FCF."
            ),
        )
        g3.metric(
            "Cena / GF Hodnota",
            fmt_multiple(suite.price_to_gf_value, 2),
            delta=suite.gf_status_label_cz,
            delta_color="off",
            help="Hodnota < 1,00x znamená obchodování pod férovou cenou, > 1,00x nad férovou cenou.",
        )
    with vg_right:
        st.plotly_chart(make_gf_value_band_chart(suite.gf_value_history_df, ccy), use_container_width=True)
        if suite.epv_share_of_price is not None and suite.growth_bet_share_of_price is not None:
            epv_s = min(max(suite.epv_share_of_price, 0.0), 2.0)
            bet_s = suite.growth_bet_share_of_price
            if bet_s > 0:
                st.caption(
                    f"📌 **Greenwald EPV (`NOPAT / WACC`):** Bez jakéhokoliv budoucího růstu má firma hodnotu "
                    f"**{fmt_price(suite.epv_per_share, ccy)}** ({fmt_percent(epv_s, 0)} dnešní ceny) – zbývajících "
                    f"**{fmt_percent(bet_s, 0)}** ceny je čistá sázka na budoucí růst."
                )
            else:
                st.caption(
                    f"📌 **Greenwald EPV (`NOPAT / WACC`):** Samotná současná síla bez růstu ({fmt_price(suite.epv_per_share, ccy)}) "
                    "plně pokrývá dnešní cenu akcie – budoucí růst je v ceně zdarma."
                )

    # --- 4. All 8 Per-Share Mathematical Models Side-by-Side ---
    m_left, m_right = st.columns([1.08, 1.15], gap="medium")
    with m_left:
        st.plotly_chart(
            make_all_models_valuation_chart(
                models=suite.models,
                gf_fair_value=suite.gf_fair_value,
                gf_buy_price=suite.gf_buy_price,
                mos=suite.margin_of_safety,
                current_price=company.current_price,
                currency=ccy,
            ),
            use_container_width=True,
        )
    with m_right:
        st.markdown("**Přehled všech matematických modelů férové hodnoty**")
        rows = []
        for m in suite.models:
            rows.append(
                {
                    "Model ocenění": m.name_cz,
                    "Férová hodnota": fmt_price(m.value_per_share, ccy),
                    "vs. Dnes": fmt_percent(m.upside_pct, 1, signed=True),
                    "Vstupy / Poznámka": m.status_note_cz if m.value_per_share is None else m.inputs_summary_cz,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=325)

    # Core vs. Growth Optionality explicit breakdown strip (when Growth Optionality or 2-stage fade is active)
    if suite.optionality_active or suite.used_two_stage_fade or suite.graham_weight_pct == 0.0:
        if suite.optionality_active:
            st.markdown("**Rozpad DCF ocenění: Hodnota jádra (Core) + Hodnota růstové opce (AI / Nové projekty)**")
            oc1, oc2, oc3, oc4 = st.columns(4)
            core_g = suite.core_dcf_gordon_per_share
            tot_g = (core_g + suite.optionality_per_share) if core_g is not None else suite.optionality_per_share
            core_e = suite.core_dcf_exit_per_share
            tot_e = (core_e + suite.optionality_per_share) if core_e is not None else suite.optionality_per_share
            oc1.metric(
                "Hodnota jádra (Core DCF Gordon)",
                fmt_price(core_g, ccy),
                help="Vnitřní hodnota stávajícího byznysu bez započtení budoucích projektů.",
            )
            oc2.metric(
                "Hodnota růstové opce",
                f"+{fmt_price(suite.optionality_per_share, ccy)}",
                help=suite.optionality_note_cz,
            )
            oc3.metric(
                "DCF Gordon celkem (Core + Opce)",
                fmt_price(tot_g, ccy),
                delta=fmt_percent((tot_g / company.current_price) - 1.0, 1, signed=True)
                if (company.current_price and company.current_price > 0)
                else None,
                help="Součet hodnoty jádra a pravděpodobností vážené hodnoty budoucích projektů.",
            )
            oc4.metric(
                "DCF Exit Multiple celkem (Core + Opce)",
                fmt_price(tot_e, ccy),
                delta=f"Core {fmt_price(core_e, ccy)}",
                delta_color="off",
                help="DCF Exit Multiple (Core) + Hodnota růstové opce.",
            )
        notes_active: list[str] = []
        if suite.used_two_stage_fade:
            notes_active.append(
                f"2fázový DCF růst aktivní: roky 1–5 = **{fmt_percent(suite.effective_stage1_growth, 1)}**, "
                "roky 6–10 lineárně klesají na terminální růst g∞."
            )
        if suite.graham_weight_pct == 0.0:
            notes_active.append("Grahamův vzorec má nastavenou váhu **0 %** (neovlivňuje celkovou GF Férovou hodnotu).")
        if notes_active:
            st.caption("⚙️ " + " · ".join(notes_active))

    # Explicit Cash Flow Decomposition Strip (FCF, FCF_adj, Buffett Owner Earnings with Delta WC)
    d = suite.cashflow_decomp
    if d.year is not None:
        st.markdown(f"**Rozklad peněžních toků (FY{d.year}) – FCF, FCF očištěné o SBC a Buffettovy Owner Earnings:**")
        cf1, cf2, cf3, cf4, cf5 = st.columns(5)
        cf1.metric(
            "FCF (OCF − CapEx)",
            fmt_large_currency(d.fcf, ccy),
            help=f"OCF ({fmt_large_currency(d.ocf, ccy)}) − CapEx ({fmt_large_currency(d.capex, ccy)}).",
        )
        cf2.metric(
            "FCF_adj (FCF − SBC)",
            fmt_large_currency(d.fcf_adj, ccy),
            help=f"FCF očištěné o akciové odměny SBC ({fmt_large_currency(d.sbc, ccy)}).",
        )
        cf3.metric(
            "Owner Earnings (Buffett)",
            fmt_large_currency(d.owner_earnings, ccy),
            help="Čistý zisk + Odpisy a amortizace − Udržovací CapEx − ΔPracovní kapitál.",
        )
        cf4.metric(
            "Udržovací CapEx / ΔWC",
            f"{fmt_large_currency(d.maintenance_capex, ccy, 1)}",
            delta=f"ΔWC: {fmt_large_currency(d.delta_working_capital, ccy, 1)}",
            delta_color="off",
            help="Udržovací CapEx (Greenwaldova metoda) a meziroční změna nepeněžního pracovního kapitálu ΔWC.",
        )
        cf5.metric(
            "Owner Earnings / akcii",
            fmt_price(d.owner_earnings_per_share, ccy),
            help="Buffettovy Owner Earnings přepočtené na 1 zředěnou akcii.",
        )

    # --- 5. Monte Carlo Crystal Ball ---
    _section_header(
        f"🔮 Věštecká koule · {monte_carlo.n_simulations:,} scénářů · Horizont {monte_carlo.horizon_years} roky/let".replace(",", "\u00a0"),
        "Pravděpodobnostní projekce budoucí ceny a ročního výnosu (IRR)",
    )
    if not monte_carlo.available:
        st.warning(monte_carlo.error_message_cz or "Monte Carlo simulaci nelze pro tuto firmu spustit.")
    else:
        mc0, mc1, mc2, mc3, mc4, mc5 = st.columns(6)
        mc0.metric(
            "Předpokládaná cena za 1 rok",
            fmt_price(monte_carlo.p50_price_1y, ccy),
            delta=fmt_percent(monte_carlo.expected_return_1y, 1, signed=True)
            if monte_carlo.expected_return_1y is not None
            else None,
            help=(
                f"Očekávaná cena akcie za 1 rok (medián P50 z 10 000 simulací včetně dividendy). "
                f"Pesimistický P10: {fmt_price(monte_carlo.p10_price_1y, ccy)} · Optimistický P90: {fmt_price(monte_carlo.p90_price_1y, ccy)}."
            ),
        )
        med_delta = (
            fmt_percent((monte_carlo.p50_price / monte_carlo.current_price) - 1.0, 0, signed=True)
            if monte_carlo.current_price > 0
            else None
        )
        mc1.metric(
            f"Medián ceny za {monte_carlo.horizon_years} r.",
            fmt_price(monte_carlo.p50_price, ccy),
            delta=med_delta,
            help="50. percentil z 10 000 simulací při mean reversion násobku P/FCF.",
        )
        mc2.metric(
            "Očekávaný výnos (Medián IRR)",
            fmt_percent(monte_carlo.p50_irr, 1, signed=True),
            delta=f"P10: {fmt_percent(monte_carlo.p10_irr, 0)} | P90: {fmt_percent(monte_carlo.p90_irr, 0)}",
            delta_color="off",
        )
        mc3.metric(
            f"Šance porazit {fmt_percent(monte_carlo.hurdle_rate, 0)}",
            fmt_percent(monte_carlo.prob_beat_hurdle, 0),
            delta=f"Kladný výnos: {fmt_percent(monte_carlo.prob_positive_return, 0)}",
            delta_color="off",
        )
        mc4.metric(
            "Riziko ztráty > 20 %",
            fmt_percent(monte_carlo.prob_loss_gt_20pct, 0),
            delta=f"Bear P10: {fmt_price(monte_carlo.p10_price, ccy)}",
            delta_color="off",
        )
        mc5.metric(
            "Asymetrie Upside / Downside",
            f"{fmt_number(monte_carlo.upside_downside_ratio, 1)} : 1"
            if monte_carlo.upside_downside_ratio is not None
            else "N/A",
            delta=f"Bull P90: {fmt_price(monte_carlo.p90_price, ccy)}",
            delta_color="off",
        )

        g_left, g_right = st.columns([1.1, 1.0], gap="medium")
        with g_left:
            st.plotly_chart(
                make_monte_carlo_fan_chart(monte_carlo, ccy, max_buy_price=suite.gf_fair_value),
                use_container_width=True,
            )
        with g_right:
            st.plotly_chart(make_monte_carlo_irr_histogram(monte_carlo), use_container_width=True)

    # --- 6. Forensic Shield Cards (Piotroski, Altman, Beneish, ROIIC) ---
    st.markdown("#### Forenzní štít (kvantitativní detektory mimo Yahoo Finance)")
    f_cols = st.columns(4)
    status_lbl = {"green": "Silné / Čisté", "amber": "Opatrnost", "red": "Varování", "na": "N/A"}
    forensic_cards = [
        ("Piotroski F-Score", forensics.piotroski.status, forensics.piotroski.label_cz, forensics.piotroski.summary_cz),
        ("Altman Z-Score (bankrot)", forensics.altman.status, forensics.altman.zone_cz, forensics.altman.summary_cz),
        ("Beneish M-Score (manipulace)", forensics.beneish.status, forensics.beneish.label_cz, forensics.beneish.summary_cz),
        (
            "Motor růstu (ROIIC × reinv.)",
            forensics.compounding.status,
            fmt_percent(forensics.compounding.intrinsic_compounding_rate, 1) + " / rok"
            if forensics.compounding.intrinsic_compounding_rate is not None
            else "N/A",
            forensics.compounding.summary_cz,
        ),
    ]
    for col, (title, st_lvl, metric_txt, desc_txt) in zip(f_cols, forensic_cards):
        with col:
            badge = _status_badge_html(st_lvl, status_lbl[st_lvl])
            st.markdown(
                f"""
                <div class="flag-card flag-{st_lvl}">
                    <div class="flag-header">
                        <span class="flag-title">{html.escape(title)}</span>
                        {badge}
                    </div>
                    <div class="flag-metric">{html.escape(metric_txt)}</div>
                    <div class="flag-desc">{html.escape(desc_txt)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_legendary_investors_section(company: CompanyData, legends: LegendsReport) -> None:
    """Render GuruFocus-style evaluation through the checklists of 5 legendary investors + Top Institutional Holders."""
    _section_header(
        "🏛️ Hodnocení největších investorů",
        "Jak by tuto akcii ohodnotily legendy hodnotového investování (čistě podle jejich tvrdých pravidel)",
    )

    cons_badge = _status_badge_html(legends.consensus_color, f"{legends.consensus_label_cz} · průměr {legends.avg_score_pct} %")
    st.markdown(f"**Konsenzus investičních legend:** &nbsp; {cons_badge}", unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(5)
    for col, inv in zip(cols, legends.investors):
        with col:
            badge = _status_badge_html(inv.color, inv.verdict_cz)
            st.markdown(
                f"""
                <div class="flag-card flag-{inv.color}">
                    <div class="flag-header">
                        <span class="flag-title">{html.escape(inv.investor_name)}</span>
                        {badge}
                    </div>
                    <div class="flag-metric">{inv.passed_count}/{inv.evaluated_count} ({inv.score_pct}\u00a0%)</div>
                    <div class="flag-desc"><em>{html.escape(inv.style_cz)}</em><br>{html.escape(inv.summary_cz)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Zobrazit detailní splněná / nesplněná kritéria jednotlivých legend", expanded=False):
        tabs = st.tabs([inv.investor_name for inv in legends.investors])
        for tab, inv in zip(tabs, legends.investors):
            with tab:
                rows = [
                    {
                        "Kritérium": c.label_cz,
                        "Stav": "✅ Splněno" if c.passed is True else ("❌ Nesplněno" if c.passed is False else "– N/A"),
                        "Naměřená hodnota": c.observed_cz,
                    }
                    for c in inv.criteria
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if company.institutional_holders:
        with st.expander(f"Největší institucionální akcionáři ({len(company.institutional_holders)} hlavních držitelů)", expanded=False):
            h_rows = [
                {
                    "Instituce / Fond": h.holder_name,
                    "Podíl": fmt_percent(h.pct_held, 2),
                    "Počet akcií": fmt_shares(h.shares),
                    "Hodnota pozice": fmt_large_currency(h.value, company.quote_currency),
                    "Datum hlášení": h.date_reported or "N/A",
                }
                for h in company.institutional_holders
            ]
            st.dataframe(pd.DataFrame(h_rows), hide_index=True, use_container_width=True)


def render_qualitative_section(
    company: CompanyData,
    verdict: DecisionVerdict,
    quality: QualityXRayReport,
    forensics: ForensicsReport,
    insiders: InsiderActivityReport,
    suite: ValuationSuiteReport,
) -> QualitativeEvaluation:
    """Render the interactive 6-pillar Non-Mathematical Criteria scorecard (auto-prefilled from hard data)."""
    _section_header(
        "🧭 Nematematická kritéria · Kvalitativní analýza",
        "6 klíčových kvalitativních pilířů (předvyplněno z tvrdých dat, volně upravitelné)",
    )

    defaults = infer_qualitative_defaults(quality, forensics, insiders, suite)
    tk_key = company.ticker

    q_left, q_right = st.columns([1.45, 0.95], gap="large")

    with q_left:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**1. Ekonomický příkop (Moat)**")
            st.caption(defaults.moat_hint_cz)
            moat_rating = st.slider(
                "Síla ekonomického příkopu (1–5)",
                1,
                5,
                value=defaults.moat_score,
                key=f"q_moat_{tk_key}",
            )
            moat_sources = st.multiselect(
                "Identifikované zdroje příkopu",
                options=list(MOAT_SOURCES_CZ),
                default=[],
                key=f"q_moat_src_{tk_key}",
                placeholder="Označte platné zdroje konkurenční výhody...",
            )

            st.markdown("**2. Cenotvorba (Pricing Power)**")
            st.caption(defaults.pricing_hint_cz)
            pricing_rating = st.slider(
                "Schopnost zvyšovat ceny bez ztráty zákazníků (1–5)",
                1,
                5,
                value=defaults.pricing_power_score,
                key=f"q_pricing_{tk_key}",
            )

            st.markdown("**3. Management a alokace kapitálu**")
            st.caption(defaults.management_hint_cz)
            mgmt_rating = st.slider(
                "Skin in the game, reinvestice, akvizice, buybacks, dividendy (1–5)",
                1,
                5,
                value=defaults.management_score,
                key=f"q_mgmt_{tk_key}",
            )

        with c2:
            st.markdown("**4. Předvídatelnost byznysu**")
            st.caption(defaults.predictability_hint_cz)
            pred_rating = st.slider(
                "Stabilita poptávky a odolnost vůči technologické změně (1–5)",
                1,
                5,
                value=defaults.predictability_score,
                key=f"q_pred_{tk_key}",
            )

            st.markdown("**5. Struktura odvětví (Porterových 5 sil)**")
            st.caption(defaults.porter_hint_cz)
            porter_ratings: dict[str, int] = {}
            with st.expander("Upravit jednotlivých 5 Porterových sil (1 = nepříznivé, 5 = příznivé)", expanded=False):
                for p_key, p_lbl in PORTER_FORCES_CZ:
                    porter_ratings[p_key] = st.slider(
                        p_lbl,
                        1,
                        5,
                        value=defaults.porter_default_score,
                        key=f"q_porter_{p_key}_{tk_key}",
                    )
            if not porter_ratings:
                porter_ratings = {k: defaults.porter_default_score for k, _ in PORTER_FORCES_CZ}

            st.markdown("**6. Rizika mimo výkazy**")
            selected_risks = st.multiselect(
                "Označte přítomná mimobilanční rizika",
                options=list(OFF_STATEMENT_RISKS_CZ),
                default=[],
                key=f"q_risks_{tk_key}",
                placeholder="Žádná zásadní rizika neoznačena...",
            )

    qual_eval = evaluate_qualitative_scorecard(
        moat_rating=moat_rating,
        selected_moat_sources=moat_sources,
        pricing_power_rating=pricing_rating,
        management_rating=mgmt_rating,
        predictability_rating=pred_rating,
        porter_ratings=porter_ratings,
        selected_risks=selected_risks,
    )

    with q_right:
        combined_score = int(round(0.70 * verdict.total_score + 0.30 * qual_eval.total_score))
        comb_color: TrafficLight = "green" if combined_score >= 65 else ("amber" if combined_score >= 46 else "red")
        st.markdown(
            f"""
            <div class="gf-box" style="margin-bottom: 0.7rem;">
                <div class="gf-box-top">
                    <span class="gf-box-title">Kvalitativní skóre</span>
                    {_status_badge_html(qual_eval.color, f"{qual_eval.total_score}/100")}
                </div>
                <div class="gf-box-metric">{html.escape(qual_eval.grade_cz)}</div>
                <div class="gf-box-detail">{html.escape(qual_eval.summary_cz)}</div>
                <hr style="border-color: #1E293B; margin: 0.55rem 0;">
                <div class="gf-box-top">
                    <span class="gf-box-title">Kombinované skóre (70 % Data + 30 % Kvalita)</span>
                    {_status_badge_html(comb_color, f"{combined_score}/100")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_qualitative_radar_chart(qual_eval.pillar_scores), use_container_width=True)

    return qual_eval


def render_deep_dive_expanders(
    company: CompanyData,
    dcf: ReverseDCFResult,
    dcf_params: ReverseDCFInputs,
    suite: ValuationSuiteReport,
    forensics: ForensicsReport,
    red_flags: RedFlagsReport,
    quality: QualityXRayReport,
    val_bands: list[ValuationBand],
    peer_df: pd.DataFrame,
    peer_tickers: list[str],
    alloc: CapitalAllocationReport,
    insiders: InsiderActivityReport,
) -> None:
    """Collapsible deep-dive sections: exact LaTeX formulas, Reverse DCF heatmaps, Forensics, ROIC, Peers, Insiders."""
    _section_header(
        "Hluboký rozbor a matematické vzorce",
        "Kompletní vzorce s dosazenými čísly, Reverse DCF matice, Forenzní detaily a 10letý rentgen",
    )

    # ------------------------------------------------------------------
    # Expander 1: Exact Mathematical Formulas (LaTeX) + Reverse DCF Heatmaps
    # ------------------------------------------------------------------
    with st.expander("📐 Detail 1: Všechny matematické vzorce (LaTeX) & Reverse DCF + Matice citlivosti", expanded=False):
        st.markdown("#### Použité matematické vzorce a konkrétní vstupy pro tuto společnost")
        f_col1, f_col2 = st.columns(2, gap="medium")
        with f_col1:
            st.markdown("**Volné peněžní toky (FCF) a FCF očištěné o SBC:**")
            st.latex(r"FCF = OCF - CapEx \qquad FCF_{\text{adj}} = OCF - CapEx - SBC")
            st.markdown("**DCF – Hodnota podniku (EV) a Vnitřní hodnota 1 akcie:**")
            st.latex(r"V_{\text{firmy}} = \sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^n}, \quad V_{\text{akcie}} = \frac{V_{\text{firmy}} + \text{Hotovost} - \text{Celkový dluh}}{\text{Počet zředěných akcií}}")
            st.markdown("**Terminální hodnota – Gordonův model vs. Exit Multiple:**")
            st.latex(r"TV_{\text{Gordon}} = \frac{FCF_n \cdot (1+g)}{r - g} \qquad TV_{\text{Exit}} = FCF_n \cdot \text{Násobek}")
            st.markdown("**Návratnost investovaného kapitálu (ROIC):**")
            st.latex(r"ROIC = \frac{NOPAT}{\text{Investovaný kapitál}}")
        with f_col2:
            st.markdown("**DDM – Dividendový diskontní model & Grahamův vzorec:**")
            st.latex(r"V_{\text{DDM}} = \frac{D_1}{r - g} \qquad V_{\text{Graham}} = \frac{EPS \cdot (8{,}5 + 2g) \cdot 4{,}4}{Y}")
            st.markdown("**EPV – Earnings Power Value (Bruce Greenwald):**")
            st.latex(r"EPV = \frac{NOPAT}{WACC}")
            st.markdown("**Buffettovy Owner Earnings:**")
            st.latex(r"\text{Owner Earnings} = \text{Čistý zisk} + \text{Odpisy a amortizace} - \text{Udržovací CapEx} - \Delta\text{Pracovní kapitál}")
            st.markdown("**Relativní ocenění (P/E a P/FCF):**")
            st.latex(r"V_{\text{P/E}} = EPS_{\text{očekávané}} \cdot \text{Férové } P/E \qquad V_{\text{P/FCF}} = \text{FCF na akcii} \cdot \text{Férové } P/FCF")

        st.divider()
        st.markdown("#### Reverse DCF – Co už dnes trh oceňuje v ceně")
        left_col, right_col = st.columns([1.4, 1.0], gap="large")
        with left_col:
            st.markdown(
                f"""
                <div class="dcf-headline-box">
                    <div class="dcf-headline-kicker">Implikovaná očekávání trhu ({html.escape(dcf.fcf_0_label)})</div>
                    <div class="dcf-headline-text">{html.escape(dcf.headline_cz)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if dcf.error_message:
                st.error(dcf.error_message)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Výchozí FCF₀", fmt_large_currency(dcf.fcf_0, company.quote_currency))
            m2.metric("FCF₀ / akcii", fmt_price(dcf.fcf_0_per_share, company.quote_currency))
            m3.metric("Čistý dluh", fmt_large_currency(dcf.net_debt, company.quote_currency))
            m4.metric("Podíl TV na EV", fmt_percent(dcf.tv_share_of_ev, 1))
        with right_col:
            v_style: dict[str, TrafficLight] = {"realistic": "green", "ambitious": "amber", "extreme": "red", "unavailable": "na"}
            st.markdown(
                f"**Test reality (Reverse DCF)** &nbsp; {_status_badge_html(v_style[dcf.verdict], dcf.verdict_label_cz)}",
                unsafe_allow_html=True,
            )
            st.caption(dcf.verdict_explanation_cz)
            rc_rows = [
                ("Implikovaný růst FCF (Reverse DCF)", fmt_percent(dcf.implied_growth, 1)),
                ("Historický růst tržeb 3Y / 5Y / 10Y", f"{fmt_percent(dcf.cagrs.revenue_cagr_3y, 1)} / {fmt_percent(dcf.cagrs.revenue_cagr_5y, 1)} / {fmt_percent(dcf.cagrs.revenue_cagr_10y, 1)}"),
                ("Historický růst FCF 3Y / 5Y / 10Y", f"{fmt_percent(dcf.cagrs.fcf_cagr_3y, 1)} / {fmt_percent(dcf.cagrs.fcf_cagr_5y, 1)} / {fmt_percent(dcf.cagrs.fcf_cagr_10y, 1)}"),
                ("Motor složeného úročení (ROIIC × reinv.)", fmt_percent(forensics.compounding.intrinsic_compounding_rate, 1)),
            ]
            st.dataframe(pd.DataFrame(rc_rows, columns=["Ukazatel", "Roční tempo (p.a.)"]), hide_index=True, use_container_width=True)

        if dcf.fcf_0 and dcf.fcf_0 > 0 and dcf.diluted_shares and dcf.diluted_shares > 0 and dcf.current_price and dcf.current_price > 0:
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                hm_mode = st.radio(
                    "Zobrazená matice citlivosti",
                    options=["Vnitřní hodnota na akcii", "Implikovaný růst (g) dle ceny a diskontní sazby"],
                    horizontal=True,
                )
            with c2:
                x_mode: Literal["discount_rate", "exit_multiple"] = "discount_rate"
                if hm_mode == "Vnitřní hodnota na akcii":
                    xc = st.radio("Osa X", options=["Diskontní sazba (r)", "Výstupní násobek (Exit Multiple)"], horizontal=True)
                    x_mode = "discount_rate" if xc.startswith("Diskontní") else "exit_multiple"
            if hm_mode == "Vnitřní hodnota na akcii":
                g_v, x_v, mat = build_intrinsic_value_grid(dcf.fcf_0, dcf.net_debt, dcf.diluted_shares, dcf_params, dcf.implied_growth, x_mode)
                st.plotly_chart(make_intrinsic_value_heatmap(g_v, x_v, mat, dcf.current_price, company.quote_currency, x_mode), use_container_width=True)
            else:
                p_v, r_v, mat_g = build_implied_growth_grid(dcf.current_price, dcf.fcf_0, dcf.net_debt, dcf.diluted_shares, dcf_params)
                st.plotly_chart(make_implied_growth_heatmap(p_v, r_v, mat_g, dcf.current_price, company.quote_currency), use_container_width=True)

    # ------------------------------------------------------------------
    # Expander 2: Forensics (F / Z / M-Score) & Accounting Red Flags
    # ------------------------------------------------------------------
    with st.expander(
        f"🧪 Detail 2: Forenzní modely (F-Score, Z-Score, M-Score) & Účetní Red Flags ({red_flags.overall_label_cz})",
        expanded=False,
    ):
        ft1, ft2, ft3, ft4, ft5 = st.tabs(
            ["Piotroski / Altman / Beneish", "Skutečné FCF (FCF − SBC)", "Počet akcií a odkupy", "Kvalita zisků a akruály", "Dluhová zeď (+2 p.b. stress)"]
        )
        with ft1:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f"**Piotroski F-Score: {forensics.piotroski.label_cz}**")
                if forensics.piotroski.criteria:
                    st.dataframe(
                        pd.DataFrame([{"Kritérium": c.name_cz, "Stav": "✅" if c.passed is True else ("❌" if c.passed is False else "–"), "Hodnota": c.detail_cz} for c in forensics.piotroski.criteria]),
                        hide_index=True,
                        use_container_width=True,
                    )
            with fc2:
                st.markdown(f"**Altman Z-Score: {forensics.altman.zone_cz}**")
                if forensics.altman.components:
                    st.dataframe(
                        pd.DataFrame([{"Složka": k, "Hodnota": fmt_number(v, 3)} for k, v in forensics.altman.components.items()]),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.caption("Prahy: > 2,99 bezpečná · 1,81–2,99 šedá · < 1,81 tíseň.")
            with fc3:
                st.markdown(f"**Beneish M-Score: {forensics.beneish.label_cz}**")
                if forensics.beneish.ratios:
                    st.dataframe(
                        pd.DataFrame([{"Index": k, "Hodnota": fmt_number(v, 3)} for k, v in forensics.beneish.ratios.items()]),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.caption("Práh: M > −1,78 značí zvýšené riziko manipulace zisků.")
        with ft2:
            st.plotly_chart(make_real_fcf_chart(red_flags.real_fcf_df, company.quote_currency), use_container_width=True)
            if not red_flags.real_fcf_df.empty:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Rok": red_flags.real_fcf_df["year"],
                            "Vykázané FCF": red_flags.real_fcf_df["fcf"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "SBC": red_flags.real_fcf_df["sbc"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "FCF_adj": red_flags.real_fcf_df["real_fcf"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "FCF_adj (% tržeb)": red_flags.real_fcf_df["real_fcf_margin"].map(lambda v: fmt_percent(v, 1)),
                            "FCF_adj / akcii": red_flags.real_fcf_df["real_fcf_per_share"].map(lambda v: fmt_price(v, company.quote_currency)),
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        with ft3:
            st.info(f"**Hodnocení zpětných odkupů:** {red_flags.buyback_verdict_cz}")
            st.plotly_chart(make_shares_chart(red_flags.shares_df), use_container_width=True)
            if not red_flags.shares_df.empty:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Rok": red_flags.shares_df["year"],
                            "Zředěné akcie": red_flags.shares_df["diluted_shares"].map(fmt_shares),
                            "Změna r/r": red_flags.shares_df["net_change_pct"].map(lambda v: fmt_percent(v, 1, signed=True)),
                            "Výdaje na odkupy": red_flags.shares_df["buyback_spend"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "Odhad. cena odkupu": red_flags.shares_df["est_buyback_price"].map(lambda v: fmt_price(v, company.quote_currency)),
                            "P/E": red_flags.shares_df["pe_ratio"].map(fmt_multiple),
                            "P/FCF": red_flags.shares_df["pfcf_ratio"].map(fmt_multiple),
                            "Posouzení": red_flags.shares_df["buyback_quality"],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        with ft4:
            st.plotly_chart(make_earnings_quality_chart(red_flags.earnings_quality_df), use_container_width=True)
            if not red_flags.earnings_quality_df.empty:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Rok": red_flags.earnings_quality_df["year"],
                            "OCF": red_flags.earnings_quality_df["ocf"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "Čistý zisk": red_flags.earnings_quality_df["net_income"].map(lambda v: fmt_large_currency(v, company.quote_currency)),
                            "OCF / NI": red_flags.earnings_quality_df["ocf_to_ni"].map(lambda v: f"{fmt_number(v, 2)}x" if pd.notna(v) else "N/A"),
                            "Akruály": red_flags.earnings_quality_df["accruals_ratio"].map(lambda v: fmt_percent(v, 1, signed=True)),
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        with ft5:
            ds = red_flags.debt_stress
            dl, dr = st.columns([1.0, 1.15])
            with dl:
                k1, k2 = st.columns(2)
                k1.metric("Čistý dluh / EBITDA", fmt_multiple(ds.net_debt_to_ebitda))
                k2.metric("Úrokové krytí", fmt_multiple(ds.interest_coverage))
                k3, k4 = st.columns(2)
                k3.metric("Stress test (+2 p.b.)", fmt_multiple(ds.stressed_interest_coverage))
                k4.metric("Efektivní úrok", fmt_percent(ds.effective_interest_rate, 1))
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Krátkodobý dluh (< 1 r.)", fmt_large_currency(ds.short_term_debt, company.quote_currency), fmt_percent(ds.short_term_share, 1)),
                            ("Dlouhodobý dluh (> 1 r.)", fmt_large_currency(ds.long_term_debt, company.quote_currency), fmt_percent(1.0 - ds.short_term_share, 1) if ds.short_term_share is not None else "N/A"),
                            ("Celkový dluh", fmt_large_currency(ds.total_debt, company.quote_currency), "100,0 %"),
                        ],
                        columns=["Složka dluhu", "Objem", "Podíl"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            with dr:
                st.plotly_chart(make_debt_stress_chart(ds.interest_coverage, ds.stressed_interest_coverage), use_container_width=True)

    # ------------------------------------------------------------------
    # Expander 3: 10-Year Quality X-Ray (ROIC = NOPAT / IC vs WACC + Margins)
    # ------------------------------------------------------------------
    with st.expander("💎 Detail 3: 10letý rentgen kvality (ROIC = NOPAT / Investovaný kapitál vs. WACC & Marže)", expanded=False):
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Průměrné ROIC", fmt_percent(quality.avg_roic, 1))
        q2.metric("WACC (CAPM)", fmt_percent(quality.wacc_components.wacc, 1))
        q3.metric("Spread ROIC − WACC", fmt_pp(quality.avg_spread, 1))
        q4.metric("Přírůstkové ROIIC", fmt_percent(forensics.compounding.roiic, 1))
        q5.metric("Trend hrubé marže", quality.gross_margin_trend_label_cz.split(" (")[0])
        c_l, c_r = st.columns(2, gap="medium")
        with c_l:
            st.plotly_chart(make_roic_wacc_chart(quality.roic_df, quality.wacc_components.wacc), use_container_width=True)
        with c_r:
            st.plotly_chart(make_margins_chart(quality.margins_df), use_container_width=True)
        st.info(f"**{quality.moat_verdict_title_cz}:** {quality.moat_verdict_cz}")
        wc = quality.wacc_components
        st.caption(
            f"CAPM WACC = {fmt_percent(wc.wacc, 2)} | Rf = {fmt_percent(wc.risk_free_rate, 2)} ({wc.risk_free_source}), "
            f"Beta = {fmt_number(wc.beta, 2)}, ERP = {fmt_percent(wc.equity_risk_premium, 1)}, "
            f"Kd = {fmt_percent(wc.cost_of_debt_pre_tax, 2)}, efektivní daň = {fmt_percent(wc.effective_tax_rate, 1)}."
        )

    # ------------------------------------------------------------------
    # Expander 4: Historical Valuation Bands & Peers
    # ------------------------------------------------------------------
    with st.expander("⚖️ Detail 4: Historická pásma ocenění (P/E, P/FCF, EV/EBITDA) a Srovnání s konkurencí", expanded=False):
        v_l, v_r = st.columns([1.05, 1.25], gap="large")
        with v_l:
            st.plotly_chart(make_valuation_bands_chart(val_bands), use_container_width=True)
            b_cols = st.columns(len(val_bands))
            for col, b in zip(b_cols, val_bands):
                col.metric(f"{b.label_cz} dnes", fmt_multiple(b.current), delta=f"{fmt_number(b.percentile, 0)}. percentil" if b.percentile is not None else "N/A", delta_color="off")
        with v_r:
            st.caption(f"Porovnávaní konkurenti: {', '.join(peer_tickers) if peer_tickers else 'žádní (upravte v postranním panelu)'}.")
            if not peer_df.empty:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ticker": peer_df["ticker"],
                            "Společnost": peer_df["name"].str.slice(0, 24),
                            "Tržní kap.": peer_df["market_cap"].map(lambda v: fmt_large_currency(v, company.quote_currency, 1)),
                            "P/E": peer_df["pe_ratio"].map(fmt_multiple),
                            "P/FCF": peer_df["pfcf_ratio"].map(fmt_multiple),
                            "EV/EBITDA": peer_df["ev_to_ebitda"].map(fmt_multiple),
                            "ROIC": peer_df["roic"].map(lambda v: fmt_percent(v, 1)),
                            "Hrubá m.": peer_df["gross_margin"].map(lambda v: fmt_percent(v, 1)),
                            "Provoz. m.": peer_df["operating_margin"].map(lambda v: fmt_percent(v, 1)),
                            "Růst 3Y": peer_df["revenue_cagr_3y"].map(lambda v: fmt_percent(v, 1)),
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    # ------------------------------------------------------------------
    # Expander 5: Capital Allocation & Open-Market Insiders
    # ------------------------------------------------------------------
    with st.expander("🏦 Detail 5: Alokace kapitálu (10 let) & Insider transakce (pouze otevřený trh)", expanded=False):
        al, ar = st.columns([1.25, 1.0], gap="medium")
        with al:
            st.plotly_chart(make_capital_allocation_annual_chart(alloc.annual_df), use_container_width=True)
        with ar:
            st.plotly_chart(make_capital_allocation_cumulative_chart(alloc.cumulative_shares_pct, alloc.cumulative_totals, company.quote_currency, alloc.years_count), use_container_width=True)
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Nákupy (open market)", fmt_large_currency(insiders.total_buy_value, company.quote_currency), delta=f"{insiders.buy_count}×", delta_color="off")
        i2.metric("Prodeje (open market)", fmt_large_currency(insiders.total_sell_value, company.quote_currency), delta=f"{insiders.sell_count}×", delta_color="off")
        i3.metric("Čistá bilance", fmt_large_currency(insiders.net_value, company.quote_currency))
        i4.metric("Odfiltrovaný šum", f"{insiders.excluded_count} zázn.")
        if insiders.cluster_buying_detected:
            st.success(f"🟢 {insiders.cluster_buying_detail_cz}")
        if insiders.consistent_monthly_selling:
            st.warning(f"🟡 {insiders.consistent_selling_detail_cz}")
        for note in insiders.notes_cz:
            st.caption(f"ℹ️ {note}")
        if insiders.filtered_trades:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Datum": t.date,
                            "Jméno": t.insider_name,
                            "Role": t.role or "Vedení / Insider",
                            "Typ": "🟢 Nákup" if t.direction == "BUY" else "🔴 Prodej",
                            "Akcie": fmt_number(t.shares, 0),
                            "Cena": fmt_price(t.price, company.quote_currency),
                            "Hodnota": fmt_large_currency(t.value, company.quote_currency),
                        }
                        for t in insiders.filtered_trades[:25]
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_disclaimer() -> None:
    """Render mandatory Czech analytical disclaimer at the bottom of the dashboard."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Upozornění (Disclaimer):</strong> Tato aplikace je kvantitativní a analytický nástroj inspirovaný
            metodikou hodnotového investování (DCF, DDM, Graham, EPV, Owner Earnings, Monte Carlo, F/Z/M-Score).
            Výsledná skóre, férové ceny ani verdikty nepředstavují investiční doporučení a nezaručují budoucí výnosy.
            Vždy ověřujte data přímo v oficiálních výkazech společnosti (10-K / výroční zpráva).
        </div>
        """,
        unsafe_allow_html=True,
    )
