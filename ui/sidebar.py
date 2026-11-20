"""Sidebar parameter controls for GuruFocus Valuation Suite, Monte Carlo Crystal Ball, and Reverse DCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from analysis.dcf import FcfBaseType, ReverseDCFInputs, TerminalMethod
from data.provider import load_fmp_api_key


@dataclass
class SidebarConfig:
    """User-selected quantitative parameters from the sidebar."""

    crystal_ball_horizon: int  # 3 (default), 5, or 10 years
    margin_of_safety_target: float  # e.g., 0.0 (pure fair value)
    dcf_inputs: ReverseDCFInputs
    equity_risk_premium: float
    aaa_yield_override_pct: Optional[float]
    graham_weight_pct: float
    custom_peers_raw: str


def render_sidebar(
    default_peers: Optional[list[str]] = None,
    default_stage1_growth_pct: Optional[float] = None,
    ticker_key: str = "",
) -> SidebarConfig:
    """Render sidebar controls in Czech for all valuation models and Monte Carlo simulation."""
    with st.sidebar:
        st.markdown("### Parametry ocenění")
        st.caption("Všechny vzorce a 10 000 scénářů se přepočítávají živě na tvrdých datech.")

        st.markdown("#### 1. Horizont & Návratnost")
        horizon_choice = st.radio(
            "Horizont věštecké koule (Monte Carlo)",
            options=["3 roky (Mean Reversion)", "5 let (Střednědobý)", "10 let (Compounder)"],
            index=0,
            help=(
                "Cílový horizont pro 10 000 simulací budoucí ceny a IRR. "
                "3 roky cílí na návrat násobků k historickému mediánu (mean reversion)."
            ),
        )
        cb_horizon = 3 if horizon_choice.startswith("3") else (5 if horizon_choice.startswith("5") else 10)

        discount_pct = st.slider(
            "Požadovaná návratnost r (%)",
            min_value=6.0,
            max_value=25.0,
            value=12.0,
            step=0.5,
            help="Diskontní sazba r pro DCF, DDM, Reverse DCF i cílový výnos (hurdle rate) pro Monte Carlo.",
        )

        fcf_base_label = st.radio(
            "Základna peněžního toku (FCF)",
            options=["FCF_adj = OCF − CapEx − SBC", "Vykázané FCF = OCF − CapEx"],
            index=0,
            help="Výchozí nastavení odečítá akciové odměny (SBC), které ředí akcionáře.",
        )
        fcf_base_type: FcfBaseType = (
            "fcf_minus_sbc" if fcf_base_label.startswith("FCF_adj") else "reported_fcf"
        )
        normalize_capex = st.checkbox(
            "Normalizovat mimořádný CapEx (dle historického průměru CapEx/Tržby)",
            value=False,
            help=(
                "Místo surového aktuálního TTM CapExu vypočte normalizovaný CapEx jako "
                "TTM Tržby × historický poměr CapEx/Tržby za posledních 4–5 let. "
                "Vhodné u firem s extrémními jednorázovými investicemi (např. AI datacentra)."
            ),
        )

        with st.expander("Růstová opce & Pokročilý scénář (AI / Nové byznysy)", expanded=False):
            model_g_default = round(
                min(max(default_stage1_growth_pct if default_stage1_growth_pct is not None else 10.0, 0.0), 60.0) * 2.0
            ) / 2.0
            stage1_g_pct = st.slider(
                "Růst FCF v 1. fázi (roky 1–5) v %",
                min_value=0.0,
                max_value=60.0,
                value=float(model_g_default),
                step=0.5,
                key=f"stage1_g_{ticker_key or 'default'}",
                help=(
                    "Nastavte tempo růstu FCF pro roky 1–5 (výchozí dle historického modelu). "
                    "V letech 6–10 model růst lineárně sníží až na terminální růst g∞."
                ),
            )
            use_custom_stage1 = abs(stage1_g_pct - float(model_g_default)) > 1e-6

            enable_opt = st.checkbox(
                "Započítat hodnotu budoucích projektů (Growth Optionality)",
                value=False,
                help=(
                    "Přičte k základní hodnotě jádra (Core DCF) pravděpodobností váženou současnou "
                    "hodnotu budoucích projektů (AI, autonomní řízení, nové produkty), které ještě nejsou v TTM FCF."
                ),
            )
            opt_fcf_b = 0.0
            opt_ramp_yrs = 5
            opt_prob_pct = 50
            if enable_opt:
                opt_fcf_b = st.number_input(
                    "Očekávané roční FCF z nových projektů (mld. USD)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f",
                    help="Odhadované roční FCF projektu v době plného rozběhu (v miliardách USD).",
                )
                opt_ramp_yrs = st.slider(
                    "Za kolik let se projekt naplno rozběhne (roky)",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Rok, ke kterému je vypočtena terminální hodnota projektu a diskontována zpět do dneška.",
                )
                opt_prob_pct = st.slider(
                    "Pravděpodobnost úspěchu projektu (%)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Váha pravděpodobnosti realizace projektu (např. 50 % = poloviční váha v ocenění).",
                )

            graham_weight_pct = st.slider(
                "Váha Grahamova vzorce v celkovém ocenění (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=5,
                help=(
                    "0 % = Grahamův vzorec je vypnut z váženého průměru GF Férové hodnoty "
                    "(doporučeno pro rychle rostoucí technologické firmy jako TSLA, NVDA, PLTR). "
                    "Výchozí váha: 10 %."
                ),
            )

        with st.expander("Pokročilé parametry (DCF, Graham Y, WACC)"):
            custom_fcf_0_millions = st.number_input(
                "Vlastní startovní FCF v mil. USD (0 = použít vypočtené FCF)",
                value=0.0,
                step=100.0,
                format="%.1f",
                help=(
                    "Pokud zadáte hodnotu různou od 0 (v milionech měny výkazů), použije se "
                    "přímo jako FCF₀ pro DCF i Reverse DCF projekci místo vypočteného TTM FCF."
                ),
            )
            dcf_horizon = st.slider(
                "Horizont DCF / Reverse DCF (roky n)",
                min_value=5,
                max_value=15,
                value=10,
                step=1,
            )
            term_label = st.radio(
                "Terminální hodnota v Reverse DCF",
                options=["Gordonův trvalý růst", "Exit Multiple (× FCF)"],
                index=0,
            )
            terminal_method: TerminalMethod = (
                "perpetual_growth" if term_label.startswith("Gordon") else "exit_multiple"
            )
            perpetual_g_pct = st.slider(
                "Trvalý růst g∞ v Gordonově modelu (%)",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                step=0.25,
            )
            exit_mult = st.slider(
                "Exit Multiple (násobek FCF_n)",
                min_value=6.0,
                max_value=35.0,
                value=15.0,
                step=1.0,
            )
            erp_pct = st.slider(
                "Riziková prémie akcií – ERP (%)",
                min_value=3.0,
                max_value=8.0,
                value=5.0,
                step=0.25,
                help="CAPM: Ke = Rf + Beta × ERP (ovlivňuje WACC, EPV a ROIC − WACC).",
            )
            use_custom_y = st.checkbox(
                "Ručně nastavit výnos AAA dluhopisů Y (Graham)",
                value=False,
                help="Výchozí Y = 10Y US Treasury + 1,0 p.b. korporátní přirážka.",
            )
            aaa_y_override: Optional[float] = None
            if use_custom_y:
                aaa_y_override = st.slider(
                    "Výnos AAA korporátních dluhopisů Y (%)",
                    min_value=2.5,
                    max_value=9.0,
                    value=5.3,
                    step=0.1,
                )

        st.divider()
        st.markdown("#### 2. Konkurenti (Peers)")
        placeholder_peers = ", ".join(default_peers) if default_peers else "např. MSFT, GOOGL, META"
        custom_peers_raw = st.text_input(
            "Vlastní tickery konkurence",
            value="",
            placeholder=placeholder_peers,
            help="Volitelně zadejte 1–5 vlastních tickerů oddělených čárkou.",
        )

        st.divider()
        fmp_key = load_fmp_api_key()
        if fmp_key:
            st.caption("Zdroj dat: **FMP API (10+ let)** + yfinance")
        else:
            st.caption("Zdroj dat: **yfinance + SEC EDGAR** (10+ let US výkazů zdarma)")

        if st.button("Vymazat mezipaměť (cache)", use_container_width=True):
            st.cache_data.clear()
            st.toast("Mezipaměť dat (6 h TTL) byla vymazána.")

    dcf_inputs = ReverseDCFInputs(
        discount_rate=discount_pct / 100.0,
        horizon_years=int(dcf_horizon),
        terminal_method=terminal_method,
        perpetual_growth_rate=perpetual_g_pct / 100.0,
        exit_multiple=float(exit_mult),
        fcf_base_type=fcf_base_type,
        normalize_capex=normalize_capex,
        custom_fcf_0_millions=float(custom_fcf_0_millions),
        stage1_growth_rate=(stage1_g_pct / 100.0) if use_custom_stage1 else None,
        enable_growth_optionality=enable_opt,
        optionality_fcf_billions=float(opt_fcf_b),
        optionality_ramp_years=int(opt_ramp_yrs),
        optionality_probability=opt_prob_pct / 100.0,
    )
    return SidebarConfig(
        crystal_ball_horizon=cb_horizon,
        margin_of_safety_target=0.0,
        dcf_inputs=dcf_inputs,
        equity_risk_premium=erp_pct / 100.0,
        aaa_yield_override_pct=aaa_y_override,
        graham_weight_pct=float(graham_weight_pct),
        custom_peers_raw=custom_peers_raw,
    )
