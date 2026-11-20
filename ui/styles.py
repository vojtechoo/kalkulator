"""GuruFocus-inspired dark-theme CSS: 5-box scorecards, valuation badges, and crisp dense layout."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1460px;
}

/* Sticky GuruFocus-style Header Bar */
.sticky-strip {
    position: sticky;
    top: 2.75rem;
    z-index: 90;
    background: rgba(11, 17, 32, 0.96);
    backdrop-filter: blur(12px);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.42);
}
.sticky-grid {
    display: grid;
    grid-template-columns: 1.6fr repeat(5, 1fr);
    gap: 0.7rem;
    align-items: center;
}
@media (max-width: 980px) {
    .sticky-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.sticky-title {
    font-size: 1.04rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.25;
}
.sticky-sub {
    font-size: 0.75rem;
    color: #94A3B8;
}
.sticky-kpi {
    border-left: 1px solid #1E293B;
    padding-left: 0.72rem;
}
.sticky-kpi-label {
    font-size: 0.69rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94A3B8;
}
.sticky-kpi-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F1F5F9;
}

/* GuruFocus 5-Box 10-Point Scorecards */
.gf-box {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 0.78rem 0.92rem;
    height: 100%;
}
.gf-box-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.32rem;
}
.gf-box-title {
    font-size: 0.80rem;
    font-weight: 700;
    color: #CBD5E1;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.gf-box-score {
    font-size: 1.08rem;
    font-weight: 800;
}
.gf-bar-track {
    width: 100%;
    height: 7px;
    background: #1E293B;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 0.42rem;
}
.gf-bar-fill {
    height: 100%;
    border-radius: 999px;
}
.gf-box-metric {
    font-size: 0.86rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.14rem;
}
.gf-box-detail {
    font-size: 0.73rem;
    color: #94A3B8;
    line-height: 1.35;
}

/* Automated Verdict Hero Card */
.verdict-hero {
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.2rem;
    flex-wrap: wrap;
}
.verdict-hero-green {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.55) 0%, rgba(15, 23, 42, 0.95) 75%);
    border: 1px solid rgba(34, 197, 94, 0.50);
}
.verdict-hero-amber {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.50) 0%, rgba(15, 23, 42, 0.95) 75%);
    border: 1px solid rgba(245, 158, 11, 0.50);
}
.verdict-hero-red {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.55) 0%, rgba(15, 23, 42, 0.95) 75%);
    border: 1px solid rgba(239, 68, 68, 0.55);
}
.verdict-kicker {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    font-weight: 700;
    margin-bottom: 0.22rem;
}
.verdict-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.22;
    margin-bottom: 0.3rem;
}
.verdict-sub {
    font-size: 0.89rem;
    color: #CBD5E1;
    margin-bottom: 0.5rem;
}
.verdict-action {
    display: inline-block;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.45rem 0.82rem;
    font-size: 0.86rem;
    font-weight: 600;
    color: #38BDF8;
}
.score-box {
    min-width: 118px;
    text-align: center;
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.7rem 1rem;
}
.score-num {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.05;
}
.score-denom {
    font-size: 0.74rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Pros / Cons list cards */
.pros-cons-box {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 0.82rem 1rem;
    height: 100%;
}
.pros-cons-box ul {
    margin: 0.38rem 0 0 0;
    padding-left: 1.15rem;
    color: #CBD5E1;
    font-size: 0.84rem;
    line-height: 1.46;
}
.pros-cons-box li {
    margin-bottom: 0.28rem;
}

/* Reverse DCF headline card */
.dcf-headline-box {
    background: linear-gradient(135deg, #0F172A 0%, #172554 100%);
    border: 1px solid #1D4ED8;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}
.dcf-headline-kicker {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #93C5FD;
    font-weight: 600;
    margin-bottom: 0.28rem;
}
.dcf-headline-text {
    font-size: 1.24rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.36;
}

/* Traffic light cards */
.flag-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    height: 100%;
}
.flag-green { border-top: 3px solid #22C55E; }
.flag-amber { border-top: 3px solid #F59E0B; }
.flag-red   { border-top: 3px solid #EF4444; }
.flag-na    { border-top: 3px solid #64748B; }
.flag-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.36rem;
    gap: 0.4rem;
}
.flag-title {
    font-size: 0.84rem;
    font-weight: 600;
    color: #E2E8F0;
}
.flag-metric {
    font-size: 1.02rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.26rem;
}
.flag-desc {
    font-size: 0.78rem;
    color: #94A3B8;
    line-height: 1.38;
}

/* Formula card */
.formula-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.65rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.58rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.badge-green { background: rgba(34, 197, 94, 0.16); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.35); }
.badge-amber { background: rgba(245, 158, 11, 0.16); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-red   { background: rgba(239, 68, 68, 0.16); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.35); }
.badge-blue  { background: rgba(56, 189, 248, 0.16); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.35); }
.badge-gray  { background: rgba(148, 163, 184, 0.16); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.30); }

/* Section header divider */
.section-header {
    border-bottom: 1px solid #1E293B;
    padding-bottom: 0.38rem;
    margin-top: 1.55rem;
    margin-bottom: 0.85rem;
}
.section-num {
    font-size: 0.73rem;
    font-weight: 700;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-title {
    font-size: 1.28rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0.08rem 0 0 0;
}

/* Disclaimer box */
.disclaimer-box {
    margin-top: 2.2rem;
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    background: #0F172A;
    border: 1px solid #1E293B;
    color: #94A3B8;
    font-size: 0.8rem;
    line-height: 1.45;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject GuruFocus-inspired dark-theme CSS into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
