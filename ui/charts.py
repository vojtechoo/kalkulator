"""Clean, minimal-clutter dark-theme Plotly charts for all dashboard sections."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.capital_allocation import ALLOCATION_COLUMNS
from analysis.crystal_ball import MonteCarloResult, TriangulationResult
from analysis.formatting import currency_symbol, fmt_large_currency, fmt_multiple, fmt_number, fmt_percent, fmt_price
from analysis.valuation import ValuationBand

DARK_TEXT = "#E2E8F0"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "rgba(148, 163, 184, 0.14)"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
BLUE = "#38BDF8"
INDIGO = "#818CF8"
TEAL = "#14B8A6"
SLATE = "#64748B"


def _apply_clean_layout(fig: go.Figure, height: int = 340, title: str = "") -> go.Figure:
    """Apply consistent minimal-clutter dark styling to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=DARK_TEXT), x=0.0, xanchor="left") if title else None,
        height=height,
        margin=dict(l=16, r=16, t=42 if title else 22, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK_TEXT, size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=MUTED_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(color=MUTED_TEXT)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(color=MUTED_TEXT)),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(color=DARK_TEXT, size=12)),
    )
    return fig


def make_intrinsic_value_heatmap(
    growth_rates: list[float],
    x_values: list[float],
    matrix: np.ndarray,
    current_price: float,
    currency: str,
    x_axis_mode: Literal["discount_rate", "exit_multiple"],
) -> go.Figure:
    """Create a Plotly heatmap of intrinsic value per share centered on current_price (red < price < green)."""
    sym = currency_symbol(currency)
    y_labels = [fmt_percent(g, decimals=1) for g in growth_rates]
    if x_axis_mode == "discount_rate":
        x_labels = [fmt_percent(x, decimals=1) for x in x_values]
        x_title = "Diskontní sazba (požadovaná návratnost)"
    else:
        x_labels = [fmt_multiple(x, decimals=0) for x in x_values]
        x_title = "Výstupní násobek (Exit Multiple × FCF)"

    # Build text annotations and relative % vs current price for hover
    text_grid: list[list[str]] = []
    hover_grid: list[list[str]] = []
    for i, g in enumerate(growth_rates):
        t_row: list[str] = []
        h_row: list[str] = []
        for j, _ in enumerate(x_values):
            val = matrix[i, j]
            if np.isnan(val):
                t_row.append("N/A")
                h_row.append("Nelze spočítat (r ≤ g_term)")
            else:
                diff_pct = (val / current_price - 1.0) if current_price > 0 else 0.0
                marker = " ★" if abs(diff_pct) <= 0.03 else ""
                t_row.append(f"{fmt_number(val, 1)} {sym}{marker}")
                h_row.append(
                    f"Růst FCF: {y_labels[i]}<br>"
                    f"{x_title}: {x_labels[j]}<br>"
                    f"Vnitřní hodnota: <b>{fmt_price(val, currency)}</b><br>"
                    f"Rozdíl vs. cena ({fmt_price(current_price, currency)}): <b>{fmt_percent(diff_pct, 1, signed=True)}</b>"
                )
        text_grid.append(t_row)
        hover_grid.append(h_row)

    # Center colorscale symmetrically around current_price so red = below price, green = above price
    z_min = max(0.0, current_price * 0.45)
    z_max = current_price * 1.55

    colorscale = [
        [0.0, "#991B1B"],
        [0.35, "#DC2626"],
        [0.50, "#1E293B"],
        [0.65, "#16A34A"],
        [1.0, "#15803D"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#F8FAFC"),
            hovertext=hover_grid,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=z_min,
            zmax=z_max,
            zmid=current_price,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text=f"Hodnota ({sym})", font=dict(color=MUTED_TEXT, size=11)),
                tickfont=dict(color=MUTED_TEXT),
            ),
        )
    )
    _apply_clean_layout(
        fig,
        height=390,
        title=f"Citlivost vnitřní hodnoty (střed = dnešní cena {fmt_price(current_price, currency)}, ★ = v pásmu ±3 %)",
    )
    fig.update_xaxes(title_text=x_title, side="bottom")
    fig.update_yaxes(title_text="Roční růst FCF (g)")
    return fig


def make_implied_growth_heatmap(
    prices: list[float],
    discount_rates: list[float],
    matrix: np.ndarray,
    current_price: float,
    currency: str,
) -> go.Figure:
    """Create a Plotly heatmap showing implied annual FCF growth rate g for (Price × Discount Rate)."""
    y_labels = [
        f"{fmt_price(p, currency)} {'(dnes)' if abs(p - current_price) < 0.02 * max(current_price, 1) else ''}".strip()
        for p in prices
    ]
    x_labels = [fmt_percent(r, decimals=1) for r in discount_rates]

    text_grid: list[list[str]] = []
    hover_grid: list[list[str]] = []
    z_pct = matrix * 100.0

    for i, p in enumerate(prices):
        t_row: list[str] = []
        h_row: list[str] = []
        for j, _ in enumerate(discount_rates):
            val = matrix[i, j]
            if np.isnan(val):
                t_row.append("N/A")
                h_row.append("Bez řešení v rozsahu modelu")
            else:
                t_row.append(fmt_percent(val, decimals=1))
                h_row.append(
                    f"Cena akcie: {fmt_price(p, currency)}<br>"
                    f"Diskontní sazba: {x_labels[j]}<br>"
                    f"Implikovaný roční růst FCF: <b>{fmt_percent(val, 1)}</b>"
                )
        text_grid.append(t_row)
        hover_grid.append(h_row)

    # Lower implied growth = easier to achieve (green); higher implied growth = demanding (red)
    colorscale = [
        [0.0, "#15803D"],
        [0.35, "#16A34A"],
        [0.55, "#1E293B"],
        [0.75, "#D97706"],
        [1.0, "#B91C1C"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_pct,
            x=x_labels,
            y=y_labels,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#F8FAFC"),
            hovertext=hover_grid,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=0.0,
            zmax=30.0,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="Růst g (%)", font=dict(color=MUTED_TEXT, size=11)),
                tickfont=dict(color=MUTED_TEXT),
            ),
        )
    )
    _apply_clean_layout(
        fig,
        height=390,
        title="Implikovaný roční růst FCF (g) dle ceny akcie a požadované návratnosti",
    )
    fig.update_xaxes(title_text="Diskontní sazba (požadovaná návratnost)")
    fig.update_yaxes(title_text="Cena akcie")
    return fig


def make_real_fcf_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    """Bar chart comparing Reported FCF, SBC, and Real FCF (FCF - SBC) over time."""
    fig = go.Figure()
    if df.empty:
        return _apply_clean_layout(fig, height=300, title="Skutečné FCF vs. SBC")

    years = df["year"].astype(str)
    sym = currency_symbol(currency)
    scale = 1e9 if df["fcf"].abs().max() >= 1e9 else 1e6
    unit_lbl = f"mld. {sym}" if scale == 1e9 else f"mil. {sym}"

    fig.add_trace(
        go.Bar(
            name="Vykázané FCF",
            x=years,
            y=df["fcf"] / scale,
            marker_color=BLUE,
            opacity=0.85,
        )
    )
    if df["sbc"].notna().any():
        fig.add_trace(
            go.Bar(
                name="Akciové odměny (SBC)",
                x=years,
                y=df["sbc"] / scale,
                marker_color=AMBER,
                opacity=0.9,
            )
        )
    fig.add_trace(
        go.Bar(
            name="Skutečné FCF (FCF − SBC)",
            x=years,
            y=df["real_fcf"] / scale,
            marker_color=[GREEN if (v is not None and pd.notna(v) and v >= 0) else RED for v in df["real_fcf"]],
        )
    )
    fig.update_layout(barmode="group")
    _apply_clean_layout(fig, height=320, title=f"Vykázané FCF, SBC a Skutečné FCF ({unit_lbl})")
    fig.update_yaxes(title_text=unit_lbl)
    return fig


def make_shares_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of diluted shares outstanding over time with YoY % change labels."""
    fig = go.Figure()
    if df.empty or df["diluted_shares"].dropna().empty:
        return _apply_clean_layout(fig, height=300, title="Počet zředěných akcií")

    years = df["year"].astype(str)
    shares_m = df["diluted_shares"] / 1e6
    colors = []
    texts = []
    for chg in df["net_change_pct"]:
        if pd.isna(chg):
            colors.append(BLUE)
            texts.append("")
        elif chg > 0.005:
            colors.append(RED)
            texts.append(fmt_percent(chg, 1, signed=True))
        elif chg < -0.005:
            colors.append(GREEN)
            texts.append(fmt_percent(chg, 1, signed=True))
        else:
            colors.append(BLUE)
            texts.append(fmt_percent(chg, 1, signed=True))

    fig.add_trace(
        go.Bar(
            name="Zředěné akcie (mil.)",
            x=years,
            y=shares_m,
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=10, color=MUTED_TEXT),
        )
    )
    _apply_clean_layout(fig, height=320, title="Vývoj počtu zředěných akcií (mil.) a meziroční změna")
    fig.update_yaxes(title_text="Miliony akcií")
    return fig


def make_earnings_quality_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of OCF / Net Income ratio per year with 0.8x warning and 1.0x reference lines."""
    fig = go.Figure()
    if df.empty or df["ocf_to_ni"].dropna().empty:
        return _apply_clean_layout(fig, height=300, title="Kvalita zisků: OCF / Čistý zisk")

    years = df["year"].astype(str)
    ratios = df["ocf_to_ni"]
    colors = [
        RED if (pd.notna(v) and v < 0.80) else (AMBER if (pd.notna(v) and v < 1.0) else GREEN)
        for v in ratios
    ]

    fig.add_trace(
        go.Bar(
            name="OCF / Čistý zisk",
            x=years,
            y=ratios,
            marker_color=colors,
            text=[fmt_number(v, 2) + "x" if pd.notna(v) else "N/A" for v in ratios],
            textposition="outside",
            textfont=dict(size=10, color=MUTED_TEXT),
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color=BLUE, annotation_text="1,0x (zdravá konverze)", annotation_font_color=BLUE)
    fig.add_hline(y=0.8, line_dash="dash", line_color=RED, annotation_text="0,8x (varovná hranice)", annotation_font_color=RED)
    _apply_clean_layout(fig, height=320, title="Konverze zisku do hotovosti (OCF / Čistý zisk)")
    fig.update_yaxes(title_text="Poměr OCF / Čistý zisk")
    return fig


def make_debt_stress_chart(current_cov: Optional[float], stressed_cov: Optional[float]) -> go.Figure:
    """Bar chart comparing current interest coverage vs. +2 p.b. refinancing stress test."""
    fig = go.Figure()
    labels = ["Aktuální úrokové krytí", "Stress test (+2 p.b. úroků)"]
    vals = [current_cov or 0.0, stressed_cov or 0.0]
    colors = [
        GREEN if v >= 6.0 else (AMBER if v >= 3.0 else RED)
        for v in vals
    ]
    text_lbls = [
        fmt_multiple(current_cov) if current_cov is not None else "N/A",
        fmt_multiple(stressed_cov) if stressed_cov is not None else "N/A",
    ]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors,
            text=text_lbls,
            textposition="outside",
            width=0.45,
        )
    )
    fig.add_hline(y=3.0, line_dash="dash", line_color=RED, annotation_text="Minimum 3,0x", annotation_font_color=RED)
    fig.add_hline(y=6.0, line_dash="dot", line_color=GREEN, annotation_text="Komfortní 6,0x", annotation_font_color=GREEN)
    _apply_clean_layout(fig, height=280, title="Úrokové krytí (EBIT / Úroky): současnost vs. +2 p.b.")
    fig.update_yaxes(title_text="Násobek (x)")
    return fig


def make_roic_wacc_chart(roic_df: pd.DataFrame, wacc: Optional[float]) -> go.Figure:
    """Clean bar chart of annual ROIC colored by spread vs. WACC, with WACC reference line."""
    fig = go.Figure()
    if roic_df.empty or roic_df["roic"].dropna().empty:
        return _apply_clean_layout(fig, height=340, title="ROIC vs. WACC (10 let)")

    years = roic_df["year"].astype(str)
    roic_pct = roic_df["roic"] * 100.0
    colors = [
        RED if below else GREEN
        for below in roic_df["below_wacc"]
    ]
    texts = [fmt_percent(v, 1) if pd.notna(v) else "N/A" for v in roic_df["roic"]]

    fig.add_trace(
        go.Bar(
            name="ROIC (%)",
            x=years,
            y=roic_pct,
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=10, color=DARK_TEXT),
        )
    )
    if wacc is not None:
        fig.add_hline(
            y=wacc * 100.0,
            line_color=AMBER,
            line_width=2.5,
            line_dash="dash",
            annotation_text=f"WACC = {fmt_percent(wacc, 1)}",
            annotation_position="top right",
            annotation_font_color=AMBER,
        )
    _apply_clean_layout(fig, height=350, title="Návratnost investovaného kapitálu (ROIC) vs. náklady kapitálu (WACC)")
    fig.update_yaxes(title_text="%")
    return fig


def make_margins_chart(margins_df: pd.DataFrame) -> go.Figure:
    """Clean grouped bar chart of Gross, Operating, and FCF margins over time."""
    fig = go.Figure()
    if margins_df.empty:
        return _apply_clean_layout(fig, height=350, title="Vývoj marží")

    years = margins_df["year"].astype(str)
    series_spec = [
        ("gross_margin", "Hrubá marže", BLUE),
        ("operating_margin", "Provozní marže (EBIT)", INDIGO),
        ("fcf_margin", "FCF marže", GREEN),
    ]
    for col, label, color in series_spec:
        if margins_df[col].notna().any():
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=years,
                    y=margins_df[col] * 100.0,
                    marker_color=color,
                    opacity=0.88,
                )
            )
    fig.update_layout(barmode="group")
    _apply_clean_layout(fig, height=350, title="Marže v čase: Hrubá, Provozní a FCF marže (%)")
    fig.update_yaxes(title_text="% tržeb")
    return fig


def make_valuation_bands_chart(bands: list[ValuationBand]) -> go.Figure:
    """Horizontal range bar chart showing historical min–max band, median, average, and current multiple."""
    valid = [b for b in bands if b.min_val is not None and b.max_val is not None]
    fig = go.Figure()
    if not valid:
        return _apply_clean_layout(fig, height=260, title="Historická pásma ocenění")

    for b in valid:
        assert b.min_val is not None and b.max_val is not None
        # Min-Max range line
        fig.add_trace(
            go.Scatter(
                x=[b.min_val, b.max_val],
                y=[b.label_cz, b.label_cz],
                mode="lines",
                line=dict(color="#334155", width=16),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Min / Max caps
    fig.add_trace(
        go.Scatter(
            x=[b.min_val for b in valid],
            y=[b.label_cz for b in valid],
            mode="markers+text",
            name="Minimum",
            marker=dict(color=GREEN, size=9, symbol="line-ns-open", line=dict(width=2.5, color=GREEN)),
            text=[fmt_multiple(b.min_val) for b in valid],
            textposition="bottom center",
            textfont=dict(size=10, color=MUTED_TEXT),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b.max_val for b in valid],
            y=[b.label_cz for b in valid],
            mode="markers+text",
            name="Maximum",
            marker=dict(color=RED, size=9, symbol="line-ns-open", line=dict(width=2.5, color=RED)),
            text=[fmt_multiple(b.max_val) for b in valid],
            textposition="bottom center",
            textfont=dict(size=10, color=MUTED_TEXT),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b.median_val for b in valid],
            y=[b.label_cz for b in valid],
            mode="markers",
            name="Medián",
            marker=dict(color=AMBER, size=11, symbol="square"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b.avg_val for b in valid],
            y=[b.label_cz for b in valid],
            mode="markers",
            name="Průměr",
            marker=dict(color="#CBD5E1", size=9, symbol="circle-open", line=dict(width=2, color="#CBD5E1")),
        )
    )

    curr_valid = [b for b in valid if b.current is not None]
    if curr_valid:
        fig.add_trace(
            go.Scatter(
                x=[b.current for b in curr_valid],
                y=[b.label_cz for b in curr_valid],
                mode="markers+text",
                name="Dnes",
                marker=dict(color=BLUE, size=14, symbol="diamond", line=dict(width=1.5, color="#F8FAFC")),
                text=[f"Dnes: {fmt_multiple(b.current)}" for b in curr_valid],
                textposition="top center",
                textfont=dict(size=11, color=BLUE),
            )
        )

    _apply_clean_layout(fig, height=270, title="Historická pásma ocenění (Min – Medián – Průměr – Max vs. Dnes)")
    fig.update_xaxes(title_text="Násobek (x)")
    return fig


def make_capital_allocation_annual_chart(annual_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing annual share (%) of cash generation (OCF + R&D) across 6 buckets."""
    fig = go.Figure()
    if annual_df.empty:
        return _apply_clean_layout(fig, height=340, title="Roční alokace kapitálu")

    years = annual_df["year"].astype(str)
    palette = {
        "capex": BLUE,
        "rd": INDIGO,
        "acquisitions": AMBER,
        "debt_repayment": SLATE,
        "dividends": TEAL,
        "buybacks": GREEN,
    }
    for key, label in ALLOCATION_COLUMNS:
        col = f"{key}_pct"
        if col in annual_df.columns and annual_df[col].sum() > 0:
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=years,
                    y=annual_df[col] * 100.0,
                    marker_color=palette[key],
                )
            )
    fig.update_layout(barmode="stack")
    _apply_clean_layout(
        fig,
        height=350,
        title="Roční alokace kapitálu (% z OCF + R&D)",
    )
    fig.update_yaxes(title_text="% z (OCF + R&D)")
    return fig


def make_capital_allocation_cumulative_chart(
    cumulative_shares: dict[str, float],
    cumulative_totals: dict[str, float],
    currency: str,
    years_count: int,
) -> go.Figure:
    """Horizontal bar chart summarizing cumulative 10Y capital allocation shares."""
    fig = go.Figure()
    palette = {
        "capex": BLUE,
        "rd": INDIGO,
        "acquisitions": AMBER,
        "debt_repayment": SLATE,
        "dividends": TEAL,
        "buybacks": GREEN,
    }
    labels = []
    values_pct = []
    colors = []
    texts = []
    for key, label in ALLOCATION_COLUMNS:
        pct = cumulative_shares.get(key, 0.0) * 100.0
        tot = cumulative_totals.get(key, 0.0)
        labels.append(label)
        values_pct.append(pct)
        colors.append(palette[key])
        texts.append(f"{fmt_number(pct, 1)} % ({fmt_large_currency(tot, currency, 1)})")

    fig.add_trace(
        go.Bar(
            y=labels,
            x=values_pct,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color=DARK_TEXT),
        )
    )
    _apply_clean_layout(
        fig,
        height=350,
        title=f"Kumulativní alokace za {years_count} let (% z OCF + R&D)",
    )
    fig.update_xaxes(title_text="% z celkového OCF + R&D", range=[0, max(values_pct + [10]) * 1.38])
    fig.update_yaxes(autorange="reversed")
    return fig


def make_monte_carlo_fan_chart(
    mc: MonteCarloResult,
    currency: str,
    max_buy_price: Optional[float] = None,
) -> go.Figure:
    """Plot Monte Carlo price trajectory fan chart (P10–P90, P25–P75, Median P50 vs Current Price)."""
    fig = go.Figure()
    if not mc.available or mc.fan_df.empty:
        return _apply_clean_layout(fig, height=340, title="Vějíř budoucích cen (Monte Carlo)")

    df = mc.fan_df
    x = df["label"]
    sym = currency_symbol(currency)

    # Outer band P10 - P90
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.12)",
            name="Pásmo P10–P90 (80 % scénářů)",
            hovertemplate=f"Pesimistický P10: %{{y:,.1f}} {sym}<extra></extra>",
        )
    )

    # Inner band P25 - P75
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p25"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.25)",
            name="Pásmo P25–P75 (50 % scénářů)",
            hovertemplate=f"Konzervativní P25: %{{y:,.1f}} {sym}<extra></extra>",
        )
    )

    # Explicit P90 & P10 lines for readability
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p90"],
            mode="lines+markers",
            name=f"Bull P90 ({fmt_price(mc.p90_price, currency)})",
            line=dict(color=GREEN, width=1.6, dash="dot"),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p50"],
            mode="lines+markers+text",
            name=f"Medián P50 ({fmt_price(mc.p50_price, currency)})",
            line=dict(color=BLUE, width=3.2),
            marker=dict(size=7, color=BLUE),
            text=[""] * (len(df) - 1) + [fmt_price(mc.p50_price, currency)],
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=11),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p10"],
            mode="lines+markers",
            name=f"Bear P10 ({fmt_price(mc.p10_price, currency)})",
            line=dict(color=RED, width=1.6, dash="dot"),
            marker=dict(size=5),
        )
    )

    fig.add_hline(
        y=mc.current_price,
        line_color="#CBD5E1",
        line_dash="dash",
        line_width=1.4,
        annotation_text=f"Dnešní cena: {fmt_price(mc.current_price, currency)}",
        annotation_position="bottom right",
        annotation_font_color="#CBD5E1",
    )
    if max_buy_price is not None and max_buy_price > 0:
        fig.add_hline(
            y=max_buy_price,
            line_color=AMBER,
            line_dash="dot",
            line_width=1.4,
            annotation_text=f"GF Férová hodnota: {fmt_price(max_buy_price, currency)}",
            annotation_position="top left",
            annotation_font_color=AMBER,
        )

    _apply_clean_layout(
        fig,
        height=360,
        title=f"Vějíř cen za {mc.horizon_years} r. ({mc.n_simulations:,} scénářů Monte Carlo)".replace(",", "\u00a0"),
    )
    fig.update_yaxes(title_text=f"Cena akcie ({sym})")
    return fig


def make_monte_carlo_irr_histogram(mc: MonteCarloResult) -> go.Figure:
    """Histogram of 10,000 simulated annualized IRR outcomes split by <0%, 0..hurdle, >=hurdle."""
    fig = go.Figure()
    if not mc.available or mc.annualized_irrs.size == 0:
        return _apply_clean_layout(fig, height=340, title="Distribuce ročního výnosu (IRR)")

    # Clip display range to P1..P99 so extreme tails don't squash the histogram
    irrs_pct = mc.annualized_irrs * 100.0
    lo, hi = float(np.percentile(irrs_pct, 1)), float(np.percentile(irrs_pct, 99))
    clipped = irrs_pct[(irrs_pct >= lo) & (irrs_pct <= hi)]
    hurdle_pct = mc.hurdle_rate * 100.0

    bins = np.linspace(lo, hi, 36)
    counts, edges = np.histogram(clipped, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs_pct = (counts / float(len(irrs_pct))) * 100.0

    colors = [
        RED if c < 0.0 else (AMBER if c < hurdle_pct else GREEN)
        for c in centers
    ]

    fig.add_trace(
        go.Bar(
            x=centers,
            y=probs_pct,
            marker_color=colors,
            opacity=0.88,
            showlegend=False,
            hovertemplate="Roční výnos IRR: %{x:.1f} %<br>Podíl scénářů: %{y:.1f} %<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.0,
        line_color=RED,
        line_dash="dash",
        line_width=1.5,
        annotation_text="0 %",
        annotation_position="top left",
        annotation_font_color=RED,
    )
    fig.add_vline(
        x=hurdle_pct,
        line_color=GREEN,
        line_dash="dash",
        line_width=2.0,
        annotation_text=f"Cíl {fmt_percent(mc.hurdle_rate, 0)}",
        annotation_position="top right",
        annotation_font_color=GREEN,
    )
    fig.add_vline(
        x=mc.p50_irr * 100.0,
        line_color=BLUE,
        line_width=2.2,
        annotation_text=f"Medián {fmt_percent(mc.p50_irr, 1)}",
        annotation_position="bottom right",
        annotation_font_color=BLUE,
    )

    _apply_clean_layout(
        fig,
        height=360,
        title=f"Pravděpodobnostní rozdělení ročního výnosu (IRR, horizont {mc.horizon_years} r.)",
    )
    fig.update_xaxes(title_text="Roční výnos IRR (% p.a.)")
    fig.update_yaxes(title_text="Podíl scénářů (%)")
    return fig


def make_triangulation_bar_chart(
    tri: TriangulationResult,
    current_price: Optional[float],
    currency: str,
) -> go.Figure:
    """Horizontal bar chart comparing 4 independent fair-value models + consensus vs current price."""
    fig = go.Figure()
    sym = currency_symbol(currency)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    texts: list[str] = []

    for m in tri.models:
        if m.value_per_share is not None and m.value_per_share > 0:
            labels.append(m.name_cz)
            values.append(m.value_per_share)
            is_above = current_price is not None and m.value_per_share >= current_price
            colors.append(BLUE if is_above else SLATE)
            up_txt = f" ({fmt_percent(m.upside_pct, 0, signed=True)})" if m.upside_pct is not None else ""
            texts.append(f"{fmt_price(m.value_per_share, currency)}{up_txt}")

    if tri.fair_value_per_share is not None:
        labels.append("Vážená férová hodnota")
        values.append(tri.fair_value_per_share)
        colors.append(INDIGO)
        up_txt = f" ({fmt_percent(tri.upside_to_fair_value, 0, signed=True)})" if tri.upside_to_fair_value is not None else ""
        texts.append(f"{fmt_price(tri.fair_value_per_share, currency)}{up_txt}")

    if tri.max_buy_price is not None:
        labels.append(f"Max. nákupní cena (MoS {int(tri.margin_of_safety_target*100)} %)")
        values.append(tri.max_buy_price)
        colors.append(GREEN)
        texts.append(fmt_price(tri.max_buy_price, currency))

    if not values:
        return _apply_clean_layout(fig, height=280, title="Triangulace vnitřní hodnoty")

    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color=DARK_TEXT),
        )
    )
    if current_price is not None and current_price > 0:
        fig.add_vline(
            x=current_price,
            line_color=AMBER,
            line_dash="dash",
            line_width=2.2,
            annotation_text=f"Dnes: {fmt_price(current_price, currency)}",
            annotation_position="top right",
            annotation_font_color=AMBER,
        )

    max_x = max(values + ([current_price] if current_price else [0.0]))
    _apply_clean_layout(fig, height=320, title="Triangulace vnitřní hodnoty ze 4 nezávislých modelů")
    fig.update_xaxes(title_text=f"Hodnota na akcii ({sym})", range=[0, max_x * 1.38])
    fig.update_yaxes(autorange="reversed")
    return fig


def make_gf_valuation_gauge(
    price_to_gf: Optional[float],
    current_price: Optional[float],
    gf_fair_value: Optional[float],
    currency: str,
    status_label: str,
) -> go.Figure:
    """GuruFocus-style semicircular Valuation Gauge showing Price / GF Fair Value ratio."""
    val = float(np.clip(price_to_gf, 0.35, 1.85)) if price_to_gf is not None else 1.0
    bar_color = (
        GREEN
        if (price_to_gf is not None and price_to_gf <= 0.92)
        else (AMBER if (price_to_gf is not None and price_to_gf <= 1.10) else RED)
    )
    if price_to_gf is None:
        bar_color = SLATE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(price_to_gf, 2) if price_to_gf is not None else 0.0,
            number=dict(
                suffix="x" if price_to_gf is not None else "",
                font=dict(size=30, color="#F8FAFC"),
            ),
            title=dict(
                text=(
                    f"<b>{status_label}</b><br>"
                    f"<span style='font-size:12px;color:#94A3B8'>Cena {fmt_price(current_price, currency)} "
                    f"vs. GF Hodnota {fmt_price(gf_fair_value, currency)}</span>"
                ),
                font=dict(size=14, color=DARK_TEXT),
            ),
            gauge=dict(
                axis=dict(
                    range=[0.4, 1.8],
                    tickvals=[0.5, 0.75, 0.92, 1.0, 1.10, 1.30, 1.7],
                    ticktext=["0,5x", "0,75x", "0,92x", "1,0x", "1,1x", "1,3x", "1,7x"],
                    tickfont=dict(size=10, color=MUTED_TEXT),
                ),
                bar=dict(color=bar_color, thickness=0.28),
                bgcolor="#0F172A",
                borderwidth=1,
                bordercolor="#1E293B",
                steps=[
                    dict(range=[0.4, 0.75], color="rgba(21, 128, 61, 0.45)"),
                    dict(range=[0.75, 0.92], color="rgba(34, 197, 94, 0.28)"),
                    dict(range=[0.92, 1.10], color="rgba(245, 158, 11, 0.28)"),
                    dict(range=[1.10, 1.30], color="rgba(234, 88, 12, 0.30)"),
                    dict(range=[1.30, 1.80], color="rgba(220, 38, 38, 0.40)"),
                ],
                threshold=dict(
                    line=dict(color="#F8FAFC", width=3),
                    thickness=0.85,
                    value=val,
                ),
            ),
        )
    )
    fig.update_layout(
        height=255,
        margin=dict(l=18, r=18, t=54, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK_TEXT),
    )
    return fig


def make_gf_value_band_chart(gf_hist_df: pd.DataFrame, currency: str) -> go.Figure:
    """Iconic GuruFocus-style Fair Value Line chart with ±10 % and ±30 % valuation corridors."""
    fig = go.Figure()
    if gf_hist_df.empty:
        return _apply_clean_layout(fig, height=300, title="GF Value koridor vs. Cena akcie")

    x = gf_hist_df["year"]
    sym = currency_symbol(currency)

    # +30% Overvalued band
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gf_hist_df["band_high30"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gf_hist_df["band_high10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.16)",
            name="Nadhodnoceno (+10 % až +30 %)",
            hoverinfo="skip",
        )
    )

    # -30% Undervalued band
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gf_hist_df["band_low10"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gf_hist_df["band_low30"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(34, 197, 94, 0.18)",
            name="Podhodnoceno (−10 % až −30 %)",
            hoverinfo="skip",
        )
    )

    # GF Fair Value central line (historical + projected through 2032)
    last_yr = str(gf_hist_df["year"].iloc[-1])
    last_fv = float(gf_hist_df["fair_value"].iloc[-1])
    fv_texts = [""] * (len(gf_hist_df) - 1) + [fmt_price(last_fv, currency)]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gf_hist_df["fair_value"],
            mode="lines+text",
            name="GF Férová hodnota (projekce do 2032)",
            line=dict(color=AMBER, width=2.4, dash="dash"),
            text=fv_texts,
            textposition="top left",
            textfont=dict(size=10, color=AMBER),
        )
    )

    # Actual stock price (stops cleanly at "Dnes")
    price_df = gf_hist_df.dropna(subset=["price"])
    fig.add_trace(
        go.Scatter(
            x=price_df["year"],
            y=price_df["price"],
            mode="lines+markers",
            name="Cena akcie",
            line=dict(color=BLUE, width=3.0),
            marker=dict(size=6, color=BLUE),
        )
    )

    # Subtle vertical divider at "Dnes" separating history from 2027-2032 projection
    if "Dnes" in list(x):
        fig.add_shape(
            type="line",
            x0="Dnes",
            x1="Dnes",
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#475569", dash="dot", width=1.4),
        )
        fig.add_annotation(
            x="Dnes",
            y=1.0,
            xref="x",
            yref="paper",
            text="Projekce →",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color=MUTED_TEXT),
        )

    _apply_clean_layout(
        fig,
        height=315,
        title=f"GF Value koridor (do roku {last_yr}): Cena akcie vs. Férová hodnota (±30 %)",
    )
    fig.update_yaxes(title_text=f"Cena ({sym})")
    return fig


def make_all_models_valuation_chart(
    models: list,
    gf_fair_value: Optional[float],
    gf_buy_price: Optional[float],
    mos: float,
    current_price: Optional[float],
    currency: str,
) -> go.Figure:
    """Horizontal bar chart comparing all mathematical valuation models vs. current price and MoS buy price."""
    fig = go.Figure()
    sym = currency_symbol(currency)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    texts: list[str] = []

    for m in models:
        if m.value_per_share is not None and m.value_per_share > 0:
            labels.append(m.name_cz)
            values.append(m.value_per_share)
            is_under = current_price is not None and m.value_per_share >= current_price
            colors.append(BLUE if is_under else SLATE)
            up_s = f" ({fmt_percent(m.upside_pct, 0, signed=True)})" if m.upside_pct is not None else ""
            texts.append(f"{fmt_price(m.value_per_share, currency)}{up_s}")

    if gf_fair_value is not None and gf_fair_value > 0:
        labels.append("GF Vážená férová hodnota")
        values.append(gf_fair_value)
        colors.append(INDIGO)
        up_g = ((gf_fair_value / current_price) - 1.0) if (current_price and current_price > 0) else None
        up_s = f" ({fmt_percent(up_g, 0, signed=True)})" if up_g is not None else ""
        texts.append(f"{fmt_price(gf_fair_value, currency)}{up_s}")

    if not values:
        return _apply_clean_layout(fig, height=320, title="Matematické modely vnitřní hodnoty")

    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color=DARK_TEXT),
        )
    )
    if current_price is not None and current_price > 0:
        fig.add_vline(
            x=current_price,
            line_color=AMBER,
            line_dash="dash",
            line_width=2.2,
            annotation_text=f"Dnes: {fmt_price(current_price, currency)}",
            annotation_position="top right",
            annotation_font_color=AMBER,
        )

    max_x = max(values + ([current_price] if current_price else [0.0]))
    _apply_clean_layout(fig, height=360, title="Srovnání všech matematických modelů ocenění vs. dnešní cena")
    fig.update_xaxes(title_text=f"Vnitřní hodnota na akcii ({sym})", range=[0, max_x * 1.42])
    fig.update_yaxes(autorange="reversed")
    return fig


def make_qualitative_radar_chart(pillar_scores: dict[str, int]) -> go.Figure:
    """Radar chart for the 6 non-mathematical value-investing criteria (0–100)."""
    labels = list(pillar_scores.keys())
    vals = list(pillar_scores.values())
    if not labels:
        return _apply_clean_layout(go.Figure(), height=310, title="Kvalitativní profil (6 kritérií)")

    # Close polygon
    labels_closed = labels + [labels[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.22)",
            line=dict(color=BLUE, width=2.4),
            marker=dict(size=6, color=BLUE),
            name="Kvalitativní skóre",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=46, r=46, t=36, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK_TEXT, size=11),
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.55)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                gridcolor=GRID_COLOR,
                tickfont=dict(size=9, color=MUTED_TEXT),
            ),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10, color=DARK_TEXT)),
        ),
        showlegend=False,
    )
    return fig
