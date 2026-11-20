"""Typed domain models for financial data retrieved from external providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnnualFinancials:
    """Normalized annual financial statement row for a single fiscal year."""

    year: int
    date: str
    currency: str
    # Income statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None  # EBIT
    interest_expense: Optional[float] = None  # Stored as positive magnitude
    income_before_tax: Optional[float] = None
    income_tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    rd_expense: Optional[float] = None  # Positive magnitude
    sga_expense: Optional[float] = None  # Positive magnitude (SG&A)
    depreciation_amortization: Optional[float] = None  # Positive magnitude (D&A)
    eps_diluted: Optional[float] = None
    weighted_avg_shares_diluted: Optional[float] = None
    # Balance sheet
    cash_and_short_term: Optional[float] = None
    accounts_receivable: Optional[float] = None
    current_assets: Optional[float] = None
    ppe_net: Optional[float] = None
    total_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    retained_earnings: Optional[float] = None
    short_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    # Cash flow statement
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None  # Stored as positive magnitude
    free_cash_flow: Optional[float] = None
    stock_based_compensation: Optional[float] = None  # Positive magnitude
    acquisitions_net: Optional[float] = None  # Positive magnitude = cash outflow
    debt_repayment: Optional[float] = None  # Positive magnitude = cash outflow
    dividends_paid: Optional[float] = None  # Positive magnitude = cash outflow
    share_buybacks: Optional[float] = None  # Positive magnitude = cash outflow
    # Historical valuation metrics (from key-metrics or derived from historical price)
    pe_ratio: Optional[float] = None
    pfcf_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    market_cap_year: Optional[float] = None
    avg_share_price: Optional[float] = None


@dataclass
class InsiderTrade:
    """Single insider transaction record."""

    date: str
    insider_name: str
    role: str
    transaction_type: str  # Raw provider transaction code/label (e.g., "P-Purchase", "S-Sale")
    shares: Optional[float] = None
    price: Optional[float] = None
    value: Optional[float] = None
    is_open_market: bool = False
    direction: str = "OTHER"  # "BUY", "SELL", "OTHER"
    link: Optional[str] = None


@dataclass
class AnalystEstimate:
    """Forward annual analyst consensus estimate."""

    year: int
    date: str
    estimated_revenue_avg: Optional[float] = None
    estimated_revenue_low: Optional[float] = None
    estimated_revenue_high: Optional[float] = None
    estimated_eps_avg: Optional[float] = None
    estimated_eps_low: Optional[float] = None
    estimated_eps_high: Optional[float] = None
    num_analysts_revenue: Optional[int] = None
    num_analysts_eps: Optional[int] = None


@dataclass
class NewsItem:
    """Single stock news article."""

    published_date: str
    title: str
    source: str
    url: str
    text: str = ""
    category: str = ""


@dataclass
class PeerSummary:
    """Valuation and quality snapshot for a peer company."""

    ticker: str
    name: str
    currency: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pfcf_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    revenue_cagr_3y: Optional[float] = None
    is_target: bool = False


@dataclass
class InstitutionalHolder:
    """Major institutional / guru shareholder record."""

    holder_name: str
    pct_held: Optional[float] = None  # decimal, e.g. 0.085 for 8.5 %
    shares: Optional[float] = None
    value: Optional[float] = None
    date_reported: str = ""


@dataclass
class CompanyData:
    """Complete data bundle for a single stock ticker."""

    ticker: str
    name: str
    exchange: str
    sector: str
    industry: str
    country: str
    quote_currency: str
    reporting_currency: str
    fx_rate_reporting_to_quote: Optional[float]  # 1 unit of reporting_currency in quote_currency
    current_price: Optional[float]
    market_cap: Optional[float]
    beta: Optional[float]
    diluted_shares_current: Optional[float]
    # TTM values (in reporting currency, converted to quote currency when FX is applied)
    ttm_fcf: Optional[float] = None
    ttm_sbc: Optional[float] = None
    ttm_capex: Optional[float] = None
    ttm_revenue: Optional[float] = None
    ttm_net_income: Optional[float] = None
    ttm_ebitda: Optional[float] = None
    ttm_ebit: Optional[float] = None
    ttm_ocf: Optional[float] = None
    # Latest balance sheet snapshot
    latest_cash: Optional[float] = None
    latest_total_debt: Optional[float] = None
    latest_net_debt: Optional[float] = None
    # Risk-free rate (10Y Treasury, decimal e.g. 0.043)
    risk_free_rate: Optional[float] = None
    risk_free_source: str = ""
    # Time series (sorted oldest -> newest)
    annuals: list[AnnualFinancials] = field(default_factory=list)
    peers: list[str] = field(default_factory=list)
    insider_trades: list[InsiderTrade] = field(default_factory=list)
    institutional_holders: list[InstitutionalHolder] = field(default_factory=list)
    analyst_estimates: list[AnalystEstimate] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    # Provider metadata & warnings
    provider_name: str = ""
    is_limited_history: bool = False
    is_financial_or_reit: bool = False
    warnings: list[str] = field(default_factory=list)
    description: str = ""
