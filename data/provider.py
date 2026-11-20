"""Abstract DataProvider interface and fallback orchestration."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from data.models import CompanyData, PeerSummary

FINANCIAL_SECTORS = {"financial services", "financial", "financials", "real estate"}
FINANCIAL_INDUSTRY_KEYWORDS = (
    "bank",
    "insurance",
    "insurer",
    "reit",
    "mortgage",
    "credit services",
    "asset management",
    "capital markets",
    "savings",
    "thrift",
)


class DataProviderError(Exception):
    """Raised when a data provider cannot fulfill a ticker request."""


class RateLimitError(DataProviderError):
    """Raised when an API provider reports a rate-limit or quota error."""


def is_financial_or_reit(sector: Optional[str], industry: Optional[str]) -> bool:
    """Return True if the company belongs to banking, insurance, financial services, or REITs."""
    sec = (sector or "").strip().lower()
    ind = (industry or "").strip().lower()
    if sec in FINANCIAL_SECTORS:
        return True
    return any(kw in ind for kw in FINANCIAL_INDUSTRY_KEYWORDS)


def load_fmp_api_key() -> Optional[str]:
    """Load FMP_API_KEY from environment, local .env file, or Streamlit secrets."""
    env_key = os.getenv("FMP_API_KEY", "").strip()
    if env_key and env_key != "your_fmp_api_key_here":
        return env_key

    # Try python-dotenv if installed, otherwise parse .env directly.
    env_path = Path.cwd() / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(dotenv_path=env_path, override=False)
            env_key = os.getenv("FMP_API_KEY", "").strip()
            if env_key and env_key != "your_fmp_api_key_here":
                return env_key
        except ImportError:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "FMP_API_KEY":
                    candidate = v.strip().strip('"').strip("'")
                    if candidate and candidate != "your_fmp_api_key_here":
                        return candidate

    try:
        import streamlit as st

        if "FMP_API_KEY" in st.secrets:
            secret_key = str(st.secrets["FMP_API_KEY"]).strip()
            if secret_key and secret_key != "your_fmp_api_key_here":
                return secret_key
    except Exception:
        pass

    return None


class DataProvider(ABC):
    """Abstract base class for fundamental market data providers."""

    name: str = "AbstractProvider"

    @abstractmethod
    def get_company_data(self, ticker: str) -> CompanyData:
        """Fetch full fundamental, valuation, insider, consensus, and news data for a ticker."""

    @abstractmethod
    def get_peer_summary(self, ticker: str) -> Optional[PeerSummary]:
        """Fetch compact valuation and quality metrics for a single peer ticker."""

    @abstractmethod
    def get_risk_free_rate(self) -> tuple[Optional[float], str]:
        """Fetch the 10-year US Treasury yield as a decimal (e.g. 0.043) and source label."""


class ProviderFactory:
    """Creates primary (FMP) and fallback (yfinance) providers and orchestrates fallback."""

    def __init__(self, fmp_api_key: Optional[str] = None) -> None:
        self.fmp_api_key = fmp_api_key if fmp_api_key is not None else load_fmp_api_key()

    def get_primary_and_fallback(self) -> tuple[Optional[DataProvider], DataProvider]:
        """Instantiate available providers."""
        from data.fmp_provider import FMPDataProvider
        from data.yfinance_provider import YFinanceDataProvider

        fallback: DataProvider = YFinanceDataProvider()
        primary: Optional[DataProvider] = (
            FMPDataProvider(api_key=self.fmp_api_key) if self.fmp_api_key else None
        )
        return primary, fallback

    def fetch_company(self, ticker: str) -> CompanyData:
        """Fetch company data using FMP first, falling back to yfinance on failure or missing key."""
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise DataProviderError("Zadejte platný burzovní ticker (např. AAPL, MSFT, BRK-B).")

        primary, fallback = self.get_primary_and_fallback()
        fallback_reason: Optional[str] = None

        if primary is not None:
            try:
                data = primary.get_company_data(clean_ticker)
                # Fill missing 10Y risk-free rate from yfinance if FMP did not return one.
                if data.risk_free_rate is None:
                    rf, rf_src = fallback.get_risk_free_rate()
                    data.risk_free_rate = rf
                    data.risk_free_source = rf_src
                return data
            except RateLimitError as exc:
                fallback_reason = f"FMP API dosáhlo limitu požadavků ({exc}); použit záložní zdroj."
            except DataProviderError as exc:
                fallback_reason = f"FMP API nevrátilo kompletní data ({exc}); použit záložní zdroj."

        data = fallback.get_company_data(clean_ticker)
        if fallback_reason:
            data.warnings.insert(0, fallback_reason)
        return data

    def fetch_peer(self, ticker: str) -> Optional[PeerSummary]:
        """Fetch peer metrics using FMP primary and yfinance fallback."""
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            return None
        primary, fallback = self.get_primary_and_fallback()
        if primary is not None:
            try:
                peer = primary.get_peer_summary(clean_ticker)
                if peer is not None:
                    return peer
            except DataProviderError:
                pass
        try:
            return fallback.get_peer_summary(clean_ticker)
        except Exception:
            return None
