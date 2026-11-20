"""Streamlit caching layer (TTL = 6 hours) for external financial data calls."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from data.models import CompanyData, PeerSummary
from data.provider import ProviderFactory

CACHE_TTL_SECONDS = 6 * 3600  # 6 hours
CACHE_SCHEMA_VERSION = "v4_capex_norm_legends"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_fetch_company(
    ticker: str,
    api_key_fingerprint: str = "",
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> CompanyData:
    """Fetch and cache full company data for 6 hours.

    `api_key_fingerprint` and `schema_version` are included in the cache key so
    switching API keys or upgrading AnnualFinancials invalidates stale entries.
    """
    _ = (api_key_fingerprint, schema_version)
    factory = ProviderFactory()
    return factory.fetch_company(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_fetch_peers(tickers: tuple[str, ...], api_key_fingerprint: str = "") -> list[PeerSummary]:
    """Fetch and cache peer summary metrics for 6 hours."""
    _ = api_key_fingerprint
    factory = ProviderFactory()
    results: list[PeerSummary] = []
    for sym in tickers:
        summary = factory.fetch_peer(sym)
        if summary is not None:
            results.append(summary)
    return results


def get_api_key_fingerprint(api_key: Optional[str]) -> str:
    """Return a non-sensitive short fingerprint indicating whether an FMP key is configured."""
    if not api_key:
        return "none"
    return f"fmp_{len(api_key)}_{api_key[-4:]}"
