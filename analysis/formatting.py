"""Czech number, percentage, currency, and safe formatting utilities."""

from __future__ import annotations

import math
from typing import Optional

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "CZK": "Kč",
    "CHF": "CHF",
    "JPY": "¥",
    "CAD": "CAD",
    "AUD": "AUD",
    "SEK": "SEK",
    "NOK": "NOK",
    "DKK": "DKK",
    "PLN": "PLN",
    "HKD": "HK$",
    "TWD": "NT$",
    "CNY": "CN¥",
    "KRW": "₩",
}


def is_valid_number(value: Optional[float]) -> bool:
    """Return True if value is a finite real number."""
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def currency_symbol(currency: Optional[str]) -> str:
    """Map an ISO currency code to a compact Czech-friendly symbol."""
    if not currency:
        return "$"
    code = currency.strip().upper()
    return CURRENCY_SYMBOLS.get(code, code)


def fmt_number(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a float in Czech style: non-breaking space thousands separator, decimal comma."""
    if not is_valid_number(value):
        return "N/A"
    assert value is not None
    sign = "+" if signed and value > 0 else ""
    formatted = f"{value:,.{decimals}f}"
    formatted = formatted.replace(",", "\u00a0").replace(".", ",")
    return f"{sign}{formatted}"


def fmt_percent(value: Optional[float], decimals: int = 1, signed: bool = False) -> str:
    """Format a decimal ratio (e.g., 0.125) as a Czech percentage string ('12,5 %')."""
    if not is_valid_number(value):
        return "N/A"
    assert value is not None
    return f"{fmt_number(value * 100.0, decimals=decimals, signed=signed)}\u00a0%"


def fmt_pp(value: Optional[float], decimals: int = 1, signed: bool = True) -> str:
    """Format a decimal difference as percentage points ('+2,5 p.b.')."""
    if not is_valid_number(value):
        return "N/A"
    assert value is not None
    return f"{fmt_number(value * 100.0, decimals=decimals, signed=signed)}\u00a0p.b."


def fmt_multiple(value: Optional[float], decimals: int = 1) -> str:
    """Format a valuation multiple (e.g., 15.2 -> '15,2x')."""
    if not is_valid_number(value):
        return "N/A"
    return f"{fmt_number(value, decimals=decimals)}x"


def fmt_price(value: Optional[float], currency: Optional[str] = "USD", decimals: int = 2) -> str:
    """Format a per-share price with currency symbol in Czech style ('150,00 $')."""
    if not is_valid_number(value):
        return "N/A"
    sym = currency_symbol(currency)
    return f"{fmt_number(value, decimals=decimals)}\u00a0{sym}"


def fmt_large_currency(value: Optional[float], currency: Optional[str] = "USD", decimals: int = 2) -> str:
    """Format large financial amounts using Czech magnitude suffixes (tis., mil., mld., bil.)."""
    if not is_valid_number(value):
        return "N/A"
    assert value is not None
    sym = currency_symbol(currency)
    abs_val = abs(value)
    if abs_val >= 1e12:
        scaled, suffix = value / 1e12, "bil."
    elif abs_val >= 1e9:
        scaled, suffix = value / 1e9, "mld."
    elif abs_val >= 1e6:
        scaled, suffix = value / 1e6, "mil."
    elif abs_val >= 1e3:
        scaled, suffix = value / 1e3, "tis."
    else:
        return f"{fmt_number(value, decimals=decimals)}\u00a0{sym}"
    return f"{fmt_number(scaled, decimals=decimals)}\u00a0{suffix}\u00a0{sym}"


def fmt_shares(value: Optional[float], decimals: int = 2) -> str:
    """Format share counts with Czech magnitude suffixes."""
    if not is_valid_number(value):
        return "N/A"
    assert value is not None
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{fmt_number(value / 1e9, decimals=decimals)}\u00a0mld."
    if abs_val >= 1e6:
        return f"{fmt_number(value / 1e6, decimals=decimals)}\u00a0mil."
    if abs_val >= 1e3:
        return f"{fmt_number(value / 1e3, decimals=decimals)}\u00a0tis."
    return fmt_number(value, decimals=0)
