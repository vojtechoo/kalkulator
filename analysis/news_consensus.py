"""Hard-news filtering (no clickbait noise) and analyst consensus contrarian comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from analysis.formatting import fmt_percent, fmt_pp
from data.models import AnalystEstimate, CompanyData, NewsItem

# Blacklist patterns: clickbait, short-term price-move noise, listicles, and opinion fluff
NOISE_BLACKLIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\bhere'?s why\b",
        r"\bshould you buy\b",
        r"\bis it time to buy\b",
        r"\btime to buy\b",
        r"\bbetter buy\b",
        r"\btop\s+\d+\s+stocks?\b",
        r"\b\d+\s+(best|top|cheap|growth|dividend|value)?\s*stocks?\s+to\b",
        r"\bstock\s+(jumps|drops|falls|soars|plunges|slides|surges|tumbles|rallies|sinks|pops|dips)\b",
        r"\bshares?\s+(jump|drop|fall|soar|plunge|slide|surge|tumble|rally|sink|pop|dip)\b",
        r"\bwhy\s+.{1,45}\s+stock\s+is\s+(up|down|moving|falling|rising|sinking|surging)\b",
        r"\bwhat you need to know\b",
        r"\bmotley fool\b",
        r"\bjim cramer\b",
        r"\bmillionaire\b",
        r"\bretire\b",
        r"\bwall\s*street\s*bets\b",
        r"\bmeme stock\b",
        r"\bunusual options?\b",
        r"\btechnical analysis\b",
        r"\bmoving average\b",
        r"\bshort squeeze\b",
        r"\bprediction\b",
        r"\bwhere will .+ be in \d+ years\b",
        r"\bprice target\s+(raised|cut|lowered|maintained|reiterated)\b",
    )
)

# Whitelist categories: hard fundamental / corporate / regulatory news
CATEGORY_WHITELIST: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Výsledky a výhled",
        re.compile(
            r"\b(earnings|quarterly results|q[1-4]\s*(20\d\d|results|earnings)|fiscal\s+20\d\d|"
            r"10-k|10-q|8-k|guidance|outlook|raises forecast|cuts forecast|revenue|eps\s+(beat|miss|of)|"
            r"annual report|financial results)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "M&A a restrukturalizace",
        re.compile(
            r"\b(acquires?|acquisition|merger|merge|buyout|takeover|divest|divestiture|"
            r"spin[- ]?off|joint venture|strategic partnership|asset sale)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Vedení a governance",
        re.compile(
            r"\b(ceo|cfo|chief executive|chief financial|board of directors|appoints?|names?\s+.+\s+as|"
            r"resigns?|steps down|retires?|succession|activist investor|proxy fight)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Kapitálová alokace",
        re.compile(
            r"\b(dividend|buyback|share repurchase|stock split|debt offering|senior notes|"
            r"credit rating|bond offering|capex|capital expenditure)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Regulace a sektor",
        re.compile(
            r"\b(fda|ema|sec\s+(filing|investigation|probe)|antitrust|doj|ftc|european commission|"
            r"lawsuit|settlement|patent|tariff|sanctions?|approval|clinical trial|phase\s+[1-3]|"
            r"contract award|recall)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass
class FilteredNewsReport:
    """Hard news results and noise filtering statistics."""

    hard_news: list[NewsItem]
    total_fetched: int
    filtered_out_count: int


@dataclass
class ConsensusContrarianReport:
    """Analyst consensus vs. Reverse DCF implied growth comparison."""

    estimates: list[AnalystEstimate]
    consensus_rev_cagr: Optional[float]
    consensus_eps_cagr: Optional[float]
    implied_dcf_growth: Optional[float]
    gap_vs_rev_cagr: Optional[float]  # implied_dcf_growth - consensus_rev_cagr
    gap_vs_eps_cagr: Optional[float]  # implied_dcf_growth - consensus_eps_cagr
    contrarian_insight_cz: str


def filter_hard_news(news_items: list[NewsItem], max_items: int = 12) -> FilteredNewsReport:
    """Keep only hard fundamental/corporate/regulatory news and strip clickbait/price-move noise."""
    hard: list[NewsItem] = []
    seen_titles: set[str] = set()

    for item in news_items:
        title_clean = item.title.strip()
        title_key = title_clean.lower()
        if not title_clean or title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # Step 1: Reject clickbait/price-move blacklist matches on the headline
        if any(pat.search(title_clean) for pat in NOISE_BLACKLIST_PATTERNS):
            continue

        # Step 2: Require whitelist match on headline (or headline + summary snippet)
        combined = f"{title_clean} {item.text[:240]}"
        matched_category = ""
        for cat_label, pat in CATEGORY_WHITELIST:
            if pat.search(title_clean) or pat.search(combined):
                matched_category = cat_label
                break

        if matched_category:
            hard.append(
                NewsItem(
                    published_date=item.published_date,
                    title=title_clean,
                    source=item.source,
                    url=item.url,
                    text=item.text,
                    category=matched_category,
                )
            )
            if len(hard) >= max_items:
                break

    return FilteredNewsReport(
        hard_news=hard,
        total_fetched=len(news_items),
        filtered_out_count=max(0, len(news_items) - len(hard)),
    )


def analyze_consensus_contrarian(
    company: CompanyData,
    implied_dcf_growth: Optional[float],
) -> ConsensusContrarianReport:
    """Compare forward 2–3Y analyst consensus growth against Reverse DCF implied FCF growth."""
    estimates = company.analyst_estimates
    latest = company.annuals[-1] if company.annuals else None

    rev_cagr: Optional[float] = None
    eps_cagr: Optional[float] = None

    if latest is not None and estimates:
        fut_rev = [
            e for e in estimates if e.year > latest.year and e.estimated_revenue_avg and e.estimated_revenue_avg > 0
        ]
        if fut_rev and latest.revenue and latest.revenue > 0:
            last_e = fut_rev[-1]
            span = last_e.year - latest.year
            if span >= 1 and last_e.estimated_revenue_avg is not None:
                rev_cagr = (last_e.estimated_revenue_avg / latest.revenue) ** (1.0 / span) - 1.0

        fut_eps = [
            e for e in estimates if e.year > latest.year and e.estimated_eps_avg and e.estimated_eps_avg > 0
        ]
        if fut_eps and latest.eps_diluted and latest.eps_diluted > 0:
            last_e = fut_eps[-1]
            span = last_e.year - latest.year
            if span >= 1 and last_e.estimated_eps_avg is not None:
                eps_cagr = (last_e.estimated_eps_avg / latest.eps_diluted) ** (1.0 / span) - 1.0

    gap_rev = (
        (implied_dcf_growth - rev_cagr)
        if (implied_dcf_growth is not None and rev_cagr is not None)
        else None
    )
    gap_eps = (
        (implied_dcf_growth - eps_cagr)
        if (implied_dcf_growth is not None and eps_cagr is not None)
        else None
    )

    # Craft Czech contrarian insight
    if not estimates:
        insight = (
            "Konsenzus analytiků (odhady tržeb a EPS na 2–3 roky) není pro tento ticker ze zdroje dat dostupný."
        )
    elif implied_dcf_growth is None:
        insight = (
            f"Dav analytiků očekává růst tržeb {fmt_percent(rev_cagr)} a EPS {fmt_percent(eps_cagr)} ročně. "
            "Srovnání s Reverse DCF není možné, protože implikovaný růst FCF nebyl vyřešen."
        )
    else:
        ref_cons = eps_cagr if eps_cagr is not None else rev_cagr
        ref_name = "EPS" if eps_cagr is not None else "tržeb"
        if ref_cons is None:
            insight = (
                "Odhady analytiků existují, ale chybí kladná výchozí základna pro výpočet tempa růstu (CAGR)."
            )
        else:
            diff = implied_dcf_growth - ref_cons
            if diff <= -0.03:
                insight = (
                    f"Potenciální kontrariánská výhoda: trh v ceně akcie předpokládá růst FCF pouze "
                    f"{fmt_percent(implied_dcf_growth)}, zatímco konsenzus analytiků čeká růst {ref_name} "
                    f"{fmt_percent(ref_cons)} (rozdíl {fmt_pp(diff)}). Pokud se konsenzus naplní, cena má bezpečnostní polštář."
                )
            elif diff >= 0.03:
                insight = (
                    f"Pozor na přepálená očekávání: cena akcie implikuje růst FCF {fmt_percent(implied_dcf_growth)}, "
                    f"což je o {fmt_pp(diff)} VÍCE, než kolik očekávají analytici u {ref_name} ({fmt_percent(ref_cons)}). "
                    "Pro ospravedlnění dnešní ceny musí firma překonat i optimismus Wall Street."
                )
            else:
                insight = (
                    f"Trh a dav jsou v souladu: Reverse DCF implikuje růst {fmt_percent(implied_dcf_growth)} "
                    f"oproti konsenzuálnímu růstu {ref_name} {fmt_percent(ref_cons)} (rozdíl {fmt_pp(diff)}). "
                    "Informační výhodu získáte pouze tehdy, máte-li odlišný názor na dlouhodobou udržitelnost marží nebo reinvestic."
                )

    return ConsensusContrarianReport(
        estimates=estimates,
        consensus_rev_cagr=rev_cagr,
        consensus_eps_cagr=eps_cagr,
        implied_dcf_growth=implied_dcf_growth,
        gap_vs_rev_cagr=gap_rev,
        gap_vs_eps_cagr=gap_eps,
        contrarian_insight_cz=insight,
    )
