"""Unit tests for the Reverse DCF solver, sensitivity grids, and core pure functions."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.dcf import (
    ReverseDCFInputs,
    build_implied_growth_grid,
    build_intrinsic_value_grid,
    compute_cagr,
    compute_dcf_value_per_share,
    evaluate_growth_verdict,
    solve_implied_growth,
    HistoricalCAGRs,
)
from analysis.news_consensus import filter_hard_news
from data.models import NewsItem


class TestReverseDCFSolver(unittest.TestCase):
    """Verify Reverse DCF mathematical accuracy, round-trip convergence, and edge cases."""

    def test_perpetual_growth_round_trip(self) -> None:
        """Solved g must reproduce the target market price within tight numerical tolerance."""
        fcf_0 = 10_000_000_000.0  # $10B
        net_debt = 15_000_000_000.0  # $15B
        shares = 1_000_000_000.0  # 1B diluted shares
        r = 0.12
        n = 10
        g_term = 0.025
        true_g = 0.145

        market_price, _, _ = compute_dcf_value_per_share(
            fcf_0=fcf_0,
            growth_rate=true_g,
            discount_rate=r,
            horizon_years=n,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method="perpetual_growth",
            perpetual_growth_rate=g_term,
        )

        solved_g, err = solve_implied_growth(
            current_price=market_price,
            fcf_0=fcf_0,
            discount_rate=r,
            horizon_years=n,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method="perpetual_growth",
            perpetual_growth_rate=g_term,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(solved_g)
        assert solved_g is not None
        self.assertAlmostEqual(solved_g, true_g, places=6)

        recomputed_price, _, _ = compute_dcf_value_per_share(
            fcf_0=fcf_0,
            growth_rate=solved_g,
            discount_rate=r,
            horizon_years=n,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method="perpetual_growth",
            perpetual_growth_rate=g_term,
        )
        self.assertAlmostEqual(recomputed_price, market_price, places=4)

    def test_exit_multiple_round_trip(self) -> None:
        """Exit-multiple terminal method must also round-trip accurately with net cash (negative net debt)."""
        fcf_0 = 5_000_000_000.0
        net_debt = -20_000_000_000.0  # $20B net cash
        shares = 500_000_000.0
        r = 0.10
        n = 8
        exit_mult = 15.0
        true_g = 0.09

        market_price, _, _ = compute_dcf_value_per_share(
            fcf_0=fcf_0,
            growth_rate=true_g,
            discount_rate=r,
            horizon_years=n,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method="exit_multiple",
            exit_multiple=exit_mult,
        )

        solved_g, err = solve_implied_growth(
            current_price=market_price,
            fcf_0=fcf_0,
            discount_rate=r,
            horizon_years=n,
            net_debt=net_debt,
            diluted_shares=shares,
            terminal_method="exit_multiple",
            exit_multiple=exit_mult,
        )
        self.assertIsNone(err)
        assert solved_g is not None
        self.assertAlmostEqual(solved_g, true_g, places=6)

    def test_negative_fcf_returns_explicit_czech_error(self) -> None:
        """Negative FCF_0 must return None and an explicit Czech explanation."""
        solved_g, err = solve_implied_growth(
            current_price=120.0,
            fcf_0=-500_000_000.0,
            discount_rate=0.12,
            horizon_years=10,
            net_debt=0.0,
            diluted_shares=100_000_000.0,
        )
        self.assertIsNone(solved_g)
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("záporný", err.lower())

    def test_discount_rate_le_perpetual_growth_fails_gracefully(self) -> None:
        """When r <= g_term in Gordon mode, solver must return None with a clear error."""
        solved_g, err = solve_implied_growth(
            current_price=100.0,
            fcf_0=1_000_000_000.0,
            discount_rate=0.025,
            horizon_years=10,
            net_debt=0.0,
            diluted_shares=100_000_000.0,
            terminal_method="perpetual_growth",
            perpetual_growth_rate=0.025,
        )
        self.assertIsNone(solved_g)
        self.assertIsNotNone(err)

    def test_unsolvable_extreme_price_returns_error(self) -> None:
        """An absurdly high price requiring >400% annual growth must return an explicit no-solution message."""
        solved_g, err = solve_implied_growth(
            current_price=1_000_000_000.0,
            fcf_0=1_000_000.0,
            discount_rate=0.12,
            horizon_years=5,
            net_debt=0.0,
            diluted_shares=100_000_000.0,
        )
        self.assertIsNone(solved_g)
        self.assertIsNotNone(err)

    def test_sensitivity_grids_shapes_and_monotonicity(self) -> None:
        """Intrinsic value increases with growth and decreases with discount rate."""
        params = ReverseDCFInputs(discount_rate=0.12, horizon_years=10)
        g_vals, r_vals, mat = build_intrinsic_value_grid(
            fcf_0=10e9,
            net_debt=5e9,
            diluted_shares=1e9,
            params=params,
            center_growth=0.10,
            x_axis_mode="discount_rate",
        )
        self.assertEqual(mat.shape, (len(g_vals), len(r_vals)))
        # Higher growth -> higher value (down columns)
        self.assertTrue(np.all(np.diff(mat[:, 3]) > 0))
        # Higher discount rate -> lower value (across rows)
        self.assertTrue(np.all(np.diff(mat[3, :]) < 0))

        # Implied growth grid: higher price -> higher implied growth
        p_vals, r_vals2, mat_g = build_implied_growth_grid(
            current_price=150.0,
            fcf_0=10e9,
            net_debt=5e9,
            diluted_shares=1e9,
            params=params,
        )
        self.assertEqual(mat_g.shape, (len(p_vals), len(r_vals2)))
        self.assertTrue(np.all(np.diff(mat_g[:, 3]) > 0))

    def test_cagr_and_verdict_classification(self) -> None:
        """Verify CAGR math and Realistic / Ambitious / Extreme verdict thresholds."""
        series = [(2019, 100.0), (2020, 110.0), (2021, 121.0), (2022, 133.1), (2023, 146.41), (2024, 161.051)]
        cagr_5y = compute_cagr(series, 5)
        self.assertIsNotNone(cagr_5y)
        assert cagr_5y is not None
        self.assertAlmostEqual(cagr_5y, 0.10, places=4)

        cagrs = HistoricalCAGRs(revenue_cagr_5y=0.10, fcf_cagr_5y=0.11)
        self.assertEqual(evaluate_growth_verdict(0.10, cagrs)[0], "realistic")
        self.assertEqual(evaluate_growth_verdict(0.16, cagrs)[0], "ambitious")
        self.assertEqual(evaluate_growth_verdict(0.28, cagrs)[0], "extreme")

    def test_hard_news_filter_removes_clickbait(self) -> None:
        """Hard news filter must reject clickbait and keep corporate/earnings news."""
        sample_news = [
            NewsItem("2026-10-01", "Apple stock drops 2% – here's why investors are worried", "Blog", "https://a.test"),
            NewsItem("2026-10-02", "Should you buy AAPL stock right now? 3 stocks to retire rich", "Motley Fool", "https://b.test"),
            NewsItem("2026-10-03", "Apple reports Q4 2026 earnings and raises FY27 revenue guidance", "Reuters", "https://c.test"),
            NewsItem("2026-10-04", "Company announces $5B acquisition of AI chip startup", "Bloomberg", "https://d.test"),
        ]
        res = filter_hard_news(sample_news)
        self.assertEqual(len(res.hard_news), 2)
        self.assertEqual(res.filtered_out_count, 2)
        titles = [n.title for n in res.hard_news]
        self.assertIn("Apple reports Q4 2026 earnings and raises FY27 revenue guidance", titles)
        self.assertIn("Company announces $5B acquisition of AI chip startup", titles)


if __name__ == "__main__":
    unittest.main()
