"""Unit tests for the Crystal Ball Monte Carlo engine, Triangulation, Forensics, and Decision Engine."""

from __future__ import annotations

import unittest

from analysis.capital_allocation import analyze_insider_activity
from analysis.crystal_ball import compute_triangulated_fair_value, run_monte_carlo_crystal_ball
from analysis.dcf import ReverseDCFInputs, run_reverse_dcf
from analysis.decision_engine import run_decision_engine
from analysis.forensics import run_forensics_analysis
from analysis.quality import run_quality_xray
from analysis.red_flags import run_red_flags_analysis
from data.models import AnnualFinancials, CompanyData


def _make_sample_company(current_price: float = 200.0, manipulative: bool = False) -> CompanyData:
    """Create a realistic 6-year synthetic company for deterministic quantitative tests."""
    annuals = []
    for i in range(6):
        rev = 10e9 * (1.12 ** i)
        gp = rev * ((0.58 + i * 0.005) if not manipulative else max(0.30, 0.60 - i * 0.05))
        ebit = rev * ((0.26 + i * 0.006) if not manipulative else 0.12)
        ni = rev * ((0.20 + i * 0.006) if not manipulative else 0.18)
        ocf = rev * (0.26 if not manipulative else 0.02)
        capex = rev * 0.05
        fcf = ocf - capex
        da = rev * 0.04
        rec = rev * (0.10 if not manipulative else 0.10 * (1.45 ** i))
        ca = rev * ((0.44 + i * 0.01) if not manipulative else 0.15)
        cl = rev * (0.20 if not manipulative else 0.55)
        ppe = rev * 0.35
        ta = rev * ((1.45 - i * 0.02) if not manipulative else 1.40)
        tot_liab = rev * (0.40 if not manipulative else 1.25)
        ret_earn = rev * (0.65 if not manipulative else -0.35)
        annuals.append(
            AnnualFinancials(
                year=2020 + i,
                date=f"{2020 + i}-12-31",
                currency="USD",
                revenue=rev,
                gross_profit=gp,
                operating_income=ebit,
                interest_expense=0.15e9,
                income_before_tax=ebit - 0.15e9,
                income_tax_expense=(ebit - 0.15e9) * 0.21,
                net_income=ni,
                ebitda=ebit + da,
                rd_expense=rev * 0.08,
                sga_expense=rev * 0.15,
                depreciation_amortization=da,
                eps_diluted=ni / (200e6 * (0.985 ** i)),
                weighted_avg_shares_diluted=200e6 * (0.985 ** i),
                cash_and_short_term=rev * 0.25,
                accounts_receivable=rec,
                current_assets=ca,
                ppe_net=ppe,
                total_assets=ta,
                current_liabilities=cl,
                total_liabilities=tot_liab,
                retained_earnings=ret_earn,
                short_term_debt=0.5e9,
                long_term_debt=2.0e9 if not manipulative else 12.0e9,
                total_debt=2.5e9 if not manipulative else 12.5e9,
                total_equity=ta - tot_liab,
                operating_cash_flow=ocf,
                capex=capex,
                free_cash_flow=fcf,
                stock_based_compensation=rev * 0.015,
                dividends_paid=rev * 0.02,
                share_buybacks=rev * 0.04,
                pe_ratio=20.0,
                pfcf_ratio=19.0,
                ev_to_ebitda=14.0,
                market_cap_year=ni * 20.0,
                avg_share_price=(ni * 20.0) / (200e6 * (0.985 ** i)),
            )
        )

    latest = annuals[-1]
    shares = latest.weighted_avg_shares_diluted or 185e6
    return CompanyData(
        ticker="QUAL" if not manipulative else "TRAP",
        name="Quality Compounder Inc." if not manipulative else "Value Trap Corp.",
        exchange="NASDAQ",
        sector="Technology",
        industry="Software",
        country="US",
        quote_currency="USD",
        reporting_currency="USD",
        fx_rate_reporting_to_quote=1.0,
        current_price=current_price,
        market_cap=current_price * shares,
        beta=1.0,
        diluted_shares_current=shares,
        ttm_fcf=latest.free_cash_flow,
        ttm_sbc=latest.stock_based_compensation,
        ttm_revenue=latest.revenue,
        ttm_net_income=latest.net_income,
        ttm_ebitda=latest.ebitda,
        ttm_ebit=latest.operating_income,
        ttm_ocf=latest.operating_cash_flow,
        latest_cash=latest.cash_and_short_term,
        latest_total_debt=latest.total_debt,
        latest_net_debt=(latest.total_debt or 0.0) - (latest.cash_and_short_term or 0.0),
        risk_free_rate=0.042,
        risk_free_source="10Y Treasury",
        annuals=annuals,
        provider_name="TestProvider",
    )


class TestCrystalBallAndDecisionEngine(unittest.TestCase):
    """Verify Piotroski, Altman, Beneish, Monte Carlo, Triangulation, and Automated Verdict."""

    def test_forensics_clean_company_scores_strong(self) -> None:
        company = _make_sample_company(current_price=220.0, manipulative=False)
        forensics = run_forensics_analysis(company)

        self.assertIsNotNone(forensics.piotroski.score)
        assert forensics.piotroski.score is not None
        self.assertGreaterEqual(forensics.piotroski.score, 6)

        self.assertIsNotNone(forensics.altman.z_score)
        assert forensics.altman.z_score is not None
        self.assertGreater(forensics.altman.z_score, 2.99)
        self.assertEqual(forensics.altman.status, "green")

        self.assertIsNotNone(forensics.beneish.m_score)
        assert forensics.beneish.m_score is not None
        self.assertLess(forensics.beneish.m_score, -1.78)
        self.assertEqual(forensics.beneish.status, "green")

        self.assertIsNotNone(forensics.compounding.intrinsic_compounding_rate)

    def test_forensics_trap_company_triggers_veto(self) -> None:
        trap = _make_sample_company(current_price=200.0, manipulative=True)
        forensics = run_forensics_analysis(trap)
        self.assertEqual(forensics.altman.status, "red")
        self.assertEqual(forensics.beneish.status, "red")

        params = ReverseDCFInputs()
        dcf = run_reverse_dcf(trap, params)
        rf = run_red_flags_analysis(trap)
        qual = run_quality_xray(trap)
        mc = run_monte_carlo_crystal_ball(trap, horizon_years=3, hurdle_rate=0.12)
        tri = compute_triangulated_fair_value(trap, params, qual, forensics.compounding, mc)
        ins = analyze_insider_activity(trap)

        verdict = run_decision_engine(trap, dcf, rf, qual, forensics, mc, tri, ins)
        self.assertTrue(verdict.veto_triggered)
        self.assertEqual(verdict.verdict_code, "avoid")
        self.assertLessEqual(verdict.total_score, 29)

    def test_monte_carlo_and_triangulation_monotonic_with_price(self) -> None:
        """Lower purchase price must strictly improve Monte Carlo IRR, beat probability, and verdict score."""
        cheap = _make_sample_company(current_price=180.0, manipulative=False)
        expensive = _make_sample_company(current_price=650.0, manipulative=False)
        params = ReverseDCFInputs(discount_rate=0.12)

        mc_cheap = run_monte_carlo_crystal_ball(cheap, horizon_years=3, hurdle_rate=0.12, n_simulations=4000)
        mc_exp = run_monte_carlo_crystal_ball(expensive, horizon_years=3, hurdle_rate=0.12, n_simulations=4000)

        self.assertTrue(mc_cheap.available)
        self.assertTrue(mc_exp.available)
        self.assertEqual(len(mc_cheap.fan_df), 4)  # Year 0, +1, +2, +3
        self.assertGreater(mc_cheap.p50_irr, mc_exp.p50_irr)
        self.assertGreater(mc_cheap.prob_beat_hurdle, mc_exp.prob_beat_hurdle)

        # Full decision engine comparison
        def _score_for(co: CompanyData, mc_res) -> int:
            dcf = run_reverse_dcf(co, params)
            rf = run_red_flags_analysis(co)
            qual = run_quality_xray(co)
            fore = run_forensics_analysis(co)
            tri = compute_triangulated_fair_value(co, params, qual, fore.compounding, mc_res, margin_of_safety_target=0.25)
            self.assertEqual(tri.valid_models_count, 4)
            self.assertIsNotNone(tri.max_buy_price)
            ins = analyze_insider_activity(co)
            return run_decision_engine(co, dcf, rf, qual, fore, mc_res, tri, ins).total_score

        self.assertGreater(_score_for(cheap, mc_cheap), _score_for(expensive, mc_exp))


if __name__ == "__main__":
    unittest.main()
