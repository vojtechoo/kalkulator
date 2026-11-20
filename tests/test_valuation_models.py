"""Unit tests for the full GuruFocus-style mathematical valuation suite and qualitative scorecard."""

from __future__ import annotations

import unittest

from analysis.capital_allocation import analyze_insider_activity
from analysis.crystal_ball import run_monte_carlo_crystal_ball
from analysis.dcf import (
    ReverseDCFInputs,
    compute_growth_optionality_per_share,
    compute_two_stage_fading_dcf_value_per_share,
    diagnose_capex_distortion,
    select_fcf_base,
)
from analysis.forensics import run_forensics_analysis
from analysis.legend_investors import evaluate_legendary_investors
from analysis.qualitative import (
    MOAT_SOURCES_CZ,
    OFF_STATEMENT_RISKS_CZ,
    evaluate_qualitative_scorecard,
    infer_qualitative_defaults,
)
from analysis.quality import run_quality_xray
from analysis.red_flags import run_red_flags_analysis
from analysis.valuation_models import (
    compute_cashflow_decomposition,
    compute_full_valuation_suite,
)
from tests.test_crystal_ball import _make_sample_company


class TestGuruFocusValuationSuite(unittest.TestCase):
    """Verify all 13 mathematical formulas, GF Value scorecards, and qualitative evaluation."""

    def test_cashflow_decomposition_and_owner_earnings_delta_wc(self) -> None:
        company = _make_sample_company(current_price=220.0, manipulative=False)
        decomp = compute_cashflow_decomposition(company)

        self.assertIsNotNone(decomp.fcf)
        self.assertIsNotNone(decomp.fcf_adj)
        self.assertIsNotNone(decomp.owner_earnings)
        self.assertIsNotNone(decomp.delta_working_capital)
        assert decomp.ocf is not None and decomp.capex is not None and decomp.sbc is not None
        assert decomp.fcf is not None and decomp.fcf_adj is not None
        self.assertAlmostEqual(decomp.fcf, decomp.ocf - decomp.capex, places=2)
        self.assertAlmostEqual(decomp.fcf_adj, decomp.fcf - decomp.sbc, places=2)

    def test_all_8_valuation_models_and_mos_buy_price(self) -> None:
        company = _make_sample_company(current_price=220.0, manipulative=False)
        params = ReverseDCFInputs(discount_rate=0.12, horizon_years=10)
        rf = run_red_flags_analysis(company)
        qual = run_quality_xray(company)
        fore = run_forensics_analysis(company)
        mc = run_monte_carlo_crystal_ball(company, horizon_years=3, hurdle_rate=0.12, n_simulations=2000)

        suite = compute_full_valuation_suite(
            company=company,
            dcf_params=params,
            quality=qual,
            red_flags=rf,
            forensics=fore,
            monte_carlo=mc,
            margin_of_safety=0.25,
            aaa_yield_override_pct=5.2,
        )

        # All 8 models must exist and produce positive values for this dividend-paying quality company
        self.assertEqual(len(suite.models), 8)
        keys = {m.key for m in suite.models}
        self.assertEqual(
            keys,
            {"dcf_gordon", "dcf_exit", "owner_earnings", "epv", "graham", "ddm", "rel_pe", "rel_pfcf"},
        )
        for m in suite.models:
            self.assertIsNotNone(m.value_per_share, f"Model {m.key} returned None")
            assert m.value_per_share is not None and m.buy_price_mos is not None
            self.assertAlmostEqual(m.buy_price_mos, m.value_per_share * 0.75, places=4)

        self.assertIsNotNone(suite.gf_fair_value)
        self.assertIsNotNone(suite.gf_buy_price)
        assert suite.gf_fair_value is not None and suite.gf_buy_price is not None
        self.assertAlmostEqual(suite.gf_buy_price, suite.gf_fair_value * 0.75, places=4)
        self.assertEqual(len(suite.gf_pillars), 5)
        self.assertFalse(suite.gf_value_history_df.empty)
        self.assertIn("2032", list(suite.gf_value_history_df["year"]))
        self.assertGreaterEqual(suite.predictability_stars, 3.5)

    def test_qualitative_defaults_and_risk_deductions(self) -> None:
        company = _make_sample_company(current_price=220.0, manipulative=False)
        params = ReverseDCFInputs()
        rf = run_red_flags_analysis(company)
        qual = run_quality_xray(company)
        fore = run_forensics_analysis(company)
        mc = run_monte_carlo_crystal_ball(company, horizon_years=3, hurdle_rate=0.12, n_simulations=1000)
        suite = compute_full_valuation_suite(company, params, qual, rf, fore, mc)
        ins = analyze_insider_activity(company)

        defaults = infer_qualitative_defaults(qual, fore, ins, suite)
        self.assertGreaterEqual(defaults.moat_score, 4)

        clean_eval = evaluate_qualitative_scorecard(
            moat_rating=5,
            selected_moat_sources=[MOAT_SOURCES_CZ[0], MOAT_SOURCES_CZ[1]],
            pricing_power_rating=5,
            management_rating=4,
            predictability_rating=5,
            porter_ratings={"customers": 4, "suppliers": 4, "barriers": 5, "substitutes": 4, "rivalry": 4},
            selected_risks=[],
        )
        risky_eval = evaluate_qualitative_scorecard(
            moat_rating=5,
            selected_moat_sources=[MOAT_SOURCES_CZ[0], MOAT_SOURCES_CZ[1]],
            pricing_power_rating=5,
            management_rating=4,
            predictability_rating=5,
            porter_ratings={"customers": 4, "suppliers": 4, "barriers": 5, "substitutes": 4, "rivalry": 4},
            selected_risks=list(OFF_STATEMENT_RISKS_CZ),
        )
        self.assertGreater(clean_eval.total_score, risky_eval.total_score)
        self.assertEqual(len(clean_eval.pillar_scores), 6)

    def test_capex_normalization_safeguard_and_custom_fcf_override(self) -> None:
        """Extreme one-off TTM CapEx spike must trigger safeguard warning and normalize cleanly."""
        co = _make_sample_company(current_price=220.0, manipulative=False)
        # Simulate massive AI/datacenter CapEx surge: 20% of revenue vs 5% historical average (4x spike!)
        assert co.ttm_revenue is not None and co.ttm_ocf is not None
        co.ttm_capex = co.ttm_revenue * 0.25
        co.ttm_fcf = co.ttm_ocf - co.ttm_capex  # Nearly zero / negative after SBC

        raw_params = ReverseDCFInputs(normalize_capex=False, custom_fcf_0_millions=0.0)
        diag = diagnose_capex_distortion(co, raw_params)
        self.assertTrue(diag.should_warn)
        self.assertIsNotNone(diag.spike_ratio)
        assert diag.spike_ratio is not None
        self.assertGreaterEqual(diag.spike_ratio, 1.5)
        self.assertIn("CapEx", diag.warning_cz)

        # Normalized FCF must recover sharply above raw FCF
        raw_fcf, _ = select_fcf_base(co, "fcf_minus_sbc", normalize_capex=False)
        norm_fcf, norm_lbl = select_fcf_base(co, "fcf_minus_sbc", normalize_capex=True)
        assert raw_fcf is not None and norm_fcf is not None
        self.assertGreater(norm_fcf, raw_fcf * 2.0)
        self.assertIn("Normalizované", norm_lbl)

        # Custom FCF_0 override in millions must be applied verbatim
        custom_fcf, custom_lbl = select_fcf_base(co, "fcf_minus_sbc", custom_fcf_0_millions=12_500.0)
        self.assertEqual(custom_fcf, 12_500_000_000.0)
        self.assertIn("Vlastní", custom_lbl)

    def test_one_year_target_and_legendary_investors_scorecard(self) -> None:
        """1-year projected price and all 5 legendary investor checklists must compute cleanly."""
        co = _make_sample_company(current_price=220.0, manipulative=False)
        params = ReverseDCFInputs()
        rf = run_red_flags_analysis(co)
        qual = run_quality_xray(co)
        fore = run_forensics_analysis(co)
        mc = run_monte_carlo_crystal_ball(co, horizon_years=3, hurdle_rate=0.12, n_simulations=1500)
        suite = compute_full_valuation_suite(co, params, qual, rf, fore, mc, margin_of_safety=0.0)
        ins = analyze_insider_activity(co)

        self.assertIsNotNone(mc.p50_price_1y)
        self.assertIsNotNone(mc.p10_price_1y)
        self.assertIsNotNone(mc.p90_price_1y)
        assert mc.p10_price_1y is not None and mc.p50_price_1y is not None and mc.p90_price_1y is not None
        self.assertLessEqual(mc.p10_price_1y, mc.p50_price_1y)
        self.assertLessEqual(mc.p50_price_1y, mc.p90_price_1y)

        legends = evaluate_legendary_investors(co, qual, rf, fore, suite, ins)
        self.assertEqual(len(legends.investors), 5)
        self.assertGreaterEqual(legends.avg_score_pct, 60)

    def test_two_stage_fade_growth_optionality_and_graham_weight(self) -> None:
        """Verify 2-stage linear fade DCF, exact Growth Optionality formula, and 0%/100% Graham weight."""
        # 1. Exact Growth Optionality math check:
        # FCF_proj = 10B, g = 2.5%, r = 12%, ramp = 5y, prob = 50%, shares = 1B
        opt_val, err = compute_growth_optionality_per_share(
            fcf_billions=10.0,
            ramp_years=5,
            probability=0.50,
            discount_rate=0.12,
            perpetual_growth_rate=0.025,
            diluted_shares=1_000_000_000.0,
        )
        self.assertIsNone(err)
        expected_tv = (10e9 * 1.025) / (0.12 - 0.025)
        expected_opt = (expected_tv / (1.12 ** 5)) * 0.50 / 1e9
        assert opt_val is not None
        self.assertAlmostEqual(opt_val, expected_opt, places=6)

        # 2. 2-Stage Fading DCF: higher stage1 growth yields strictly higher value
        v_low, _, _ = compute_two_stage_fading_dcf_value_per_share(
            fcf_0=10e9,
            stage1_growth_rate=0.10,
            discount_rate=0.12,
            horizon_years=10,
            net_debt=0.0,
            diluted_shares=1e9,
            perpetual_growth_rate=0.025,
        )
        v_high, _, _ = compute_two_stage_fading_dcf_value_per_share(
            fcf_0=10e9,
            stage1_growth_rate=0.30,
            discount_rate=0.12,
            horizon_years=10,
            net_debt=0.0,
            diluted_shares=1e9,
            perpetual_growth_rate=0.025,
        )
        self.assertGreater(v_high, v_low)

        # 3. Full suite integration: Core + Optionality and Graham weight 0% vs 100%
        co = _make_sample_company(current_price=220.0, manipulative=False)
        rf = run_red_flags_analysis(co)
        qual = run_quality_xray(co)
        fore = run_forensics_analysis(co)
        mc = run_monte_carlo_crystal_ball(co, horizon_years=3, hurdle_rate=0.12, n_simulations=1000)

        opt_params = ReverseDCFInputs(
            discount_rate=0.12,
            horizon_years=10,
            perpetual_growth_rate=0.025,
            stage1_growth_rate=0.25,
            enable_growth_optionality=True,
            optionality_fcf_billions=5.0,
            optionality_ramp_years=5,
            optionality_probability=0.50,
        )
        suite_no_graham = compute_full_valuation_suite(
            co, opt_params, qual, rf, fore, mc, margin_of_safety=0.0, graham_weight_pct=0.0
        )
        self.assertTrue(suite_no_graham.optionality_active)
        self.assertTrue(suite_no_graham.used_two_stage_fade)
        dcf_g_row = next(m for m in suite_no_graham.models if m.key == "dcf_gordon")
        assert dcf_g_row.value_per_share is not None and suite_no_graham.core_dcf_gordon_per_share is not None
        self.assertAlmostEqual(
            dcf_g_row.value_per_share,
            suite_no_graham.core_dcf_gordon_per_share + suite_no_graham.optionality_per_share,
            places=5,
        )

        # 100% Graham weight must set gf_fair_value == graham model value
        suite_100_graham = compute_full_valuation_suite(
            co, opt_params, qual, rf, fore, mc, margin_of_safety=0.0, graham_weight_pct=100.0
        )
        graham_row = next(m for m in suite_100_graham.models if m.key == "graham")
        assert suite_100_graham.gf_fair_value is not None and graham_row.value_per_share is not None
        self.assertAlmostEqual(suite_100_graham.gf_fair_value, graham_row.value_per_share, places=5)
        self.assertNotAlmostEqual(suite_no_graham.gf_fair_value or 0.0, suite_100_graham.gf_fair_value, places=2)


if __name__ == "__main__":
    unittest.main()
