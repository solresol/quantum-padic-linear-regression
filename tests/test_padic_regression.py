"""Tests for the padic_regression library."""

import pytest
from padic_regression import (
    DataPoint,
    padic_valuation,
    valuation_sum,
    bits_needed,
    tournament_winner,
    run_tournament,
    find_optimal,
    find_optimal_monte_carlo,
    brute_force_optimal,
    analyze_tournament,
)


class TestCore:
    """Tests for core p-adic functions."""

    def test_padic_valuation_basic(self):
        """Test basic 2-adic valuation."""
        assert padic_valuation(1) == 0   # 1 is odd
        assert padic_valuation(2) == 1   # 2 = 2^1
        assert padic_valuation(4) == 2   # 4 = 2^2
        assert padic_valuation(8) == 3   # 8 = 2^3
        assert padic_valuation(12) == 2  # 12 = 4 * 3 = 2^2 * 3

    def test_padic_valuation_zero(self):
        """Zero has infinite valuation."""
        assert padic_valuation(0) >= 1000

    def test_padic_valuation_negative(self):
        """Valuation works for negative numbers."""
        assert padic_valuation(-8) == 3

    def test_padic_valuation_other_primes(self):
        """Test p-adic valuation for p != 2."""
        assert padic_valuation(9, p=3) == 2   # 9 = 3^2
        assert padic_valuation(18, p=3) == 2  # 18 = 2 * 3^2
        assert padic_valuation(25, p=5) == 2  # 25 = 5^2

    def test_bits_needed(self):
        """Test bit calculation."""
        assert bits_needed(1) == 1
        assert bits_needed(2) == 2
        assert bits_needed(3) == 2
        assert bits_needed(7) == 3
        assert bits_needed(8) == 4
        assert bits_needed(255) == 8
        assert bits_needed(256) == 9

    def test_valuation_sum_perfect_fit(self):
        """Perfect fit should have high valuation sum."""
        data = [DataPoint(1, 5), DataPoint(2, 10)]
        val = valuation_sum(data, m=5, b=0)
        assert val >= 2000  # Two points with zero residual

    def test_valuation_sum_imperfect_fit(self):
        """Imperfect fit has lower valuation."""
        data = [DataPoint(1, 5), DataPoint(2, 10)]
        val = valuation_sum(data, m=4, b=0)  # Wrong slope
        assert val < 2000


class TestTournament:
    """Tests for tournament algorithm."""

    def test_tournament_winner_basic(self):
        """Test basic winner selection."""
        data = [DataPoint(1, 2), DataPoint(2, 4)]
        # Coefficient 2 is perfect fit
        assert tournament_winner(data, 2, 3) == 2
        assert tournament_winner(data, 1, 2) == 2

    def test_run_tournament_perfect_data(self):
        """Tournament finds optimal for perfect data."""
        data = [DataPoint(1, 5), DataPoint(2, 10)]
        # From any start, should reach 5
        assert run_tournament(data, 3, start=0) == 5
        assert run_tournament(data, 3, start=7) == 5
        assert run_tournament(data, 3, start=3) == 5

    def test_find_optimal_perfect_data(self):
        """find_optimal finds the correct coefficient."""
        data = [DataPoint(1, 7), DataPoint(2, 14), DataPoint(3, 21)]
        optimal = find_optimal(data, coeff_bits=4)
        assert optimal == 7

    def test_find_optimal_matches_brute_force(self):
        """Tournament result matches brute force."""
        data = [DataPoint(1, 3), DataPoint(2, 6)]
        tournament_result = find_optimal(data, coeff_bits=3)
        bf_result, _ = brute_force_optimal(data, coeff_bits=3)
        assert tournament_result == bf_result

    def test_monte_carlo_finds_optimal(self):
        """Monte Carlo approach finds optimal."""
        data = [DataPoint(1, 9), DataPoint(2, 18)]
        winner, counts = find_optimal_monte_carlo(data, 4, n_samples=10)
        assert winner == 9

    def test_analyze_tournament_structure(self):
        """Test tournament analysis."""
        data = [DataPoint(1, 2), DataPoint(2, 4)]
        analysis = analyze_tournament(data, coeff_bits=2)

        assert 'valuations' in analysis
        assert 'basins' in analysis
        assert 'brute_force_optimal' in analysis
        assert analysis['brute_force_optimal'] == 2


class TestScaling:
    """Tests for scaling behavior."""

    @pytest.mark.parametrize("bits", [4, 6, 8, 10])
    def test_scaling_perfect_data(self, bits):
        """Tournament works for various bit sizes."""
        true_a = (1 << bits) // 2 + 1
        data = [DataPoint(1, true_a), DataPoint(2, 2 * true_a)]

        result = find_optimal(data, coeff_bits=bits)
        assert result == true_a

    def test_large_coefficient_space(self):
        """Test with large (16-bit) coefficient space."""
        true_a = 12345
        data = [DataPoint(1, true_a), DataPoint(2, 2 * true_a)]

        result = find_optimal(data, coeff_bits=16)
        assert result == true_a


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_data_point(self):
        """Works with single data point."""
        data = [DataPoint(2, 8)]
        optimal = find_optimal(data, coeff_bits=3)
        # a=4 gives y=8 exactly
        assert optimal == 4

    def test_datapoint_unpacking(self):
        """DataPoint supports unpacking."""
        pt = DataPoint(3, 7)
        x, y = pt
        assert x == 3
        assert y == 7

    def test_zero_coefficient(self):
        """Handles coefficient = 0."""
        data = [DataPoint(1, 0), DataPoint(2, 0)]
        optimal = find_optimal(data, coeff_bits=2)
        assert optimal == 0


class TestOtherPrimes:
    """Tests for primes other than 2."""

    def test_3adic_valuation(self):
        """3-adic valuation works."""
        data = [DataPoint(1, 3), DataPoint(2, 6)]
        # Perfect fit with m=3
        val = valuation_sum(data, m=3, b=0, p=3)
        assert val >= 2000

    def test_3adic_brute_force(self):
        """Brute force works for 3-adic (tournament is 2-adic specific)."""
        data = [DataPoint(1, 3), DataPoint(2, 6)]
        # Note: Tournament uses binary bit-flipping, so it's 2-adic specific
        # For 3-adic, use brute force
        optimal, val = brute_force_optimal(data, coeff_bits=3, p=3)
        assert optimal == 3
