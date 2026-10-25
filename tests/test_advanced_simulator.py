"""
test_advanced_simulator.py
==========================
Comprehensive test suite for the Quantum Tensor RBM package.

Tests cover:
  - Truncated SVD: right/left singular vectors, reconstruction correctness
  - Frobenius-norm truncation error
  - Entanglement analysis: Von Neumann and Rényi entropy
  - Analytical GHZ state entanglement verification
  - Entanglement growth classification: area-law vs. volume-law
  - Mutual information calculations
"""

import math

from simulator.entanglement import (
    classify_entanglement,
    ghz_entanglement_entropy,
    mutual_information,
    renyi_entropy,
    truncation_error,
    von_neumann_entropy,
)
from simulator.svd import truncated_svd


class TestSVD:
    def test_identity_svd(self):
        A = [[complex(1 if i == j else 0) for j in range(3)] for i in range(3)]
        s, U, V = truncated_svd(A)
        assert len(s) == 3
        for val in s:
            assert abs(val - 1.0) < 1e-7

    def test_reconstruction_exact(self):
        # A simple 2x2 complex matrix
        A = [[complex(1, 2), complex(3, -1)], [complex(0, 1), complex(-2, 4)]]
        s, U, V = truncated_svd(A)
        # Reconstruct: U @ diag(s) @ V†
        # V is list of eigenvectors v_k, so V† elements are v_k[j].conjugate()
        reconstructed = [[complex(0) for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(len(s)):
                    reconstructed[i][j] += U[k][i] * s[k] * V[k][j].conjugate()

        # Check equivalence
        for i in range(2):
            for j in range(2):
                assert abs(A[i][j] - reconstructed[i][j]) < 1e-4

    def test_truncation_k(self):
        A = [[complex(1, 1), complex(2, 0)], [complex(0, -1), complex(3, 4)]]
        s_all, _, _ = truncated_svd(A)
        s_tr, _, _ = truncated_svd(A, keep_k=1)
        assert len(s_tr) == 1
        assert abs(s_tr[0] - s_all[0]) < 1e-7


class TestEntanglement:
    def test_entropy_pure_state_is_zero(self):
        # A product state has a single singular value = 1.0
        assert von_neumann_entropy([1.0]) == 0.0

    def test_entropy_bell_state(self):
        # Bell state has singular values [1/√2, 1/√2]
        s = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        # S = log(2) ≈ 0.693 nats
        assert abs(von_neumann_entropy(s) - math.log(2.0)) < 1e-7

    def test_renyi_order_two(self):
        s = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        # S_2 = -log(∑ λ_i⁴) = -log( (1/2)² + (1/2)² ) = -log(0.5) = log(2)
        assert abs(renyi_entropy(s, alpha=2.0) - math.log(2.0)) < 1e-7

    def test_truncation_error_calculation(self):
        s = [0.8, 0.5, 0.3, 0.1]
        err = truncation_error(s, keep_k=2)
        # Discarded: 0.3, 0.1
        # Error = 0.3^2 + 0.1^2 = 0.09 + 0.01 = 0.10
        assert abs(err - 0.10) < 1e-7

    def test_mutual_information_subadditivity(self):
        # Bipartite state: S(A) = 0.6, S(B) = 0.6, S(AB) = 0.4
        # I(A:B) = 0.6 + 0.6 - 0.4 = 0.8
        assert abs(mutual_information(0.6, 0.6, 0.4) - 0.8) < 1e-7

    def test_ghz_reference_entropy(self):
        # GHZ state analytical result must be log(2) for all valid bipartitions
        for size in [1, 2, 3]:
            assert abs(ghz_entanglement_entropy(4, size) - math.log(2.0)) < 1e-7

    def test_classify_area_law(self):
        # Area-law: S(L) is roughly constant
        entropies = [0.693, 0.694, 0.692, 0.693]
        assert classify_entanglement(entropies, system_size=8) == "area-law"

    def test_classify_volume_law(self):
        # Volume-law: S(L) scales linearly with L
        entropies = [0.5, 1.0, 1.5, 2.0]
        assert classify_entanglement(entropies, system_size=8) == "volume-law"
