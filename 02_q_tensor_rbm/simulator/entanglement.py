"""
entanglement.py
===============
Quantum entanglement analysis for multi-state register systems.

Computes:
  - Von Neumann entropy S(ρ) = -Tr(ρ log ρ) from Schmidt decomposition
  - Mutual information I(A:B) between bipartitions
  - Entanglement classification (area-law vs. volume-law)
  - Rényi entropy S_α for comparison with MPS truncation error

Reference:
  - Vidal, "Efficient Classical Simulation of Slightly Entangled Quantum Computations"
    PRL 91, 147902 (2003)
  - Eisert, "Colloquium: Area laws for the entanglement entropy"
    Rev. Mod. Phys. 82, 277 (2010)
"""

from __future__ import annotations

import math
import cmath
from typing import List, Optional, Tuple


def von_neumann_entropy(singular_values: List[float], eps: float = 1e-12) -> float:
    """
    Computes Von Neumann entanglement entropy from Schmidt singular values.

    S(ρ) = -∑_i λ_i² log(λ_i²)

    where λ_i are the Schmidt coefficients (singular values normalised so that
    ∑ λ_i² = 1).

    Args:
        singular_values: List of non-negative Schmidt singular values.
        eps: Numerical zero threshold (values below this are ignored).

    Returns:
        Von Neumann entropy in nats (base-e). Multiply by log2(e) for bits.
    """
    svs = [s for s in singular_values if s > eps]
    if not svs:
        return 0.0

    # Normalise squared singular values
    total = sum(s * s for s in svs)
    if total < eps:
        return 0.0

    entropy = 0.0
    for s in svs:
        p = (s * s) / total
        entropy -= p * math.log(p)

    return entropy


def renyi_entropy(singular_values: List[float], alpha: float = 2.0, eps: float = 1e-12) -> float:
    """
    Computes Rényi entropy of order α.

    S_α(ρ) = (1 / (1 - α)) * log(∑_i λ_i^(2α))

    At α → 1: reduces to Von Neumann entropy.
    At α = 2: S_2 = -log(Tr(ρ²)) = -log(purity).
    At α → ∞: S_∞ = -log(max_i λ_i²).

    Args:
        singular_values: Schmidt singular values.
        alpha: Rényi order (must be ≠ 1 and > 0).

    Returns:
        Rényi entropy in nats.
    """
    if abs(alpha - 1.0) < 1e-9:
        return von_neumann_entropy(singular_values, eps)

    svs = [s for s in singular_values if s > eps]
    if not svs:
        return 0.0

    total = sum(s * s for s in svs)
    renyi_sum = sum((s * s / total) ** alpha for s in svs)

    if renyi_sum < eps:
        return 0.0

    return math.log(renyi_sum) / (1.0 - alpha)


def truncation_error(singular_values: List[float], keep_k: int) -> float:
    """
    Computes the Frobenius-norm truncation error when keeping only the top-k
    singular values.

    Error = ||ρ - ρ_trunc||_F² = ∑_{i > k} λ_i²

    A standard convergence criterion for MPS approximation quality.

    Args:
        singular_values: All singular values (sorted descending or unsorted).
        keep_k: Number of singular values to retain.

    Returns:
        Truncation error (squared Frobenius norm of discarded components).
    """
    sorted_svs = sorted(singular_values, reverse=True)
    discarded = sorted_svs[keep_k:]
    return sum(s * s for s in discarded)


def classify_entanglement(
    entropies: List[float],
    system_size: int,
) -> str:
    """
    Classifies the entanglement growth pattern for a 1D quantum system.

    For area-law systems (e.g. gapped ground states): S ~ const (independent of L).
    For volume-law systems (e.g. random states, generic excited states): S ~ L.

    Args:
        entropies: Entanglement entropy S(L) for bipartition sizes L = 1..N//2.
        system_size: Total system size N.

    Returns:
        "area-law", "volume-law", or "sub-volume-law" classification string.
    """
    if len(entropies) < 3:
        return "insufficient-data"

    # Fit linear regression S(L) ≈ a * L + b
    n = len(entropies)
    L_vals = list(range(1, n + 1))
    L_mean = sum(L_vals) / n
    S_mean = sum(entropies) / n

    cov = sum((L_vals[i] - L_mean) * (entropies[i] - S_mean) for i in range(n))
    var = sum((L - L_mean) ** 2 for L in L_vals)

    if var < 1e-10:
        return "constant"

    slope = cov / var

    # Classification thresholds (heuristic)
    if slope < 0.05:
        return "area-law"
    elif slope > 0.4:
        return "volume-law"
    else:
        return "sub-volume-law"


def mutual_information(
    s_a: float,
    s_b: float,
    s_ab: float,
) -> float:
    """
    Computes bipartite mutual information I(A:B) = S(A) + S(B) - S(AB).

    A non-negative measure of total correlations between subsystems A and B.

    Args:
        s_a: Entropy of subsystem A (marginal).
        s_b: Entropy of subsystem B (marginal).
        s_ab: Entropy of joint system AB.

    Returns:
        Mutual information in nats (≥ 0 by subadditivity).
    """
    return max(0.0, s_a + s_b - s_ab)


def ghz_entanglement_entropy(n: int, bipartition_size: int) -> float:
    """
    Computes the exact analytical entanglement entropy for an n-qubit GHZ state.

    |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

    For any non-trivial bipartition (1 ≤ k < n), the Schmidt decomposition has
    exactly 2 non-zero singular values: λ₁ = λ₂ = 1/√2.
    Therefore S = log(2) ≈ 0.693 nats (1 ebit), independent of bipartition size.

    This function returns the exact value for test validation.

    Args:
        n: Total number of qubits.
        bipartition_size: Size of subsystem A (1 ≤ bipartition_size < n).

    Returns:
        Exact Von Neumann entropy (log(2) for all valid bipartitions).
    """
    if bipartition_size <= 0 or bipartition_size >= n:
        return 0.0
    # GHZ state: two Schmidt coefficients, each 1/√2
    return von_neumann_entropy([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
