"""Legacy v1 API: ``MultiStateQuantumSimulator`` is now a thin wrapper around :class:`qtensor.MPS`.

v1's ``apply_two_state_gate`` **did not apply the gate**: it generated random
"discarded singular values" and fed them into the neural cache, and
``calculate_reconstructed_entropy`` summed RBM weight magnitudes. Both now do
real work: the gate is applied with SVD truncation and the entropy is the sum
of von Neumann entanglement entropies over all bonds.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from qtensor.gates import csum, dft
from qtensor.mps import MPS
from simulator.neural_cache import NeuralEntanglementCache


class MultiStateQuantumSimulator:
    """``n_qstates`` registers of dimension ``physical_dim`` with bond cap ``max_bond_dim``."""

    def __init__(self, n_qstates: int, physical_dim: int = 10, max_bond_dim: int = 64):
        self.n = n_qstates
        self.d = physical_dim
        self.max_bond_dim = max_bond_dim
        self.mps = MPS(n_qstates, physical_dim, max_bond=max_bond_dim)
        self.neural_cache = NeuralEntanglementCache(self.n, self.d)  # legacy; unused by the physics

    @property
    def tensors(self) -> List[np.ndarray]:
        return self.mps.tensors

    def apply_single_state_gate(self, i: int, gate_matrix) -> None:
        self.mps.apply_1site(np.asarray(gate_matrix, dtype=complex), i)

    def apply_two_state_gate(self, i: int, j: int, gate_matrix: Optional[List[List[complex]]] = None) -> float:
        """Apply a ``(d^2, d^2)`` gate to registers ``i, j``; returns the discarded weight.

        With no gate, applies the entangling pair ``CSUM . (DFT x I)`` (v1 applied nothing).
        """
        if gate_matrix is None:
            g = csum(self.d) @ np.kron(dft(self.d), np.eye(self.d))
        else:
            g = np.asarray(gate_matrix, dtype=complex)
        return self.mps.apply_2site(g, i, j)

    def calculate_reconstructed_entropy(self) -> float:
        """Sum over bonds of the von Neumann entanglement entropy (nats)."""
        return float(sum(self.mps.entropies())) if self.n > 1 else 0.0
