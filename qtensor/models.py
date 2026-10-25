"""Spin-chain Hamiltonians, exact diagonalisation, and TEBD time evolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qtensor.gates import I2, Array, X, Z, expm_hermitian
from qtensor.mps import MPS


@dataclass(frozen=True)
class TFIM:
    """Open transverse-field Ising chain ``H = -J sum Z_i Z_{i+1} - h sum X_i`` (critical at ``h = J``)."""

    n: int
    J: float = 1.0
    h: float = 1.0

    def bond_terms(self) -> list[Array]:
        """Two-site terms; the field is split so each site's ``-h X`` is counted exactly once."""
        terms = []
        for b in range(self.n - 1):
            wl = 1.0 if b == 0 else 0.5
            wr = 1.0 if b == self.n - 2 else 0.5
            terms.append(-self.J * np.kron(Z, Z) - self.h * (wl * np.kron(X, I2) + wr * np.kron(I2, X)))
        return terms

    def dense(self) -> Array:
        if self.n > 14:
            raise MemoryError("dense TFIM limited to n <= 14")
        dim = 2**self.n
        h = np.zeros((dim, dim), dtype=complex)

        def op(single: dict[int, Array]) -> Array:
            m = np.ones((1, 1), dtype=complex)
            for i in range(self.n):
                m = np.kron(m, single.get(i, I2))
            return m

        for i in range(self.n - 1):
            h -= self.J * op({i: Z, i + 1: Z})
        for i in range(self.n):
            h -= self.h * op({i: X})
        return h

    def energy(self, psi: MPS) -> float:
        return float(sum(psi.expectation_2site(t, b).real for b, t in enumerate(self.bond_terms())))


def exact_ground_state(model: TFIM) -> tuple[float, Array]:
    w, v = np.linalg.eigh(model.dense())
    return float(w[0]), v[:, 0]


def tebd(psi: MPS, bond_terms: list[Array], dt: float, steps: int, imaginary: bool = False) -> MPS:
    """Second-order (symmetric) Trotter sweeps: half steps left-to-right then right-to-left.

    ``imaginary=True`` evolves with ``exp(-dt H)`` and renormalises: projects onto the ground state.
    """
    t = -1j * dt / 2 if imaginary else dt / 2
    gates = [expm_hermitian(h, t) for h in bond_terms]
    for _ in range(steps):
        for b in range(len(gates)):
            psi.apply_2site(gates[b], b)
        for b in reversed(range(len(gates))):
            psi.apply_2site(gates[b], b)
    return psi
