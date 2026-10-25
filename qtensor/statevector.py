"""Dense state-vector reference simulator for qudit chains (exact, memory ``d^n``)."""

from __future__ import annotations

import numpy as np

from qtensor.gates import Array


class StateVector:
    """``psi`` stored as an ``n``-axis tensor of shape ``(d,)*n``."""

    def __init__(self, n: int, d: int = 2, config: list[int] | None = None) -> None:
        if d**n > 2**26:
            raise MemoryError(f"d^n = {d}^{n} amplitudes is too large for a dense simulation")
        self.n, self.d = n, d
        self.psi = np.zeros((d,) * n, dtype=complex)
        self.psi[tuple(config or [0] * n)] = 1.0

    def apply_1site(self, gate: Array, i: int) -> None:
        self.psi = np.moveaxis(np.tensordot(gate, self.psi, axes=([1], [i])), 0, i)

    def apply_2site(self, gate: Array, i: int, j: int) -> None:
        d = self.d
        g = gate.reshape(d, d, d, d)
        self.psi = np.moveaxis(np.tensordot(g, self.psi, axes=([2, 3], [i, j])), [0, 1], [i, j])

    def vector(self) -> Array:
        return self.psi.reshape(-1)

    def entropy(self, cut: int) -> float:
        """Von Neumann entropy (nats) of sites ``[0, cut)`` vs the rest."""
        m = self.psi.reshape(self.d**cut, -1)
        s = np.linalg.svd(m, compute_uv=False)
        p = s**2
        p = p[p > 1e-15]
        return float(-(p * np.log(p)).sum())

    def expectation_1site(self, op: Array, i: int) -> complex:
        phi = np.moveaxis(np.tensordot(op, self.psi, axes=([1], [i])), 0, i)
        return complex(np.vdot(self.psi, phi))
