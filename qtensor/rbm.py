"""Restricted Boltzmann Machine neural-network quantum state (Carleo & Troyer, Science 2017).

``psi(s) = exp(a . s) * prod_j 2 cosh(b_j + W_j . s)`` with spins ``s_i in {-1, +1}``
(``s = +1`` is ``|0>``, the Z eigenvalue). Parameters are real, which is
sufficient for stoquastic Hamiltonians such as the TFIM, whose ground state
is positive.

Training here is *exact-enumeration* variational optimisation: all ``2^n``
amplitudes are evaluated, so the energy and its gradient carry no Monte Carlo
noise. That restricts it to small ``n`` (≈ 16) and is meant for validating the
ansatz against exact diagonalisation, not for scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qtensor.gates import Array


def all_spins(n: int) -> np.ndarray:
    """All ``2^n`` configurations, row ``k`` = binary of k (MSB = site 0), mapped 0->+1, 1->-1."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return 1 - 2 * bits.astype(np.float64)


@dataclass
class RBMState:
    a: np.ndarray  # (n,)
    b: np.ndarray  # (m,)
    W: np.ndarray  # (m, n)

    @classmethod
    def random(cls, n: int, alpha: int = 2, scale: float = 0.01, seed: int = 0) -> RBMState:
        rng = np.random.default_rng(seed)
        m = alpha * n
        return cls(rng.normal(0, scale, n), rng.normal(0, scale, m), rng.normal(0, scale, (m, n)))

    @property
    def n_params(self) -> int:
        return self.a.size + self.b.size + self.W.size

    def log_psi(self, s: np.ndarray) -> np.ndarray:
        theta = s @ self.W.T + self.b
        return s @ self.a + np.sum(np.logaddexp(theta, -theta), axis=-1)  # log 2cosh

    def amplitudes(self) -> Array:
        n = self.a.size
        lp = self.log_psi(all_spins(n))
        psi = np.exp(lp - lp.max())
        return (psi / np.linalg.norm(psi)).astype(complex)

    def log_derivatives(self, s: np.ndarray) -> np.ndarray:
        """``d log psi / d theta`` for each config; shape ``(batch, n_params)``."""
        t = np.tanh(s @ self.W.T + self.b)
        return np.concatenate([s, t, (t[:, :, None] * s[:, None, :]).reshape(len(s), -1)], axis=1)

    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.a, self.b, self.W.ravel()])

    def set_flat(self, v: np.ndarray) -> None:
        n, m = self.a.size, self.b.size
        self.a, self.b, self.W = v[:n].copy(), v[n : n + m].copy(), v[n + m :].reshape(m, n).copy()


def train_exact(
    state: RBMState, H: Array, steps: int = 1000, lr: float = 0.01, sr_shift: float = 1e-3, sr: bool = True
) -> list[float]:
    """Minimise ``<psi|H|psi>`` by exact gradients; returns the energy curve.

    With ``sr=True`` uses stochastic reconfiguration (natural gradient with the
    quantum geometric tensor, diagonal shift ``sr_shift``); otherwise plain
    gradient descent.
    """
    n = state.a.size
    spins = all_spins(n)
    Hr = np.real(H) if np.allclose(H.imag, 0) else H
    energies = []
    for _ in range(steps):
        lp = state.log_psi(spins)
        psi = np.exp(lp - lp.max())
        prob = psi**2 / np.sum(psi**2)
        hpsi = Hr @ psi
        eloc = np.real(hpsi / psi)
        E = float(prob @ eloc)
        energies.append(E)
        Ok = state.log_derivatives(spins)
        Obar = prob @ Ok
        dO = Ok - Obar
        grad = 2 * (prob * (eloc - E)) @ dO
        if sr:
            S = (dO * prob[:, None]).T @ dO
            S += sr_shift * np.eye(S.shape[0])
            step = np.linalg.solve(S, grad)
        else:
            step = grad
        state.set_flat(state.get_flat() - lr * step)
    return energies
