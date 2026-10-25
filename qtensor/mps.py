"""Matrix Product State simulator for chains of d-level systems (TEBD-style gate application).

Tensors are ``A[i]`` with shape ``(chi_left, d, chi_right)``. The state is kept in
mixed-canonical form with a tracked orthogonality centre, so every SVD
truncation is optimal in the 2-norm (Vidal 2003; Schollwöck 2011) and the
discarded weight is an exact measure of the error introduced by that step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from qtensor.gates import Array, swap

log = logging.getLogger(__name__)


@dataclass
class TruncationStats:
    """Accumulated truncation information."""

    steps: int = 0
    discarded_weights: list[float] = field(default_factory=list)

    @property
    def fidelity_estimate(self) -> float:
        """``prod(1 - eps_k)``: standard estimate of ``|<psi_exact|psi_mps>|^2``."""
        return float(np.prod([1.0 - e for e in self.discarded_weights])) if self.discarded_weights else 1.0

    @property
    def total_discarded(self) -> float:
        return float(sum(self.discarded_weights))


class MPS:
    """Open-boundary MPS with bond-dimension cap ``max_bond`` and singular-value ``cutoff``."""

    def __init__(
        self,
        n: int,
        d: int = 2,
        max_bond: int | None = 64,
        cutoff: float = 1e-12,
        config: list[int] | None = None,
    ) -> None:
        if n < 1 or d < 2:
            raise ValueError("need n >= 1 and d >= 2")
        if max_bond is not None and max_bond < 1:
            raise ValueError("max_bond must be >= 1")
        self.n, self.d, self.max_bond, self.cutoff = n, d, max_bond, cutoff
        config = config or [0] * n
        if len(config) != n or any(not 0 <= c < d for c in config):
            raise ValueError("invalid product-state configuration")
        self.tensors: list[Array] = []
        for c in config:
            t = np.zeros((1, d, 1), dtype=complex)
            t[0, c, 0] = 1.0
            self.tensors.append(t)
        self.center = 0
        self.stats = TruncationStats()

    # ------------------------------------------------------------------ canonical form
    def _shift_center_right(self) -> None:
        i = self.center
        a = self.tensors[i]
        dl, d, dr = a.shape
        q, r = np.linalg.qr(a.reshape(dl * d, dr))
        self.tensors[i] = q.reshape(dl, d, -1)
        self.tensors[i + 1] = np.tensordot(r, self.tensors[i + 1], axes=([1], [0]))
        self.center = i + 1

    def _shift_center_left(self) -> None:
        i = self.center
        a = self.tensors[i]
        dl, d, dr = a.shape
        q, r = np.linalg.qr(a.reshape(dl, d * dr).T)
        self.tensors[i] = q.T.reshape(-1, d, dr)
        self.tensors[i - 1] = np.tensordot(self.tensors[i - 1], r.T, axes=([2], [0]))
        self.center = i - 1

    def move_center(self, i: int) -> None:
        if not 0 <= i < self.n:
            raise IndexError(i)
        while self.center < i:
            self._shift_center_right()
        while self.center > i:
            self._shift_center_left()

    # ------------------------------------------------------------------ gates
    def apply_1site(self, gate: Array, i: int) -> None:
        self.tensors[i] = np.einsum("ps,lsr->lpr", gate, self.tensors[i])

    def apply_2site(self, gate: Array, i: int, j: int | None = None, normalize: bool = True) -> float:
        """Apply a ``(d^2, d^2)`` gate to sites ``(i, j)``; returns the discarded weight.

        Non-adjacent pairs are routed with SWAP gates (each SWAP may truncate too).
        ``normalize=False`` keeps the norm change (for non-unitary gates where it matters).
        """
        j = i + 1 if j is None else j
        if i == j or not (0 <= i < self.n and 0 <= j < self.n):
            raise ValueError("need two distinct valid sites")
        if i > j:
            sw = swap(self.d)
            return self.apply_2site(sw @ gate @ sw, j, i, normalize)
        disc = 0.0
        sw = swap(self.d)
        for k in range(j - 1, i, -1):  # bring site j next to i
            disc += self._apply_adjacent(sw, k, normalize)
        disc += self._apply_adjacent(gate, i, normalize)
        for k in range(i + 1, j):
            disc += self._apply_adjacent(sw, k, normalize)
        return disc

    def _apply_adjacent(self, gate: Array, i: int, normalize: bool) -> float:
        d = self.d
        self.move_center(i)
        a, b = self.tensors[i], self.tensors[i + 1]
        dl, dr = a.shape[0], b.shape[2]
        theta = np.tensordot(a, b, axes=([2], [0]))  # (dl, d, d, dr)
        g = gate.reshape(d, d, d, d)
        theta = np.einsum("pqst,lstr->lpqr", g, theta).reshape(dl * d, d * dr)
        try:
            u, s, vh = np.linalg.svd(theta, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - LAPACK fallback
            u, s, vh = np.linalg.svd(theta + 1e-14 * np.random.default_rng(0).standard_normal(theta.shape), False)
        total = float((s**2).sum())
        if total == 0.0:
            raise FloatingPointError("state collapsed to zero norm")
        keep = int(np.count_nonzero(s > self.cutoff * s[0])) if s[0] > 0 else 1
        if self.max_bond is not None:
            keep = min(keep, self.max_bond)
        keep = max(keep, 1)
        discarded = float((s[keep:] ** 2).sum()) / total
        s = s[:keep]
        if normalize:
            s = s / np.linalg.norm(s)
        self.tensors[i] = u[:, :keep].reshape(dl, d, keep)
        self.tensors[i + 1] = (s[:, None] * vh[:keep]).reshape(keep, d, dr)
        self.center = i + 1
        self.stats.steps += 1
        if discarded > 0:
            self.stats.discarded_weights.append(discarded)
        return discarded

    # ------------------------------------------------------------------ observables
    @property
    def bond_dims(self) -> list[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    def norm(self) -> float:
        c = self.tensors[self.center]
        return float(np.linalg.norm(c))

    def schmidt_values(self, bond: int) -> Array:
        """Schmidt coefficients across the cut between sites ``bond`` and ``bond+1``."""
        if not 0 <= bond < self.n - 1:
            raise IndexError(bond)
        self.move_center(bond)
        a = self.tensors[bond]
        s = np.linalg.svd(a.reshape(-1, a.shape[2]), compute_uv=False)
        return s / np.linalg.norm(s)

    def entropy(self, bond: int) -> float:
        """Von Neumann entanglement entropy (nats) across ``bond``."""
        p = self.schmidt_values(bond) ** 2
        p = p[p > 1e-15]
        return float(-(p * np.log(p)).sum())

    def entropies(self) -> list[float]:
        return [self.entropy(b) for b in range(self.n - 1)]

    def amplitude(self, config: list[int]) -> complex:
        v = np.ones(1, dtype=complex)
        for t, s in zip(self.tensors, config, strict=True):
            v = v @ t[:, s, :]
        return complex(v[0])

    def to_dense(self) -> Array:
        if self.d**self.n > 2**24:
            raise MemoryError("state too large to densify")
        psi = self.tensors[0]
        for t in self.tensors[1:]:
            psi = np.tensordot(psi, t, axes=([-1], [0]))
        return psi.reshape(-1)

    def expectation_1site(self, op: Array, i: int) -> complex:
        self.move_center(i)
        a = self.tensors[i]
        return complex(np.einsum("lsr,ps,lpr->", a.conj(), op, a))

    def expectation_2site(self, op: Array, i: int) -> complex:
        """``<op>`` for a ``(d^2, d^2)`` operator on adjacent sites ``(i, i+1)``."""
        d = self.d
        self.move_center(i)
        theta = np.tensordot(self.tensors[i], self.tensors[i + 1], axes=([2], [0]))
        g = op.reshape(d, d, d, d)
        return complex(np.einsum("lpqr,pqst,lstr->", theta.conj(), g, theta))

    def sample(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Exact (perfect) sampling of computational-basis outcomes; shape ``(shots, n)``."""
        self.move_center(0)
        out = np.zeros((shots, self.n), dtype=np.int64)
        for k in range(shots):
            env = np.ones(1, dtype=complex)
            for i, t in enumerate(self.tensors):
                vs = np.einsum("l,lsr->sr", env, t)
                p = np.real(np.einsum("sr,sr->s", vs.conj(), vs))
                p = p / p.sum()
                s = int(rng.choice(self.d, p=p))
                out[k, i] = s
                env = vs[s] / np.linalg.norm(vs[s])
        return out

    def memory_bytes(self) -> int:
        return int(sum(t.nbytes for t in self.tensors))


def overlap(a: MPS, b: MPS) -> complex:
    """``<a|b>`` by transfer-matrix contraction, ``O(n d chi^3)``."""
    if a.n != b.n or a.d != b.d:
        raise ValueError("incompatible MPS")
    env = np.ones((1, 1), dtype=complex)
    for x, y in zip(a.tensors, b.tensors, strict=True):
        env = np.einsum("ab,asc,bsd->cd", env, x.conj(), y)
    return complex(env[0, 0])
