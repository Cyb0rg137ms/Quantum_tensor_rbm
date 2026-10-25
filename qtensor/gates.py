"""Gates for d-level systems (qudits) as NumPy arrays.

Two-site gates are ``(d*d, d*d)`` matrices in the basis ``|a b> -> a*d + b``
(left site is the most significant index).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]  # complex in practice; float arrays are accepted


def dft(d: int) -> Array:
    """Generalised Hadamard: the d-point DFT, ``F[r, c] = w^(rc) / sqrt(d)``."""
    k = np.arange(d)
    return np.exp(2j * np.pi * np.outer(k, k) / d) / np.sqrt(d)


def shift(d: int, k: int = 1) -> Array:
    """Generalised Pauli-X: ``|j> -> |j + k mod d>``."""
    return np.roll(np.eye(d, dtype=complex), k, axis=0)


def clock(d: int) -> Array:
    """Generalised Pauli-Z: ``|j> -> w^j |j>``."""
    return np.diag(np.exp(2j * np.pi * np.arange(d) / d))


def phase(d: int, phi: float) -> Array:
    """Phase ``e^{i phi}`` on level ``d-1`` only."""
    g = np.eye(d, dtype=complex)
    g[-1, -1] = np.exp(1j * phi)
    return g


def csum(d: int) -> Array:
    """Generalised CNOT (SUM gate): ``|a, b> -> |a, a + b mod d>``. For d=2 this is CNOT."""
    g = np.zeros((d * d, d * d), dtype=complex)
    for a in range(d):
        for b in range(d):
            g[a * d + (a + b) % d, a * d + b] = 1
    return g


def controlled_increment(d: int) -> Array:
    """v1 "CNOT": increment the target only when the control is ``d-1``."""
    g = np.zeros((d * d, d * d), dtype=complex)
    for a in range(d):
        for b in range(d):
            g[a * d + ((b + 1) % d if a == d - 1 else b), a * d + b] = 1
    return g


def swap(d: int) -> Array:
    g = np.zeros((d * d, d * d), dtype=complex)
    for a in range(d):
        for b in range(d):
            g[b * d + a, a * d + b] = 1
    return g


def haar_unitary(n: int, rng: np.random.Generator) -> Array:
    """Haar-random ``n x n`` unitary (QR of a complex Gaussian with phase fix)."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def expm_hermitian(h: Array, t: complex) -> Array:
    """``exp(-i t H)`` for Hermitian ``H`` (``t`` may be complex: ``t = -i tau`` gives ``exp(-tau H)``)."""
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * t * w)) @ v.conj().T


# Pauli matrices for spin-1/2 models
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
H = dft(2)
CNOT = csum(2)


def is_unitary(g: Array, atol: float = 1e-10) -> bool:
    return bool(np.allclose(g @ g.conj().T, np.eye(g.shape[0]), atol=atol))
