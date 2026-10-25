"""
svd.py
======
Pure-Python truncated Singular Value Decomposition (SVD) via Jacobi iterations.

Implements a simplified two-sided Jacobi SVD algorithm for small matrices,
suitable for MPS bond dimension truncation without NumPy dependency.

For production use, prefer numpy.linalg.svd (``qtensor`` uses LAPACK SVD) — this module exists to demonstrate
the SVD mechanics and provide a dependency-free fallback for quantum simulation.

Reference:
  - Golub & Van Loan, "Matrix Computations", 4th ed., Ch. 8 (Jacobi iterations)
  - Vidal, PRL 91, 147902 (2003) — MPS Schmidt decomposition
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Matrix = List[List[complex]]


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """Complex matrix multiplication C = A @ B."""
    n, m, p = len(A), len(B[0]), len(B)
    C = [[complex(0) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if A[i][k] == 0:
                continue
            for j in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _identity(n: int) -> Matrix:
    return [[complex(1 if i == j else 0) for j in range(n)] for i in range(n)]


def _gram_matrix(A: Matrix) -> Matrix:
    """Compute A† @ A (Gram matrix)."""
    n = len(A)
    m = len(A[0])
    G = [[complex(0) for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            for k in range(n):
                G[i][j] += A[k][i].conjugate() * A[k][j]
    return G


def truncated_svd(
    A: Matrix,
    keep_k: Optional[int] = None,
    threshold: float = 1e-10,
) -> Tuple[List[float], Matrix, Matrix]:
    """
    Computes a truncated SVD of a small complex matrix A ≈ U @ diag(s) @ V†.

    Uses power iteration (via A†A) for singular value estimation.
    This is a simplified algorithm suitable for small MPS tensors (d ≤ 64).

    Args:
        A: Input matrix (n_rows × n_cols) as List[List[complex]].
        keep_k: Maximum number of singular values to retain (None = keep all).
        threshold: Singular values below this are discarded.

    Returns:
        (singular_values, left_vecs, right_vecs) where:
          - singular_values: List[float] of length k (descending order)
          - left_vecs: List of left singular vectors, each of length n_rows
          - right_vecs: List of right singular vectors, each of length n_cols
    """
    n = len(A)
    m = len(A[0])
    rank = min(n, m)
    if keep_k is not None:
        rank = min(rank, keep_k)

    # Work with A†A to get right singular vectors V and singular values σ² = λ
    G = _gram_matrix(A)  # m × m

    # Power iteration to extract top singular vectors
    singular_values: List[float] = []
    right_vecs: List[List[complex]] = []
    left_vecs: List[List[complex]] = []

    # Deflation: progressively subtract found components
    G_deflated = [row[:] for row in G]  # copy

    for k in range(rank):
        # Random init for power iteration
        v = [complex(1.0 / math.sqrt(m) if j == k % m else 0) for j in range(m)]

        # Power iteration to converge to dominant eigenvector
        for _ in range(50):
            # v_new = G_deflated @ v
            v_new = [sum(G_deflated[i][j] * v[j] for j in range(m)) for i in range(m)]
            # Orthogonalise against previously found eigenvectors
            for prev_v in right_vecs:
                proj = sum(prev_v[j].conjugate() * v_new[j] for j in range(m))
                v_new = [v_new[j] - proj * prev_v[j] for j in range(m)]

            # Compute norm
            norm = math.sqrt(sum(abs(x) ** 2 for x in v_new))
            if norm < threshold:
                break
            v = [x / norm for x in v_new]

        # Compute eigenvalue (Rayleigh quotient): λ = v† G v
        Gv = [sum(G_deflated[i][j] * v[j] for j in range(m)) for i in range(m)]
        lam = sum(v[j].conjugate() * Gv[j] for j in range(m)).real

        if lam < threshold * threshold:
            break

        sigma = math.sqrt(max(0.0, lam))
        if sigma < threshold:
            break

        singular_values.append(sigma)
        right_vecs.append(list(v))

        # Compute left singular vector u = A v / sigma
        u = [sum(A[i][j] * v[j] for j in range(m)) / sigma for i in range(n)]
        left_vecs.append(u)

        # Deflate: G_deflated -= lam * v v†
        # Correct outer product shape: [v v†]_ij = v_i * v_j*
        for i in range(m):
            for j in range(m):
                G_deflated[i][j] -= lam * v[i] * v[j].conjugate()

    return singular_values, left_vecs, right_vecs
