"""
LIMITATIONS.md — Quantum Tensor RBM
===================================

# Known Limitations and Future Work

## Tensor Networks (MPS)

### 1. 1D Dimensionality Constraint
Matrix Product States (MPS) are inherently designed for 1D topology:
- Entanglement scaling is limited by the bond dimension $\chi$.
- For 2D systems (e.g. PEPS), boundary truncation error grows exponentially.
- Bipartite cuts are simple slices. For complex partitions, computing partial traces becomes computationally expensive.

### 2. Jacobi Power Iteration SVD Performance
Our pure-Python SVD uses a power iteration with Gram-Schmidt orthogonalisation and deflation. While correct:
- **Complexity**: $O(K \times \text{iters} \times D^2)$ where $D$ is the matrix dimension.
- In Python, this is ~40× slower than LAPACK-based implementations (`numpy.linalg.svd`).
- On bond dimension $\chi > 64$, the simulation slows down significantly.

---

## Neural Cache (RBM)

### 1. Contrastive Divergence CDs Approximations
Contrastive Divergence (CD-k) approximates the gradient of the log-likelihood:
- It replaces the true model expectation (which requires computing the partition function $Z$, an NP-hard sum over $2^{N+H}$ states) with a short Gibbs chain starting at data.
- This introduces a **systematic bias** that can prevent the RBM from learning the exact phase distribution of highly entangled states.
- Using Persistent Contrastive Divergence (PCD) or Parallel Tempering would reduce this bias at the cost of higher sampling times.

### 2. Representational Capacity: Area-Law vs. Volume-Law
An RBM with a modest number of hidden units $H$ can only capture area-law entanglement (or simple volume-law structures):
- For random quantum circuits, entanglement grows linearly with system size (volume-law).
- To represent a volume-law state with high fidelity, the required number of hidden units scales exponentially, causing representational collapse.
"""
