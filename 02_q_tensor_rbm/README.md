# ⚛️ Q-Tensor-RBM: Multi-State Quantum System Emulator

Q-Tensor-RBM is a high-performance classical emulator for multi-state quantum systems ($d$-dimensional states / registers). It utilizes Matrix Product States (MPS) combined with a Restricted Boltzmann Machine (RBM) neural cache to represent non-local quantum correlations and volume-law entanglement in linear memory space.

---

## 📐 Mathematical Framework

### 1. Matrix Product States (MPS)
A quantum state vector of $N$ registers with dimension $d$ typically requires $d^N$ complex coefficients. MPS decomposes this exponential state vector into a chain of rank-3 tensors:
$$|\Psi\rangle = \sum_{s_1, \dots, s_N} \operatorname{Tr}(A^{s_1}_1 A^{s_2}_2 \dots A^{s_N}_N) |s_1 s_2 \dots s_N\rangle$$

where $A^{s_i}_i$ is a matrix of size $\chi_{i-1} \times \chi_i$, and $\chi$ is the bond dimension (representing entanglement scale).

### 2. SVD Truncation & Information Loss
When applying 2-site gates, the bond dimension grows:
$$\chi \leftarrow \chi \cdot d$$

To prevent exponential complexity, Singular Value Decomposition (SVD) splits and truncates the bond dimension back to a fixed maximum $\chi_{\max}$:
$$M = U S V^\dagger \approx U_{\text{trunc}} S_{\text{trunc}} V^\dagger_{\text{trunc}}$$

This SVD truncation drops singular values $S_{\text{discarded}}$, causing a loss of non-local entanglement information.

### 3. RBM Entanglement Recovery
Rather than discarding this correlation, Q-Tensor-RBM projects the truncated modes into a Restricted Boltzmann Machine (RBM). The probability amplitude correction for a configuration $v$ is modeled as:
$$P(v) = \frac{1}{Z} e^{b^T v} \prod_{h=1}^H \left(1 + e^{c_h + W_h v}\right)$$

where:
- $v$ is the visible vector representing the physical state config.
- $W$ represents the complex coupling weights between register sites and hidden neurons.
- $b, c$ are the visible and hidden bias vectors.

During truncation, weight increments are applied via a Hebbian learning update:
$$\Delta W_{i, p, h} \propto S_{\text{dropped}} \cdot l_p \cdot e^{i \phi}$$

---

## 🛠️ Workings & Pipeline

```mermaid
graph TD
    A[Register State |00...0>] --> B[Apply Quantum Gates]
    B --> C[Contract Adjacent Tensors]
    C --> D[Perform SVD Matrix Splitting]
    D -->|Bond Dim > max_bond| E[Truncate Singular Values]
    E -->|Dropped Modes| F[Update RBM Entanglement Cache]
    E -->|Retained Modes| G[Update MPS Tensors]
    F --> H[Reconstructed State Amplitude]
    G --> H
    H --> I[Measure Entanglement Entropy & State]
```

1. **Gate Application**: Contracts adjacent tensors and applies generalized unitary gates (Fourier/Hadamard, CNOT, phase-shifts).
2. **Entanglement Caching**: During SVD factorization, discarded singular vectors are mapped to the RBM visible-hidden connection weights.
3. **Reconstructed Probability**: The total state amplitude is computed as the product of the MPS boundary contraction and the neural cache correction, preventing loss of volume-law correlations.

---

## 💎 Key Advantages

- **Linear Scaling**: Memory usage scales as $O(N \cdot d \cdot \chi^2 + N \cdot d \cdot H)$ where $H$ is the number of hidden units.
- **Volume-Law Recovery**: Standard tensor networks are limited to area-law entanglement. The RBM cache stores non-local volume-law correlations classically.
- **Generalized State Dimension**: Simulates multi-state registers ($d > 2$), opening up advanced research on high-dimensional quantum computing.

---

## 📦 How to Install and Run

### Prerequisites
- Python 3.9 or higher
- [Poetry](https://python-poetry.org/) (optional)

### Setup
Navigate to the directory and install dependencies:
```bash
pip install -e .
```
Or with Poetry:
```bash
poetry install
```

### Running Tests
Run the test suite using `pytest`:
```bash
pytest tests/
```

### Running the Example
To run the high-dimensional Bell State cascade example:
```bash
python examples/bell_state.py
```

---

<div align="center">
  <a href="https://q.com">
    <img src="../../assets/https_q_com.png" alt="Q Logo" width="100" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
  </a>
  <br/>
  <small>Ecosystem mapping and validation protocols courtesy of <a href="https://q.com">q.com</a></small>
</div>

## Performance Benchmark

*Benchmark not available:* No benchmark script found
