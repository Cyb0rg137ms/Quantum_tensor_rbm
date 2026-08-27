# Q-Tensor-RBM — Architecture & Technical Reference

> **Full Project Name:** Q-Tensor-RBM — Multi-State Quantum System Emulator
> **Category:** Quantum Computing Simulation / Tensor Networks
> **Language:** Python 3.9+, NumPy
> **Test Coverage:** 5/5 unit tests passing ✅

---

## 1. Architecture Overview

```
02_q_tensor_rbm/
├── simulator/
│   ├── mps.py           # Matrix Product States contraction engine
│   ├── neural_cache.py  # RBM entanglement restoration cache
│   ├── gates.py         # Quantum gates: DFT-Hadamard, CNOT, Phase Shift
│   └── __init__.py
├── examples/
│   └── bell_state.py    # Bell state cascade demonstration
├── tests/
│   └── test_simulator.py
└── README.md
```

### Component Interaction

```
┌───────────────────────────────────────────────────────────────┐
│                  Q-TENSOR-RBM SIMULATOR PIPELINE             │
│                                                               │
│  Initial State ──► MPS Initializer                           │
│                         │                                    │
│                         ▼                                    │
│                    Gate Application                           │
│               (Hadamard / CNOT / Phase)                      │
│                         │                                    │
│                         ▼                                    │
│                    SVD Truncation                             │
│                  (bond dim cutoff)                            │
│                         │                                    │
│                    Entropy Loss ──► RBM Neural Cache         │
│                                    (Hebbian reconstruction)  │
│                         │                                    │
│                         ▼                                    │
│                  Volume-Law Entropy Estimate                  │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework

### 2.1 Matrix Product State (MPS) Representation

A quantum state |ψ⟩ across N sites, each with physical dimension `d` (number of
possible states per site), is stored as a chain of matrices:

```
|ψ⟩ = SUM over all basis states (s_1, s_2, ..., s_N):
          A_1[s_1] × A_2[s_2] × ... × A_N[s_N]  ×  |s_1 s_2 ... s_N⟩

Each A_k[s_k]  is a matrix of shape  (χ_{k-1}  ×  χ_k)
χ_k  = bond dimension at bond k  (capped at χ_max)
```

Larger `χ_max` → more accurate representation but more memory and compute.

### 2.2 SVD Truncation

After applying a gate, the bond tensor M is decomposed using Singular Value Decomposition:

```
M  =  U  ×  Σ  ×  V†

U  = left unitary matrix
Σ  = diagonal matrix of singular values  σ_1 ≥ σ_2 ≥ ... ≥ 0
V† = conjugate transpose of right unitary matrix
```

Singular values below threshold `ε` are discarded, keeping at most `χ_max` values.

**Truncation error** (how much quantum information is lost):

```
ε_trunc = sqrt( σ_{χ_max+1}^2  +  σ_{χ_max+2}^2  +  ... )
        = sqrt( sum of squared discarded singular values )
```

### 2.3 RBM Entanglement Cache

To recover entropy lost during SVD truncation, a Restricted Boltzmann Machine (RBM)
learns the correlation between visible units `v` and hidden units `h`.

Weight update (contrastive-divergence / Hebbian rule):

```
ΔW_ij  =  η × ( <v_i * h_j>_data  -  <v_i * h_j>_model )

η             = learning rate
<...>_data    = expectation from training data
<...>_model   = expectation from model reconstruction
```

The trained RBM outputs a **volume-law entropy correction δS** that estimates
the missing long-range entanglement that MPS truncation discarded.

### 2.4 Volume-Law Entropy Estimation

For a bipartition of the system at site k, the entanglement entropy from the SVD is:

```
S_k  =  - SUM over j:  σ_j^2 × log(σ_j^2)

where σ_j are the singular values at bond k
```

The RBM adds correction `δS` so the final entropy better matches the true
volume-law scaling (entropy grows proportional to volume, not just boundary area).

### 2.5 Quantum Gates

| Gate | Description | Matrix Form |
|------|-------------|-------------|
| **Hadamard (DFT)** | Discrete Fourier Transform | Entry (j,k) = (1/√d) × exp(2πi × j×k / d) |
| **CNOT** | Controlled-NOT generalized to d states | Block permutation on d² space |
| **Phase Shift** | Rotational phase encoding | Diagonal: diag(1, e^(iφ), e^(2iφ), ...) |

---

## 3. Workflow

```
Initialize:  N registers, physical dim d, bond dim χ_max
        │
        ▼
Apply quantum gate (Hadamard / CNOT / Phase)
        │
        ▼
Reshape gate output to bond matrix M
        │
        ▼
SVD decompose:  M = U × Σ × V†
        │
        ├─── singular values σ_j < ε  →  discard
        └─── keep up to χ_max values  →  update MPS bond tensors
        │
        ▼
Compute truncation error  ε_trunc
        │
        ▼
RBM Cache: compute entropy correction δS
        │
        ▼
More gates? ── Yes ──► loop back to gate application
        │
       No
        ▼
Compute final volume-law entropy estimate  S
```

---

## 4. System Design

| Component | Module | Role |
|-----------|--------|------|
| **State Engine** | `mps.py` | Tensor initialization, gate application, SVD truncation |
| **Entanglement Cache** | `neural_cache.py` | RBM weights, Hebbian updates, entropy reconstruction |
| **Gate Library** | `gates.py` | Unitary matrices for DFT-Hadamard, CNOT, Phase Shift |
| **Demo** | `examples/bell_state.py` | 10-register Bell cascade with entropy tracking |
| **Tests** | `tests/test_simulator.py` | Gate correctness, initialization, RBM fidelity |

---

## 5. Key Advantages

| Advantage | Description |
|-----------|-------------|
| **d-dimensional generalization** | Handles d > 2 physical dimensions beyond standard qubits |
| **Efficient SVD truncation** | Polynomial-time contraction via bond compression |
| **Neural entanglement recovery** | RBM compensates for volume-law entropy lost in MPS |
| **Pure NumPy** | No heavy quantum framework dependency (no Qiskit required) |
| **Extensible gate library** | Simple 2D unitary matrix interface for custom gates |

---

## 6. Test Results

```
tests/test_simulator.py::test_simulator_initialization     PASSED
tests/test_simulator.py::test_gate_generation              PASSED
tests/test_simulator.py::test_gate_cnot                    PASSED
tests/test_simulator.py::test_neural_entanglement_cache    PASSED
tests/test_simulator.py::test_simulator_gate_application   PASSED
────────────────────────────────────────────────────────────
5 passed in 0.13s
```

---

## 7. Quick Start

```bash
pip install -e .
pytest tests/
python examples/bell_state.py
```

<div align="center">
  <a href="https://q.com"><img src="../../assets/https_q_com.png" width="80" /></a>
</div>
