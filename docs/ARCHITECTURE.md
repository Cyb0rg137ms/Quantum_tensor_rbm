# Architecture

```mermaid
flowchart LR
    G[qtensor.gates<br/>DFT, shift, clock, CSUM, SWAP, Haar] --> M
    G --> S
    subgraph MPS engine
      M[qtensor.mps.MPS] -->|move_center: QR sweeps| C[mixed-canonical form]
      C -->|theta = A_i A_i+1, apply gate| SVD[SVD + truncate chi / cutoff]
      SVD -->|discarded weight eps| T[TruncationStats<br/>fidelity ≈ Π(1-eps)]
      SVD --> M
    end
    S[qtensor.statevector<br/>dense reference] -->|tests: fidelity, entropies| M
    H[qtensor.models.TFIM] -->|bond terms| TE[tebd: 2nd-order Trotter<br/>real / imaginary time]
    TE --> M
    H -->|dense H| ED[exact diagonalisation]
    H -->|dense H| R[qtensor.rbm: RBM ansatz<br/>exact-enumeration SR]
    ED -->|reference E0| B[benchmarks / CLI]
    R --> B
    M --> B
```

| module | contents |
|---|---|
| `qtensor.mps` | MPS with tracked orthogonality centre, 1-/2-site gates (non-adjacent through SWAPs), Schmidt values, entropies, amplitudes, `to_dense`, local expectations, exact sampling, `overlap` |
| `qtensor.statevector` | dense `d^n` reference simulator used to validate the MPS |
| `qtensor.models` | open TFIM, exact ground state, TEBD |
| `qtensor.rbm` | Carleo–Troyer RBM wavefunction, analytic log-derivatives, exact-enumeration stochastic reconfiguration |
| `qtensor.benchmarks`, `qtensor.cli` | `qtensor bench / ghz / tfim` |
| `simulator.*` | v1 API: `MultiStateQuantumSimulator` now wraps `MPS`; pure-Python SVD, entanglement helpers and gate lists are unchanged; `NeuralEntanglementCache` is kept but documented as non-physical |
