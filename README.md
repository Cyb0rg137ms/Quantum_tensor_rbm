# Q-Tensor-RBM: MPS simulation of qudit chains, plus RBM neural quantum states

A small, tested NumPy library for simulating chains of **d-level quantum systems (qudits)**:

- **`qtensor.MPS`**: a matrix product state in mixed-canonical form with optimal SVD truncation, exact discarded-weight
  tracking, non-adjacent gates, entanglement entropies, amplitudes, local expectations and exact sampling.
- **`qtensor.tebd`**: real- and imaginary-time evolution using second-order Trotter.
- **`qtensor.RBMState`**: the Carleo–Troyer restricted-Boltzmann-machine wavefunction, trained with stochastic
  reconfiguration.
- **`qtensor.StateVector`**: a dense reference simulator. Every MPS result is checked against it, or against exact
  diagonalisation, in 38 tests.

> **v2 rewrite, read this first.** In v1, `apply_two_state_gate` **never applied the gate**. It generated random numbers
> and fed them into an "RBM entanglement cache", and the reported "entropy" was a sum of weight magnitudes. The claim
> that the RBM "recovers volume-law entanglement in linear memory" was false. Information discarded by truncation is
> gone, and MPS cost grows with entanglement. v2 implements the real physics and measures where it breaks down
> (see [LIMITATIONS](docs/LIMITATIONS.md)).

## Install and run

```bash
pip install -e ".[dev]"
qtensor bench            # every table below (about 7 s on CPU)
qtensor ghz --n 1000 --d 10
python examples/bell_state.py
pytest
```

```python
import numpy as np
from qtensor import MPS, dft, csum, haar_unitary

psi = MPS(n=50, d=3, max_bond=32)
psi.apply_1site(dft(3), 0)
for i in range(49):
    psi.apply_2site(csum(3), i)  # qutrit GHZ, exact at chi = 3
psi.apply_2site(haar_unitary(9, np.random.default_rng(0)), 0, 7)  # non-adjacent, routed by SWAPs
print(psi.entropies()[:3], psi.stats.fidelity_estimate, psi.sample(4, np.random.default_rng(1)))
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Measured results (`qtensor bench`, seed 0, CPU)

**Correctness.** Untruncated MPS against the dense state vector on random Haar circuits that include non-adjacent
gates:

| n | d | gates | fidelity | max entropy error |
|---|---|---|---|---|
| 10 | 2 | 60 | 0.999999999999994 | 9.8e-15 |
| 6 | 3 | 40 | 0.999999999999997 | 4.0e-15 |

**Low entanglement is cheap.** A qudit GHZ state with d=10 is exact at χ = d, and every bond carries exactly ln d:

| n | seconds | MPS bytes | dense state would need |
|---|---|---|---|
| 10 | 0.002 | 131,200 | 10^11.2 B |
| 100 | 0.018 | 1,571,200 | 10^101.2 B |
| 1000 | 0.19 | 15,971,200 | 10^1001.2 B |

**High entanglement is not cheap.** A Haar brick-wall circuit with n=16 and depth 12, compared with the exact state
(dense size 1,048,576 B):

| χ | true fidelity | estimate Π(1−ε) | mid-chain S | seconds | MPS bytes |
|---|---|---|---|---|---|
| 2 | 0.0059 | 0.0079 | 0.509 | 0.010 | 1,920 |
| 8 | 0.2957 | 0.2975 | 1.753 | 0.016 | 23,168 |
| 16 | 0.6698 | 0.6714 | 2.356 | 0.021 | 76,416 |
| 32 | 0.9204 | 0.9207 | 2.822 | 0.043 | 240,256 |
| 64 | 0.9944 | 0.9944 | 3.043 | 0.173 | 699,008 |
| 256 | 1.0000 | 1.0000 | 3.079 | 0.413 | 2,796,160 |

The discarded-weight estimate tracks the true fidelity closely. Near-exact simulation needs an MPS *larger* than the
dense vector, which is the volume-law wall.

**Ground state.** Critical TFIM, n=10, J=h=1, exact E₀ = −12.3814899997:

| method | energy | rel. error | seconds |
|---|---|---|---|
| imaginary-time TEBD, χ=2 | −12.3721291107 | 7.6e-4 | 0.23 |
| imaginary-time TEBD, χ=4 | −12.3814818996 | 6.5e-7 | 0.23 |
| imaginary-time TEBD, χ=8 | −12.3814899850 | 1.2e-9 (Trotter floor) | 0.26 |
| RBM α=1, 120 params | −12.3814155921 | 6.0e-6 | 1.40 |
| RBM α=2, 230 params | −12.3814788218 | 9.0e-7 | 2.30 |

## v1 compatibility

`from simulator import MultiStateQuantumSimulator, ...` still works, and all 16 original tests pass unchanged.
`apply_two_state_gate` now really applies the gate; it defaults to `CSUM·(DFT⊗I)` when no gate is given.
`calculate_reconstructed_entropy` returns the sum of the true bond entropies.

v1's `benchmark.py` imported a function that did not exist, and `BENCHMARK_RESULTS.md` contained no measurements. Both
were removed.

## CI

`ci/github-actions-ci.yml` runs ruff, mypy, pytest and a quick benchmark. It sits under `ci/` because the push token
lacks the `workflow` scope; move it to `.github/workflows/` to enable it.

## License

MIT
