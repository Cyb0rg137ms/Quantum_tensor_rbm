# Limitations

- **MPS cost follows entanglement.** Memory is O(n d χ²) and a gate costs O(d³ χ³). Area-law states, such as gapped
  ground states, GHZ states and short circuits, are cheap. Volume-law states, such as deep random circuits and long
  quenches, need χ that grows exponentially with n. The brick-wall benchmark shows this directly: at n=16 and depth
  12, fidelity is 0.30 at χ=8 and reaches 0.99 only at χ=64. **No classical method here simulates generic deep
  circuits efficiently.**
- **The v1 "RBM entanglement cache" claim was false.** Truncated Schmidt components cannot be recovered from their
  singular values, and v1's cache added random phases to weights that nothing read. The class is kept only for API
  compatibility.
- **The RBM is trained by exact enumeration** over 2ⁿ configurations, so it works up to n ≈ 16. It is a validated
  reference implementation of the ansatz, not a scalable VMC code; there is no Metropolis sampler. Its parameters are
  real, which is enough for stoquastic models like the TFIM but not for sign-problem Hamiltonians.
- **Only 1-D open chains are supported.** There is no PEPS, no periodic boundaries and no MPO/DMRG. Ground states come
  from imaginary-time TEBD, which carries Trotter error O(dt²). In the benchmark that error sets a floor of about 1e-9
  relative error at χ ≥ 8.
- **CPU and NumPy only.** There is no GPU backend.
- `simulator.svd.truncated_svd` (pure Python, power iteration) remains for v1 users. It is slow and approximate;
  `qtensor` uses LAPACK.
