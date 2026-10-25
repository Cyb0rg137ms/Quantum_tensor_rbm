"""Qudit GHZ state (d-level generalisation of a Bell/GHZ cascade) on an MPS, verified against theory.

|GHZ_d> = (1/sqrt d) sum_k |k k ... k>: every bond carries exactly ln(d) nats of entanglement,
so bond dimension d is exact (no truncation) and memory is linear in n.
"""

import numpy as np

from qtensor import MPS, csum, dft


def main(n: int = 10, d: int = 10) -> None:
    psi = MPS(n, d, max_bond=d)
    psi.apply_1site(dft(d), 0)
    for i in range(n - 1):
        psi.apply_2site(csum(d), i)
    print(f"{n} registers x {d} levels, bond dims {psi.bond_dims}")
    print(f"entropy per bond {psi.entropies()[0]:.6f} nats (theory ln d = {np.log(d):.6f})")
    print(f"amplitude <3...3|psi> = {psi.amplitude([3] * n):.6f} (theory {1 / np.sqrt(d):.6f})")
    print(f"amplitude <0...01|psi> = {abs(psi.amplitude([0] * (n - 1) + [1])):.1e}")
    print("samples:", psi.sample(5, np.random.default_rng(0))[:, :6].tolist(), "...")
    print(f"discarded weight: {psi.stats.total_discarded}, MPS memory {psi.memory_bytes()} B vs dense 16*{d}^{n} B")


if __name__ == "__main__":
    main()
