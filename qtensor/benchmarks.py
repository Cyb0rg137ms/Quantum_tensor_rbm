"""Reproducible CPU benchmarks; every README number comes from ``qtensor bench``."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from qtensor.gates import H, csum, dft, haar_unitary
from qtensor.models import TFIM, exact_ground_state, tebd
from qtensor.mps import MPS
from qtensor.rbm import RBMState, train_exact
from qtensor.statevector import StateVector


@dataclass
class BenchConfig:
    seed: int = 0
    brick_n: int = 16
    brick_depth: int = 12
    tfim_n: int = 10
    quick: bool = False


def random_circuit_agreement(n: int, d: int, gates: int, seed: int) -> dict:
    """Untruncated MPS vs dense state vector on random (incl. non-adjacent) 2-site Haar gates."""
    rng = np.random.default_rng(seed)
    m, sv = MPS(n, d, max_bond=None), StateVector(n, d)
    for _ in range(gates):
        i, j = (int(x) for x in rng.choice(n, 2, replace=False))
        u = haar_unitary(d * d, rng)
        m.apply_2site(u, i, j)
        sv.apply_2site(u, i, j)
    fid = abs(np.vdot(sv.vector(), m.to_dense())) ** 2
    ent_err = max(abs(m.entropy(b) - sv.entropy(b + 1)) for b in range(n - 1))
    return {
        "n": n,
        "d": d,
        "gates": gates,
        "fidelity": float(fid),
        "max_entropy_err": float(ent_err),
        "max_bond": max(m.bond_dims),
    }


def qudit_ghz(n: int, d: int) -> dict:
    t0 = time.perf_counter()
    m = MPS(n, d, max_bond=d)
    m.apply_1site(dft(d), 0)
    for i in range(n - 1):
        m.apply_2site(csum(d), i)
    dt = time.perf_counter() - t0
    ent = m.entropies()
    return {
        "n": n,
        "d": d,
        "seconds": dt,
        "mps_bytes": m.memory_bytes(),
        "dense_bytes_log10": float(n * np.log10(d) + np.log10(16)),
        "entropy_min": min(ent),
        "entropy_max": max(ent),
        "log_d": float(np.log(d)),
        "discarded": m.stats.total_discarded,
    }


def brick_circuit(n: int, depth: int, chis: list[int], seed: int) -> list[dict]:
    """Haar brick-wall circuit: true fidelity vs bond dimension (the volume-law limit of MPS)."""
    rng = np.random.default_rng(seed)
    layers = [[(i, haar_unitary(4, rng)) for i in range(L % 2, n - 1, 2)] for L in range(depth)]
    sv = StateVector(n, 2)
    for layer in layers:
        for i, u in layer:
            sv.apply_2site(u, i, i + 1)
    exact = sv.vector()
    rows = []
    for chi in chis:
        m = MPS(n, 2, max_bond=chi)
        t0 = time.perf_counter()
        for layer in layers:
            for i, u in layer:
                m.apply_2site(u, i)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "chi": chi,
                "true_fidelity": float(abs(np.vdot(exact, m.to_dense())) ** 2),
                "estimated_fidelity": m.stats.fidelity_estimate,
                "seconds": dt,
                "mps_bytes": m.memory_bytes(),
                "dense_bytes": exact.nbytes,
                "mid_entropy": m.entropy(n // 2 - 1),
            }
        )
    return rows


def tfim_ground_state(n: int, quick: bool, seed: int) -> dict:
    model = TFIM(n, 1.0, 1.0)
    t0 = time.perf_counter()
    e0, _ = exact_ground_state(model)
    out: dict = {"n": n, "exact_energy": e0, "ed_seconds": time.perf_counter() - t0, "tebd": [], "rbm": []}
    for chi in [2, 4, 8, 16]:
        psi = MPS(n, 2, max_bond=chi)
        for i in range(n):
            psi.apply_1site(H, i)
        t0 = time.perf_counter()
        tebd(psi, model.bond_terms(), 0.1, 40 if quick else 100, imaginary=True)
        tebd(psi, model.bond_terms(), 0.02, 40 if quick else 100, imaginary=True)
        e = model.energy(psi)
        out["tebd"].append(
            {"chi": chi, "energy": e, "rel_err": abs(e - e0) / abs(e0), "seconds": time.perf_counter() - t0}
        )
    Hd = model.dense()
    for alpha in [1, 2]:
        st = RBMState.random(n, alpha, seed=seed)
        t0 = time.perf_counter()
        curve = train_exact(st, Hd, steps=150 if quick else 600, lr=0.05)
        e = curve[-1]
        out["rbm"].append(
            {
                "alpha": alpha,
                "params": st.n_params,
                "energy": e,
                "rel_err": abs(e - e0) / abs(e0),
                "seconds": time.perf_counter() - t0,
            }
        )
    return out


def run_all(cfg: BenchConfig | None = None) -> dict:
    cfg = cfg or BenchConfig()
    t0 = time.perf_counter()
    chis = [2, 4, 8, 16, 32] if cfg.quick else [2, 4, 8, 16, 32, 64, 128, 256]
    res: dict = {
        "config": asdict(cfg),
        "agreement": [random_circuit_agreement(10, 2, 60, cfg.seed), random_circuit_agreement(6, 3, 40, cfg.seed)],
        "ghz": [qudit_ghz(n, 10) for n in ([10, 50] if cfg.quick else [10, 100, 1000])],
        "brick": brick_circuit(cfg.brick_n if not cfg.quick else 12, cfg.brick_depth, chis, cfg.seed),
        "tfim": tfim_ground_state(cfg.tfim_n if not cfg.quick else 8, cfg.quick, cfg.seed),
    }
    res["total_s"] = time.perf_counter() - t0
    return res
