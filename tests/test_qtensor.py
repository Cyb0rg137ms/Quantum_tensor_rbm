"""Tests for the v2 ``qtensor`` package (validated against dense simulation and exact diagonalisation)."""

from __future__ import annotations

import numpy as np
import pytest

from qtensor import (
    MPS,
    TFIM,
    RBMState,
    StateVector,
    csum,
    dft,
    exact_ground_state,
    haar_unitary,
    overlap,
    tebd,
    train_exact,
)
from qtensor import gates as G
from qtensor.benchmarks import BenchConfig, brick_circuit, run_all
from qtensor.cli import main
from qtensor.rbm import all_spins
from simulator import MultiStateQuantumSimulator


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_gates_unitary(d: int, rng) -> None:
    for g in [
        G.dft(d),
        G.shift(d),
        G.clock(d),
        G.phase(d, 0.3),
        G.csum(d),
        G.controlled_increment(d),
        G.swap(d),
        G.haar_unitary(d * d, rng),
    ]:
        assert G.is_unitary(g)
    assert np.allclose(G.csum(2), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    # Weyl commutation: Z X = w X Z
    assert np.allclose(G.clock(d) @ G.shift(d), np.exp(2j * np.pi / d) * G.shift(d) @ G.clock(d))


def test_expm_hermitian() -> None:
    h = np.kron(G.Z, G.Z)
    assert np.allclose(G.expm_hermitian(h, 0.3), np.diag(np.exp(-0.3j * np.diag(h))))
    assert np.allclose(G.expm_hermitian(h, -1j * 0.5), np.diag(np.exp(-0.5 * np.diag(h))))


@pytest.mark.parametrize("n,d", [(8, 2), (5, 3)])
def test_mps_matches_dense_on_random_circuits(n: int, d: int, rng) -> None:
    m, sv = MPS(n, d, max_bond=None), StateVector(n, d)
    for k in range(25):
        i, j = (int(x) for x in rng.choice(n, 2, replace=False))
        u = haar_unitary(d * d, rng)
        m.apply_2site(u, i, j)
        sv.apply_2site(u, i, j)
        v = haar_unitary(d, rng)
        m.apply_1site(v, k % n)
        sv.apply_1site(v, k % n)
    assert abs(np.vdot(sv.vector(), m.to_dense())) ** 2 == pytest.approx(1.0, abs=1e-10)
    for b in range(n - 1):
        assert m.entropy(b) == pytest.approx(sv.entropy(b + 1), abs=1e-9)
    assert m.norm() == pytest.approx(1.0)
    assert m.stats.total_discarded < 1e-20
    config = [int(x) for x in rng.integers(0, d, n)]
    assert m.amplitude(config) == pytest.approx(sv.psi[tuple(config)], abs=1e-10)
    op = G.clock(d)
    assert m.expectation_1site(op, 2) == pytest.approx(sv.expectation_1site(op, 2), abs=1e-10)


def test_reversed_site_order_gate(rng) -> None:
    u = haar_unitary(4, rng)
    m, sv = MPS(4, 2, None), StateVector(4, 2)
    m.apply_1site(G.H, 3)
    sv.apply_1site(G.H, 3)
    m.apply_2site(u, 3, 1)
    sv.apply_2site(u, 3, 1)
    assert abs(np.vdot(sv.vector(), m.to_dense())) ** 2 == pytest.approx(1.0)


@pytest.mark.parametrize("d", [2, 4, 10])
def test_qudit_ghz_exact(d: int) -> None:
    n = 12
    m = MPS(n, d, max_bond=d)
    m.apply_1site(dft(d), 0)
    for i in range(n - 1):
        m.apply_2site(csum(d), i)
    assert m.bond_dims == [d] * (n - 1)
    assert np.allclose(m.entropies(), np.log(d))
    assert m.amplitude([d - 1] * n) == pytest.approx(1 / np.sqrt(d))
    s = m.sample(50, np.random.default_rng(1))
    assert np.all(s == s[:, :1])  # perfectly correlated
    assert m.stats.total_discarded == 0


def test_truncation_bounded_and_fidelity_estimate() -> None:
    rows = brick_circuit(10, 8, [2, 8, 32], seed=0)
    assert rows[0]["true_fidelity"] < rows[1]["true_fidelity"] < rows[2]["true_fidelity"]
    assert rows[2]["true_fidelity"] == pytest.approx(1.0, abs=1e-9)
    for r in rows:
        assert r["true_fidelity"] == pytest.approx(r["estimated_fidelity"], abs=0.05)


def test_max_bond_is_respected(rng) -> None:
    m = MPS(10, 2, max_bond=4)
    for _ in range(5):
        for i in range(9):
            m.apply_2site(haar_unitary(4, rng), i)
    assert max(m.bond_dims) <= 4 and m.norm() == pytest.approx(1.0)
    assert 0 < m.stats.fidelity_estimate < 1


def test_sampling_distribution_matches_born_rule(rng) -> None:
    m = MPS(3, 2, None)
    m.apply_1site(G.haar_unitary(2, rng), 0)
    m.apply_2site(haar_unitary(4, rng), 0)
    m.apply_2site(haar_unitary(4, rng), 1)
    probs = np.abs(m.to_dense()) ** 2
    s = m.sample(20000, np.random.default_rng(2))
    idx = s[:, 0] * 4 + s[:, 1] * 2 + s[:, 2]
    emp = np.bincount(idx, minlength=8) / len(idx)
    assert np.abs(emp - probs).max() < 0.015


def test_overlap(rng) -> None:
    a, b = MPS(5, 2, None), MPS(5, 2, None)
    assert overlap(a, b) == pytest.approx(1.0)
    b.apply_1site(G.X, 2)
    assert abs(overlap(a, b)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        overlap(a, MPS(4, 2))


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        MPS(0)
    with pytest.raises(ValueError):
        MPS(3, 2, max_bond=0)
    with pytest.raises(ValueError):
        MPS(3, 2, config=[0, 2, 0])
    m = MPS(3)
    with pytest.raises(ValueError):
        m.apply_2site(G.CNOT, 1, 1)
    with pytest.raises(IndexError):
        m.entropy(2)
    with pytest.raises(MemoryError):
        StateVector(30, 2)


def test_tfim_dense_matches_bond_terms() -> None:
    model = TFIM(6, 1.0, 0.7)
    e0, v0 = exact_ground_state(model)
    psi = MPS(6, 2, None)
    for i in range(6):
        psi.apply_1site(G.H, i)
    tebd(psi, model.bond_terms(), 0.05, 150, imaginary=True)
    assert model.energy(psi) == pytest.approx(e0, rel=1e-5)
    assert abs(np.vdot(v0, psi.to_dense())) ** 2 > 0.999


def test_real_time_tebd_matches_exact_evolution() -> None:
    model = TFIM(6, 1.0, 1.0)
    psi = MPS(6, 2, None)
    tebd(psi, model.bond_terms(), 0.01, 100)  # t = 1
    exact = G.expm_hermitian(model.dense(), 1.0)[:, 0]
    assert abs(np.vdot(exact, psi.to_dense())) ** 2 > 0.9999


def test_rbm_reaches_ground_state() -> None:
    model = TFIM(6)
    e0, _ = exact_ground_state(model)
    st = RBMState.random(6, alpha=2)
    curve = train_exact(st, model.dense(), steps=300, lr=0.05)
    assert curve[-1] < curve[0]
    assert abs(curve[-1] - e0) / abs(e0) < 1e-3


def test_rbm_log_derivatives_match_finite_differences() -> None:
    st = RBMState.random(4, 2, scale=0.3, seed=1)
    s = all_spins(4)[:3]
    an = st.log_derivatives(s)
    v = st.get_flat()
    eps = 1e-6
    for k in [0, 5, 13]:
        vp = v.copy()
        vp[k] += eps
        st.set_flat(vp)
        up = st.log_psi(s)
        vp[k] -= 2 * eps
        st.set_flat(vp)
        dn = st.log_psi(s)
        st.set_flat(v)
        assert np.allclose((up - dn) / (2 * eps), an[:, k], atol=1e-6)


def test_rbm_plain_gradient_mode() -> None:
    curve = train_exact(RBMState.random(4), TFIM(4).dense(), steps=50, lr=0.02, sr=False)
    assert curve[-1] < curve[0]


def test_legacy_simulator_now_applies_real_gates() -> None:
    sim = MultiStateQuantumSimulator(4, 3, 9)
    assert sim.calculate_reconstructed_entropy() == 0.0
    sim.apply_two_state_gate(0, 1)
    assert sim.calculate_reconstructed_entropy() == pytest.approx(np.log(3))


def test_bench_quick_and_cli(capsys) -> None:
    r = run_all(BenchConfig(quick=True))
    assert all(a["fidelity"] > 1 - 1e-9 for a in r["agreement"])
    assert main(["ghz", "--n", "20", "--d", "3"]) == 0
    assert '"discarded": 0.0' in capsys.readouterr().out
