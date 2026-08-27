import pytest
import cmath
from simulator.mps import MultiStateQuantumSimulator
from simulator.neural_cache import NeuralEntanglementCache
from simulator.gates import (
    get_d_dimensional_hadamard,
    get_d_dimensional_cnot,
    get_d_dimensional_phase_shift
)

def test_simulator_initialization():
    sim = MultiStateQuantumSimulator(n_qstates=5, physical_dim=10, max_bond_dim=16)
    assert sim.n == 5
    assert sim.d == 10
    assert len(sim.tensors) == 5
    # Verify bounds are set correctly
    assert len(sim.tensors[0]) == 1  # Boundary left dimension
    assert len(sim.tensors[-1][0][0]) == 1  # Boundary right dimension

def test_gate_generation():
    d = 3
    hadamard = get_d_dimensional_hadamard(d)
    assert len(hadamard) == d
    assert len(hadamard[0]) == d
    
    # Check unitarity: H * H_dagger = I
    for r in range(d):
        for c in range(d):
            val = sum(hadamard[r][k] * hadamard[c][k].conjugate() for k in range(d))
            if r == c:
                assert abs(val - 1.0) < 1e-9
            else:
                assert abs(val) < 1e-9

def test_gate_cnot():
    d = 2
    cnot = get_d_dimensional_cnot(d)
    assert len(cnot) == 4
    # State basis order: |00>, |01>, |10>, |11>
    # CNOT on qubit: control is 1 (second half of state), target is flipped.
    # index 2 (|10>) -> output index 3 (|11>)
    assert cnot[3][2] == 1.0
    assert cnot[2][3] == 1.0
    # control is 0: identity
    assert cnot[0][0] == 1.0
    assert cnot[1][1] == 1.0

def test_neural_entanglement_cache():
    cache = NeuralEntanglementCache(n_qstates=3, physical_dim=4, hidden_units=10)
    # Perform update
    cache.ingest_truncated_modes(
        site=1, 
        left_modes=[complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)],
        right_modes=[complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)],
        singular_values=[0.2]
    )
    # Check energy correction factor is complex number
    corr = cache.compute_energy_correction([0, 0, 0])
    assert isinstance(corr, complex)
    assert not cmath.isnan(corr)

def test_simulator_gate_application():
    sim = MultiStateQuantumSimulator(n_qstates=5, physical_dim=10, max_bond_dim=16)
    initial_entropy = sim.calculate_reconstructed_entropy()
    
    # Apply nearest-neighbor gates
    for i in range(4):
        sim.apply_two_state_gate(i, i + 1)
        
    final_entropy = sim.calculate_reconstructed_entropy()
    # Entanglement should increase after SVD truncations are cached
    assert final_entropy > initial_entropy
