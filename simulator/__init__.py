"""Legacy v1 API (kept for compatibility). New code should use :mod:`qtensor`."""

from simulator.gates import get_d_dimensional_cnot, get_d_dimensional_hadamard, get_d_dimensional_phase_shift
from simulator.mps import MultiStateQuantumSimulator
from simulator.neural_cache import NeuralEntanglementCache

__all__ = [
    "MultiStateQuantumSimulator",
    "NeuralEntanglementCache",
    "get_d_dimensional_cnot",
    "get_d_dimensional_hadamard",
    "get_d_dimensional_phase_shift",
]
