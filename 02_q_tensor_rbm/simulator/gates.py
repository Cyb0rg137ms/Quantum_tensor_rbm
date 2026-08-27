"""
gates.py
========
Unitary gate operators for high-dimensional quantum states (d-state registers).
Exposes matrices for d-dimensional Hadamard, controlled-NOT, and Phase-Shift gates.
"""

import cmath
import math
from typing import List

def get_d_dimensional_hadamard(d: int) -> List[List[complex]]:
    """
    Generates the discrete Fourier transform (DFT) matrix, which acts
    as the generalized Hadamard gate in d-dimensional state spaces.
    """
    omega = cmath.exp(2j * cmath.pi / d)
    factor = 1.0 / math.sqrt(d)
    
    gate = []
    for r in range(d):
        row = []
        for c in range(d):
            row.append(factor * (omega ** (r * c)))
        gate.append(row)
    return gate

def get_d_dimensional_cnot(d: int) -> List[List[complex]]:
    """
    Generates the generalized CNOT gate for two d-dimensional registers,
    where the target register is incremented modulo d if the control is in the state d-1.
    Shape: (d^2, d^2)
    """
    size = d * d
    gate = [[complex(0, 0) for _ in range(size)] for _ in range(size)]
    
    for ctrl in range(d):
        for target in range(d):
            # Input index in combined basis
            idx_in = ctrl * d + target
            
            if ctrl == d - 1:
                # Target is shifted modulo d
                idx_out = ctrl * d + ((target + 1) % d)
            else:
                # Identity
                idx_out = idx_in
                
            gate[idx_out][idx_in] = complex(1.0, 0.0)
            
    return gate

def get_d_dimensional_phase_shift(d: int, phi: float) -> List[List[complex]]:
    """
    Generates a phase-shift gate that applies a phase phi to the last state (d-1).
    Shape: (d, d)
    """
    gate = [[complex(0, 0) for _ in range(d)] for _ in range(d)]
    for i in range(d - 1):
        gate[i][i] = complex(1.0, 0.0)
    gate[d - 1][d - 1] = cmath.exp(1j * phi)
    return gate
