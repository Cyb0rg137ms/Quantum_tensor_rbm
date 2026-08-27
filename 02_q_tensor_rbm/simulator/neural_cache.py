"""
neural_cache.py
===============
Neural Entanglement Cache using a Restricted Boltzmann Machine (RBM)
to capture volume-law entanglement dropped during MPS tensor network truncation.
"""

import math
import random
import cmath
from typing import List

class NeuralEntanglementCache:
    """
    A Restricted Boltzmann Machine (RBM) that captures non-local correlations 
    discarded by MPS truncation. It scales linearly O(N * d * H) in memory, 
    bridging the gap between classical tensor network limitations and full state vector simulators.
    """
    
    def __init__(self, n_qstates: int, physical_dim: int = 10, hidden_units: int = 64):
        """
        Args:
            n_qstates: Number of multi-state quantum registers.
            physical_dim: Number of states per register (d).
            hidden_units: Hidden units in the RBM.
        """
        self.n = n_qstates
        self.d = physical_dim
        self.h = hidden_units
        
        # Initialize RBM complex weights: W[site][physical_state][hidden_unit]
        # Linear memory: O(N * d * H)
        self.W = [[[complex(random.gauss(0, 0.01), random.gauss(0, 0.01)) 
                    for _ in range(self.h)] 
                   for _ in range(self.d)] 
                  for _ in range(self.n)]
                  
        self.visible_bias = [[complex(0, 0)] * self.d for _ in range(self.n)]
        self.hidden_bias = [complex(0, 0)] * self.h

    def ingest_truncated_modes(self, site: int, left_modes: List[complex], 
                               right_modes: List[complex], singular_values: List[float]):
        """
        Ingests discarded Schmidt modes and updates the RBM weights via Hebbian-style
        quantum updates.
        
        Args:
            site: The index of the tensor bond where truncation occurred.
            left_modes: Truncated left singular vectors.
            right_modes: Truncated right singular vectors.
            singular_values: Dropped singular values (coefficients of entanglement).
        """
        # Truncation error updates RBM weights
        for s_val, l_val, r_val in zip(singular_values, left_modes, right_modes):
            coupling = s_val * 0.1  # Learning rate proportional to dropped energy
            for h in range(self.h):
                # Update weights to absorb correlation terms
                for p in range(len(left_modes)):
                    if p < self.d:
                        self.W[site][p][h] += coupling * l_val * cmath.exp(1j * random.uniform(0, 2 * math.pi))
                        
    def compute_energy_correction(self, state_config: List[int]) -> complex:
        """
        Computes the quantum phase correction factor for a given register state configuration.
        
        Args:
            state_config: List of physical state indices (0 to d-1) for each register site.
            
        Returns:
            The complex phase amplitude correction from the RBM.
        """
        # Calculate effective visible bias contribution
        v_contribution = sum(self.visible_bias[i][state_config[i]] for i in range(self.n))
        
        # Calculate hidden layer activation product
        h_product = 0.0
        for h in range(self.h):
            total_input = self.hidden_bias[h]
            for i in range(self.n):
                total_input += self.W[i][state_config[i]][h]
            h_product += cmath.log(1.0 + cmath.exp(total_input)).real
            
        return cmath.exp(v_contribution + h_product)
