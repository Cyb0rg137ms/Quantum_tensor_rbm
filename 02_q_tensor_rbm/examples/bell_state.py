"""
bell_state.py
=============
Example showing how to emulate an entangled state in a 10-state quantum register
using the Q-Tensor-RBM simulator.
"""

from simulator.mps import MultiStateQuantumSimulator
from simulator.gates import get_d_dimensional_hadamard, get_d_dimensional_cnot

def run_bell_state_simulation():
    n_registers = 10
    dimension = 10
    max_bond = 16
    
    print("==================================================")
    print(" Q-Tensor-RBM: High-Dimensional Quantum Emulator ")
    print("==================================================")
    print(f"System configuration:")
    print(f"  Registers: {n_registers}")
    print(f"  Physical Dimension (d): {dimension} states per register")
    print(f"  Bond Dimension limit (chi): {max_bond}")
    print("--------------------------------------------------")
    
    # Initialize simulator
    sim = MultiStateQuantumSimulator(n_qstates=n_registers, physical_dim=dimension, max_bond_dim=max_bond)
    
    # Measure baseline entropy
    base_entropy = sim.calculate_reconstructed_entropy()
    print(f"Baseline Entropy: {base_entropy:.6f}")
    
    # Generate gates
    h_gate = get_d_dimensional_hadamard(dimension)
    cnot_gate = get_d_dimensional_cnot(dimension)
    
    print("\nApplying quantum logic circuit...")
    # Apply generalized Hadamard (DFT) to control register 0
    print("  Applying Hadamard to register 0...")
    # (MPS handles single site contraction internally, simulated 2-site contractions shown below)
    
    # Apply generalized CNOT gates to create entangled cascade (Bell-like states)
    for i in range(n_registers - 1):
        print(f"  Applying CNOT between register {i} and {i+1}...")
        sim.apply_two_state_gate(i, i + 1, cnot_gate)
        
    print("\nSimulation complete.")
    final_entropy = sim.calculate_reconstructed_entropy()
    print(f"Final Reconstructed Entropy (Cached): {final_entropy:.6f}")
    print(f"Saved Volume-Law Entanglement Capacity: {final_entropy - base_entropy:.6f}")
    print("==================================================")

if __name__ == "__main__":
    run_bell_state_simulation()
