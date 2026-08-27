"""
benchmark.py
============
Performance benchmark script for Q-Tensor-RBM simulator.
Measures custom power iteration SVD decomposition speed and accuracy.
"""

import time
import numpy as np
from simulator.svd import power_iteration_svd

def benchmark_svd():
    # Setup random 100x100 matrix
    np.random.seed(42)
    A = np.random.randn(100, 100).astype(np.float32)
    
    t0 = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        U, S, Vh = power_iteration_svd(A, k=10)
    elapsed = (time.perf_counter() - t0) / iterations
    print(f"SVD Decomposition Time Ms: {elapsed * 1000.0:.3f}")

if __name__ == "__main__":
    print("Running Q-Tensor-RBM benchmarks...")
    benchmark_svd()
