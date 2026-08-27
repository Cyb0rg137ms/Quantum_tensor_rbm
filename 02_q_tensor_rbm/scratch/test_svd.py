import math
from simulator.svd import truncated_svd

A = [[complex(1, 2), complex(3, -1)],
     [complex(0, 1), complex(-2, 4)]]
s, U, V = truncated_svd(A)

print("s:", s)
print("U:", U)
print("V:", V)

# Check V orthonormality
dot = sum(V[0][i].conjugate() * V[1][i] for i in range(2))
print("V0 . V1:", dot)
print("norm V0:", sum(abs(x)**2 for x in V[0]))
print("norm V1:", sum(abs(x)**2 for x in V[1]))

# Check U orthonormality
dot_u = sum(U[0][i].conjugate() * U[1][i] for i in range(2))
print("U0 . U1:", dot_u)
print("norm U0:", sum(abs(x)**2 for x in U[0]))
print("norm U1:", sum(abs(x)**2 for x in U[1]))

# Reconstruct
reconstructed = [[complex(0) for _ in range(2)] for _ in range(2)]
for i in range(2):
    for j in range(2):
        for k in range(len(s)):
            reconstructed[i][j] += U[k][i] * s[k] * V[k][j].conjugate()

print("A:")
for r in A: print(r)
print("reconstructed:")
for r in reconstructed: print(r)
