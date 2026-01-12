import torch

def solve_equilibrium(f, z0, policy, max_iter=200, tol=1e-4):
    z = z0
    for _ in range(max_iter):
        z_next = f(z, policy)
        if torch.norm(z_next - z) < tol:
            break
        z = z_next
    return z