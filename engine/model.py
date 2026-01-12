import torch
import torch.nn as nn
from .solver import solve_equilibrium
from .dynamics import CommunityDynamics

class ModerationDEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = CommunityDynamics()

    def forward(self, z0, policy):
        # 1. Solve equilibrium WITHOUT tracking solver path
        with torch.no_grad():
            z_star = solve_equilibrium(self.f, z0, policy)

        # 2. Re-attach graph at equilibrium
        z_star = z_star.detach().requires_grad_(True)

        # 3. One extra application to create dependency on params
        f_z = self.f(z_star, policy)

        return z_star, f_z
