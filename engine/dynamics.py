import torch
import torch.nn as nn

STATE_DIM = 4  # quality, toxicity, mod_pressure, engagement
POLICY_DIM = 6 # strictness, threshold

class CommunityDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + POLICY_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, STATE_DIM)
        )

    def forward(self, z, policy):
        policy = policy.unsqueeze(0).expand(z.size(0), -1)
        policy_embed = torch.cat([
            policy,
            policy ** 2,
            torch.sin(policy)], dim=-1)
        inp = torch.cat([z, policy_embed], dim=1)
        delta = self.net(inp)
        z_next = z + 0.2 * delta
        z_next = 0.9 * z_next + 0.1 * z  # inertia
        return torch.sigmoid(z_next)
