import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib.pyplot as plt
from engine.model import ModerationDEQ
                                                                  
model = ModerationDEQ()
model.load_state_dict(torch.load("model.pt"))
model.eval()

strictness_values = torch.linspace(0.1, 0.9, 20)
tox_eq = []
eng_eq = []

z0 = torch.tensor([[0.6, 0.4, 0.3, 0.7]])

for s in strictness_values:
    policy = torch.tensor([s.item(), 0.5])
    z_star, _ = model(z0, policy)
    z_star = torch.clamp(z_star, 0, 1)
    tox_eq.append(z_star[0, 1].item())
    eng_eq.append(z_star[0, 3].item())
    print(s.item(), z_star.tolist())

plt.plot(strictness_values, tox_eq, label="Toxicity")
plt.plot(strictness_values, eng_eq, label="Engagement")
plt.xlabel("Moderation Strictness")
plt.ylabel("Equilibrium Value")
plt.legend()
plt.title("Policy → Equilibrium Regimes")
plt.show()
