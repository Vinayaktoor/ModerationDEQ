import torch
from engine.model import ModerationDEQ

model = ModerationDEQ()
model.load_state_dict(torch.load("model.pt"))
model.eval()

z0 = torch.tensor([[0.6, 0.4, 0.3, 0.7]])
policy = torch.tensor([0.6, 0.4])

with torch.no_grad():
    z_star = model(z0, policy)

print("Stable community state:", z_star)
