import torch
from engine.model import ModerationDEQ
from engine.loss import system_loss
from torch.utils.data import DataLoader
from engine.dataloader import CommunityDataset

dataset = CommunityDataset("data/community_dynamics.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ModerationDEQ()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for z, policy, z_next in loader:
        z_star, f_z = model(z, policy[0])

        # 1️⃣ Equilibrium objective
        eq_loss = system_loss(z_star)

        # 2️⃣ One-step realism objective
        step_loss = torch.mean((f_z - z_next) ** 2)

        
        eq_reg = torch.mean((f_z - z_star) ** 2)
        loss = eq_loss + 0.5 * step_loss + 0.1 * eq_reg

        # 1️⃣ ∂L / ∂z*
        dL_dz = torch.autograd.grad(
            loss, z_star, retain_graph=True)[0]

        # 2️⃣ ∂L / ∂θ via implicit chain rule
        grads = torch.autograd.grad(
            outputs=f_z,
            inputs=model.parameters(),
            grad_outputs=dL_dz,
            retain_graph=False
        )
        optimizer.zero_grad()
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        optimizer.step()
        
        with torch.no_grad():
            residual = torch.norm(f_z - z_star).item()
    print("equilibrium residual:", residual)    

    print(f"epoch {epoch} | loss {loss.item():.4f}")
    
torch.save(model.state_dict(), "model.pt")