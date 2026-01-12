import torch

def system_loss(z_star):
    quality, toxicity, pressure, engagement = z_star.T

    loss = (
        0.3*toxicity.mean() +
        torch.relu(0.4 - engagement).mean() +
        pressure.mean()
    )
    return loss
