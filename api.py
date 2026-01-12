import torch
from fastapi import FastAPI
from pydantic import BaseModel
from engine.model import ModerationDEQ

app = FastAPI()
model = ModerationDEQ()
model.load_state_dict(torch.load("model.pt"))
model.eval()

class Request(BaseModel):
    content_quality: float
    toxicity: float
    pressure: float
    engagement: float
    strictness: float
    threshold: float

@app.post("/simulate")
def simulate(req: Request):
    z0 = torch.tensor([[req.content_quality,
                        req.toxicity,
                        req.pressure,
                        req.engagement]])
    policy = torch.tensor([req.strictness, req.threshold])

    with torch.no_grad():
        z_star = model(z0, policy)

    q, t, p, e = z_star[0].tolist()

    return {
        "stable_state": {
            "content_quality": q,
            "toxicity": t,
            "moderation_pressure": p,
            "engagement": e
        }
    }
