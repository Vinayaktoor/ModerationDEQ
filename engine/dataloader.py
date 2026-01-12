import torch
from torch.utils.data import Dataset
import pandas as pd

class CommunityDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.z = torch.tensor(df[
            ["content_quality",
             "user_toxicity",
             "moderation_pressure",
             "engagement"]
        ].values, dtype=torch.float32)

        self.policy = torch.tensor(df[
            ["strictness", "threshold"]
        ].values, dtype=torch.float32)

        self.z_next = torch.tensor(df[
            ["next_content_quality",
             "next_user_toxicity",
             "next_moderation_pressure",
             "next_engagement"]
        ].values, dtype=torch.float32)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.policy[idx], self.z_next[idx]
