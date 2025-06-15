import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ActionLatentPairDataset(Dataset):
    def __init__(self, json_path, n_actions=9):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.n_actions = n_actions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        action = item['action']
        latent_code = item['latent_code']
        # One-hot encode action
        action_one_hot = torch.zeros(self.n_actions, dtype=torch.float32)
        action_one_hot[action] = 1.0
        # Convert latent code to tensor
        latent_code = torch.tensor(latent_code, dtype=torch.long)
        return action_one_hot, latent_code

def get_action_latent_dataloaders(json_path, batch_size=32, num_workers=4, train_split=0.8, n_actions=9):
    dataset = ActionLatentPairDataset(json_path, n_actions=n_actions)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
