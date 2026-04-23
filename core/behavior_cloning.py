# core/behavior_cloning.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class DemoDataset(Dataset):
    def __init__(self, demonstrations: list):
        obs_list     = []
        action_list  = []

        for demo in demonstrations:
            obs_list.append(demo["obs"])
            action_list.append(demo["actions"])

        self.obs     = torch.tensor(
            np.concatenate(obs_list, axis=0), dtype=torch.float32)
        self.actions = torch.tensor(
            np.concatenate(action_list, axis=0), dtype=torch.float32)

        print(f"[BC] Dataset: {len(self.obs)} transitions")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# ── Behavior cloning trainer ──────────────────────────────────────────────────

class BehaviorCloningTrainer:
    """
    Trains a policy to imitate demonstrated actions via supervised learning.
    Works with SB3 PPO models — extracts and fine-tunes the feature extractor
    and policy network.
    """

    def __init__(self, model, device="cpu", lr=1e-4):
        self.model  = model
        self.device = torch.device(device)
        self.lr     = lr

        # extract the policy network
        self.policy = model.policy.to(self.device)

        # optimizer over all policy parameters
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def train(
        self,
        demonstrations: list,
        epochs:     int   = 20,
        batch_size: int   = 256,
        save_path:  str   = None,
    ) -> list:
        dataset    = DemoDataset(demonstrations)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=0,
        )

        self.policy.train()
        losses = []

        print(f"[BC] Training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_losses = []

            for obs_batch, action_batch in dataloader:
                obs_batch    = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                # forward pass through SB3 policy
                features = self.policy.features_extractor(obs_batch)
                latent   = self.policy.mlp_extractor.forward_actor(features)
                actions  = self.policy.action_net(latent)

                # MSE loss against demonstrated actions
                loss = self.loss_fn(actions, action_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = np.mean(epoch_losses)
            losses.append(mean_loss)
            print(f"[BC] Epoch {epoch+1:3d}/{epochs} | Loss: {mean_loss:.6f}")

        self.policy.eval()

        if save_path:
            self.model.save(save_path)
            print(f"[BC] Saved BC model → {save_path}")

        return losses