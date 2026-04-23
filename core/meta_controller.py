# core/meta_controller.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


# ── Meta-state builder ────────────────────────────────────────────────────────

def build_meta_state(reward_history: deque, current_arch: int, t: int,
                     episode_length: int, lift_history: deque = None) -> np.ndarray:
    """
    Extended meta-state now includes lift signal — gives meta-controller
    direct task success information to make better switching decisions.

    Components:
        mean_r, std_r, trend  — reward statistics
        lift_rate             — fraction of recent steps with cube lifted
        norm_t                — normalized time
        arch_onehot (2)       — current architecture
    Total: 7-dim meta-state
    """
    rewards = np.array(reward_history, dtype=np.float32)

    if len(rewards) == 0:
        mean_r, std_r, trend = 0.0, 0.0, 0.0
    else:
        mean_r = float(np.mean(rewards))
        std_r  = float(np.std(rewards))
        half   = max(1, len(rewards) // 2)
        trend  = float(np.mean(rewards[half:]) - np.mean(rewards[:half]))

    # lift rate — how often cube was above table recently
    if lift_history is not None and len(lift_history) > 0:
        lift_rate = float(np.mean(list(lift_history)))
    else:
        lift_rate = 0.0

    norm_t      = t / episode_length
    arch_onehot = np.zeros(2, dtype=np.float32)
    arch_onehot[current_arch] = 1.0

    return np.array([mean_r, std_r, trend, lift_rate, norm_t, *arch_onehot],
                    dtype=np.float32)


# ── Meta-controller network ───────────────────────────────────────────────────

class MetaControllerNet(nn.Module):
    def __init__(self, meta_state_dim: int = 7, n_archs: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_archs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ── Meta-controller agent ─────────────────────────────────────────────────────

class MetaController:
    def __init__(
        self,
        n_archs:           int   = 2,
        meta_state_dim:    int   = 7,
        hidden:            int   = 64,
        lr:                float = 0.001,
        gamma:             float = 0.99,
        switching_cost:    float = 0.1,
        decision_interval: int   = 20,
        history_window:    int   = 50,
        device:            str   = "cpu",
    ):
        self.n_archs           = n_archs
        self.gamma             = gamma
        self.switching_cost    = switching_cost
        self.decision_interval = decision_interval
        self.history_window    = history_window
        self.device            = torch.device(device)

        self.net       = MetaControllerNet(meta_state_dim, n_archs, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.log_probs      = []
        self.meta_rewards   = []
        self.current_arch   = 0
        self.reward_history = deque(maxlen=history_window)
        self.lift_history   = deque(maxlen=history_window)

    def select_arch(self, meta_state: np.ndarray, deterministic: bool = False) -> int:
        x     = torch.tensor(meta_state, dtype=torch.float32).to(self.device)
        probs = self.net(x)
        dist  = torch.distributions.Categorical(probs)

        action = torch.argmax(probs).item() if deterministic else dist.sample().item()
        self.log_probs.append(dist.log_prob(torch.tensor(action).to(self.device)))
        return int(action)

    def compute_meta_reward(self, task_reward: float, new_arch: int,
                            lifted: bool = False) -> float:
        switched = int(new_arch != self.current_arch)
        # bonus for switching when cube is being lifted — encourage useful switches
        lift_bonus = 0.1 if lifted and switched else 0.0
        return task_reward - self.switching_cost * switched + lift_bonus

    def store_reward(self, meta_reward: float):
        self.meta_rewards.append(meta_reward)

    def update_reward_history(self, task_reward: float, lifted: bool = False):
        self.reward_history.append(task_reward)
        self.lift_history.append(float(lifted))

    def update(self):
        if len(self.log_probs) == 0:
            return 0.0

        returns = []
        G = 0.0
        for r in reversed(self.meta_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = (-torch.stack(self.log_probs) * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.log_probs    = []
        self.meta_rewards = []
        return float(loss.item())

    def reset_episode(self):
        self.log_probs      = []
        self.meta_rewards   = []
        self.reward_history = deque(maxlen=self.history_window)
        self.lift_history   = deque(maxlen=self.history_window)
        self.current_arch   = 0

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
        print(f"Meta-controller saved → {path}")

    def load(self, path: str):
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print(f"Meta-controller loaded ← {path}")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    meta = MetaController()

    for r in [0.1, 0.3, -0.1, 0.5, 0.2]:
        meta.update_reward_history(r, lifted=False)

    state = build_meta_state(
        meta.reward_history, current_arch=0,
        t=50, episode_length=500,
        lift_history=meta.lift_history,
    )
    print(f"Meta-state: {state}")
    print(f"Meta-state dim: {len(state)}")

    arch   = meta.select_arch(state)
    meta_r = meta.compute_meta_reward(task_reward=0.4, new_arch=arch, lifted=False)
    meta.store_reward(meta_r)
    loss   = meta.update()

    print(f"Selected arch: {arch}")
    print(f"Meta loss:     {loss:.4f}")