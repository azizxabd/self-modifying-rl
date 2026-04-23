# core/policy.py
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# ── MLP Policy (memoryless) ───────────────────────────────────────────────────

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── LSTM Policy (memory-aware) ────────────────────────────────────────────────

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 128, hidden_size: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim        = int(np.prod(observation_space.shape))
        self.hidden_size = hidden_size
        self.pre_lstm    = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.lstm        = nn.LSTM(256, hidden_size, batch_first=True)
        self.fc          = nn.Sequential(
            nn.Linear(hidden_size, features_dim),
            nn.ReLU(),
        )
        self.hidden      = None

    def reset_hidden(self, batch_size: int = 1):
        device      = next(self.parameters()).device
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.pre_lstm(obs).unsqueeze(1)
        if self.hidden is None or self.hidden[0].shape[1] != x.shape[0]:
            self.reset_hidden(x.shape[0])
        out, self.hidden = self.lstm(x, self.hidden)
        self.hidden      = (self.hidden[0].detach(), self.hidden[1].detach())
        return self.fc(out[:, -1, :])


# ── Factory ───────────────────────────────────────────────────────────────────

def make_policy(arch: str, env, device: str = "cpu",
                cfg: dict = None) -> PPO:
    assert arch in ("mlp", "lstm"), "arch must be 'mlp' or 'lstm'"

    # use config values if provided, else defaults
    if cfg is not None:
        pol_cfg = cfg["inner_policies"][arch]
        lr          = pol_cfg.get("learning_rate", 0.0003)
        n_steps     = pol_cfg.get("n_steps",       1024)
        batch_size  = pol_cfg.get("batch_size",    128)
        n_epochs    = pol_cfg.get("n_epochs",      10)
        gamma       = pol_cfg.get("gamma",         0.99)
        gae_lambda  = pol_cfg.get("gae_lambda",    0.95)
        clip_range  = pol_cfg.get("clip_range",    0.2)
        ent_coef    = pol_cfg.get("ent_coef",      0.01)
        features_dim = pol_cfg.get("features_dim", 128)
        hidden_size  = pol_cfg.get("hidden_size",  128)
    else:
        lr, n_steps, batch_size, n_epochs  = 0.0003, 1024, 128, 10
        gamma, gae_lambda, clip_range      = 0.99, 0.95, 0.2
        ent_coef, features_dim, hidden_size = 0.01, 128, 128

    if arch == "mlp":
        policy_kwargs = dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[],
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_size=hidden_size,
            ),
            net_arch=[],
        )

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
    )


def save_policy(model: PPO, path: str):
    model.save(path)
    print(f"Saved → {path}")


def load_policy(arch: str, path: str, env, device: str = "cpu") -> PPO:
    model = PPO.load(path, env=env, device=device)
    print(f"Loaded ← {path}")
    return model


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.robosuite_env import RobosuitePhased
    env = RobosuitePhased(seed=42)

    for arch in ("mlp", "lstm"):
        model     = make_policy(arch, env)
        obs, _    = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"[{arch.upper()}] action shape: {action.shape} | sample: {action[:3]}")

    env.close()