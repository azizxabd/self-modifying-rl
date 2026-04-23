# core/lunarlander_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Phase definitions ─────────────────────────────────────────────────────────

LUNARLANDER_PHASES = {
    0: {
        "name":        "Normal",
        "gravity":     -10.0,
        "wind_power":  0.0,
        "turbulence":  0.0,
        "color":       "#DBEAFE",
    },
    1: {
        "name":        "Heavy",
        "gravity":     -11.5,   # max allowed is just under -12
        "wind_power":  0.0,
        "turbulence":  0.0,
        "color":       "#FEF9C3",
    },
    2: {
        "name":        "Windy",
        "gravity":     -10.0,
        "wind_power":  15.0,
        "turbulence":  1.5,
        "color":       "#DCFCE7",
    },
}

PHASE_SCHEDULE = [0, 1, 2, 0, 1]


# ── Wrapper ───────────────────────────────────────────────────────────────────

class LunarLanderPhased(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        episode_length: int = 500,
        phase_duration: int = 100,
        render_mode:    str = None,
        seed:           int = None,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.phase_duration = phase_duration
        self.render_mode    = render_mode
        self.phase_schedule = PHASE_SCHEDULE
        self.rng            = np.random.default_rng(seed)
        self.history        = []
        self.t              = 0
        self._seed          = seed

        self._make_env(phase_id=0)

        # obs: 8 lunar lander obs + phase indicator = 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(9,), dtype=np.float32,
        )
        self.action_space = self._env.action_space

    def _make_env(self, phase_id):
        phase = LUNARLANDER_PHASES[phase_id]
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
        self._env = gym.make(
            "LunarLander-v3",
            render_mode=self.render_mode,
            gravity=phase["gravity"],
            enable_wind=phase["wind_power"] > 0,
            wind_power=phase["wind_power"],
            turbulence_power=phase["turbulence"],
        )

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _current_phase_idx(self):
        return min(self.t // self.phase_duration, len(self.phase_schedule) - 1)

    def _current_phase_id(self):
        return self.phase_schedule[self._current_phase_idx()]

    def _phase_indicator(self):
        return (self.t % self.phase_duration) / self.phase_duration

    def _get_obs(self, obs):
        return np.append(obs, self._phase_indicator()).astype(np.float32)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng   = np.random.default_rng(seed)
        self.t       = 0
        self.history = []
        self._make_env(phase_id=0)
        obs, info = self._env.reset(seed=self._seed)
        return self._get_obs(obs), info

    def step(self, action):
        phase_id     = self._current_phase_id()
        prev_phase   = self.phase_schedule[
            min((self.t - 1) // self.phase_duration,
                len(self.phase_schedule) - 1)
        ] if self.t > 0 else 0

        # recreate env when phase changes
        if self.t > 0 and phase_id != prev_phase:
            self._make_env(phase_id)
            obs, _ = self._env.reset(seed=self._seed)
            self.t += 1
            self.history.append({
                "t":          self.t,
                "reward":     0.0,
                "phase_id":   phase_id,
                "phase_name": LUNARLANDER_PHASES[phase_id]["name"],
            })
            return self._get_obs(obs), 0.0, False, False, {
                "phase_id":   phase_id,
                "phase_name": LUNARLANDER_PHASES[phase_id]["name"],
            }

        obs, reward, terminated, truncated, info = self._env.step(action)

        self.t    += 1
        truncated  = truncated or (self.t >= self.episode_length)

        self.history.append({
            "t":          self.t,
            "reward":     float(reward),
            "phase_id":   phase_id,
            "phase_name": LUNARLANDER_PHASES[phase_id]["name"],
        })

        return self._get_obs(obs), reward, terminated, truncated, {
            "phase_id":   phase_id,
            "phase_name": LUNARLANDER_PHASES[phase_id]["name"],
        }

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def get_phase_boundaries(self):
        boundaries = []
        for i, phase_id in enumerate(self.phase_schedule):
            boundaries.append({
                "start":      i * self.phase_duration,
                "end":        min((i+1) * self.phase_duration, self.episode_length),
                "phase_id":   phase_id,
                "phase_name": LUNARLANDER_PHASES[phase_id]["name"],
                "color":      LUNARLANDER_PHASES[phase_id]["color"],
            })
        return boundaries

    def get_history_arrays(self):
        if not self.history:
            return {}
        result = {}
        for k in self.history[0].keys():
            try:
                result[k] = np.array([h[k] for h in self.history], dtype=np.float32)
            except Exception:
                result[k] = np.array([h[k] for h in self.history])
        return result


# ── Parallel env factory ──────────────────────────────────────────────────────

def make_lunarlander_env(seed: int, episode_length: int = 500,
                         phase_duration: int = 100):
    def _init():
        return LunarLanderPhased(
            episode_length=episode_length,
            phase_duration=phase_duration,
            seed=seed,
        )
    return _init


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env    = LunarLanderPhased(seed=42)
    obs, _ = env.reset()
    print(f"Obs shape:    {obs.shape}")
    print(f"Action space: {env.action_space}")

    done  = False
    total = 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        total += reward
        done   = terminated or truncated

    print(f"Total reward:     {total:.3f}")
    print(f"Phase boundaries: {env.get_phase_boundaries()}")
    print("LunarLander environment test passed.")