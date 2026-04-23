# core/cartpole_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Phase definitions ─────────────────────────────────────────────────────────

CARTPOLE_PHASES = {
    0: {
        "name":          "Normal",
        "gravity":       9.8,
        "pole_mass":     0.1,
        "cart_mass":     1.0,
        "disturbance":   0.0,
        "color":         "#DBEAFE",
    },
    1: {
        "name":          "Heavy",
        "gravity":       9.8,
        "pole_mass":     0.3,   # 3x heavier pole
        "cart_mass":     1.0,
        "disturbance":   0.0,
        "color":         "#FEF9C3",
    },
    2: {
        "name":          "Windy",
        "gravity":       9.8,
        "pole_mass":     0.1,
        "cart_mass":     1.0,
        "disturbance":   0.5,   # random force on cart
        "color":         "#DCFCE7",
    },
}

PHASE_SCHEDULE = [0, 1, 2, 0, 1]


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CartPolePhased(gym.Env):
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

        self._env = gym.make(
            "CartPole-v1",
            render_mode=render_mode,
        )

        # obs: [cart_pos, cart_vel, pole_angle, pole_vel, phase_indicator]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(5,), dtype=np.float32,
        )
        self.action_space = self._env.action_space

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _current_phase_idx(self):
        return min(self.t // self.phase_duration, len(self.phase_schedule) - 1)

    def _current_phase_id(self):
        return self.phase_schedule[self._current_phase_idx()]

    def _phase_indicator(self):
        return (self.t % self.phase_duration) / self.phase_duration

    def _apply_phase(self, phase_id):
        phase = CARTPOLE_PHASES[phase_id]
        self._env.unwrapped.masscart  = phase["cart_mass"]
        self._env.unwrapped.masspole  = phase["pole_mass"]
        self._env.unwrapped.gravity   = phase["gravity"]
        self._env.unwrapped.total_mass = phase["cart_mass"] + phase["pole_mass"]
        self._env.unwrapped.polemass_length = phase["pole_mass"] * self._env.unwrapped.length

    def _apply_disturbance(self, phase_id):
        dist = CARTPOLE_PHASES[phase_id]["disturbance"]
        if dist > 0 and self.t % 10 == 0:
            force = self.rng.uniform(-dist, dist)
            self._env.unwrapped.state[1] += force  # add to cart velocity

    def _get_obs(self, obs):
        return np.append(obs, self._phase_indicator()).astype(np.float32)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t       = 0
        self.history = []
        obs, info    = self._env.reset(seed=seed)
        self._apply_phase(self._current_phase_id())
        return self._get_obs(obs), info

    def step(self, action):
        phase_id = self._current_phase_id()
        self._apply_phase(phase_id)
        self._apply_disturbance(phase_id)

        obs, reward, terminated, truncated, info = self._env.step(action)

        self.t    += 1
        truncated  = truncated or (self.t >= self.episode_length)

        self.history.append({
            "t":          self.t,
            "reward":     float(reward),
            "phase_id":   phase_id,
            "phase_name": CARTPOLE_PHASES[phase_id]["name"],
        })

        return self._get_obs(obs), reward, terminated, truncated, {
            "phase_id":   phase_id,
            "phase_name": CARTPOLE_PHASES[phase_id]["name"],
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
                "phase_name": CARTPOLE_PHASES[phase_id]["name"],
                "color":      CARTPOLE_PHASES[phase_id]["color"],
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

def make_cartpole_env(seed: int, episode_length: int = 500,
                      phase_duration: int = 100):
    def _init():
        return CartPolePhased(
            episode_length=episode_length,
            phase_duration=phase_duration,
            seed=seed,
        )
    return _init


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env    = CartPolePhased(seed=42)
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
    print("CartPole environment test passed.")