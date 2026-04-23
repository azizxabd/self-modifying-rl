# core/robosuite_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import os

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# ── Phase definitions ─────────────────────────────────────────────────────────

ROBOSUITE_PHASES = {
    0: {
        "name":           "Normal",
        "object_mass":    0.1,
        "disturbance":    0.0,
        "reward_weights": {"lift": 1.0, "grasp": 0.5, "efficiency": 0.1},
        "color":          "#DBEAFE",
        "lift_threshold": 0.04,
    },
    1: {
        "name":           "Heavy",
        "object_mass":    0.3,
        "disturbance":    0.0,
        "reward_weights": {"lift": 1.5, "grasp": 0.8, "efficiency": 0.05},
        "color":          "#FEF9C3",
        "lift_threshold": 0.04,
    },
    2: {
        "name":           "Disturbed",
        "object_mass":    0.1,
        "disturbance":    0.3,
        "reward_weights": {"lift": 1.2, "grasp": 0.6, "efficiency": 0.05},
        "color":          "#DCFCE7",
        "lift_threshold": 0.04,
    },
}

PHASE_SCHEDULE = [0, 1, 2, 0, 1]
TABLE_Z        = 0.8059   # actual robosuite table surface height


# ── Wrapper ───────────────────────────────────────────────────────────────────

class RobosuitePhased(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        episode_length: int = 500,
        phase_duration: int = 100,
        render_mode:    str = None,
        seed:           int = None,
        control_freq:   int = 20,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.phase_duration = phase_duration
        self.render_mode    = render_mode
        self.phase_schedule = PHASE_SCHEDULE
        self.rng            = np.random.default_rng(seed)
        self.history        = []
        self.t              = 0

        controller_config = load_composite_controller_config(
            controller="BASIC",
            robot="Panda",
        )

        self._env = suite.make(
            env_name               = "Lift",
            robots                 = "Panda",
            controller_configs     = controller_config,
            has_renderer           = render_mode == "human",
            has_offscreen_renderer = render_mode == "rgb_array",
            use_camera_obs         = False,
            use_object_obs         = True,
            reward_shaping         = True,
            control_freq           = control_freq,
            horizon                = episode_length,
            ignore_done            = True,
        )

        obs_dict = self._env.reset()
        obs_flat = self._flatten_obs(obs_dict)
        obs_dim  = len(obs_flat) + 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self._env.action_spec[0].astype(np.float32),
            high=self._env.action_spec[1].astype(np.float32),
            dtype=np.float32,
        )

        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        self._arm_geom_ids     = self._get_arm_geom_ids()
        self._gripper_geom_ids = self._get_gripper_geom_ids()
        self._cube_geom_id     = self._get_geom_id("cube_g0")
        self._table_geom_id    = self._get_geom_id("table_collision")

    # ── Geom ID helpers ───────────────────────────────────────────────────────

    def _get_geom_id(self, name):
        try:
            return self._env.sim.model.geom_name2id(name)
        except Exception:
            return None

    def _get_arm_geom_ids(self):
        arm_geoms = [
            "robot0_link0_collision", "robot0_link1_collision",
            "robot0_link2_collision", "robot0_link3_collision",
            "robot0_link4_collision", "robot0_link5_collision",
            "robot0_link6_collision",
        ]
        return [gid for g in arm_geoms
                if (gid := self._get_geom_id(g)) is not None]

    def _get_gripper_geom_ids(self):
        gripper_geoms = [
            "robot0_rightfinger_collision",
            "robot0_leftfinger_collision",
        ]
        return [gid for g in gripper_geoms
                if (gid := self._get_geom_id(g)) is not None]

    # ── Contact checkers ──────────────────────────────────────────────────────

    def _check_contact(self, geom_ids_a, geom_id_b) -> bool:
        if geom_id_b is None or not geom_ids_a:
            return False
        try:
            for i in range(self._env.sim.data.ncon):
                contact = self._env.sim.data.contact[i]
                g1, g2  = contact.geom1, contact.geom2
                if (g1 in geom_ids_a and g2 == geom_id_b) or \
                   (g2 in geom_ids_a and g1 == geom_id_b):
                    return True
        except Exception:
            pass
        return False

    def _check_arm_table_contact(self) -> bool:
        return self._check_contact(self._arm_geom_ids, self._table_geom_id)

    def _check_gripper_cube_contact(self) -> bool:
        return self._check_contact(self._gripper_geom_ids, self._cube_geom_id)

    # ── Observation helpers ───────────────────────────────────────────────────

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        parts = []
        for k in ["robot0_proprio-state", "object-state"]:
            if k in obs_dict:
                parts.append(np.array(obs_dict[k], dtype=np.float32).flatten())
        return np.concatenate(parts) if parts else np.zeros(60, dtype=np.float32)

    def _current_phase_idx(self):
        return min(self.t // self.phase_duration, len(self.phase_schedule) - 1)

    def _current_phase_id(self):
        return self.phase_schedule[self._current_phase_idx()]

    def _phase_indicator(self):
        return (self.t % self.phase_duration) / self.phase_duration

    def _get_obs(self, obs_dict):
        return np.append(
            self._flatten_obs(obs_dict),
            self._phase_indicator()
        ).astype(np.float32)

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _apply_disturbance(self, phase_id):
        dist = ROBOSUITE_PHASES[phase_id]["disturbance"]
        try:
            body_id = self._env.sim.model.body_name2id("cube_main")
            if dist > 0:
                if self.t % 10 == 0:
                    force = self.rng.uniform(-dist, dist, size=3).astype(np.float64)
                    self._env.sim.data.xfrc_applied[body_id, :3] = force
            else:
                self._env.sim.data.xfrc_applied[body_id, :3] = 0.0
        except Exception:
            pass

    def _apply_mass(self, phase_id):
        try:
            geom_id      = self._env.sim.model.geom_name2id("cube_g0")
            base_density = 1000.0
            multiplier   = ROBOSUITE_PHASES[phase_id]["object_mass"] / 0.1
            self._env.sim.model.geom_density[geom_id] = base_density * multiplier
        except Exception:
            pass

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, base_reward, action, phase_id, obs_dict):
        w         = ROBOSUITE_PHASES[phase_id]["reward_weights"]
        threshold = ROBOSUITE_PHASES[phase_id]["lift_threshold"]
        cube_pos  = np.array(obs_dict.get("cube_pos",           [0, 0, TABLE_Z]))
        grip_pos  = np.array(obs_dict.get("robot0_eef_pos",     [0, 0, 0]))
        gripper   = float(obs_dict.get("robot0_gripper_qpos",   [0])[0])

        reward = 0.0

        # 1. Sparse lift reward
        lift_height = cube_pos[2] - TABLE_Z
        if lift_height > threshold:
            reward += lift_height * w["lift"] * 20.0

        # 2. Grasp reward
        if self._check_gripper_cube_contact():
            reward += w["grasp"] * 2.0

        # 3. Reaching reward
        dist    = np.linalg.norm(grip_pos - cube_pos)
        reward += max(0.0, 1.0 - dist * 3.0) * 0.5

        # 4. Gripper closed when near cube
        if dist < 0.05 and gripper < 0.0155:
            reward += 1.0

        # 5. Gripper open penalty when far
        if dist > 0.1 and gripper < 0.016:
            reward -= 0.3

        # 6. Arm table contact penalty
        if self._check_arm_table_contact():
            reward -= 2.0

        # 7. Efficiency penalty
        reward -= np.linalg.norm(action) * w["efficiency"] * 0.001

        return float(reward)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t           = 0
        self.history     = []
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        obs_dict         = self._env.reset()

        # pre-open gripper — critical for learning to grasp
        # gripper: -1.0 = open, +1.0 = close
        pre_open = np.zeros(self.action_space.shape[0], dtype=np.float32)
        pre_open[-1] = -1.0
        for _ in range(100):
            obs_dict, _, _, _ = self._env.step(pre_open)

        return self._get_obs(obs_dict), {}

    def step(self, action):
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        # smooth arm joints only — gripper passes through unmodified
        # gripper: -1.0 = open, +1.0 = close
        smoothed      = 0.85 * action_clipped + 0.15 * self.prev_action
        smoothed[:-1] = np.clip(smoothed[:-1], -0.5, 0.5)
        smoothed[-1]  = action_clipped[-1]
        action        = smoothed

        self.prev_action = action.copy()
        phase_id         = self._current_phase_id()

        self._apply_mass(phase_id)
        self._apply_disturbance(phase_id)

        obs_dict, base_reward, done, info = self._env.step(action)

        obs_candidate = self._get_obs(obs_dict)
        if not np.isfinite(obs_candidate).all():
            obs_dict   = self._env.reset()
            reward     = -1.0
            terminated = True
            truncated  = False
            cube_pos   = np.array([0, 0, TABLE_Z])
            grasped    = False
            lifted     = False
        else:
            cube_pos   = np.array(obs_dict.get("cube_pos", [0, 0, TABLE_Z]))
            grasped    = self._check_gripper_cube_contact()
            lifted     = bool(cube_pos[2] > TABLE_Z + 0.04)
            reward     = self._compute_reward(base_reward, action, phase_id, obs_dict)
            terminated = bool(done)
            truncated  = self.t >= self.episode_length

        self.t += 1

        self.history.append({
            "t":           self.t,
            "reward":      reward,
            "phase_id":    phase_id,
            "phase_name":  ROBOSUITE_PHASES[phase_id]["name"],
            "action_norm": float(np.linalg.norm(action)),
            "cube_height": float(cube_pos[2]),
            "lifted":      float(lifted),
            "grasped":     float(grasped),
        })

        if self.render_mode == "human":
            self._env.render()

        return self._get_obs(obs_dict), reward, terminated, truncated, {
            "phase_id":   phase_id,
            "phase_name": ROBOSUITE_PHASES[phase_id]["name"],
            "lifted":     lifted,
            "grasped":    grasped,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.sim.render(
                camera_name="agentview",
                width=640, height=480,
            )[::-1]
        self._env.render()

    def close(self):
        self._env.close()

    def get_phase_boundaries(self):
        boundaries = []
        for i, phase_id in enumerate(self.phase_schedule):
            boundaries.append({
                "start":      i * self.phase_duration,
                "end":        min((i+1) * self.phase_duration, self.episode_length),
                "phase_id":   phase_id,
                "phase_name": ROBOSUITE_PHASES[phase_id]["name"],
                "color":      ROBOSUITE_PHASES[phase_id]["color"],
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


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env    = RobosuitePhased(seed=42)
    obs, _ = env.reset()
    print(f"Obs shape:    {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")

    done   = False
    total  = 0.0
    lifts  = 0
    grasps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        total  += reward
        lifts  += int(info["lifted"])
        grasps += int(info["grasped"])
        done    = terminated or truncated

    print(f"Total reward:     {total:.3f}")
    print(f"Lift steps:       {lifts}")
    print(f"Grasp steps:      {grasps}")
    print(f"Phase boundaries: {env.get_phase_boundaries()}")
    print("Environment test passed.")


# ── Parallel env factory ──────────────────────────────────────────────────────

def make_env(seed: int, episode_length: int = 500,
             phase_duration: int = 100, control_freq: int = 20):
    def _init():
        env = RobosuitePhased(
            episode_length=episode_length,
            phase_duration=phase_duration,
            control_freq=control_freq,
            seed=seed,
        )
        return env
    return _init