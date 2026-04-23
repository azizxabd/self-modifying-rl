# training/baselines.py
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from core.robosuite_env import RobosuitePhased, make_env
from core.policy import make_policy, save_policy, load_policy
from training.train import (
    load_config,
    run_fixed_episode,
    run_heuristic_episode,
    run_random_episode,
)


# ── High capacity network ─────────────────────────────────────────────────────

def make_high_capacity_policy(env, cfg):
    hidden = cfg["baselines"]["high_capacity_hidden"]
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device=cfg["inner_policies"]["device"],
        policy_kwargs=dict(net_arch=[hidden, hidden, hidden]),
    )


# ── Fixed episode with height tracking ───────────────────────────────────────

def run_fixed_episode_with_height(env, model, cfg, deterministic=False):
    episode_length = cfg["environment"]["episode_length"]
    obs, _         = env.reset()
    total_reward   = 0.0

    for _ in range(episode_length):
        action, _  = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    history     = env.get_history_arrays()
    max_height  = float(np.max(history["cube_height"]))  if "cube_height" in history else 0.0
    mean_height = float(np.mean(history["cube_height"])) if "cube_height" in history else 0.0
    return total_reward, max_height, mean_height


# ── Pretrain baselines ────────────────────────────────────────────────────────

def pretrain_baselines(cfg, seed=0):
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    bl_cfg   = cfg["baselines"]
    save_dir = cfg["training"]["models_dir"]
    n_envs   = cfg["training"].get("n_envs", 4)
    os.makedirs(save_dir, exist_ok=True)

    # ── Fixed PPO ─────────────────────────────────────────────────────────────
    fixed_path = os.path.join(save_dir, f"baseline_fixed_{bl_cfg['fixed_ppo_arch']}")
    if os.path.exists(fixed_path + ".zip"):
        print(f"[baseline] Fixed PPO already exists, skipping.")
    else:
        print(f"[baseline] Pretraining Fixed PPO with {n_envs} parallel envs...")
        env = VecMonitor(SubprocVecEnv([
            make_env(
                seed=seed + i,
                episode_length=env_cfg["episode_length"],
                phase_duration=env_cfg["phase_duration"],
                control_freq=env_cfg["control_freq"],
            )
            for i in range(n_envs)
        ]))
        model = make_policy(bl_cfg["fixed_ppo_arch"], env, pol_cfg["device"], cfg=cfg)
        model.learn(total_timesteps=bl_cfg["pretrain_timesteps"], progress_bar=True)
        save_policy(model, fixed_path)
        env.close()

    # ── High Capacity PPO ─────────────────────────────────────────────────────
    hc_path = os.path.join(save_dir, "baseline_high_capacity")
    if os.path.exists(hc_path + ".zip"):
        print(f"[baseline] High Capacity PPO already exists, skipping.")
    else:
        print(f"[baseline] Pretraining High Capacity PPO with {n_envs} parallel envs...")
        env = VecMonitor(SubprocVecEnv([
            make_env(
                seed=seed + i,
                episode_length=env_cfg["episode_length"],
                phase_duration=env_cfg["phase_duration"],
                control_freq=env_cfg["control_freq"],
            )
            for i in range(n_envs)
        ]))
        model = make_high_capacity_policy(env, cfg)
        model.learn(total_timesteps=bl_cfg["high_capacity_timesteps"], progress_bar=True)
        save_policy(model, hc_path)
        env.close()

    print("[baseline] Pretraining done.")


# ── Single seed runner (parallelizable) ──────────────────────────────────────

def run_single_seed(seed, cfg):
    cfg = dict(cfg)
    cfg["inner_policies"] = dict(cfg["inner_policies"])
    cfg["inner_policies"]["device"] = "cpu"

    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    bl_cfg   = cfg["baselines"]
    tr_cfg   = cfg["training"]
    arch     = bl_cfg["fixed_ppo_arch"]

    writers = {
        agent: SummaryWriter(
            os.path.join(tr_cfg["logs_dir"], "tensorboard", f"{agent}_seed{seed}")
        )
        for agent in ("fixed_ppo", "high_capacity", "heuristic", "random")
    }

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        seed=seed,
    )

    fixed_model = load_policy(
        arch,
        os.path.join(tr_cfg["models_dir"], f"baseline_fixed_{arch}"),
        env, pol_cfg["device"],
    )
    hc_model = PPO.load(
        os.path.join(tr_cfg["models_dir"], "baseline_high_capacity"),
        env=env, device=pol_cfg["device"],
    )
    print(f"[seed {seed}] Loaded ← baseline_high_capacity")

    policies = {
        "mlp":  load_policy("mlp",  os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),  env, pol_cfg["device"]),
        "lstm": load_policy("lstm", os.path.join(tr_cfg["models_dir"], "lstm_pretrained"), env, pol_cfg["device"]),
    }

    log_rows       = []
    reward_windows = {a: [] for a in ("fixed_ppo", "high_capacity", "heuristic", "random")}

    for ep in tqdm(
        range(tr_cfg["total_episodes"]),
        desc=f"Seed {seed}",
        ncols=100,
        ascii=True,
        position=seed,
        leave=True,
    ):
        # ── Fixed PPO ─────────────────────────────────────────────────────────
        r, max_h, mean_h = run_fixed_episode_with_height(env, fixed_model, cfg)
        reward_windows["fixed_ppo"].append(r)
        if len(reward_windows["fixed_ppo"]) > 20: reward_windows["fixed_ppo"].pop(0)
        writers["fixed_ppo"].add_scalar("reward/episode",        r,                                    ep)
        writers["fixed_ppo"].add_scalar("reward/smoothed_20ep",  np.mean(reward_windows["fixed_ppo"]), ep)
        writers["fixed_ppo"].add_scalar("robot/max_cube_height", max_h,                                ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "fixed_ppo",
                         "reward": r, "switches": 0, "mlp_frac": 1.0, "lstm_frac": 0.0,
                         "max_height": max_h, "mean_height": mean_h})

        # ── High Capacity ─────────────────────────────────────────────────────
        r, max_h, mean_h = run_fixed_episode_with_height(env, hc_model, cfg)
        reward_windows["high_capacity"].append(r)
        if len(reward_windows["high_capacity"]) > 20: reward_windows["high_capacity"].pop(0)
        writers["high_capacity"].add_scalar("reward/episode",        r,                                        ep)
        writers["high_capacity"].add_scalar("reward/smoothed_20ep",  np.mean(reward_windows["high_capacity"]), ep)
        writers["high_capacity"].add_scalar("robot/max_cube_height", max_h,                                    ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "high_capacity",
                         "reward": r, "switches": 0, "mlp_frac": 0.0, "lstm_frac": 0.0,
                         "max_height": max_h, "mean_height": mean_h})

        # ── Heuristic ─────────────────────────────────────────────────────────
        r, arch_log = run_heuristic_episode(env, policies, cfg)
        arch_arr    = np.array(arch_log)
        switches    = int(np.sum(np.diff(arch_arr) != 0)) if len(arch_arr) > 1 else 0
        history     = env.get_history_arrays()
        max_h       = float(np.max(history["cube_height"]))  if "cube_height" in history else 0.0
        mean_h      = float(np.mean(history["cube_height"])) if "cube_height" in history else 0.0
        reward_windows["heuristic"].append(r)
        if len(reward_windows["heuristic"]) > 20: reward_windows["heuristic"].pop(0)
        writers["heuristic"].add_scalar("reward/episode",        r,                                    ep)
        writers["heuristic"].add_scalar("reward/smoothed_20ep",  np.mean(reward_windows["heuristic"]), ep)
        writers["heuristic"].add_scalar("meta/switches",         switches,                             ep)
        writers["heuristic"].add_scalar("robot/max_cube_height", max_h,                                ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "heuristic",
                         "reward": r, "switches": switches,
                         "mlp_frac": float(np.mean(arch_arr == 0)),
                         "lstm_frac": float(np.mean(arch_arr == 1)),
                         "max_height": max_h, "mean_height": mean_h})

        # ── Random ────────────────────────────────────────────────────────────
        r, arch_log = run_random_episode(env, policies, cfg)
        arch_arr    = np.array(arch_log)
        switches    = int(np.sum(np.diff(arch_arr) != 0)) if len(arch_arr) > 1 else 0
        history     = env.get_history_arrays()
        max_h       = float(np.max(history["cube_height"]))  if "cube_height" in history else 0.0
        mean_h      = float(np.mean(history["cube_height"])) if "cube_height" in history else 0.0
        reward_windows["random"].append(r)
        if len(reward_windows["random"]) > 20: reward_windows["random"].pop(0)
        writers["random"].add_scalar("reward/episode",        r,                                  ep)
        writers["random"].add_scalar("reward/smoothed_20ep",  np.mean(reward_windows["random"]),  ep)
        writers["random"].add_scalar("meta/switches",         switches,                           ep)
        writers["random"].add_scalar("robot/max_cube_height", max_h,                              ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "random",
                         "reward": r, "switches": switches,
                         "mlp_frac": float(np.mean(arch_arr == 0)),
                         "lstm_frac": float(np.mean(arch_arr == 1)),
                         "max_height": max_h, "mean_height": mean_h})

    for w in writers.values():
        w.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(tr_cfg["logs_dir"], f"baselines_seed{seed}.csv"), index=False)
    print(f"\n[baselines] Seed {seed} done.")
    env.close()


# ── Run all baselines in parallel ─────────────────────────────────────────────

def run_baselines(cfg):
    os.makedirs(cfg["training"]["logs_dir"],   exist_ok=True)
    os.makedirs(cfg["training"]["models_dir"], exist_ok=True)

    seeds = cfg["training"]["seeds"]
    print(f"\n[baselines] Running {len(seeds)} seeds in parallel...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(seeds)) as pool:
        pool.starmap(run_single_seed, [(seed, cfg) for seed in seeds])

    print("[baselines] All seeds done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = load_config()
    pretrain_baselines(cfg, seed=0)
    run_baselines(cfg)