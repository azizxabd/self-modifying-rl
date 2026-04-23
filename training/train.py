# training/train.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from core.robosuite_env import RobosuitePhased, make_env
from core.policy import make_policy, save_policy, load_policy
from core.meta_controller import MetaController, build_meta_state
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


# ── Load config ───────────────────────────────────────────────────────────────

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Pretrain inner policies ───────────────────────────────────────────────────

def pretrain_inner_policies(cfg, seed=0):
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    save_dir = cfg["training"]["models_dir"]
    n_envs   = cfg["training"].get("n_envs", 4)
    os.makedirs(save_dir, exist_ok=True)

    for arch in ("mlp", "lstm"):
        save_path = os.path.join(save_dir, f"{arch}_pretrained")
        if os.path.exists(save_path + ".zip"):
            print(f"[pretrain] {arch.upper()} already exists, skipping.")
            continue

        print(f"[pretrain] Training {arch.upper()} with {n_envs} parallel envs...")
        env = VecMonitor(SubprocVecEnv([
            make_env(
                seed=seed + i,
                episode_length=env_cfg["episode_length"],
                phase_duration=env_cfg["phase_duration"],
                control_freq=env_cfg["control_freq"],
            )
            for i in range(n_envs)
        ]))

        model = make_policy(arch, env, device=pol_cfg["device"], cfg=cfg)
        model.learn(
            total_timesteps=pol_cfg[arch]["pretrain_timesteps"],
            progress_bar=True,
        )
        save_policy(model, save_path)
        env.close()

    print("[pretrain] Done.")


# ── Single episode: self-modifying ───────────────────────────────────────────

def run_self_modifying_episode(env, policies, meta, cfg, deterministic=False):
    meta_cfg          = cfg["meta_controller"]
    decision_interval = meta_cfg["decision_interval"]
    episode_length    = cfg["environment"]["episode_length"]

    obs, _       = env.reset()
    meta.reset_episode()

    total_reward = 0.0
    arch_log     = []
    switch_count = 0
    current_arch = 0
    cube_heights = []

    for t in range(episode_length):
        if t % decision_interval == 0:
            meta_state = build_meta_state(
                meta.reward_history,
                current_arch=current_arch,
                t=t,
                episode_length=episode_length,
                lift_history=meta.lift_history,
            )
            new_arch = meta.select_arch(meta_state, deterministic=deterministic)

            if new_arch != current_arch:
                switch_count += 1
                if new_arch == 1:
                    policies["lstm"].policy.features_extractor.reset_hidden()

            meta_r = meta.compute_meta_reward(
                task_reward=np.mean(list(meta.reward_history)) if meta.reward_history else 0.0,
                new_arch=new_arch,
                lifted=bool(np.mean(list(meta.lift_history))) > 0.1 if meta.lift_history else False,
            )
            meta.store_reward(meta_r)
            current_arch      = new_arch
            meta.current_arch = current_arch

        arch_key  = "mlp" if current_arch == 0 else "lstm"
        action, _ = policies[arch_key].predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        meta.update_reward_history(float(reward), lifted=info.get("lifted", False))
        total_reward += reward
        arch_log.append(current_arch)

        if env.history:
            cube_heights.append(env.history[-1]["cube_height"])

        if terminated or truncated:
            break

    return total_reward, arch_log, switch_count, cube_heights


# ── Single episode: fixed PPO ─────────────────────────────────────────────────

def run_fixed_episode(env, model, cfg, deterministic=False):
    episode_length = cfg["environment"]["episode_length"]
    obs, _         = env.reset()
    total_reward   = 0.0

    for _ in range(episode_length):
        action, _  = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


# ── Single episode: heuristic ─────────────────────────────────────────────────

def run_heuristic_episode(env, policies, cfg, deterministic=False):
    threshold         = cfg["baselines"]["heuristic_threshold"]
    decision_interval = cfg["meta_controller"]["decision_interval"]
    episode_length    = cfg["environment"]["episode_length"]

    obs, _         = env.reset()
    total_reward   = 0.0
    current_arch   = 0
    recent_rewards = []
    arch_log       = []

    for t in range(episode_length):
        if t % decision_interval == 0 and len(recent_rewards) > 0:
            current_arch = 0 if np.mean(recent_rewards) >= threshold else 1

        arch_key  = "mlp" if current_arch == 0 else "lstm"
        action, _ = policies[arch_key].predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward    += reward
        recent_rewards.append(reward)
        arch_log.append(current_arch)

        if terminated or truncated:
            break

    return total_reward, arch_log


# ── Single episode: random ────────────────────────────────────────────────────

def run_random_episode(env, policies, cfg, deterministic=False):
    decision_interval = cfg["meta_controller"]["decision_interval"]
    episode_length    = cfg["environment"]["episode_length"]

    obs, _       = env.reset()
    total_reward = 0.0
    arch_log     = []
    current_arch = 0

    for t in range(episode_length):
        if t % decision_interval == 0:
            current_arch = np.random.randint(0, 2)

        arch_key  = "mlp" if current_arch == 0 else "lstm"
        action, _ = policies[arch_key].predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        arch_log.append(current_arch)

        if terminated or truncated:
            break

    return total_reward, arch_log


# ── Single seed training (parallelizable) ────────────────────────────────────

def train_single_seed(seed: int, cfg: dict):
    cfg = dict(cfg)
    cfg["inner_policies"] = dict(cfg["inner_policies"])
    cfg["inner_policies"]["device"] = "cpu"
    cfg["meta_controller"] = dict(cfg["meta_controller"])
    cfg["meta_controller"]["device"] = "cpu"

    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    meta_cfg = cfg["meta_controller"]
    tr_cfg   = cfg["training"]

    os.makedirs(tr_cfg["logs_dir"],   exist_ok=True)
    os.makedirs(tr_cfg["models_dir"], exist_ok=True)

    tb_dir = os.path.join(tr_cfg["logs_dir"], "tensorboard", f"self_modifying_seed{seed}")
    writer = SummaryWriter(tb_dir)

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        seed=seed,
    )

    policies = {
        "mlp":  load_policy("mlp",  os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),  env, pol_cfg["device"]),
        "lstm": load_policy("lstm", os.path.join(tr_cfg["models_dir"], "lstm_pretrained"), env, pol_cfg["device"]),
    }

    meta = MetaController(
        n_archs=meta_cfg["n_archs"],
        meta_state_dim=meta_cfg["meta_state_dim"],
        hidden=meta_cfg["hidden"],
        lr=meta_cfg["lr"],
        gamma=meta_cfg["gamma"],
        switching_cost=meta_cfg["switching_cost"],
        decision_interval=meta_cfg["decision_interval"],
        history_window=meta_cfg["history_window"],
        device=meta_cfg["device"],
    )

    log_rows      = []
    reward_window = []

    print(f"\n[train] Seed {seed} started — {tr_cfg['total_episodes']} episodes")

    for ep in tqdm(
        range(tr_cfg["total_episodes"]),
        desc=f"Seed {seed}",
        ncols=100,
        ascii=True,
        position=seed,
        leave=True,
    ):
        ep_reward, arch_log, switches, cube_heights = run_self_modifying_episode(
            env, policies, meta, cfg
        )
        meta_loss = meta.update()

        arch_arr    = np.array(arch_log)
        mlp_frac    = float(np.mean(arch_arr == 0))
        lstm_frac   = float(np.mean(arch_arr == 1))
        max_height  = float(np.max(cube_heights))  if cube_heights else 0.0
        mean_height = float(np.mean(cube_heights)) if cube_heights else 0.0

        reward_window.append(ep_reward)
        if len(reward_window) > 20:
            reward_window.pop(0)

        writer.add_scalar("reward/episode",         ep_reward,              ep)
        writer.add_scalar("reward/smoothed_20ep",   np.mean(reward_window), ep)
        writer.add_scalar("meta/loss",              meta_loss,              ep)
        writer.add_scalar("meta/switches",          switches,               ep)
        writer.add_scalar("meta/mlp_fraction",      mlp_frac,               ep)
        writer.add_scalar("meta/lstm_fraction",     lstm_frac,              ep)
        writer.add_scalar("robot/max_cube_height",  max_height,             ep)
        writer.add_scalar("robot/mean_cube_height", mean_height,            ep)

        log_rows.append({
            "seed":        seed,
            "episode":     ep,
            "agent":       "self_modifying",
            "reward":      ep_reward,
            "meta_loss":   meta_loss,
            "switches":    switches,
            "mlp_frac":    mlp_frac,
            "lstm_frac":   lstm_frac,
            "max_height":  max_height,
            "mean_height": mean_height,
        })

        if (ep + 1) % tr_cfg["save_interval"] == 0:
            meta.save(os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_ep{ep+1}.pt"))

    meta.save(os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_final.pt"))
    writer.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(tr_cfg["logs_dir"], f"self_modifying_seed{seed}.csv"), index=False)
    print(f"\n[train] Seed {seed} done.")
    env.close()


# ── Keep old train() as alias for compatibility ───────────────────────────────

def train(seed: int, cfg: dict):
    train_single_seed(seed, cfg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = load_config()
    pretrain_inner_policies(cfg, seed=0)

    seeds = cfg["training"]["seeds"]
    print(f"\n[train] Running {len(seeds)} seeds in parallel...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(seeds)) as pool:
        pool.starmap(train_single_seed, [(seed, cfg) for seed in seeds])

    print("[train] All seeds done.")