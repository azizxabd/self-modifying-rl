# training/train_cartpole.py
import os
import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from core.cartpole_env import CartPolePhased, make_cartpole_env, CARTPOLE_PHASES, PHASE_SCHEDULE
from core.policy import make_policy, save_policy, load_policy
from core.meta_controller import MetaController, build_meta_state


# ── Config ────────────────────────────────────────────────────────────────────

CARTPOLE_CFG = {
    "environment": {
        "episode_length": 500,
        "phase_duration":  100,
    },
    "inner_policies": {
        "device": "cuda",
        "mlp": {
            "features_dim":       64,
            "learning_rate":      0.0003,
            "n_steps":            512,
            "batch_size":         64,
            "n_epochs":           10,
            "gamma":              0.99,
            "gae_lambda":         0.95,
            "clip_range":         0.2,
            "ent_coef":           0.01,
            "pretrain_timesteps": 100000,
        },
        "lstm": {
            "features_dim":       64,
            "hidden_size":        64,
            "learning_rate":      0.0003,
            "n_steps":            512,
            "batch_size":         64,
            "n_epochs":           10,
            "gamma":              0.99,
            "gae_lambda":         0.95,
            "clip_range":         0.2,
            "ent_coef":           0.01,
            "pretrain_timesteps": 100000,
        },
    },
    "meta_controller": {
        "n_archs":           2,
        "meta_state_dim":    7,
        "hidden":            64,
        "lr":                0.001,
        "gamma":             0.99,
        "switching_cost":    0.3,
        "decision_interval": 20,
        "history_window":    50,
        "device":            "cpu",
    },
    "training": {
        "seeds":          [0, 1, 2, 3, 4],
        "total_episodes": 500,
        "n_envs":         4,
        "save_interval":  100,
        "logs_dir":       "results/cartpole/logs",
        "models_dir":     "results/cartpole/models",
    },
    "baselines": {
        "fixed_ppo_arch":            "mlp",
        "heuristic_threshold":       0.0,
        "pretrain_timesteps":        100000,
        "high_capacity_hidden":      256,
        "high_capacity_timesteps":   100000,
    },
}


# ── Pretrain inner policies ───────────────────────────────────────────────────

def pretrain_inner_policies(cfg):
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    save_dir = cfg["training"]["models_dir"]
    n_envs   = cfg["training"]["n_envs"]
    os.makedirs(save_dir, exist_ok=True)

    for arch in ("mlp", "lstm"):
        save_path = os.path.join(save_dir, f"{arch}_pretrained")
        if os.path.exists(save_path + ".zip"):
            print(f"[pretrain] {arch.upper()} already exists, skipping.")
            continue

        print(f"[pretrain] Training {arch.upper()} with {n_envs} parallel envs...")
        env = VecMonitor(SubprocVecEnv([
            make_cartpole_env(
                seed=i,
                episode_length=env_cfg["episode_length"],
                phase_duration=env_cfg["phase_duration"],
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


# ── Episode runners ───────────────────────────────────────────────────────────

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
            )
            meta.store_reward(meta_r)
            current_arch      = new_arch
            meta.current_arch = current_arch

        arch_key  = "mlp" if current_arch == 0 else "lstm"
        action, _ = policies[arch_key].predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        meta.update_reward_history(float(reward))
        total_reward += reward
        arch_log.append(current_arch)

        if terminated or truncated:
            break

    return total_reward, arch_log, switch_count


def run_fixed_episode(env, model, cfg, deterministic=False):
    obs, _       = env.reset()
    total_reward = 0.0
    for _ in range(cfg["environment"]["episode_length"]):
        action, _  = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


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
        if t % decision_interval == 0 and recent_rewards:
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


# ── Single seed training ──────────────────────────────────────────────────────

def train_single_seed(seed, cfg):
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

    env = CartPolePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        seed=seed,
    )

    policies = {
        "mlp":  load_policy("mlp",  os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),  env, "cpu"),
        "lstm": load_policy("lstm", os.path.join(tr_cfg["models_dir"], "lstm_pretrained"), env, "cpu"),
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

    for ep in tqdm(
        range(tr_cfg["total_episodes"]),
        desc=f"Seed {seed}",
        ncols=100, ascii=True, position=seed, leave=True,
    ):
        ep_reward, arch_log, switches = run_self_modifying_episode(
            env, policies, meta, cfg
        )
        meta_loss = meta.update()

        arch_arr  = np.array(arch_log)
        mlp_frac  = float(np.mean(arch_arr == 0))
        lstm_frac = float(np.mean(arch_arr == 1))

        reward_window.append(ep_reward)
        if len(reward_window) > 20:
            reward_window.pop(0)

        writer.add_scalar("reward/episode",       ep_reward,              ep)
        writer.add_scalar("reward/smoothed_20ep", np.mean(reward_window), ep)
        writer.add_scalar("meta/loss",            meta_loss,              ep)
        writer.add_scalar("meta/switches",        switches,               ep)
        writer.add_scalar("meta/mlp_fraction",    mlp_frac,               ep)
        writer.add_scalar("meta/lstm_fraction",   lstm_frac,              ep)

        log_rows.append({
            "seed":       seed,
            "episode":    ep,
            "agent":      "self_modifying",
            "reward":     ep_reward,
            "meta_loss":  meta_loss,
            "switches":   switches,
            "mlp_frac":   mlp_frac,
            "lstm_frac":  lstm_frac,
            "max_height": 0.0,
            "mean_height": 0.0,
        })

        if (ep + 1) % tr_cfg["save_interval"] == 0:
            meta.save(os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_ep{ep+1}.pt"))

    meta.save(os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_final.pt"))
    writer.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(tr_cfg["logs_dir"], f"self_modifying_seed{seed}.csv"), index=False)
    print(f"\n[train] Seed {seed} done.")
    env.close()


# ── Baselines single seed ─────────────────────────────────────────────────────

def run_baseline_seed(seed, cfg):
    cfg = dict(cfg)
    cfg["inner_policies"] = dict(cfg["inner_policies"])
    cfg["inner_policies"]["device"] = "cpu"

    env_cfg = cfg["environment"]
    pol_cfg = cfg["inner_policies"]
    bl_cfg  = cfg["baselines"]
    tr_cfg  = cfg["training"]

    writers = {
        agent: SummaryWriter(
            os.path.join(tr_cfg["logs_dir"], "tensorboard", f"{agent}_seed{seed}")
        )
        for agent in ("fixed_ppo", "high_capacity", "heuristic", "random")
    }

    env = CartPolePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        seed=seed,
    )

    fixed_model = load_policy(
        bl_cfg["fixed_ppo_arch"],
        os.path.join(tr_cfg["models_dir"], f"baseline_fixed_{bl_cfg['fixed_ppo_arch']}"),
        env, "cpu",
    )
    hc_model = PPO.load(
        os.path.join(tr_cfg["models_dir"], "baseline_high_capacity"),
        env=env, device="cpu",
    )
    policies = {
        "mlp":  load_policy("mlp",  os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),  env, "cpu"),
        "lstm": load_policy("lstm", os.path.join(tr_cfg["models_dir"], "lstm_pretrained"), env, "cpu"),
    }

    log_rows       = []
    reward_windows = {a: [] for a in ("fixed_ppo", "high_capacity", "heuristic", "random")}

    for ep in tqdm(
        range(tr_cfg["total_episodes"]),
        desc=f"BL Seed {seed}",
        ncols=100, ascii=True, position=seed, leave=True,
    ):
        # Fixed PPO
        r = run_fixed_episode(env, fixed_model, cfg)
        reward_windows["fixed_ppo"].append(r)
        if len(reward_windows["fixed_ppo"]) > 20: reward_windows["fixed_ppo"].pop(0)
        writers["fixed_ppo"].add_scalar("reward/episode",       r, ep)
        writers["fixed_ppo"].add_scalar("reward/smoothed_20ep", np.mean(reward_windows["fixed_ppo"]), ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "fixed_ppo",
                         "reward": r, "switches": 0, "mlp_frac": 1.0, "lstm_frac": 0.0,
                         "max_height": 0.0, "mean_height": 0.0})

        # High Capacity
        r = run_fixed_episode(env, hc_model, cfg)
        reward_windows["high_capacity"].append(r)
        if len(reward_windows["high_capacity"]) > 20: reward_windows["high_capacity"].pop(0)
        writers["high_capacity"].add_scalar("reward/episode",       r, ep)
        writers["high_capacity"].add_scalar("reward/smoothed_20ep", np.mean(reward_windows["high_capacity"]), ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "high_capacity",
                         "reward": r, "switches": 0, "mlp_frac": 0.0, "lstm_frac": 0.0,
                         "max_height": 0.0, "mean_height": 0.0})

        # Heuristic
        r, arch_log = run_heuristic_episode(env, policies, cfg)
        arch_arr    = np.array(arch_log)
        switches    = int(np.sum(np.diff(arch_arr) != 0)) if len(arch_arr) > 1 else 0
        reward_windows["heuristic"].append(r)
        if len(reward_windows["heuristic"]) > 20: reward_windows["heuristic"].pop(0)
        writers["heuristic"].add_scalar("reward/episode",       r, ep)
        writers["heuristic"].add_scalar("reward/smoothed_20ep", np.mean(reward_windows["heuristic"]), ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "heuristic",
                         "reward": r, "switches": switches,
                         "mlp_frac": float(np.mean(arch_arr == 0)),
                         "lstm_frac": float(np.mean(arch_arr == 1)),
                         "max_height": 0.0, "mean_height": 0.0})

        # Random
        r, arch_log = run_random_episode(env, policies, cfg)
        arch_arr    = np.array(arch_log)
        switches    = int(np.sum(np.diff(arch_arr) != 0)) if len(arch_arr) > 1 else 0
        reward_windows["random"].append(r)
        if len(reward_windows["random"]) > 20: reward_windows["random"].pop(0)
        writers["random"].add_scalar("reward/episode",       r, ep)
        writers["random"].add_scalar("reward/smoothed_20ep", np.mean(reward_windows["random"]), ep)
        log_rows.append({"seed": seed, "episode": ep, "agent": "random",
                         "reward": r, "switches": switches,
                         "mlp_frac": float(np.mean(arch_arr == 0)),
                         "lstm_frac": float(np.mean(arch_arr == 1)),
                         "max_height": 0.0, "mean_height": 0.0})

    for w in writers.values():
        w.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(tr_cfg["logs_dir"], f"baselines_seed{seed}.csv"), index=False)
    print(f"\n[baselines] Seed {seed} done.")
    env.close()


# ── Pretrain baselines ────────────────────────────────────────────────────────

def pretrain_baselines(cfg):
    env_cfg  = cfg["environment"]
    bl_cfg   = cfg["baselines"]
    tr_cfg   = cfg["training"]
    n_envs   = tr_cfg["n_envs"]
    os.makedirs(tr_cfg["models_dir"], exist_ok=True)

    # Fixed PPO
    fixed_path = os.path.join(tr_cfg["models_dir"], f"baseline_fixed_{bl_cfg['fixed_ppo_arch']}")
    if os.path.exists(fixed_path + ".zip"):
        print("[baseline] Fixed PPO exists, skipping.")
    else:
        print("[baseline] Pretraining Fixed PPO...")
        env = VecMonitor(SubprocVecEnv([
            make_cartpole_env(seed=i,
                              episode_length=env_cfg["episode_length"],
                              phase_duration=env_cfg["phase_duration"])
            for i in range(n_envs)
        ]))
        model = make_policy(bl_cfg["fixed_ppo_arch"], env, "cuda", cfg=cfg)
        model.learn(total_timesteps=bl_cfg["pretrain_timesteps"], progress_bar=True)
        save_policy(model, fixed_path)
        env.close()

    # High Capacity
    hc_path = os.path.join(tr_cfg["models_dir"], "baseline_high_capacity")
    if os.path.exists(hc_path + ".zip"):
        print("[baseline] High Capacity exists, skipping.")
    else:
        print("[baseline] Pretraining High Capacity PPO...")
        env = VecMonitor(SubprocVecEnv([
            make_cartpole_env(seed=i,
                              episode_length=env_cfg["episode_length"],
                              phase_duration=env_cfg["phase_duration"])
            for i in range(n_envs)
        ]))
        hidden = bl_cfg["high_capacity_hidden"]
        model  = PPO(
            policy="MlpPolicy", env=env,
            learning_rate=0.0003, n_steps=512, batch_size=64,
            n_epochs=10, gamma=0.99, verbose=0, device="cuda",
            policy_kwargs=dict(net_arch=[hidden, hidden, hidden]),
        )
        model.learn(total_timesteps=bl_cfg["high_capacity_timesteps"], progress_bar=True)
        save_policy(model, hc_path)
        env.close()

    print("[baseline] Pretraining done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = CARTPOLE_CFG

    os.makedirs(cfg["training"]["logs_dir"],   exist_ok=True)
    os.makedirs(cfg["training"]["models_dir"], exist_ok=True)

    print("=" * 50)
    print("  CartPole Phased — Full Pipeline")
    print("=" * 50)

    # Step 1: Pretrain inner policies
    pretrain_inner_policies(cfg)

    # Step 2: Train self-modifying agent
    seeds = cfg["training"]["seeds"]
    print(f"\n[train] Running {len(seeds)} seeds in parallel...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(seeds)) as pool:
        pool.starmap(train_single_seed, [(seed, cfg) for seed in seeds])
    print("[train] All seeds done.")

    # Step 3: Pretrain baselines
    pretrain_baselines(cfg)

    # Step 4: Run baselines
    print(f"\n[baselines] Running {len(seeds)} seeds in parallel...")
    with ctx.Pool(processes=len(seeds)) as pool:
        pool.starmap(run_baseline_seed, [(seed, cfg) for seed in seeds])
    print("[baselines] All done.")

    print("\n[pipeline] Complete. Run evaluation:")
    print("  python evaluation/eval_cartpole.py")