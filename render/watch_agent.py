# render/watch_agent.py
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.robosuite_env   import RobosuitePhased, ROBOSUITE_PHASES, PHASE_SCHEDULE
from core.policy          import load_policy
from core.meta_controller import MetaController, build_meta_state
from training.train       import load_config


# ── Watch single agent ────────────────────────────────────────────────────────

def watch_fixed(arch="mlp", seed=42, deterministic=True):
    """
    Watch a pretrained fixed PPO agent live in MuJoCo viewer.
    """
    cfg     = load_config()
    env_cfg = cfg["environment"]
    pol_cfg = cfg["inner_policies"]
    tr_cfg  = cfg["training"]

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        render_mode="human",
        seed=seed,
    )

    model_path = os.path.join(tr_cfg["models_dir"], f"{arch}_pretrained")
    model      = load_policy(arch, model_path, env, pol_cfg["device"])

    obs, _       = env.reset(seed=seed)
    total_reward = 0.0
    done         = False

    print(f"\n[watch] Fixed {arch.upper()} agent | seed={seed}")
    print("Phase schedule:", [ROBOSUITE_PHASES[p]["name"] for p in PHASE_SCHEDULE])
    print("Close the MuJoCo window to stop.\n")

    while not done:
        action, _  = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done          = terminated or truncated

        print(
            f"\rStep {env.t:4d} | "
            f"Phase: {info['phase_name']:10s} | "
            f"Cube height: {env.history[-1]['cube_height']:.4f}m | "
            f"Reward: {reward:+.3f} | "
            f"Total: {total_reward:+.3f}",
            end="", flush=True,
        )

    print(f"\n[watch] Done. Total reward: {total_reward:.3f}")
    env.close()


# ── Watch self-modifying agent ────────────────────────────────────────────────

def watch_self_modifying(seed=0, deterministic=True):
    """
    Watch the trained self-modifying agent live.
    Prints which architecture is active at each step.
    """
    cfg      = load_config()
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    meta_cfg = cfg["meta_controller"]
    tr_cfg   = cfg["training"]

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        render_mode="human",
        seed=seed,
    )

    policies = {
        "mlp":  load_policy("mlp",
                    os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),
                    env, pol_cfg["device"]),
        "lstm": load_policy("lstm",
                    os.path.join(tr_cfg["models_dir"], "lstm_pretrained"),
                    env, pol_cfg["device"]),
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

    meta_path = os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_final.pt")
    if os.path.exists(meta_path):
        meta.load(meta_path)
    else:
        print(f"[watch] No trained meta-controller found at {meta_path}")
        print("[watch] Running with untrained meta-controller.")

    obs, _       = env.reset(seed=seed)
    meta.reset_episode()
    total_reward = 0.0
    current_arch = 0
    switch_count = 0
    done         = False

    print(f"\n[watch] Self-Modifying agent | seed={seed}")
    print("Phase schedule:", [ROBOSUITE_PHASES[p]["name"] for p in PHASE_SCHEDULE])
    print("Close the MuJoCo window to stop.\n")

    arch_labels = {0: "MLP ", 1: "LSTM"}

    while not done:
        t = env.t

        if t % meta_cfg["decision_interval"] == 0:
            meta_state = build_meta_state(
                meta.reward_history,
                current_arch=current_arch,
                t=t,
                episode_length=env_cfg["episode_length"],
                lift_history= meta.lift_history,
            )
            new_arch = meta.select_arch(meta_state, deterministic=deterministic)

            if new_arch != current_arch:
                switch_count += 1
                print(f"\n  ↻ Switch at step {t:4d}: "
                      f"{arch_labels[current_arch]} → {arch_labels[new_arch]} "
                      f"| Phase: {ROBOSUITE_PHASES[env._current_phase_id()]['name']}")
                if new_arch == 1:
                    policies["lstm"].policy.features_extractor.reset_hidden()

            current_arch      = new_arch
            meta.current_arch = current_arch

        arch_key   = "mlp" if current_arch == 0 else "lstm"
        action, _  = policies[arch_key].predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        meta.update_reward_history(float(reward), lifted=info.get("lifted", False))
        total_reward += reward
        done          = terminated or truncated

        print(
            f"\rStep {t:4d} | "
            f"Arch: {arch_labels[current_arch]} | "
            f"Phase: {info['phase_name']:10s} | "
            f"Cube: {env.history[-1]['cube_height']:.4f}m | "
            f"Switches: {switch_count:2d} | "
            f"Total: {total_reward:+.3f}",
            end="", flush=True,
        )

    print(f"\n[watch] Done. Total reward: {total_reward:.3f} | "
          f"Total switches: {switch_count}")
    env.close()


# ── Watch all agents back to back ─────────────────────────────────────────────

def watch_comparison(seed=42):
    """
    Runs all agents sequentially so you can visually compare them.
    """
    print("\n" + "="*55)
    print("  AGENT COMPARISON — Robosuite Panda Lift")
    print("="*55)

    print("\n[1/3] Fixed MLP agent")
    watch_fixed(arch="mlp", seed=seed)
    time.sleep(1)

    print("\n[2/3] Fixed LSTM agent")
    watch_fixed(arch="lstm", seed=seed)
    time.sleep(1)

    print("\n[3/3] Self-Modifying agent")
    watch_self_modifying(seed=seed)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch trained agents live.")
    parser.add_argument("--agent", type=str, default="self_modifying",
                        choices=["mlp", "lstm", "self_modifying", "compare"],
                        help="Which agent to watch")
    parser.add_argument("--seed",  type=int, default=0,
                        help="Seed for environment")
    args = parser.parse_args()

    if args.agent == "self_modifying":
        watch_self_modifying(seed=args.seed)
    elif args.agent == "compare":
        watch_comparison(seed=args.seed)
    else:
        watch_fixed(arch=args.agent, seed=args.seed)