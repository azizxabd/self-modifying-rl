# training/train_bc.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.robosuite_env   import RobosuitePhased, make_env
from core.policy          import make_policy, save_policy, load_policy
from core.behavior_cloning import BehaviorCloningTrainer
from core.demo_collector   import collect_demonstrations, load_demonstrations
from training.train        import load_config, train_single_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import multiprocessing as mp


# ── Step 1: Collect demonstrations ───────────────────────────────────────────

def step1_collect_demos(cfg, n_demos=500):
    demo_path = "results/demos/demonstrations.pkl"
    if os.path.exists(demo_path):
        print(f"[BC pipeline] Demos already exist, skipping.")
        return load_demonstrations(demo_path)

    print(f"[BC pipeline] Step 1: Collecting {n_demos} demos...")
    from core.demo_collector import collect_demonstrations
    return collect_demonstrations(
        n_demos=n_demos,
        episode_length=1500,
        save_path=demo_path,
        n_workers=4,
    )


# ── Step 2: Behavior cloning pretraining ──────────────────────────────────────

def step2_behavior_cloning(cfg, demonstrations):
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    save_dir = cfg["training"]["models_dir"]
    n_envs   = cfg["training"].get("n_envs", 4)
    os.makedirs(save_dir, exist_ok=True)

    for arch in ("mlp", "lstm"):
        bc_path  = os.path.join(save_dir, f"{arch}_bc")
        ppo_path = os.path.join(save_dir, f"{arch}_pretrained")

        if os.path.exists(ppo_path + ".zip"):
            print(f"[BC pipeline] {arch.upper()} PPO already exists, skipping BC.")
            continue

        print(f"\n[BC pipeline] Step 2: Behavior cloning for {arch.upper()}...")

        # create env for policy init
        env = VecMonitor(SubprocVecEnv([
            make_env(
                seed=i,
                episode_length=env_cfg["episode_length"],
                phase_duration=env_cfg["phase_duration"],
                control_freq=env_cfg["control_freq"],
            )
            for i in range(n_envs)
        ]))

        # init fresh policy
        model = make_policy(arch, env, device=pol_cfg["device"], cfg=cfg)

        # behavior cloning
        trainer = BehaviorCloningTrainer(
            model,
            device=pol_cfg["device"],
            lr=1e-4,
        )
        losses = trainer.train(
            demonstrations=demonstrations,
            epochs=30,
            batch_size=256,
            save_path=bc_path,
        )

        # plot BC loss
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Behavior Cloning Loss — {arch.upper()}")
        plt.tight_layout()
        plt.savefig(f"results/bc_loss_{arch}.png", dpi=150)
        plt.close()
        print(f"[BC pipeline] Loss plot saved → results/bc_loss_{arch}.png")

        # PPO fine-tuning on top of BC weights
        print(f"\n[BC pipeline] Step 2b: PPO fine-tuning {arch.upper()}...")
        model.learn(
            total_timesteps=pol_cfg[arch]["pretrain_timesteps"],
            progress_bar=True,
        )
        save_policy(model, ppo_path)
        env.close()

    print("[BC pipeline] Step 2 done.")


# ── Step 3: Train meta-controller ─────────────────────────────────────────────

def step3_train_meta(cfg):
    seeds = cfg["training"]["seeds"]
    print(f"\n[BC pipeline] Step 3: Training meta-controller ({len(seeds)} seeds)...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(seeds)) as pool:
        pool.starmap(train_single_seed, [(seed, cfg) for seed in seeds])

    print("[BC pipeline] Step 3 done.")


# ── Full pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = load_config()

    os.makedirs(cfg["training"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["logs_dir"],   exist_ok=True)
    os.makedirs("results/demos",               exist_ok=True)

    # Step 1 — collect demos
    demos = step1_collect_demos(cfg, n_demos=500)

    # Step 2 — BC pretraining + PPO fine-tuning
    step2_behavior_cloning(cfg, demos)

    # Step 3 — meta-controller training
    step3_train_meta(cfg)

    print("\n[BC pipeline] All done. Run baselines.py next.")