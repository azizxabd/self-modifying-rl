# core/demo_collector.py
import os
import numpy as np
import pickle
import multiprocessing as mp
from tqdm import tqdm


def _collect_worker(args):
    n_demos, episode_length, seed_start = args

    from core.robosuite_env import RobosuitePhased, TABLE_Z
    from core.scripted_policy import ScriptedLiftPolicy

    demonstrations = []
    seed = seed_start

    while len(demonstrations) < n_demos:
        env    = RobosuitePhased(seed=seed, episode_length=episode_length)
        obs, _ = env.reset()
        policy = ScriptedLiftPolicy()
        policy.reset()
        policy.stage = 1

        episode_obs     = []
        episode_actions = []
        max_height      = 0.0

        for t in range(episode_length):
            raw_obs = env._env._get_observations()
            action  = policy.act(raw_obs)
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            obs, reward, term, trunc, info = env.step(action)
            cube_h     = env.history[-1]["cube_height"]
            max_height = max(max_height, cube_h)
            if term or trunc:
                break

        env.close()
        seed += 1

        if max_height > TABLE_Z + 0.04:
            demonstrations.append({
                "obs":        np.array(episode_obs),
                "actions":    np.array(episode_actions),
                "max_height": max_height,
                "length":     len(episode_obs),
            })

    return demonstrations


def collect_demonstrations(
    n_demos:        int = 500,
    episode_length: int = 1500,
    save_path:      str = "results/demos/demonstrations.pkl",
    seed_start:     int = 0,
    n_workers:      int = 4,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    demos_per_worker = n_demos // n_workers
    remainder        = n_demos % n_workers

    # distribute demos across workers
    worker_args = []
    for i in range(n_workers):
        extra  = 1 if i < remainder else 0
        worker_args.append((
            demos_per_worker + extra,
            episode_length,
            seed_start + i * 2000,
        ))

    print(f"[demo] Collecting {n_demos} demos with {n_workers} workers...")
    print(f"[demo] Each worker collects ~{demos_per_worker} demos")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_collect_worker, worker_args),
            total=n_workers,
            desc="Workers",
        ))

    # merge all demos
    all_demos = []
    for demos in results:
        all_demos.extend(demos)
    all_demos = all_demos[:n_demos]

    with open(save_path, "wb") as f:
        pickle.dump(all_demos, f)

    print(f"\n[demo] Saved {len(all_demos)} demos to {save_path}")
    print(f"[demo] Mean episode length: {np.mean([d['length'] for d in all_demos]):.0f}")
    print(f"[demo] Mean max height:     {np.mean([d['max_height'] for d in all_demos]):.4f}m")

    return all_demos


def load_demonstrations(path: str = "results/demos/demonstrations.pkl"):
    with open(path, "rb") as f:
        demos = pickle.load(f)
    print(f"[demo] Loaded {len(demos)} demonstrations from {path}")
    return demos


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    collect_demonstrations(
        n_demos=500,
        episode_length=1500,
        save_path="results/demos/demonstrations.pkl",
        n_workers=4,
    )