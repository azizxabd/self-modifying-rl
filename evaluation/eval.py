# evaluation/eval.py
import os
import sys
import numpy as np
import pandas as pd
from glob import glob

from training.train import load_config


# ── Load all logs ─────────────────────────────────────────────────────────────

def load_all_logs(cfg):
    logs_dir = cfg["training"]["logs_dir"]
    all_dfs  = []

    for pattern in [
        os.path.join(logs_dir, "self_modifying_seed*.csv"),
        os.path.join(logs_dir, "baselines_seed*.csv"),
    ]:
        for fpath in sorted(glob(pattern)):
            all_dfs.append(pd.read_csv(fpath))

    if not all_dfs:
        raise FileNotFoundError(f"No logs found in {logs_dir}.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"[eval] Loaded {len(combined)} rows from {len(all_dfs)} files.")
    return combined


# ── Summary stats ─────────────────────────────────────────────────────────────

def compute_summary(df):
    rows = []
    for agent in df["agent"].unique():
        sub = df[df["agent"] == agent]
        final_rewards = (
            sub.groupby("seed")
               .apply(lambda x: x.nlargest(50, "episode")["reward"].mean(),
                      include_groups=False)
        )
        rows.append({
            "agent":          agent,
            "mean_reward":    round(final_rewards.mean(), 4),
            "std_reward":     round(final_rewards.std(),  4),
            "mean_switches":  round(sub["switches"].mean(),    2),
            "mean_mlp_frac":  round(sub["mlp_frac"].mean(),   3),
            "mean_lstm_frac": round(sub["lstm_frac"].mean(),  3),
            "mean_max_height":  round(sub["max_height"].mean(),  4),
            "mean_mean_height": round(sub["mean_height"].mean(), 4),
            "lift_success_rate": round((sub["max_height"] > 0.86).mean(), 3),
        })

    return (pd.DataFrame(rows)
              .sort_values("mean_reward", ascending=False)
              .reset_index(drop=True))


# ── Learning curves ───────────────────────────────────────────────────────────

def compute_learning_curves(df, window=20):
    curves = {}
    for agent in df["agent"].unique():
        sub   = df[df["agent"] == agent]
        seeds = sub["seed"].unique()
        seed_curves = []
        for seed in seeds:
            rewards  = sub[sub["seed"] == seed].sort_values("episode")["reward"].values
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            seed_curves.append(smoothed)
        min_len = min(len(c) for c in seed_curves)
        arr     = np.array([c[:min_len] for c in seed_curves])
        curves[agent] = {
            "mean":     arr.mean(axis=0),
            "std":      arr.std(axis=0),
            "episodes": np.arange(min_len),
        }
    return curves


# ── Arch usage ────────────────────────────────────────────────────────────────

def compute_arch_usage(df, agent="self_modifying", window=20):
    sub   = df[df["agent"] == agent]
    seeds = sub["seed"].unique()
    mlp_curves, lstm_curves = [], []

    for seed in seeds:
        seed_df = sub[sub["seed"] == seed].sort_values("episode")
        mlp_curves.append( np.convolve(seed_df["mlp_frac"].values,  np.ones(window)/window, mode="valid"))
        lstm_curves.append(np.convolve(seed_df["lstm_frac"].values, np.ones(window)/window, mode="valid"))

    min_len  = min(len(c) for c in mlp_curves)
    mlp_arr  = np.array([c[:min_len] for c in mlp_curves])
    lstm_arr = np.array([c[:min_len] for c in lstm_curves])

    return {
        "episodes":  np.arange(min_len),
        "mlp_mean":  mlp_arr.mean(axis=0),
        "mlp_std":   mlp_arr.std(axis=0),
        "lstm_mean": lstm_arr.mean(axis=0),
        "lstm_std":  lstm_arr.std(axis=0),
    }


# ── Switch frequency ──────────────────────────────────────────────────────────

def compute_switch_frequency(df, window=20):
    result = {}
    for agent in ["self_modifying", "heuristic", "random"]:
        if agent not in df["agent"].unique():
            continue
        sub   = df[df["agent"] == agent]
        seeds = sub["seed"].unique()
        seed_curves = []
        for seed in seeds:
            seed_df  = sub[sub["seed"] == seed].sort_values("episode")
            smoothed = np.convolve(seed_df["switches"].values, np.ones(window)/window, mode="valid")
            seed_curves.append(smoothed)
        min_len = min(len(c) for c in seed_curves)
        arr     = np.array([c[:min_len] for c in seed_curves])
        result[agent] = {
            "mean":     arr.mean(axis=0),
            "std":      arr.std(axis=0),
            "episodes": np.arange(min_len),
        }
    return result


# ── Cube height curves ────────────────────────────────────────────────────────

def compute_height_curves(df, window=20):
    curves = {}
    for agent in df["agent"].unique():
        sub   = df[df["agent"] == agent]
        seeds = sub["seed"].unique()
        seed_curves = []
        for seed in seeds:
            seed_df  = sub[sub["seed"] == seed].sort_values("episode")
            heights  = seed_df["max_height"].values
            smoothed = np.convolve(heights, np.ones(window)/window, mode="valid")
            seed_curves.append(smoothed)
        min_len = min(len(c) for c in seed_curves)
        arr     = np.array([c[:min_len] for c in seed_curves])
        curves[agent] = {
            "mean":     arr.mean(axis=0),
            "std":      arr.std(axis=0),
            "episodes": np.arange(min_len),
        }
    return curves


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(summary, arch_usage):
    print("\n" + "="*60)
    print("  EVALUATION REPORT — Robosuite Panda Lift")
    print("="*60)

    print("\n── Final Performance (mean ± std over last 50 eps) ──")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['mean_reward']:+.4f} ± {row['std_reward']:.4f}")

    print("\n── Mean Switches Per Episode ─────────────────────────")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['mean_switches']:.2f}")

    print("\n── Mean Max Cube Height ──────────────────────────────")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['mean_max_height']:.4f} m")

    print("\n── Architecture Usage (self_modifying) ───────────────")
    print(f"  MLP  mean fraction: {arch_usage['mlp_mean'].mean():.3f}")
    print(f"  LSTM mean fraction: {arch_usage['lstm_mean'].mean():.3f}")

    print("\n── Best Agent ────────────────────────────────────────")
    best = summary.iloc[0]
    print(f"  {best['agent']} with mean reward {best['mean_reward']:+.4f}")
    print("="*60 + "\n")

    print("\n── Lift Success Rate (cube > 0.86m) ──────────────────")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['lift_success_rate']*100:.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg          = load_config()
    df           = load_all_logs(cfg)
    summary      = compute_summary(df)
    curves       = compute_learning_curves(df)
    arch_usage   = compute_arch_usage(df)
    switch_freq  = compute_switch_frequency(df)
    height_curves = compute_height_curves(df)

    print_report(summary, arch_usage)

    out = os.path.join(cfg["training"]["logs_dir"], "summary.csv")
    summary.to_csv(out, index=False)
    print(f"[eval] Summary saved → {out}")