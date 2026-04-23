# evaluation/eval_lunarlander.py
import os
import sys
import numpy as np
import pandas as pd
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train_lunarlander import LUNARLANDER_CFG


def load_all_logs():
    logs_dir = LUNARLANDER_CFG["training"]["logs_dir"]
    all_dfs  = []
    for pattern in [
        os.path.join(logs_dir, "self_modifying_seed*.csv"),
        os.path.join(logs_dir, "baselines_seed*.csv"),
    ]:
        for fpath in sorted(glob(pattern)):
            all_dfs.append(pd.read_csv(fpath))
    if not all_dfs:
        raise FileNotFoundError(f"No logs found in {logs_dir}")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"[eval] Loaded {len(combined)} rows from {len(all_dfs)} files.")
    return combined


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
            "agent":         agent,
            "mean_reward":   round(final_rewards.mean(), 4),
            "std_reward":    round(final_rewards.std(),  4),
            "mean_switches": round(sub["switches"].mean(), 2),
            "mean_mlp_frac": round(sub["mlp_frac"].mean(), 3),
            "mean_lstm_frac":round(sub["lstm_frac"].mean(), 3),
        })
    return (pd.DataFrame(rows)
              .sort_values("mean_reward", ascending=False)
              .reset_index(drop=True))


if __name__ == "__main__":
    df      = load_all_logs()
    summary = compute_summary(df)

    print("\n" + "="*55)
    print("  EVALUATION REPORT — LunarLander Phased")
    print("="*55)
    print("\n── Final Performance (mean ± std over last 50 eps) ──")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['mean_reward']:+.4f} ± {row['std_reward']:.4f}")
    print("\n── Mean Switches Per Episode ─────────────────────────")
    for _, row in summary.iterrows():
        print(f"  {row['agent']:<20} {row['mean_switches']:.2f}")
    print("\n── Architecture Usage (self_modifying) ───────────────")
    sm = summary[summary["agent"] == "self_modifying"]
    if not sm.empty:
        print(f"  MLP  mean fraction: {sm['mean_mlp_frac'].values[0]:.3f}")
        print(f"  LSTM mean fraction: {sm['mean_lstm_frac'].values[0]:.3f}")
    print("="*55)

    out = os.path.join(LUNARLANDER_CFG["training"]["logs_dir"], "summary.csv")
    summary.to_csv(out, index=False)
    print(f"\n[eval] Saved → {out}")