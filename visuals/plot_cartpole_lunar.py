# visuals/plot_cartpole_lunar.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train_cartpole import CARTPOLE_CFG
from training.train_lunarlander import LUNARLANDER_CFG

# ── Style ─────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "self_modifying": "#E63946",
    "high_capacity":  "#2196F3",
    "fixed_ppo":      "#4CAF50",
    "heuristic":      "#FF9800",
    "random":         "#9E9E9E",
}

AGENT_LABELS = {
    "self_modifying": "Self-Modifying",
    "high_capacity":  "High Capacity",
    "fixed_ppo":      "Fixed PPO",
    "heuristic":      "Heuristic",
    "random":         "Random",
}

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_logs(logs_dir):
    all_dfs = []
    for pattern in [
        os.path.join(logs_dir, "self_modifying_seed*.csv"),
        os.path.join(logs_dir, "baselines_seed*.csv"),
    ]:
        for fpath in sorted(glob(pattern)):
            all_dfs.append(pd.read_csv(fpath))
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


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
            "mean_reward":   final_rewards.mean(),
            "std_reward":    final_rewards.std(),
            "mean_switches": sub["switches"].mean(),
            "mean_mlp_frac": sub["mlp_frac"].mean(),
            "mean_lstm_frac":sub["lstm_frac"].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_reward", ascending=False).reset_index(drop=True)


# ── Plot 1: Learning curves side by side ──────────────────────────────────────

def plot_learning_curves_comparison(cp_curves, ll_curves, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Learning Curves — CartPole vs LunarLander (Phased)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, curves, title in zip(
        axes,
        [cp_curves, ll_curves],
        ["CartPole Phased", "LunarLander Phased"]
    ):
        agent_order = ["self_modifying", "high_capacity", "fixed_ppo", "heuristic", "random"]
        for agent in agent_order:
            if agent not in curves:
                continue
            c     = curves[agent]
            color = AGENT_COLORS[agent]
            label = AGENT_LABELS[agent]
            ax.plot(c["episodes"], c["mean"], color=color, label=label, linewidth=2)
            ax.fill_between(
                c["episodes"],
                c["mean"] - c["std"],
                c["mean"] + c["std"],
                color=color, alpha=0.15,
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Plot 2: Final performance bar chart ───────────────────────────────────────

def plot_final_comparison(cp_summary, ll_summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Final Performance Comparison (Mean ± Std, Last 50 Episodes)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, summary, title in zip(
        axes,
        [cp_summary, ll_summary],
        ["CartPole Phased", "LunarLander Phased"]
    ):
        agents = summary["agent"].tolist()
        means  = summary["mean_reward"].tolist()
        stds   = summary["std_reward"].tolist()
        colors = [AGENT_COLORS.get(a, "#888888") for a in agents]
        labels = [AGENT_LABELS.get(a, a) for a in agents]

        bars = ax.bar(labels, means, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.5)
        ax.errorbar(labels, means, yerr=stds, fmt="none",
                    color="black", capsize=5, linewidth=1.5)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(stds)*0.05,
                    f"{mean:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean Reward")
        ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Plot 3: Switching behavior ────────────────────────────────────────────────

def plot_switching_comparison(cp_summary, ll_summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Mean Switches Per Episode",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, summary, title in zip(
        axes,
        [cp_summary, ll_summary],
        ["CartPole Phased", "LunarLander Phased"]
    ):
        agents   = summary["agent"].tolist()
        switches = summary["mean_switches"].tolist()
        colors   = [AGENT_COLORS.get(a, "#888888") for a in agents]
        labels   = [AGENT_LABELS.get(a, a) for a in agents]

        bars = ax.bar(labels, switches, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.5)

        for bar, sw in zip(bars, switches):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"{sw:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean Switches / Episode")
        ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Plot 4: Architecture usage ────────────────────────────────────────────────

def plot_arch_usage_comparison(cp_summary, ll_summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Self-Modifying Agent — Architecture Usage",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, summary, title in zip(
        axes,
        [cp_summary, ll_summary],
        ["CartPole Phased", "LunarLander Phased"]
    ):
        sm = summary[summary["agent"] == "self_modifying"]
        if sm.empty:
            continue

        mlp_frac  = sm["mean_mlp_frac"].values[0]
        lstm_frac = sm["mean_lstm_frac"].values[0]

        wedges, texts, autotexts = ax.pie(
            [mlp_frac, lstm_frac],
            labels=["MLP\n(Memoryless)", "LSTM\n(Memory-Aware)"],
            colors=["#4CAF50", "#E63946"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight("bold")

        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Plot 5: Combined summary table ────────────────────────────────────────────

def plot_summary_table(cp_summary, ll_summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle("Results Summary Table",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, summary, title in zip(
        axes,
        [cp_summary, ll_summary],
        ["CartPole Phased", "LunarLander Phased"]
    ):
        ax.axis("off")
        table_data = []
        for _, row in summary.iterrows():
            table_data.append([
                AGENT_LABELS.get(row["agent"], row["agent"]),
                f"{row['mean_reward']:+.1f} ± {row['std_reward']:.1f}",
                f"{row['mean_switches']:.2f}",
                f"{row['mean_mlp_frac']:.2f} / {row['mean_lstm_frac']:.2f}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["Agent", "Reward (mean±std)", "Switches/ep", "MLP/LSTM"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # color self_modifying row
        for i, row in enumerate(summary["agent"].tolist()):
            color = AGENT_COLORS.get(row, "#ffffff")
            for j in range(4):
                table[(i+1, j)].set_facecolor(color + "33")

        # header style
        for j in range(4):
            table[(0, j)].set_facecolor("#37474F")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_style()

    os.makedirs("results/plots", exist_ok=True)

    print("[plots] Loading CartPole data...")
    cp_df = load_logs(CARTPOLE_CFG["training"]["logs_dir"])

    print("[plots] Loading LunarLander data...")
    ll_df = load_logs(LUNARLANDER_CFG["training"]["logs_dir"])

    if cp_df is None or ll_df is None:
        print("ERROR: Missing data. Run train_cartpole.py and train_lunarlander.py first.")
        exit(1)

    cp_curves  = compute_learning_curves(cp_df)
    ll_curves  = compute_learning_curves(ll_df)
    cp_summary = compute_summary(cp_df)
    ll_summary = compute_summary(ll_df)

    print("\n[plots] Generating plots...")

    plot_learning_curves_comparison(
        cp_curves, ll_curves,
        "results/plots/learning_curves_comparison.png"
    )
    plot_final_comparison(
        cp_summary, ll_summary,
        "results/plots/final_comparison.png"
    )
    plot_switching_comparison(
        cp_summary, ll_summary,
        "results/plots/switching_comparison.png"
    )
    plot_arch_usage_comparison(
        cp_summary, ll_summary,
        "results/plots/arch_usage_comparison.png"
    )
    plot_summary_table(
        cp_summary, ll_summary,
        "results/plots/summary_table.png"
    )

    print("\n[plots] All saved to results/plots/")
    print("  learning_curves_comparison.png")
    print("  final_comparison.png")
    print("  switching_comparison.png")
    print("  arch_usage_comparison.png")
    print("  summary_table.png")