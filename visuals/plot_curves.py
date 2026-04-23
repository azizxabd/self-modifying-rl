# visuals/plot_curves.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visuals.plot_utils import (
    set_style, plot_band, format_ax, save_fig,
    AGENTS, ARCH_COLORS
)


# ── Main: learning curves all agents ─────────────────────────────────────────

def plot_learning_curves(curves, save_path=None, show=True):
    set_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    for agent, data in curves.items():
        if agent not in AGENTS:
            continue
        cfg = AGENTS[agent]
        plot_band(
            ax,
            x=data["episodes"],
            mean=data["mean"],
            std=data["std"],
            color=cfg["color"],
            label=cfg["label"],
            dash=cfg["dash"],
        )

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--", zorder=1)
    format_ax(
        ax,
        xlabel="Episode",
        ylabel="Total Reward",
        title="Learning Curves — All Agents (mean ± 1 std across seeds)",
        legend=True,
        legend_loc="lower right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Secondary: reward vs cube height dual axis ────────────────────────────────

def plot_reward_vs_height(df, agent="self_modifying", window=20,
                          save_path=None, show=True):
    """
    Dual-axis plot showing reward and cube height on the same figure
    for one agent — shows correlation between reward and task success.
    """
    set_style()

    sub    = df[df["agent"] == agent]
    seeds  = sub["seed"].unique()

    reward_curves = []
    height_curves = []

    for seed in seeds:
        seed_df = sub[sub["seed"] == seed].sort_values("episode")
        r = np.convolve(seed_df["reward"].values,     np.ones(window)/window, mode="valid")
        h = np.convolve(seed_df["max_height"].values, np.ones(window)/window, mode="valid")
        reward_curves.append(r)
        height_curves.append(h)

    min_len = min(len(c) for c in reward_curves)
    r_arr   = np.array([c[:min_len] for c in reward_curves])
    h_arr   = np.array([c[:min_len] for c in height_curves])
    eps     = np.arange(min_len)

    cfg  = AGENTS.get(agent, {"color": "#2563EB", "label": agent})
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2  = ax1.twinx()

    ax1.plot(eps, r_arr.mean(axis=0),
             color=cfg["color"], linewidth=1.8, label="Reward")
    ax1.fill_between(eps,
                     r_arr.mean(axis=0) - r_arr.std(axis=0),
                     r_arr.mean(axis=0) + r_arr.std(axis=0),
                     color=cfg["color"], alpha=0.15)

    ax2.plot(eps, h_arr.mean(axis=0),
             color="#16A34A", linewidth=1.8, linestyle="--",
             label="Max cube height")
    ax2.fill_between(eps,
                     h_arr.mean(axis=0) - h_arr.std(axis=0),
                     h_arr.mean(axis=0) + h_arr.std(axis=0),
                     color="#16A34A", alpha=0.12)

    ax1.set_xlabel("Episode", labelpad=6)
    ax1.set_ylabel("Total Reward",      color=cfg["color"], labelpad=6)
    ax2.set_ylabel("Max Cube Height (m)", color="#16A34A",  labelpad=6)
    ax1.tick_params(axis="y", labelcolor=cfg["color"])
    ax2.tick_params(axis="y", labelcolor="#16A34A")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", frameon=False, fontsize=9)

    ax1.set_title(
        f"Reward vs Cube Height — {cfg['label']}",
        fontsize=12, fontweight="bold", pad=8,
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Tertiary: per-seed curves ─────────────────────────────────────────────────

def plot_per_seed_curves(df, agent="self_modifying", window=20,
                         save_path=None, show=True):
    set_style()

    cfg   = AGENTS.get(agent, {"color": "#2563EB", "label": agent, "dash": "-"})
    sub   = df[df["agent"] == agent]
    seeds = sorted(sub["seed"].unique())

    fig, ax = plt.subplots(figsize=(10, 4))

    for seed in seeds:
        seed_df  = sub[sub["seed"] == seed].sort_values("episode")
        smoothed = np.convolve(
            seed_df["reward"].values,
            np.ones(window)/window, mode="valid"
        )
        ax.plot(np.arange(len(smoothed)), smoothed,
                color=cfg["color"], linewidth=1.0,
                alpha=0.5, label=f"Seed {seed}")

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Episode",
        ylabel="Total Reward (smoothed)",
        title=f"Per-Seed Learning Curves — {cfg['label']}",
        legend=True,
        legend_loc="lower right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Quaternary: reward distributions violin ───────────────────────────────────

def plot_reward_distributions(df, save_path=None, show=True):
    import seaborn as sns
    set_style()

    agents_present = [a for a in AGENTS if a in df["agent"].unique()]
    labels         = {a: AGENTS[a]["label"] for a in agents_present}

    plot_df          = df[df["agent"].isin(agents_present)].copy()
    plot_df["Agent"] = plot_df["agent"].map(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=plot_df,
        x="Agent", y="reward",
        palette={labels[a]: AGENTS[a]["color"] for a in agents_present},
        inner="quartile",
        linewidth=0.8,
        ax=ax,
    )
    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Agent",
        ylabel="Episode Reward",
        title="Reward Distribution — All Agents",
        legend=False,
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from evaluation.eval import load_all_logs, compute_learning_curves
    from training.train  import load_config

    cfg    = load_config()
    df     = load_all_logs(cfg)
    curves = compute_learning_curves(df)

    os.makedirs("results", exist_ok=True)

    plot_learning_curves(curves,
        save_path="results/learning_curves.png")

    plot_per_seed_curves(df, agent="self_modifying",
        save_path="results/per_seed_curves.png")

    plot_reward_distributions(df,
        save_path="results/reward_distributions.png")

    plot_reward_vs_height(df, agent="self_modifying",
        save_path="results/reward_vs_height.png")