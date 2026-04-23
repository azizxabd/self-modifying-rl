# visuals/plot_comparison.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visuals.plot_utils import (
    set_style, format_ax, save_fig,
    AGENTS
)


# ── Main: final performance bar chart ────────────────────────────────────────

def plot_final_comparison(summary, save_path=None, show=True):
    set_style()

    agents  = [a for a in summary["agent"] if a in AGENTS]
    means   = [summary.loc[summary["agent"] == a, "mean_reward"].values[0]  for a in agents]
    stds    = [summary.loc[summary["agent"] == a, "std_reward"].values[0]   for a in agents]
    colors  = [AGENTS[a]["color"] for a in agents]
    labels  = [AGENTS[a]["label"] for a in agents]
    x       = np.arange(len(agents))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        x, means,
        yerr=stds,
        color=colors,
        alpha=0.85,
        width=0.5,
        capsize=5,
        error_kw={"linewidth": 1.2, "ecolor": "#475569"},
        zorder=3,
    )

    for bar, mean, std in zip(bars, means, stds):
        ypos = bar.get_height() + std + 0.01
        if mean < 0:
            ypos = std + 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{mean:+.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="#1E293B",
        )

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    format_ax(
        ax,
        ylabel="Mean Final Reward",
        title="Final Performance Comparison\n(mean ± std over last 50 episodes across seeds)",
        legend=False,
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Secondary: cube height comparison ────────────────────────────────────────

def plot_height_comparison(summary, save_path=None, show=True):
    """
    Bar chart of mean max cube height per agent.
    Most direct measure of task success.
    """
    set_style()

    agents  = [a for a in summary["agent"] if a in AGENTS]
    heights = [summary.loc[summary["agent"] == a, "mean_max_height"].values[0] for a in agents]
    colors  = [AGENTS[a]["color"] for a in agents]
    labels  = [AGENTS[a]["label"] for a in agents]
    x       = np.arange(len(agents))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        x, heights,
        color=colors,
        alpha=0.85,
        width=0.5,
        zorder=3,
    )

    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{h:.4f}m",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="#1E293B",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    format_ax(
        ax,
        ylabel="Mean Max Cube Height (m)",
        title="Cube Lift Height Comparison — All Agents",
        legend=False,
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Tertiary: switches vs reward scatter ─────────────────────────────────────

def plot_switches_vs_reward(df, save_path=None, show=True):
    set_style()

    agents = ["self_modifying", "heuristic", "random"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for agent in agents:
        if agent not in AGENTS:
            continue
        cfg = AGENTS[agent]
        sub = df[df["agent"] == agent]
        ax.scatter(
            sub["switches"], sub["reward"],
            color=cfg["color"], alpha=0.2,
            s=12, label=cfg["label"], zorder=2,
        )
        x_vals = sub["switches"].values
        y_vals = sub["reward"].values
        if len(np.unique(x_vals)) > 1:
            z      = np.polyfit(x_vals, y_vals, 1)
            p      = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, p(x_line),
                    color=cfg["color"],
                    linewidth=1.6,
                    linestyle=cfg["dash"],
                    zorder=3)

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Switches per Episode",
        ylabel="Episode Reward",
        title="Switch Count vs Reward\n(scatter + trend per agent)",
        legend=True,
        legend_loc="upper right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Quaternary: switching cost ablation ──────────────────────────────────────

def plot_switching_cost_ablation(ablation_data, save_path=None, show=True):
    set_style()

    costs   = sorted(ablation_data.keys())
    rewards = [ablation_data[c]["mean"] for c in costs]
    stds    = [ablation_data[c]["std"]  for c in costs]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(costs, rewards,
            color=AGENTS["self_modifying"]["color"],
            linewidth=1.8, marker="o", markersize=6,
            zorder=3, label="Self-Modifying")
    ax.fill_between(
        costs,
        np.array(rewards) - np.array(stds),
        np.array(rewards) + np.array(stds),
        color=AGENTS["self_modifying"]["color"],
        alpha=0.15, zorder=2,
    )
    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Switching Cost (λ)",
        ylabel="Mean Final Reward",
        title="Ablation: Switching Cost vs Final Performance",
        legend=True,
        legend_loc="upper right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Quinary: full summary grid ────────────────────────────────────────────────

def plot_summary_grid(summary, save_path=None, show=True):
    set_style()

    agents  = [a for a in summary["agent"] if a in AGENTS]
    labels  = [AGENTS[a]["label"] for a in agents]
    colors  = [AGENTS[a]["color"] for a in agents]
    x       = np.arange(len(agents))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Summary Metrics — All Agents (Robosuite Panda Lift)",
                 fontsize=14, fontweight="bold", y=1.01)

    def _bar(ax, values, title, ylabel, show_zero=True):
        ax.bar(x, values, color=colors, alpha=0.85, width=0.5, zorder=3)
        if show_zero:
            ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        format_ax(ax, ylabel=ylabel, title=title, legend=False)

    means      = [summary.loc[summary["agent"] == a, "mean_reward"].values[0]    for a in agents]
    stds       = [summary.loc[summary["agent"] == a, "std_reward"].values[0]     for a in agents]
    switches   = [summary.loc[summary["agent"] == a, "mean_switches"].values[0]  for a in agents]
    mlp_frac   = [summary.loc[summary["agent"] == a, "mean_mlp_frac"].values[0]  for a in agents]
    max_height = [summary.loc[summary["agent"] == a, "mean_max_height"].values[0] for a in agents]
    mean_height= [summary.loc[summary["agent"] == a, "mean_mean_height"].values[0] for a in agents]

    _bar(axes[0, 0], means,       "Mean Final Reward",        "Reward")
    _bar(axes[0, 1], stds,        "Reward Std (variability)", "Std",          show_zero=False)
    _bar(axes[0, 2], switches,    "Mean Switches / Episode",  "Switches",     show_zero=False)
    _bar(axes[1, 0], mlp_frac,    "MLP Usage Fraction",       "Fraction",     show_zero=False)
    _bar(axes[1, 1], max_height,  "Mean Max Cube Height",     "Height (m)",   show_zero=False)
    _bar(axes[1, 2], mean_height, "Mean Avg Cube Height",     "Height (m)",   show_zero=False)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from evaluation.eval import load_all_logs, compute_summary
    from training.train  import load_config

    cfg     = load_config()
    df      = load_all_logs(cfg)
    summary = compute_summary(df)

    os.makedirs("results", exist_ok=True)

    plot_final_comparison(summary,
        save_path="results/final_comparison.png")

    plot_height_comparison(summary,
        save_path="results/height_comparison.png")

    plot_switches_vs_reward(df,
        save_path="results/switches_vs_reward.png")

    plot_summary_grid(summary,
        save_path="results/summary_grid.png")