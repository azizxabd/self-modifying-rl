# visuals/plot_env.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visuals.plot_utils import (
    set_style, format_ax, save_fig,
    PHASE_FILL, PHASE_BORDER, PHASE_NAMES
)
from core.robosuite_env import ROBOSUITE_PHASES, PHASE_SCHEDULE


# ── Main: 4-panel episode visualization ──────────────────────────────────────

def plot_episode(history, phase_boundaries, title="Robosuite Episode", save_path=None, show=True):
    """
    4-panel figure:
      Panel 1 — Reward signal with phase bands
      Panel 2 — Cube height over time
      Panel 3 — Action norm (effort)
      Panel 4 — Phase indicator
    """
    set_style()

    t           = history["t"]
    reward      = history["reward"]
    cube_height = history["cube_height"]
    action_norm = history["action_norm"]
    phase_ids   = history["phase_id"]

    window        = 10
    smooth_reward = np.convolve(reward,      np.ones(window)/window, mode="same")
    smooth_height = np.convolve(cube_height, np.ones(window)/window, mode="same")

    fig = plt.figure(figsize=(12, 9))
    gs  = GridSpec(4, 1, figure=fig, hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    axes = [ax1, ax2, ax3, ax4]

    # ── Phase bands ───────────────────────────────────────────────────────────
    legend_patches = []
    seen_phases    = set()

    for pb in phase_boundaries:
        pid = pb["phase_id"]
        for ax in axes:
            ax.axvspan(pb["start"], pb["end"],
                       color=PHASE_FILL[pid], alpha=0.6, zorder=0)
            ax.axvline(pb["start"],
                       color="#CBD5E1", linewidth=0.8, zorder=1)

        mid = (pb["start"] + pb["end"]) / 2
        ax1.text(mid, 1.02, pb["phase_name"],
                 transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom",
                 fontsize=8, style="italic",
                 color=PHASE_BORDER[pid], fontweight="bold")

        if pid not in seen_phases:
            seen_phases.add(pid)
            legend_patches.append(mpatches.Patch(
                facecolor=PHASE_FILL[pid],
                edgecolor=PHASE_BORDER[pid],
                linewidth=1.2,
                label=f"{pb['phase_name']}",
            ))

    # ── Panel 1: Reward ───────────────────────────────────────────────────────
    ax1.plot(t, reward,        color="#94A3B8", linewidth=0.6, alpha=0.5, zorder=2)
    ax1.plot(t, smooth_reward, color="#1E3A5F", linewidth=1.8, zorder=3,
             label="Reward (smoothed)")
    ax1.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(ax1, ylabel="Reward", title="Reward Signal",
              legend=True, legend_loc="upper right")
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: Cube height ──────────────────────────────────────────────────
    ax2.plot(t, cube_height,   color="#93C5FD", linewidth=0.6, alpha=0.5, zorder=2)
    ax2.plot(t, smooth_height, color="#1D4ED8", linewidth=1.8, zorder=3,
             label="Cube height (smoothed)")
    ax2.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(ax2, ylabel="Height (m)", title="Cube Height",
              legend=True, legend_loc="upper right")
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: Action norm ──────────────────────────────────────────────────
    ax3.fill_between(t, 0, action_norm,
                     color="#2563EB", alpha=0.4, zorder=2, label="Action norm")
    ax3.plot(t, action_norm, color="#1E40AF", linewidth=0.8, zorder=3)
    format_ax(ax3, ylabel="‖action‖", title="Control Effort",
              legend=True, legend_loc="upper right")
    ax3.tick_params(labelbottom=False)

    # ── Panel 4: Phase indicator ──────────────────────────────────────────────
    ax4.plot(t, phase_ids, color="#475569", linewidth=1.2,
             drawstyle="steps-post", label="Phase ID")
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["Normal", "Heavy", "Disturbed"], fontsize=8)
    format_ax(ax4, xlabel="Timestep", ylabel="Phase",
              title="Active Phase", legend=False)

    # ── Phase legend ──────────────────────────────────────────────────────────
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        title="Environment Phases",
        title_fontsize=9,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Secondary: phase reward structure heatmap ─────────────────────────────────

def plot_phase_reward_structure(save_path=None, show=True):
    import seaborn as sns
    set_style()

    phase_ids  = sorted(ROBOSUITE_PHASES.keys())
    components = ["lift", "efficiency", "stability"]
    data       = np.array([
        [ROBOSUITE_PHASES[pid]["reward_weights"][c] for c in components]
        for pid in phase_ids
    ])
    phase_labels = [ROBOSUITE_PHASES[pid]["name"] for pid in phase_ids]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        data,
        annot=True, fmt=".1f",
        xticklabels=components,
        yticklabels=phase_labels,
        cmap="Blues",
        linewidths=0.5,
        linecolor="#E2E8F0",
        cbar_kws={"shrink": 0.8, "label": "Weight"},
        ax=ax,
    )
    ax.set_title("Reward Weights per Phase",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Reward Component", labelpad=6)
    ax.set_ylabel("Phase", labelpad=6)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Tertiary: cube height comparison across policies ─────────────────────────

def plot_height_curves(height_curves, save_path=None, show=True):
    from visuals.plot_utils import plot_band, AGENTS
    set_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    for agent, data in height_curves.items():
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

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Episode",
        ylabel="Max Cube Height (m)",
        title="Cube Lift Height Over Training — All Agents",
        legend=True,
        legend_loc="upper left",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.robosuite_env import RobosuitePhased

    env    = RobosuitePhased(seed=42)
    obs, _ = env.reset(seed=42)
    done   = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated

    history    = env.get_history_arrays()
    boundaries = env.get_phase_boundaries()
    env.close()

    os.makedirs("results", exist_ok=True)

    plot_episode(
        history, boundaries,
        title="Robosuite Episode — Random Policy (seed=42)",
        save_path="results/env_episode.png",
    )

    plot_phase_reward_structure(
        save_path="results/phase_reward_structure.png",
    )