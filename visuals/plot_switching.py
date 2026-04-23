# visuals/plot_switching.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visuals.plot_utils import (
    set_style, plot_band, format_ax, save_fig,
    AGENTS, ARCH_COLORS, PHASE_FILL, PHASE_BORDER
)
from core.robosuite_env import ROBOSUITE_PHASES, PHASE_SCHEDULE


# ── Main: architecture usage over training ────────────────────────────────────

def plot_arch_usage(arch_usage, save_path=None, show=True):
    set_style()

    episodes  = arch_usage["episodes"]
    mlp_mean  = arch_usage["mlp_mean"]
    lstm_mean = arch_usage["lstm_mean"]
    mlp_std   = arch_usage["mlp_std"]
    lstm_std  = arch_usage["lstm_std"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(
        episodes,
        mlp_mean, lstm_mean,
        labels=["MLP (memoryless)", "LSTM (memory)"],
        colors=[ARCH_COLORS["mlp"], ARCH_COLORS["lstm"]],
        alpha=0.6,
    )

    plot_band(ax, episodes, mlp_mean, mlp_std,
              ARCH_COLORS["mlp"],  label=None, alpha=0.2)
    plot_band(ax, episodes, 1 - lstm_mean, lstm_std,
              ARCH_COLORS["lstm"], label=None, alpha=0.2)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{int(y*100)}%")
    )
    format_ax(
        ax,
        xlabel="Episode",
        ylabel="Architecture Usage",
        title="Architecture Usage Over Training — Self-Modifying Agent",
        legend=True,
        legend_loc="center right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Secondary: switch frequency ───────────────────────────────────────────────

def plot_switch_frequency(switch_freq, save_path=None, show=True):
    set_style()

    fig, ax = plt.subplots(figsize=(10, 4))

    for agent, data in switch_freq.items():
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

    format_ax(
        ax,
        xlabel="Episode",
        ylabel="Switches per Episode",
        title="Architecture Switch Frequency Over Training",
        legend=True,
        legend_loc="upper right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Tertiary: episode switching timeline ──────────────────────────────────────

def plot_episode_timeline(env_history, arch_log, save_path=None, show=True):
    """
    3-panel detailed episode view:
      Top    — reward with arch background coloring
      Middle — cube height with arch background
      Bottom — active architecture band
    """
    set_style()

    t           = env_history["t"]
    reward      = env_history["reward"]
    cube_height = env_history["cube_height"]
    arch_arr    = np.array(arch_log[:len(t)])

    window   = 10
    smoothed_r = np.convolve(reward,      np.ones(window)/window, mode="same")
    smoothed_h = np.convolve(cube_height, np.ones(window)/window, mode="same")

    fig = plt.figure(figsize=(12, 7))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 3, 1], hspace=0.2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── Arch background ───────────────────────────────────────────────────────
    for i in range(len(t) - 1):
        color = ARCH_COLORS["mlp"] if arch_arr[i] == 0 else ARCH_COLORS["lstm"]
        for ax in [ax1, ax2]:
            ax.axvspan(t[i], t[i+1], color=color, alpha=0.07, zorder=0)

    # ── Switch markers ────────────────────────────────────────────────────────
    switches = np.where(np.diff(arch_arr) != 0)[0]
    for sw in switches:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(t[sw], color="#475569", linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=4)

    # ── Panel 1: Reward ───────────────────────────────────────────────────────
    ax1.plot(t, reward,     color="#94A3B8", linewidth=0.6, alpha=0.5, zorder=2)
    ax1.plot(t, smoothed_r, color="#1E3A5F", linewidth=1.8, zorder=3,
             label="Reward (smoothed)")
    ax1.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(ax1, ylabel="Reward",
              title="Episode Timeline — Architecture Switching",
              legend=True, legend_loc="upper right")
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: Cube height ──────────────────────────────────────────────────
    ax2.plot(t, cube_height, color="#93C5FD", linewidth=0.6, alpha=0.5, zorder=2)
    ax2.plot(t, smoothed_h,  color="#1D4ED8", linewidth=1.8, zorder=3,
             label="Cube height")
    ax2.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(ax2, ylabel="Height (m)",
              title="Cube Height During Switching",
              legend=True, legend_loc="upper right")
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: Active arch band ─────────────────────────────────────────────
    for i in range(len(t) - 1):
        color = ARCH_COLORS["mlp"] if arch_arr[i] == 0 else ARCH_COLORS["lstm"]
        ax3.axvspan(t[i], t[i+1], color=color, alpha=0.7, zorder=1)

    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels(["MLP", "LSTM"], fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Timestep", labelpad=6)
    ax3.set_ylabel("Active\nArch", labelpad=6)
    ax3.grid(False)

    patches = [
        mpatches.Patch(color=ARCH_COLORS["mlp"],  alpha=0.7, label="MLP active"),
        mpatches.Patch(color=ARCH_COLORS["lstm"], alpha=0.7, label="LSTM active"),
    ]
    ax3.legend(handles=patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Quaternary: switching per phase analysis ──────────────────────────────────

def plot_switching_per_phase(df, save_path=None, show=True):
    """
    Shows how often the meta-controller switches architecture
    broken down by which phase the environment is in.
    Reveals whether switching correlates with phase transitions.
    """
    set_style()

    sub = df[df["agent"] == "self_modifying"]

    phase_switch_data = {
        ROBOSUITE_PHASES[pid]["name"]: [] for pid in ROBOSUITE_PHASES
    }

    for _, row in sub.iterrows():
        phase_idx = int(row["episode"] % len(PHASE_SCHEDULE))
        phase_name = ROBOSUITE_PHASES[PHASE_SCHEDULE[phase_idx]]["name"]
        phase_switch_data[phase_name].append(row["switches"])

    phases  = list(phase_switch_data.keys())
    means   = [np.mean(v) for v in phase_switch_data.values()]
    stds    = [np.std(v)  for v in phase_switch_data.values()]
    colors  = [PHASE_FILL[i] for i in range(len(phases))]
    borders = [PHASE_BORDER[i] for i in range(len(phases))]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(phases))
    bars = ax.bar(x, means, yerr=stds,
                  color=colors, edgecolor=borders,
                  linewidth=1.2, width=0.5, capsize=5,
                  error_kw={"linewidth": 1.2, "ecolor": "#475569"},
                  zorder=3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"{mean:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1E293B")

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=10)
    format_ax(
        ax,
        ylabel="Mean Switches",
        title="Architecture Switches per Phase\n(Self-Modifying Agent)",
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
    from evaluation.eval import load_all_logs, compute_arch_usage, compute_switch_frequency
    from training.train  import load_config

    cfg         = load_config()
    df          = load_all_logs(cfg)
    arch_usage  = compute_arch_usage(df)
    switch_freq = compute_switch_frequency(df)

    os.makedirs("results", exist_ok=True)

    plot_arch_usage(arch_usage,
        save_path="results/arch_usage.png")

    plot_switch_frequency(switch_freq,
        save_path="results/switch_frequency.png")

    plot_switching_per_phase(df,
        save_path="results/switching_per_phase.png")