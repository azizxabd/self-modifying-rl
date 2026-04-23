# visuals/plot_robot.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visuals.plot_utils import (
    set_style, plot_band, format_ax, save_fig,
    AGENTS, ARCH_COLORS, PHASE_FILL, PHASE_BORDER
)
from core.robosuite_env import ROBOSUITE_PHASES, PHASE_SCHEDULE


# ── Main: cube trajectory colored by phase ───────────────────────────────────

def plot_cube_trajectory(env_history, phase_boundaries,
                         save_path=None, show=True):
    """
    3D-style 2-panel trajectory plot:
      Left  — XY plane (top-down view of cube position)
      Right — Z over time (height profile colored by phase)
    """
    set_style()

    t           = env_history["t"]
    cube_height = env_history["cube_height"]
    phase_ids   = env_history["phase_id"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: height profile colored by phase ────────────────────────────────
    ax = axes[0]
    for pb in phase_boundaries:
        pid   = pb["phase_id"]
        mask  = (t >= pb["start"]) & (t < pb["end"])
        t_seg = t[mask]
        h_seg = cube_height[mask]
        if len(t_seg) > 0:
            ax.plot(t_seg, h_seg,
                    color=PHASE_BORDER[pid],
                    linewidth=1.8, zorder=3,
                    label=pb["phase_name"])
            ax.fill_between(t_seg, 0, h_seg,
                            color=PHASE_FILL[pid],
                            alpha=0.4, zorder=2)

    for pb in phase_boundaries:
        ax.axvline(pb["start"], color="#CBD5E1",
                   linewidth=0.8, zorder=1)

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    format_ax(
        ax,
        xlabel="Timestep",
        ylabel="Cube Height (m)",
        title="Cube Height Profile\n(colored by phase)",
        legend=True,
        legend_loc="upper left",
    )

    # ── Right: height distribution per phase (violin) ────────────────────────
    ax2 = axes[1]
    import seaborn as sns

    phase_data   = []
    phase_labels = []
    phase_colors = []

    for pb in phase_boundaries:
        pid  = pb["phase_id"]
        mask = (t >= pb["start"]) & (t < pb["end"])
        h    = cube_height[mask]
        if len(h) > 0:
            phase_data.append(h)
            phase_labels.append(pb["phase_name"])
            phase_colors.append(PHASE_BORDER[pid])

    parts = ax2.violinplot(
        phase_data,
        positions=range(len(phase_data)),
        showmedians=True,
        showextrema=True,
    )

    for i, (pc, color) in enumerate(zip(parts["bodies"], phase_colors)):
        pc.set_facecolor(PHASE_FILL[i % len(PHASE_FILL)])
        pc.set_edgecolor(color)
        pc.set_alpha(0.7)

    parts["cmedians"].set_color("#1E293B")
    parts["cbars"].set_color("#94A3B8")
    parts["cmaxes"].set_color("#94A3B8")
    parts["cmins"].set_color("#94A3B8")

    ax2.set_xticks(range(len(phase_labels)))
    ax2.set_xticklabels(phase_labels, fontsize=10)
    format_ax(
        ax2,
        xlabel="Phase",
        ylabel="Cube Height (m)",
        title="Height Distribution per Phase",
        legend=False,
    )

    fig.suptitle("Cube Trajectory Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Secondary: action analysis per phase ─────────────────────────────────────

def plot_action_analysis(env_history, phase_boundaries,
                         save_path=None, show=True):
    """
    Shows control effort (action norm) broken down by phase.
    Reveals whether agent uses more effort in harder phases.
    """
    set_style()

    t           = env_history["t"]
    action_norm = env_history["action_norm"]
    phase_ids   = env_history["phase_id"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: action norm over time colored by phase ──────────────────────────
    ax = axes[0]
    for pb in phase_boundaries:
        pid  = pb["phase_id"]
        mask = (t >= pb["start"]) & (t < pb["end"])
        t_s  = t[mask]
        a_s  = action_norm[mask]
        if len(t_s) > 0:
            ax.fill_between(t_s, 0, a_s,
                            color=PHASE_FILL[pid], alpha=0.5, zorder=2)
            ax.plot(t_s, a_s,
                    color=PHASE_BORDER[pid], linewidth=0.8, zorder=3)
        ax.axvline(pb["start"], color="#CBD5E1",
                   linewidth=0.8, zorder=1)

    format_ax(
        ax,
        xlabel="Timestep",
        ylabel="‖action‖",
        title="Control Effort Over Time\n(colored by phase)",
        legend=False,
    )

    patches = [
        mpatches.Patch(facecolor=PHASE_FILL[pb["phase_id"]],
                       edgecolor=PHASE_BORDER[pb["phase_id"]],
                       label=pb["phase_name"])
        for pb in phase_boundaries[:3]
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    # ── Right: mean action norm per phase bar chart ───────────────────────────
    ax2    = axes[1]
    labels = []
    means  = []
    stds   = []
    colors = []

    seen = set()
    for pb in phase_boundaries:
        pid  = pb["phase_id"]
        if pid in seen:
            continue
        seen.add(pid)
        mask = (phase_ids == pid)
        a    = action_norm[mask]
        labels.append(pb["phase_name"])
        means.append(float(np.mean(a)))
        stds.append(float(np.std(a)))
        colors.append(PHASE_FILL[pid])

    x    = np.arange(len(labels))
    bars = ax2.bar(x, means, yerr=stds,
                   color=colors,
                   edgecolor=[PHASE_BORDER[i] for i in range(len(labels))],
                   linewidth=1.2, width=0.5, capsize=5,
                   error_kw={"linewidth": 1.2, "ecolor": "#475569"},
                   zorder=3)

    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{mean:.2f}",
                 ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#1E293B")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    format_ax(
        ax2,
        ylabel="Mean ‖action‖",
        title="Mean Control Effort per Phase",
        legend=False,
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Tertiary: arch preference per phase ──────────────────────────────────────

def plot_arch_preference_per_phase(df, save_path=None, show=True):
    """
    For each phase, shows MLP vs LSTM usage fraction
    for the self-modifying agent.
    Strongest evidence that the meta-controller learned
    phase-appropriate architecture selection.
    """
    set_style()

    sub = df[df["agent"] == "self_modifying"]

    phase_mlp  = {ROBOSUITE_PHASES[pid]["name"]: [] for pid in ROBOSUITE_PHASES}
    phase_lstm = {ROBOSUITE_PHASES[pid]["name"]: [] for pid in ROBOSUITE_PHASES}

    for _, row in sub.iterrows():
        phase_idx  = int(row["episode"] % len(PHASE_SCHEDULE))
        phase_name = ROBOSUITE_PHASES[PHASE_SCHEDULE[phase_idx]]["name"]
        phase_mlp[phase_name].append(row["mlp_frac"])
        phase_lstm[phase_name].append(row["lstm_frac"])

    phases     = list(phase_mlp.keys())
    mlp_means  = [np.mean(phase_mlp[p])  for p in phases]
    lstm_means = [np.mean(phase_lstm[p]) for p in phases]
    x          = np.arange(len(phases))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width/2, mlp_means,  width,
           label="MLP",  color=ARCH_COLORS["mlp"],
           alpha=0.8, zorder=3)
    ax.bar(x + width/2, lstm_means, width,
           label="LSTM", color=ARCH_COLORS["lstm"],
           alpha=0.8, zorder=3)

    for i, (m, l) in enumerate(zip(mlp_means, lstm_means)):
        ax.text(i - width/2, m + 0.01, f"{m:.2f}",
                ha="center", fontsize=8, fontweight="bold",
                color=ARCH_COLORS["mlp"])
        ax.text(i + width/2, l + 0.01, f"{l:.2f}",
                ha="center", fontsize=8, fontweight="bold",
                color=ARCH_COLORS["lstm"])

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{int(y*100)}%")
    )
    format_ax(
        ax,
        ylabel="Usage Fraction",
        title="Architecture Preference per Phase\n(Self-Modifying Agent)",
        legend=True,
        legend_loc="upper right",
    )

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Quaternary: multi-agent cube height over training ─────────────────────────

def plot_lift_progress(height_curves, save_path=None, show=True):
    """
    Shows how cube lift height improves over training
    for all agents — your clearest task success figure.
    """
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
        title="Lift Progress Over Training — All Agents\n"
              "(mean ± 1 std across seeds)",
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
    from core.robosuite_env      import RobosuitePhased
    from evaluation.eval         import load_all_logs, compute_height_curves
    from training.train          import load_config

    cfg = load_config()

    # Sample episode for trajectory plots
    env    = RobosuitePhased(seed=42)
    obs, _ = env.reset(seed=42)
    done   = False
    while not done:
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated

    history    = env.get_history_arrays()
    boundaries = env.get_phase_boundaries()
    env.close()

    os.makedirs("results", exist_ok=True)

    plot_cube_trajectory(history, boundaries,
        save_path="results/cube_trajectory.png")

    plot_action_analysis(history, boundaries,
        save_path="results/action_analysis.png")

    # Needs training logs
    try:
        df            = load_all_logs(cfg)
        height_curves = compute_height_curves(df)
        plot_arch_preference_per_phase(df,
            save_path="results/arch_preference_per_phase.png")
        plot_lift_progress(height_curves,
            save_path="results/lift_progress.png")
    except FileNotFoundError:
        print("No training logs found — skipping log-based plots.")
        print("Run training first then re-run this script.")