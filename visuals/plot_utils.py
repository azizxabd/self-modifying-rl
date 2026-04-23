# visuals/plot_utils.py
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np


# ── Agent registry ────────────────────────────────────────────────────────────

AGENTS = {
    "self_modifying": {
        "label":  "Self-Modifying",
        "color":  "#2563EB",
        "dash":   "-",
        "marker": "o",
    },
    "fixed_ppo": {
        "label":  "Fixed PPO",
        "color":  "#DC2626",
        "dash":   "--",
        "marker": "s",
    },
    "high_capacity": {
        "label":  "High Capacity PPO",
        "color":  "#EA580C",
        "dash":   "-.",
        "marker": "D",
    },
    "heuristic": {
        "label":  "Heuristic",
        "color":  "#16A34A",
        "dash":   "-.",
        "marker": "^",
    },
    "random": {
        "label":  "Random",
        "color":  "#9333EA",
        "dash":   ":",
        "marker": "x",
    },
}

ARCH_COLORS = {
    "mlp":  "#2563EB",
    "lstm": "#EA580C",
}

PHASE_FILL   = ["#DBEAFE", "#FEF9C3", "#DCFCE7"]
PHASE_BORDER = ["#3B82F6", "#EAB308", "#22C55E"]
PHASE_NAMES  = ["Normal",  "Heavy",   "Disturbed"]


# ── Global style ──────────────────────────────────────────────────────────────

def set_style():
    sns.set_theme(style="whitegrid", font="Georgia")
    plt.rcParams.update({
        "figure.dpi":          150,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "font.family":         "serif",
        "font.serif":          ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size":           11,
        "axes.titlesize":      12,
        "axes.titleweight":    "bold",
        "axes.labelsize":      11,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    True,
        "axes.spines.bottom":  True,
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.alpha":          0.35,
        "grid.linewidth":      0.5,
        "grid.linestyle":      "--",
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     9,
        "legend.title_fontsize": 9,
        "lines.linewidth":     1.8,
        "lines.markersize":    5,
        "savefig.dpi":         150,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   "white",
    })


# ── Confidence band ───────────────────────────────────────────────────────────

def plot_band(ax, x, mean, std, color, label, dash="-", alpha=0.15, n_std=1):
    ax.plot(x, mean, color=color, linestyle=dash,
            linewidth=1.8, label=label, zorder=3)
    ax.fill_between(x,
                    mean - n_std * std,
                    mean + n_std * std,
                    color=color, alpha=alpha, zorder=2)


# ── Phase bands ───────────────────────────────────────────────────────────────

def draw_phase_bands(ax, phase_schedule, phase_duration,
                     episode_length=500, show_labels=True):
    from core.robosuite_env import ROBOSUITE_PHASES
    seen = set()
    for i, pid in enumerate(phase_schedule):
        start = i * phase_duration
        end   = min((i + 1) * phase_duration, episode_length)
        ax.axvspan(start, end,
                   color=ROBOSUITE_PHASES[pid]["color"],
                   alpha=0.5, zorder=0)
        ax.axvline(start, color="#CBD5E1", linewidth=0.7, zorder=1)
        if show_labels:
            mid = (start + end) / 2
            ax.text(mid, 1.01, ROBOSUITE_PHASES[pid]["name"],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8, style="italic",
                    color=PHASE_BORDER[pid], fontweight="bold")


# ── Axis formatting ───────────────────────────────────────────────────────────

def format_ax(ax, xlabel=None, ylabel=None, title=None,
              legend=True, legend_loc="best"):
    if xlabel: ax.set_xlabel(xlabel, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, labelpad=6)
    if title:  ax.set_title(title,   pad=8)
    if legend: ax.legend(loc=legend_loc, handlelength=2.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", length=3)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved → {path}")