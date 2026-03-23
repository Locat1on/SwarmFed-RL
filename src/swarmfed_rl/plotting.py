"""Plotting module aligned with three-layer core metrics.

Layer 1 — Learning effectiveness: episode return, success rate
Layer 2 — Federation value: multi-mode convergence comparison, communication cost
Layer 3 — Defense effectiveness: detection precision, performance under attack
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_COLORS = {
    "local": "#1f77b4",
    "centralized": "#ff7f0e",
    "p2p": "#2ca02c",
}
_FIGSIZE = (8, 4.5)


def _style_ax(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()


# ---------------------------------------------------------------------------
# Public entry: single-run plots (backward compatible)
# ---------------------------------------------------------------------------

def generate_plots(csv_path: str, out_dir: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    outputs = [
        _plot_episode_return(df, out_dir),
        _plot_success_rate(df, out_dir),
        _plot_communication(df, out_dir),
    ]

    # Defense plots only when defense data is present
    has_defense = (
        df.get("defense_rejected_malicious") is not None
        and (df["defense_rejected_malicious"] + df["defense_accepted_malicious"]).sum() > 0
    )
    if has_defense:
        outputs.append(_plot_defense_precision(df, out_dir))

    # Epoch-level plots
    epoch_csv = csv_path.replace(".csv", "_epoch.csv")
    if Path(epoch_csv).exists():
        try:
            outputs.extend(generate_epoch_plots(epoch_csv, out_dir))
        except Exception as e:
            print(f"Warning: Failed to generate epoch plots: {e}")

    return outputs


def generate_epoch_plots(epoch_csv_path: str, out_dir: str) -> list[str]:
    df = pd.read_csv(epoch_csv_path)
    if df.empty:
        return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return [
        _plot_epoch_return(df, out_dir),
        _plot_epoch_success_collision(df, out_dir),
    ]


# ---------------------------------------------------------------------------
# Public entry: multi-run comparison plots
# ---------------------------------------------------------------------------

def generate_comparison_plots(
    csv_paths: dict[str, str],
    out_dir: str,
) -> list[str]:
    """Generate comparison plots from multiple experiment CSVs.

    Args:
        csv_paths: mapping of label -> csv_path, e.g.
                   {"local": "artifacts/logs/local/run.csv",
                    "p2p": "artifacts/logs/p2p/run.csv"}
        out_dir: output directory for plots.
    """
    dfs = {}
    for label, path in csv_paths.items():
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping label '{label}'")
            continue
        dfs[label] = pd.read_csv(p)
    if not dfs:
        raise ValueError("No valid CSVs found for comparison")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    outputs = [
        _plot_compare_episode_return(dfs, out_dir),
        _plot_compare_success_rate(dfs, out_dir),
    ]

    # Communication comparison (skip labels with 0 bytes, e.g. local mode)
    comm_dfs = {k: v for k, v in dfs.items() if v["communication_bytes"].sum() > 0}
    if comm_dfs:
        outputs.append(_plot_compare_communication(comm_dfs, out_dir))

    # Defense comparison if any run has defense data
    def_dfs = {
        k: v for k, v in dfs.items()
        if v.get("defense_rejected_malicious") is not None
        and (v["defense_rejected_malicious"] + v["defense_accepted_malicious"]).sum() > 0
    }
    if def_dfs:
        outputs.append(_plot_compare_defense(def_dfs, out_dir))

    return outputs


# ===================================================================
# Layer 1: Learning Effectiveness
# ===================================================================

def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _plot_episode_return(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    window = max(int(len(df) * 0.05), 50)

    ax.plot(df["step"], df["episode_return_mean"],
            label="Episode Return (running avg)", color="#1f77b4", linewidth=2)
    ax.fill_between(
        df["step"],
        _smooth(df["step_reward"], window) * 0.8,
        _smooth(df["step_reward"], window) * 1.2,
        alpha=0.1, color="#1f77b4",
    )
    _style_ax(ax, "Timesteps", "Return", "Layer 1: Episode Return Convergence")
    path = str(Path(out_dir) / "L1_episode_return.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_success_rate(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    episodes = df["episodes"].replace(0, 1)
    success_rate = df["successes"] / episodes
    collision_rate = df["collisions"] / episodes

    ax.plot(df["step"], success_rate, label="Success Rate", color="green", linewidth=2)
    ax.plot(df["step"], collision_rate, label="Collision Rate", color="red", linewidth=1.5, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, "Timesteps", "Rate", "Layer 1: Success / Collision Rate")
    path = str(Path(out_dir) / "L1_success_rate.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ===================================================================
# Layer 2: Federation Value
# ===================================================================

def _plot_communication(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(df["step"], df["communication_bytes"] / 1e6,
            label="Communication", color="#ff7f0e", linewidth=2)
    _style_ax(ax, "Timesteps", "MB", "Layer 2: Communication Overhead")
    path = str(Path(out_dir) / "L2_communication.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_compare_episode_return(dfs: dict[str, pd.DataFrame], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for label, df in dfs.items():
        color = _COLORS.get(label, None)
        ax.plot(df["step"], df["episode_return_mean"],
                label=label, color=color, linewidth=2)
    _style_ax(ax, "Timesteps", "Episode Return Mean",
              "Layer 2: Convergence Comparison — Episode Return")
    path = str(Path(out_dir) / "L2_compare_episode_return.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_compare_success_rate(dfs: dict[str, pd.DataFrame], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for label, df in dfs.items():
        episodes = df["episodes"].replace(0, 1)
        rate = df["successes"] / episodes
        color = _COLORS.get(label, None)
        ax.plot(df["step"], rate, label=label, color=color, linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, "Timesteps", "Success Rate",
              "Layer 2: Convergence Comparison — Success Rate")
    path = str(Path(out_dir) / "L2_compare_success_rate.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_compare_communication(dfs: dict[str, pd.DataFrame], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for label, df in dfs.items():
        color = _COLORS.get(label, None)
        episodes = df["episodes"].replace(0, 1)
        rate = df["successes"] / episodes
        mb = df["communication_bytes"] / 1e6
        ax.plot(mb, rate, label=label, color=color, linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, "Communication (MB)", "Success Rate",
              "Layer 2: Communication Efficiency")
    path = str(Path(out_dir) / "L2_compare_comm_efficiency.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ===================================================================
# Layer 3: Defense Effectiveness
# ===================================================================

def _plot_defense_precision(df: pd.DataFrame, out_dir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    rejected_m = df["defense_rejected_malicious"]
    accepted_m = df["defense_accepted_malicious"]
    total_m = rejected_m + accepted_m
    precision = np.where(total_m > 0, rejected_m / total_m, np.nan)

    # Left: precision over time
    ax1.plot(df["step"], precision, color="purple", linewidth=2, label="Detection Precision")
    ax1.set_ylim(-0.05, 1.05)
    _style_ax(ax1, "Timesteps", "Precision",
              "Layer 3: Malicious Update Detection Precision")

    # Right: cumulative counts
    ax2.plot(df["step"], rejected_m, label="Rejected Malicious", color="green", linewidth=2)
    ax2.plot(df["step"], accepted_m, label="Accepted Malicious", color="red", linewidth=1.5)
    _style_ax(ax2, "Timesteps", "Count",
              "Layer 3: Defense Cumulative Counts")

    path = str(Path(out_dir) / "L3_defense_precision.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_compare_defense(dfs: dict[str, pd.DataFrame], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for label, df in dfs.items():
        episodes = df["episodes"].replace(0, 1)
        rate = df["successes"] / episodes
        ax.plot(df["step"], rate, label=label, linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, "Timesteps", "Success Rate",
              "Layer 3: Success Rate Under Attack (by config)")
    path = str(Path(out_dir) / "L3_compare_defense_success.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ===================================================================
# Epoch-level plots
# ===================================================================

def _plot_epoch_return(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(df["epoch"], df["epoch_reward"], marker='o', label="Epoch Total Reward", color="#1f77b4")
    ax.plot(df["epoch"], df["episode_return_mean"], marker='s', label="Avg Episode Return", color="#ff7f0e")
    _style_ax(ax, "Epoch", "Reward", "Epoch Performance: Reward")
    path = str(Path(out_dir) / "epoch_reward.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _plot_epoch_success_collision(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(df["epoch"], df["epoch_successes"], marker='o', label="Successes", color="green")
    ax.plot(df["epoch"], df["epoch_collisions"], marker='x', label="Collisions", color="red")
    _style_ax(ax, "Epoch", "Count per Epoch", "Epoch Performance: Success vs Collision")
    path = str(Path(out_dir) / "epoch_success_collision.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path
