from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(csv_path: str, out_dir: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs = [
        _plot_reward_curve(df, out_dir),
        _plot_success_collision(df, out_dir),
        _plot_communication(df, out_dir),
        _plot_defense(df, out_dir),
        _plot_convergence_rates(df, out_dir),
    ]
    
    # Check for epoch file
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
        _plot_epoch_reward(df, out_dir),
        _plot_epoch_metrics(df, out_dir),
    ]


def _plot_reward_curve(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate rolling average of step reward to show trends
    window = max(int(len(df) * 0.05), 100)
    rolling_reward = df["step_reward"].rolling(window=window).mean()
    
    ax.plot(df["step"], df["step_reward"], label="Step Reward (Raw)", alpha=0.3, color="gray")
    ax.plot(df["step"], rolling_reward, label=f"Step Reward ({window}-step Avg)", color="blue", linewidth=2)
    
    if "episode_return_mean" in df.columns:
        ax.plot(df["step"], df["episode_return_mean"], label="Episode Return Mean", color="orange", alpha=0.8)
    
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Training Stability: Reward Trend")
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    path = str(Path(out_dir) / "reward_trend.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_epoch_reward(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["epoch_reward"], marker='o', label="Epoch Total Reward", color="blue")
    ax.plot(df["epoch"], df["episode_return_mean"], marker='s', label="Avg Episode Return", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Epoch Performance: Reward")
    ax.grid(True, alpha=0.2)
    ax.legend()
    path = str(Path(out_dir) / "epoch_reward.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_epoch_metrics(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["epoch_successes"], marker='o', label="Successes", color="green")
    ax.plot(df["epoch"], df["epoch_collisions"], marker='x', label="Collisions", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count per Epoch")
    ax.set_title("Epoch Performance: Success vs Collision")
    ax.grid(True, alpha=0.2)
    ax.legend()
    path = str(Path(out_dir) / "epoch_metrics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_convergence_rates(df: pd.DataFrame, out_dir: str) -> str:
    # Avoid division by zero
    episodes = df["episodes"].replace(0, 1)
    
    success_rate = df["successes"] / episodes
    collision_rate = df["collisions"] / episodes
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], success_rate, label="Success Rate", color="green")
    ax.plot(df["step"], collision_rate, label="Collision Rate", color="red")
    
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate (Cumulative)")
    ax.set_title("Convergence: Success & Collision Rates")
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    path = str(Path(out_dir) / "convergence_rates.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_success_collision(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["successes"], label="successes")
    ax.plot(df["step"], df["collisions"], label="collisions")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Count")
    ax.set_title("Success/Collision Count")
    ax.grid(True, alpha=0.2)
    ax.legend()
    path = str(Path(out_dir) / "success_collision.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_communication(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["communication_bytes"], label="communication bytes")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Bytes")
    ax.set_title("Communication Overhead")
    ax.grid(True, alpha=0.2)
    ax.legend()
    path = str(Path(out_dir) / "communication_overhead.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_defense(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["defense_rejected_malicious"], label="rejected malicious")
    ax.plot(df["step"], df["defense_accepted_malicious"], label="accepted malicious")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Count")
    ax.set_title("Defense Robustness")
    ax.grid(True, alpha=0.2)
    ax.legend()
    path = str(Path(out_dir) / "defense_robustness.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
