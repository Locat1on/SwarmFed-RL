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
    return outputs


def _plot_reward_curve(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate rolling average of step reward to show trends
    # Window size = 5% of total steps or 100, whichever is larger
    window = max(int(len(df) * 0.05), 100)
    rolling_reward = df["step_reward"].rolling(window=window).mean()
    
    ax.plot(df["step"], df["step_reward"], label="Step Reward (Raw)", alpha=0.3, color="gray")
    ax.plot(df["step"], rolling_reward, label=f"Step Reward ({window}-step Avg)", color="blue", linewidth=2)
    
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward per Step (All Robots)")
    ax.set_title("Training Stability: Reward Trend")
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    path = str(Path(out_dir) / "reward_trend.png")
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
