"""
plotting.py
-----------
Matplotlib plotting utilities for 2048 training and evaluation.

All functions accept either a path to a CSV log file or a list of dicts
(as returned by EpisodeLogger.get_records()).

Usage
-----
    from utils.plotting import plot_training, plot_agent_comparison

    plot_training("logs/dqn_run_20240101/episodes.csv", save_dir="logs/plots")
    plot_agent_comparison(results_dict, save_dir="logs/plots")
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Lazy matplotlib import so the module doesn't crash in headless envs
def _plt():
    import matplotlib
    matplotlib.use("Agg")        # non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> List[Dict[str, Any]]:
    """Load a CSV log into a list of dicts."""
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            records.append(parsed)
    return records


def _col(records: List[dict], key: str) -> np.ndarray:
    vals = [r.get(key, 0.0) for r in records]
    try:
        return np.array(vals, dtype=np.float64)
    except Exception:
        return np.zeros(len(vals))


def _moving_avg(arr: np.ndarray, w: int = 50) -> np.ndarray:
    kernel  = np.ones(w) / w
    padded  = np.pad(arr, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_rewards(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Episode rewards + moving average."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    rewards = _col(records, "reward")
    eps     = np.arange(len(rewards))
    ma      = _moving_avg(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, rewards, alpha=0.25, color="steelblue", label="Episode reward")
    ax.plot(eps, ma,      color="steelblue", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "rewards.png", show)


def plot_scores(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Episode game scores + moving average."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    scores = _col(records, "score")
    eps    = np.arange(len(scores))
    ma     = _moving_avg(scores, window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, scores, alpha=0.25, color="coral", label="Episode score")
    ax.plot(eps, ma,     color="coral", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Game Scores Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "scores.png", show)


def plot_loss(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Training loss curve."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    loss = _col(records, "loss")
    # Filter out zero-loss episodes (before learning starts)
    nonzero = loss[loss > 0]
    eps_nz  = np.where(loss > 0)[0]

    if len(nonzero) < 2:
        print("Not enough loss data to plot.")
        return

    ma = _moving_avg(nonzero, min(window, len(nonzero)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(eps_nz, nonzero, alpha=0.2, color="mediumseagreen", label="Loss")
    ax.semilogy(eps_nz, ma,      color="mediumseagreen", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "loss.png", show)


def plot_epsilon(
    records: Union[str, List[dict]],
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Epsilon decay curve."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    eps_vals = _col(records, "epsilon")
    x        = np.arange(len(eps_vals))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, eps_vals, color="orchid", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "epsilon.png", show)


def plot_max_tile_frequency(
    records: Union[str, List[dict]],
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Histogram of the maximum tile reached per episode."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    tiles = _col(records, "max_tile").astype(int)
    unique, counts = np.unique(tiles, return_counts=True)
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        [str(t) for t in unique], pct,
        color="goldenrod", edgecolor="white", linewidth=0.8,
    )
    ax.set_xlabel("Max Tile")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Max Tile Distribution")
    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{p:.1f}%", ha="center", va="bottom", fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "max_tile_freq.png", show)


def plot_q_values(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Average Q-values over training."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    q_vals = _col(records, "avg_q")
    nonzero = q_vals[q_vals != 0]
    eps_nz  = np.where(q_vals != 0)[0]
    if len(nonzero) < 2:
        return

    ma = _moving_avg(nonzero, min(window, len(nonzero)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps_nz, nonzero, alpha=0.2, color="royalblue", label="Avg Q")
    ax.plot(eps_nz, ma,      color="royalblue", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.set_title("Average Q-Values During Training")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "q_values.png", show)


def plot_invalid_moves(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Invalid move count per episode (should decrease during training)."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    invalids = _col(records, "invalid_moves")
    eps      = np.arange(len(invalids))
    ma       = _moving_avg(invalids, window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, invalids, alpha=0.25, color="tomato", label="Invalid moves")
    ax.plot(eps, ma,       color="tomato", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Invalid Moves Per Episode")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "invalid_moves.png", show)


def plot_action_distribution(
    records: Union[str, List[dict]],
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Stacked area chart of action distribution over training."""
    if isinstance(records, str):
        records = load_csv(records)
    plt = _plt()

    ups    = _col(records, "action_up")
    downs  = _col(records, "action_down")
    lefts  = _col(records, "action_left")
    rights = _col(records, "action_right")

    totals = ups + downs + lefts + rights
    totals[totals == 0] = 1  # avoid div/0

    x = np.arange(len(ups))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        x,
        ups / totals * 100,
        downs / totals * 100,
        lefts / totals * 100,
        rights / totals * 100,
        labels=["Up", "Down", "Left", "Right"],
        alpha=0.75,
        colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Action Distribution Over Training")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "action_distribution.png", show)


# ---------------------------------------------------------------------------
# Composite / convenience
# ---------------------------------------------------------------------------

def plot_training(
    records: Union[str, List[dict]],
    window:  int = 50,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Generate all standard training plots in one call."""
    _ensure_dir(save_dir)
    if isinstance(records, str):
        records = load_csv(records)

    plot_rewards(records,            window=window, save_dir=save_dir, show=show)
    plot_scores(records,             window=window, save_dir=save_dir, show=show)
    plot_loss(records,               window=window, save_dir=save_dir, show=show)
    plot_epsilon(records,                           save_dir=save_dir, show=show)
    plot_max_tile_frequency(records,                save_dir=save_dir, show=show)
    plot_q_values(records,           window=window, save_dir=save_dir, show=show)
    plot_invalid_moves(records,      window=window, save_dir=save_dir, show=show)
    plot_action_distribution(records,               save_dir=save_dir, show=show)

    if save_dir:
        print(f"All plots saved → {save_dir}/")


def plot_agent_comparison(
    results: Dict[str, dict],
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """
    Side-by-side comparison bar charts for multiple agents.

    Parameters
    ----------
    results : dict mapping agent_name → stats_dict with keys:
              avg_score, win_rate, avg_steps, avg_max_tile
    """
    _ensure_dir(save_dir)
    plt = _plt()

    agents   = list(results.keys())
    metrics  = ["avg_score", "win_rate", "avg_steps", "avg_max_tile"]
    titles   = ["Avg Score", "Win Rate (2048)", "Avg Steps Survived", "Avg Max Tile"]
    colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        values = [results[a].get(metric, 0.0) for a in agents]
        bars   = ax.bar(agents, values, color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            label = f"{v:.1%}" if metric == "win_rate" else f"{v:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                label, ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Agent Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(plt, save_dir, "agent_comparison.png", show)


def plot_tile_frequency_comparison(
    results: Dict[str, dict],
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """
    Grouped bar chart comparing the max-tile distribution across agents.

    results[agent]["tile_frequency"] must be a dict {tile: count}.
    """
    _ensure_dir(save_dir)
    plt = _plt()

    # Gather all tile values
    all_tiles = sorted(set(
        t for r in results.values()
        for t in r.get("tile_frequency", {})
    ))

    agents = list(results.keys())
    x      = np.arange(len(all_tiles))
    width  = 0.8 / len(agents)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (agent, color) in enumerate(zip(agents, colors)):
        freq  = results[agent].get("tile_frequency", {})
        total = results[agent].get("n_games", 1)
        vals  = [freq.get(t, 0) / total * 100 for t in all_tiles]
        ax.bar(x + i * width, vals, width, label=agent, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels([str(t) for t in all_tiles])
    ax.set_xlabel("Max Tile")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Max-Tile Distribution by Agent")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_dir, "tile_freq_comparison.png", show)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(plt, save_dir: Optional[str], filename: str, show: bool):
    if save_dir:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()
