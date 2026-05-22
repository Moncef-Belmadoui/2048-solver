"""
rewards.py
----------
Modular reward functions for the 2048 environment.

Each function has the signature:
    reward_fn(board_before, board_after, score_before, score_after,
              moved, done, invalid) -> float

Pass any of these to Game2048Env(reward_fn=...).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# Type alias
RewardFn = Callable[
    [np.ndarray, np.ndarray, int, int, bool, bool, bool], float
]


# ---------------------------------------------------------------------------
# Basic reward functions
# ---------------------------------------------------------------------------

def reward_sparse(
    board_before, board_after, score_before, score_after,
    moved, done, invalid
) -> float:
    """
    +1 if 2048 is reached (win), -1 if game over without win, 0 otherwise.
    """
    if done:
        return 1.0 if board_after.max() >= 2048 else -1.0
    return 0.0


def reward_score_delta(
    board_before, board_after, score_before, score_after,
    moved, done, invalid
) -> float:
    """Raw score gain this step (0 if invalid move)."""
    if invalid:
        return 0.0
    return float(score_after - score_before)


def reward_log_score_delta(
    board_before, board_after, score_before, score_after,
    moved, done, invalid
) -> float:
    """log2 of score gain (smooths large rewards)."""
    if invalid:
        return 0.0
    delta = score_after - score_before
    return math.log2(delta + 1)


def reward_survival(
    board_before, board_after, score_before, score_after,
    moved, done, invalid
) -> float:
    """
    +0.1 for every surviving step, -1 for game over,
    small penalty for invalid moves.
    """
    if invalid:
        return -0.1
    if done:
        return -1.0
    return 0.1


# ---------------------------------------------------------------------------
# Advanced shaped reward
# ---------------------------------------------------------------------------

def _monotonicity(board: np.ndarray) -> float:
    """
    Reward boards where tile values are monotonically increasing or
    decreasing along rows / columns.
    """
    score = 0.0
    for row in board:
        for direction in (1, -1):
            cur = row if direction == 1 else row[::-1]
            mono = 0.0
            for i in range(len(cur) - 1):
                a = math.log2(cur[i]) if cur[i] else 0
                b = math.log2(cur[i + 1]) if cur[i + 1] else 0
                if a > b:
                    mono += b - a   # negative penalty
                else:
                    mono += a - b   # negative penalty
            score = max(score, mono)

    for col in board.T:
        for direction in (1, -1):
            cur = col if direction == 1 else col[::-1]
            mono = 0.0
            for i in range(len(cur) - 1):
                a = math.log2(cur[i]) if cur[i] else 0
                b = math.log2(cur[i + 1]) if cur[i + 1] else 0
                if a > b:
                    mono += b - a
                else:
                    mono += a - b
            score = max(score, mono)

    return score  # ≤ 0


def _smoothness(board: np.ndarray) -> float:
    """Penalise large differences between neighbouring tiles (log2 space)."""
    penalty = 0.0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                continue
            val = math.log2(board[r, c])
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4 and board[nr, nc] != 0:
                    penalty -= abs(val - math.log2(board[nr, nc]))
    return penalty


def _free_tiles(board: np.ndarray) -> float:
    """Reward proportional to number of empty cells."""
    return float(np.sum(board == 0))


def _max_tile(board: np.ndarray) -> float:
    """log2 of the maximum tile."""
    m = int(board.max())
    return math.log2(m) if m else 0.0


def _corner_bonus(board: np.ndarray) -> float:
    """Reward max tile being in a corner."""
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    if board.max() in corners:
        return math.log2(int(board.max())) if board.max() else 0.0
    return 0.0


def make_shaped_reward(
    w_score: float = 1.0,
    w_mono: float = 1.0,
    w_smooth: float = 0.1,
    w_free: float = 2.7,
    w_max: float = 1.0,
    w_corner: float = 2.0,
    invalid_penalty: float = -0.5,
    game_over_penalty: float = -2.0,
) -> RewardFn:
    """
    Factory that returns a fully configured shaped reward function.

    The returned function combines:
        - score delta (log-scaled)
        - monotonicity
        - smoothness
        - free tile count
        - max tile (log2)
        - corner bonus
        - invalid move penalty
        - game-over penalty
    """

    def reward_shaped(
        board_before, board_after, score_before, score_after,
        moved, done, invalid
    ) -> float:
        if invalid:
            return invalid_penalty

        if done:
            return game_over_penalty

        delta = score_after - score_before
        r_score  = w_score  * math.log2(delta + 1)
        r_mono   = w_mono   * _monotonicity(board_after)
        r_smooth = w_smooth * _smoothness(board_after)
        r_free   = w_free   * _free_tiles(board_after)
        r_max    = w_max    * _max_tile(board_after)
        r_corner = w_corner * _corner_bonus(board_after)

        return r_score + r_mono + r_smooth + r_free + r_max + r_corner

    return reward_shaped


# Default shaped reward (ready to use)
reward_shaped_v1: RewardFn = make_shaped_reward()