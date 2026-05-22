"""
heuristics.py
-------------
Individual board evaluation heuristics for the Expectiminimax agent.

Every function takes a (4, 4) numpy int32 board and returns a float.
Higher is always better.
"""

from __future__ import annotations
from typing import Optional
import math

import numpy as np

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _log2_board(board: np.ndarray) -> np.ndarray:
    """Return element-wise log2 of the board; empty cells → 0."""
    out = np.zeros((4, 4), dtype=np.float64)
    mask = board > 0
    out[mask] = np.log2(board[mask].astype(np.float64))
    return out


# ---------------------------------------------------------------------------
# Individual heuristics
# ---------------------------------------------------------------------------

# def monotonicity(board: np.ndarray) -> float:
#     """
#     Reward boards where tile values are monotonically ordered along rows
#     and columns.

#     We compute the best monotonic score along all 4 directions
#     (left→right, right→left for rows; top→bottom, bottom→top for cols)
#     and take the maximum so the snake / corner strategy scores well.
#     """
#     log_b = _log2_board(board)
#     score = 0.0

#     # Rows
#     for row in log_b:
#         inc = dec = 0.0
#         for i in range(3):
#             diff = row[i + 1] - row[i]
#             if diff > 0:
#                 inc += diff
#             else:
#                 dec -= diff          # keep positive
#         score += max(inc, dec)

#     # Columns
#     for col in log_b.T:
#         inc = dec = 0.0
#         for i in range(3):
#             diff = col[i + 1] - col[i]
#             if diff > 0:
#                 inc += diff
#             else:
#                 dec -= diff
#         score += max(inc, dec)

#     return score


# SNAKE_MATRIX_BASE = np.array([
#     [65536, 32768, 16384, 8192],
#     [512,   1024,  2048,  4096],
#     [256,   128,     64,    32],
#     [2,       4,      8,    16],
# ], dtype=np.float64)

SNAKE_MATRIX_BASE = np.array([
    [65536, 16384, 4096, 1024],
    [16384, 4096, 1024, 256],
    [4096, 1024, 256, 64],
    [1024, 256, 64, 16],
], dtype=np.float64)

SNAKE_MATRICES = [np.rot90(SNAKE_MATRIX_BASE, k) for k in range(4)]


def monotonicity(board: np.ndarray) -> float:
    """
    Strictly directional monotonicity — no rotation, fixed direction.

    Rewards decrease left→right and top→bottom.
    Penalises the reverse (increases add negative contribution).

    Why no rotation?
    ----------------
    Trying all 4 rotations and taking the max allows BOTH
        128  64  32  16   (good – decreasing L→R)
        16   32  64  128  (bad  – increasing L→R)
    to achieve the same score, because the bad board simply "picks" the
    rotation where it looks decreasing.  The fix is to commit to ONE
    direction.  The snake_score() heuristic handles corner selection via
    its own 4-rotation max, so monotonicity only needs to be a consistent
    directional signal.

    Result
    ------
        128 64 32 16  →  row contribution = +3  (correct, rewarded)
        16 32 64 128  →  row contribution = -3  (incorrect, penalised)
    """
    log_b = _log2_board(board)
    score = 0.0

    # Rows: reward decrease left → right
    for row in log_b:
        for i in range(3):
            score += row[i] - row[i + 1]   # + if decreasing, − if increasing

    # Columns: reward decrease top → bottom
    for col in log_b.T:
        for i in range(3):
            score += col[i] - col[i + 1]

    return score


def snake_score(
    board: np.ndarray,
    weight_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Dot-product of the log2 board with a positional snake weight matrix.

    The best of all 4 rotations is returned so the agent can commit to
    whichever corner the max tile currently occupies.

    Parameters
    ----------
    board :          (4, 4) int32 tile values.
    weight_matrix :  Custom weight matrix (4×4 float64).  If None the
                     default SNAKE_MATRIX_BASE is used.

    Why it works
    ------------
    Each cell is pre-assigned a desirability score.  Multiplying by log2
    of the tile value means large tiles in high-weight positions (near the
    chosen corner) contribute the most.  The snake layout ensures the
    path from high to low weight is a legal merge sequence.
    """
    log_b = _log2_board(board)

    if weight_matrix is not None:
        matrices = [np.rot90(weight_matrix, k) for k in range(4)]
    else:
        matrices = SNAKE_MATRICES

    return float(max(np.sum(log_b * m) for m in matrices))




def smoothness(board: np.ndarray) -> float:
    """
    Penalise large differences between adjacent tiles (log2 space).
    Returns a non-positive number (0 = perfectly smooth).
    """
    log_b = _log2_board(board)
    penalty = 0.0

    for r in range(4):
        for c in range(4):
            if log_b[r, c] == 0:
                continue
            # Right neighbour
            if c + 1 < 4 and log_b[r, c + 1] != 0:
                penalty -= abs(log_b[r, c] - log_b[r, c + 1])
            # Down neighbour
            if r + 1 < 4 and log_b[r + 1, c] != 0:
                penalty -= abs(log_b[r, c] - log_b[r + 1, c])

    return penalty


def free_tiles(board: np.ndarray) -> float:
    """
    Number of empty cells.  Logarithmically scaled so the bonus doesn't
    overwhelm everything when the board is nearly empty.
    """
    n_free = int(np.sum(board == 0))
    return math.log(n_free + 1)


def max_tile(board: np.ndarray) -> float:
    """log2 of the maximum tile value."""
    m = int(board.max())
    return math.log2(m) if m > 0 else 0.0


def corner_bonus(board: np.ndarray) -> float:
    """
    Reward having the maximum tile in one of the four corners.
    The bonus is proportional to the tile's log2 value so larger tiles
    produce a proportionally larger incentive.
    """
    m = int(board.max())
    if m == 0:
        return 0.0
    corners = (board[0, 0], board[0, 3], board[3, 0], board[3, 3])
    if m in corners:
        return math.log2(m)
    return 0.0


def merge_potential(board: np.ndarray) -> float:
    """
    Count the number of pairs of adjacent equal tiles (potential merges).
    Rewards boards that set up many simultaneous merges.
    """
    count = 0
    for r in range(4):
        for c in range(3):
            if board[r, c] != 0 and board[r, c] == board[r, c + 1]:
                count += 1
    for r in range(3):
        for c in range(4):
            if board[r, c] != 0 and board[r, c] == board[r + 1, c]:
                count += 1
    return float(count)


# ---------------------------------------------------------------------------
# Weighted combination
# ---------------------------------------------------------------------------

# DEFAULT_WEIGHTS = dict(
#     w_mono    = 5.0,
#     w_smooth  = 0.1,
#     w_free    = 2.7,
#     w_max     = 1.0,
#     w_corner  = 2.0,
#     w_merge   = 0.5,
# )


# def evaluate(
#     board: np.ndarray,
#     w_mono:   float = DEFAULT_WEIGHTS["w_mono"],
#     w_smooth: float = DEFAULT_WEIGHTS["w_smooth"],
#     w_free:   float = DEFAULT_WEIGHTS["w_free"],
#     w_max:    float = DEFAULT_WEIGHTS["w_max"],
#     w_corner: float = DEFAULT_WEIGHTS["w_corner"],
#     w_merge:  float = DEFAULT_WEIGHTS["w_merge"],
# ) -> float:
#     """
#     Combined weighted heuristic evaluation.

#     Note: the raw game score is intentionally *not* included because
#     delayed merges can produce better long-term positions despite a
#     lower immediate score.

#     Returns
#     -------
#     float  – Higher is better.
#     """
#     return (
#         w_mono   * monotonicity(board)
#       + w_smooth * smoothness(board)
#       + w_free   * free_tiles(board)
#       + w_max    * max_tile(board)
#       + w_corner * corner_bonus(board)
#       + w_merge  * merge_potential(board)
#     )


DEFAULT_WEIGHTS = dict(
    w_mono   = 0.5,   # reduced – snake already encodes directional preference
    w_smooth = 0.1,
    w_free   = 2.7,
    w_max    = 1.0,
    w_corner = 0.5,   # reduced – snake already rewards the corner cell heavily
    w_merge  = 0.5,
    w_snake  = 3.0,   # primary structural heuristic
)


# ---------------------------------------------------------------------------
# Combined evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    board: np.ndarray,
    w_mono:        float               = DEFAULT_WEIGHTS["w_mono"],
    w_smooth:      float               = DEFAULT_WEIGHTS["w_smooth"],
    w_free:        float               = DEFAULT_WEIGHTS["w_free"],
    w_max:         float               = DEFAULT_WEIGHTS["w_max"],
    w_corner:      float               = DEFAULT_WEIGHTS["w_corner"],
    w_merge:       float               = DEFAULT_WEIGHTS["w_merge"],
    w_snake:       float               = DEFAULT_WEIGHTS["w_snake"],
    snake_matrix:  Optional[np.ndarray] = None,
) -> float:
    """
    Weighted combination of all heuristics.

    Raw game score is intentionally excluded – delayed merges can build
    better long-term structure even at a lower immediate score.

    Parameters
    ----------
    snake_matrix : Optional custom 4×4 weight matrix for snake_score().
                   Pass None to use the built-in SNAKE_MATRIX_BASE.
    """
    return (
        w_mono   * monotonicity(board)
      + w_smooth * smoothness(board)
      + w_free   * free_tiles(board)
      + w_max    * max_tile(board)
      + w_corner * corner_bonus(board)
      + w_merge  * merge_potential(board)
      + w_snake  * snake_score(board, snake_matrix)
    )
