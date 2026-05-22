"""
expectiminimax_agent.py
-----------------------
Expectiminimax search agent for 2048.

Tree structure
--------------
  MAX nodes    – the player picks the best move.
  CHANCE nodes – the environment places a tile (2 w.p. 0.9, 4 w.p. 0.1)
                 uniformly in each empty cell.

Optimisations
-------------
* Alpha-beta pruning on MAX nodes.
* Move ordering: UP → LEFT → RIGHT → DOWN (empirically good for 2048).
* Probability-threshold pruning on CHANCE nodes.
* All board operations via numpy (no Python object overhead).

Changes vs original
-------------------
* _eval() now passes w_snake and snake_matrix to heuristics.evaluate().
* ExpectiminimaxAgent accepts snake_matrix kwarg for CMA-ES tuning.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from game.game2048 import Game2048, UP, DOWN, LEFT, RIGHT
from agents.expectiminimax.heuristics import evaluate, DEFAULT_WEIGHTS, SNAKE_MATRIX_BASE

_MOVE_ORDER = [UP, LEFT, RIGHT, DOWN]
_SPAWN      = [(2, 0.9), (4, 0.1)]


class ExpectiminimaxAgent:
    """
    Parameters
    ----------
    depth :          Search depth in player moves.
    prob_threshold : Chance branches below this cumulative probability
                     are pruned.
    weights :        Dict of heuristic weights (see heuristics.DEFAULT_WEIGHTS).
    snake_matrix :   Optional (4, 4) float64 positional weight matrix for
                     snake_score().  Pass None to use the built-in default.
    """

    def __init__(
        self,
        depth:          int              = 3,
        prob_threshold: float            = 0.005,
        weights:        Optional[dict]   = None,
        snake_matrix:   Optional[np.ndarray] = None,
    ):
        self.depth          = depth
        self.prob_threshold = prob_threshold
        self.weights        = weights or dict(DEFAULT_WEIGHTS)
        self.snake_matrix   = snake_matrix   # None → use SNAKE_MATRIX_BASE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """
        Pick the best action for a log2-encoded obs (1, 4, 4).
        Decodes back to raw tile values internally.
        """
        board = np.where(obs[0] > 0, 2.0 ** obs[0], 0).astype(np.int32)
        return self._best_move(board)

    def act_game(self, game: Game2048) -> int:
        """Pick the best action directly from a Game2048 object (faster)."""
        return self._best_move(game.get_state())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _best_move(self, board: np.ndarray) -> int:
        best_action = UP
        best_value  = -math.inf

        for action in _MOVE_ORDER:
            new_board, _ = Game2048._apply_direction(board.copy(), action)
            if np.array_equal(new_board, board):
                continue
            value = self._chance_node(new_board, self.depth - 1, prob=1.0)
            if value > best_value:
                best_value  = value
                best_action = action

        return best_action

    def _max_node(
        self,
        board: np.ndarray,
        depth: int,
        prob:  float,
        alpha: float,
        beta:  float,
    ) -> float:
        if depth == 0 or self._is_terminal(board):
            return self._eval(board)

        value = -math.inf
        for action in _MOVE_ORDER:
            new_board, _ = Game2048._apply_direction(board.copy(), action)
            if np.array_equal(new_board, board):
                continue
            child = self._chance_node(new_board, depth - 1, prob)
            value = max(value, child)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value if value != -math.inf else self._eval(board)

    def _chance_node(
        self,
        board: np.ndarray,
        depth: int,
        prob:  float,
    ) -> float:
        if depth == 0 or self._is_terminal(board):
            return self._eval(board)

        empty = list(zip(*np.where(board == 0)))
        if not empty:
            return self._eval(board)

        total     = 0.0
        n_empty   = len(empty)
        cell_prob = 1.0 / n_empty

        for (r, c) in empty:
            for tile, tile_prob in _SPAWN:
                branch_prob = prob * cell_prob * tile_prob
                if branch_prob < self.prob_threshold:
                    continue
                board[r, c] = tile
                value = self._max_node(
                    board, depth, branch_prob,
                    alpha=-math.inf, beta=math.inf,
                )
                total   += cell_prob * tile_prob * value
                board[r, c] = 0

        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eval(self, board: np.ndarray) -> float:
        return evaluate(board, snake_matrix=self.snake_matrix, **self.weights)

    @staticmethod
    def _is_terminal(board: np.ndarray) -> bool:
        for action in (UP, DOWN, LEFT, RIGHT):
            nb, _ = Game2048._apply_direction(board.copy(), action)
            if not np.array_equal(nb, board):
                return False
        return True
