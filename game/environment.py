"""
environment.py
--------------
Gymnasium-style wrapper around Game2048.

Usage
-----
    from game.environment import Game2048Env
    from game.rewards import reward_shaped_v1

    env = Game2048Env(reward_fn=reward_shaped_v1)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from game.game2048 import Game2048
from game.rewards import reward_score_delta, RewardFn

# Observation modes
OBS_RAW   = "raw"    # actual tile values, shape (4,4)
OBS_LOG2  = "log2"   # log2 encoded (0 = empty, 1 = tile-2, …), shape (4,4)
OBS_1HOT  = "onehot" # one-hot encoded, shape (16, 4, 4)

MAX_LOG2_TILE = 16   # 2^16 = 65536 (well beyond 2048)


class Game2048Env:
    """
    Gym-compatible environment for 2048.

    Parameters
    ----------
    reward_fn :
        Callable with signature
        ``reward_fn(board_before, board_after, score_before, score_after,
                    moved, done, invalid) -> float``
        Defaults to reward_score_delta.
    obs_mode :
        One of ``'raw'``, ``'log2'``, ``'onehot'``.
    penalize_invalid :
        If True, invalid moves do not advance the game and return a penalty.
    max_invalid_moves :
        Episode is truncated after this many consecutive invalid moves.
    seed :
        RNG seed for the underlying game.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        reward_fn: RewardFn = reward_score_delta,
        obs_mode: str = OBS_LOG2,
        penalize_invalid: bool = True,
        max_invalid_moves: int = 100,
        seed: Optional[int] = None,
    ):
        assert obs_mode in (OBS_RAW, OBS_LOG2, OBS_1HOT), \
            f"obs_mode must be one of {OBS_RAW!r}, {OBS_LOG2!r}, {OBS_1HOT!r}"

        self.reward_fn = reward_fn
        self.obs_mode = obs_mode
        self.penalize_invalid = penalize_invalid
        self.max_invalid_moves = max_invalid_moves
        self.seed_val = seed

        self.game: Game2048 = Game2048(seed=seed)

        # Episode stats
        self.episode_invalid_moves: int = 0
        self.episode_steps: int = 0
        self.episode_merges: int = 0

        # Spaces (informational – mirrors gym.spaces API surface)
        self.action_space_n: int = 4
        self.observation_shape: Tuple[int, ...] = self._obs_shape()

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed_val = seed
        self.game = Game2048(seed=self.seed_val)
        self.episode_invalid_moves = 0
        self.episode_steps = 0
        self.episode_merges = 0
        return self._get_obs(), self._info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Parameters
        ----------
        action : int  (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        board_before = self.game.get_state()
        score_before = self.game.score

        moved, gained = self.game.move(action)
        board_after = self.game.get_state()
        score_after = self.game.score

        invalid = not moved
        done = self.game.is_over()

        if invalid:
            self.episode_invalid_moves += 1
        else:
            self.episode_steps += 1
            if gained > 0:
                # each merge produces score, count merges roughly
                self.episode_merges += int(math.log2(gained + 1))

        truncated = (
            self.episode_invalid_moves >= self.max_invalid_moves
        )

        reward = self.reward_fn(
            board_before, board_after,
            score_before, score_after,
            moved, done, invalid,
        )

        return self._get_obs(), reward, done, truncated, self._info(gained)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode == "ansi":
            return str(self.game)
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_shape(self) -> Tuple[int, ...]:
        if self.obs_mode == OBS_RAW:
            return (4, 4)
        if self.obs_mode == OBS_LOG2:
            return (1, 4, 4)
        # onehot
        return (MAX_LOG2_TILE, 4, 4)

    def _get_obs(self) -> np.ndarray:
        board = self.game.get_state()
        if self.obs_mode == OBS_RAW:
            return board.astype(np.float32)
        if self.obs_mode == OBS_LOG2:
            return self._log2_encode(board)
        return self._onehot_encode(board)

    @staticmethod
    def _log2_encode(board: np.ndarray) -> np.ndarray:
        """Returns shape (1, 4, 4) float32 with values 0, 1, 2, …"""
        encoded = np.zeros((4, 4), dtype=np.float32)
        mask = board > 0
        encoded[mask] = np.log2(board[mask])
        return encoded[np.newaxis]  # (1, 4, 4)

    @staticmethod
    def _onehot_encode(board: np.ndarray) -> np.ndarray:
        """Returns shape (MAX_LOG2_TILE, 4, 4) float32."""
        encoded = np.zeros((MAX_LOG2_TILE, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                if board[r, c] > 0:
                    idx = int(math.log2(board[r, c]))
                    if idx < MAX_LOG2_TILE:
                        encoded[idx, r, c] = 1.0
        return encoded

    @staticmethod
    def normalize_obs(obs: np.ndarray) -> np.ndarray:
        """Normalise log2 observation into [0, 1]."""
        return obs / MAX_LOG2_TILE

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _info(self, gained: int = 0) -> Dict[str, Any]:
        return {
            "score": self.game.score,
            "max_tile": self.game.max_tile(),
            "empty_cells": self.game.empty_cells(),
            "moves": self.game.moves,
            "invalid_moves": self.episode_invalid_moves,
            "merges": self.episode_merges,
            "won": self.game.has_won(),
            "valid_moves": self.game.get_valid_moves(),
            "score_gained": gained,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def board(self) -> np.ndarray:
        return self.game.get_state()

    @property
    def score(self) -> int:
        return self.game.score