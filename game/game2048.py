"""
game2048.py
-----------
Pure game logic for 2048.  No RL, no rendering, no reward shaping.
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

import numpy as np

# Direction constants
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

DIRECTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


class Game2048:
    """
    Self-contained 2048 game engine.

    Board is stored as a (4, 4) numpy int32 array where each cell holds
    the actual tile value (0 = empty, 2, 4, 8, …).
    """

    SIZE = 4

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.board: np.ndarray = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self.score: int = 0
        self.moves: int = 0
        self._game_over: bool = False
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the board and spawn two tiles. Returns the new board."""
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self._game_over = False
        self.spawn_tile()
        self.spawn_tile()
        return self.get_state()

    def move(self, direction: int) -> Tuple[bool, int]:
        """
        Execute a move in *direction*.

        Returns
        -------
        moved : bool   – Whether the board changed.
        gained : int   – Score points gained this move.
        """
        if self._game_over:
            return False, 0

        old_board = self.board.copy()
        gained = 0

        if direction == UP:
            self.board, gained = self._move_up(self.board)
        elif direction == DOWN:
            self.board, gained = self._move_down(self.board)
        elif direction == LEFT:
            self.board, gained = self._move_left(self.board)
        elif direction == RIGHT:
            self.board, gained = self._move_right(self.board)
        else:
            raise ValueError(f"Unknown direction {direction}")

        moved = not np.array_equal(self.board, old_board)

        if moved:
            self.score += gained
            self.moves += 1
            self.spawn_tile()
            if not self.can_move():
                self._game_over = True

        return moved, gained

    def spawn_tile(self) -> bool:
        """
        Place a new tile (90 % → 2, 10 % → 4) in a random empty cell.
        Returns False if there is no empty cell.
        """
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return False
        row, col = self._rng.choice(empty)
        self.board[row, col] = 4 if self._rng.random() < 0.1 else 2
        return True

    def can_move(self) -> bool:
        """Return True if at least one valid move exists."""
        return len(self.get_valid_moves()) > 0

    def get_valid_moves(self) -> List[int]:
        """Return list of directions that would actually change the board."""
        valid = []
        for direction in (UP, DOWN, LEFT, RIGHT):
            tmp = self.board.copy()
            new_board, _ = self._apply_direction(tmp, direction)
            if not np.array_equal(new_board, tmp):
                valid.append(direction)
        return valid

    def is_over(self) -> bool:
        return self._game_over

    def has_won(self) -> bool:
        return int(self.board.max()) >= 2048

    def get_state(self) -> np.ndarray:
        """Return a copy of the current board."""
        return self.board.copy()

    def clone(self) -> "Game2048":
        """Return a deep copy of this game (cheap – no rng state needed)."""
        new = Game2048.__new__(Game2048)
        new.board = self.board.copy()
        new.score = self.score
        new.moves = self.moves
        new._game_over = self._game_over
        new._rng = random.Random()
        return new

    def max_tile(self) -> int:
        return int(self.board.max())

    def empty_cells(self) -> int:
        return int(np.sum(self.board == 0))

    def __repr__(self) -> str:
        rows = []
        for row in self.board:
            rows.append("\t".join(str(v) if v else "." for v in row))
        return "\n".join(rows) + f"\nScore: {self.score}"

    # ------------------------------------------------------------------
    # Internal move mechanics
    # ------------------------------------------------------------------

    @staticmethod
    def _slide_and_merge(row: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Slide a 1-D row LEFT: compact non-zeros, merge adjacent equals.
        Returns new row and score gained.
        """
        gained = 0
        tiles = row[row != 0].tolist()          # compact
        merged: List[int] = []
        skip = False
        for i, t in enumerate(tiles):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                val = t * 2
                merged.append(val)
                gained += val
                skip = True
            else:
                merged.append(t)
        # Pad with zeros
        merged += [0] * (len(row) - len(merged))
        return np.array(merged, dtype=np.int32), gained

    @classmethod
    def _move_left(cls, board: np.ndarray) -> Tuple[np.ndarray, int]:
        new_board = np.zeros_like(board)
        total_gain = 0
        for r in range(cls.SIZE):
            new_board[r], gain = cls._slide_and_merge(board[r])
            total_gain += gain
        return new_board, total_gain

    @classmethod
    def _move_right(cls, board: np.ndarray) -> Tuple[np.ndarray, int]:
        flipped = np.fliplr(board)
        new_board, gain = cls._move_left(flipped)
        return np.fliplr(new_board), gain

    @classmethod
    def _move_up(cls, board: np.ndarray) -> Tuple[np.ndarray, int]:
        transposed = board.T
        new_board, gain = cls._move_left(transposed)
        return new_board.T, gain

    @classmethod
    def _move_down(cls, board: np.ndarray) -> Tuple[np.ndarray, int]:
        transposed = board.T
        new_board, gain = cls._move_right(transposed)
        return new_board.T, gain

    @classmethod
    def _apply_direction(cls, board: np.ndarray, direction: int):
        if direction == UP:
            return cls._move_up(board)
        elif direction == DOWN:
            return cls._move_down(board)
        elif direction == LEFT:
            return cls._move_left(board)
        elif direction == RIGHT:
            return cls._move_right(board)
        raise ValueError(f"Unknown direction {direction}")

