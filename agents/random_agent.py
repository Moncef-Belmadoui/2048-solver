"""
random_agent.py
---------------
A uniformly random baseline agent.

API
---
    agent.act(obs)  -> int   (random action from 0-3)
    agent.act_valid(game) -> int   (random valid action)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


class RandomAgent:
    """Sample uniformly from {0, 1, 2, 3}."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray) -> int:
        return self._rng.randint(0, 3)

    def act_valid(self, game) -> int:
        """Use the game object to only sample valid moves."""
        valid = game.get_valid_moves()
        if valid:
            return self._rng.choice(valid)
        return self._rng.randint(0, 3)