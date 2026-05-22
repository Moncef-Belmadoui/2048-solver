"""
replay_buffer.py
----------------
Proportional Prioritized Experience Replay (PER).

Reference: Schaul et al., 2015 – "Prioritized Experience Replay"

Uses a 1-indexed binary sum-tree for O(log N) priority sampling.
Works correctly for any capacity (not just powers of 2).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1-indexed Sum-tree
# ---------------------------------------------------------------------------

class SumTree:
    """
    1-indexed binary sum-tree.

    Layout (capacity=4):
        index:   1  2  3  4  5  6  7  (8 = 2*cap+1 slots allocated)
                 [root][internal][   leaves    ]
        Leaves:  indices  capacity  ..  2*capacity-1
        pointer 0..capacity-1  ->  leaf = pointer + capacity

    This avoids off-by-one issues with non-power-of-2 capacities.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity + 1, dtype=np.float64)

    # ── internal ─────────────────────────────────────────────────────

    def _propagate(self, leaf_idx: int):
        idx = leaf_idx
        while idx > 1:
            parent = idx >> 1
            self._tree[parent] = self._tree[2 * parent] + self._tree[2 * parent + 1]
            idx = parent

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left  = 2 * idx
            right = left + 1
            if left >= 2 * self.capacity:   # reached leaf level
                return idx
            if s <= self._tree[left]:
                idx = left
            else:
                s  -= self._tree[left]
                idx = right

    # ── public ───────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def add(self, pointer: int, priority: float):
        leaf_idx = pointer + self.capacity
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx)

    def sample(self, s: float) -> Tuple[int, float]:
        leaf_idx = self._retrieve(1, max(s, 1e-8))
        data_ptr = leaf_idx - self.capacity
        data_ptr = int(np.clip(data_ptr, 0, self.capacity - 1))
        return data_ptr, float(self._tree[leaf_idx])


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Proportional PER buffer.

    Parameters
    ----------
    capacity :    Maximum number of transitions stored.
    alpha :       Priority exponent (0 = uniform, 1 = full PER).
    beta_start :  Initial importance-sampling exponent.
    beta_end :    Final IS exponent (annealed over *anneal_steps* steps).
    anneal_steps: Steps over which beta is linearly annealed.
    eps :         Small constant added to priorities to avoid zero.
    """

    def __init__(
        self,
        capacity:     int   = 100_000,
        alpha:        float = 0.6,
        beta_start:   float = 0.4,
        beta_end:     float = 1.0,
        anneal_steps: int   = 500_000,
        eps:          float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha    = alpha
        self.beta     = beta_start
        self.beta_end = beta_end
        self.beta_inc = (beta_end - beta_start) / max(anneal_steps, 1)
        self.eps      = eps

        self._tree         = SumTree(capacity)
        self._size         = 0
        self._ptr          = 0
        self._max_priority = 1.0

        self._states      = np.zeros((capacity, 1, 4, 4), dtype=np.float32)
        self._next_states = np.zeros((capacity, 1, 4, 4), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Add a transition with the current maximum priority."""
        idx = self._ptr
        self._states[idx]      = state
        self._next_states[idx] = next_state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._dones[idx]       = float(done)

        self._tree.add(idx, self._max_priority ** self.alpha)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample *batch_size* transitions proportional to priority.

        Returns
        -------
        states, actions, rewards, next_states, dones, IS-weights, indices
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        total   = self._tree.total
        segment = total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            s = float(np.clip(s, 1e-8, total - 1e-8))

            idx, p   = self._tree.sample(s)
            indices[i] = idx
            prob       = p / total if total > 0 else 1.0 / self._size
            weights[i] = (self._size * prob) ** (-self.beta)

        max_w = weights.max()
        if max_w > 0:
            weights /= max_w

        self.beta = min(self.beta_end, self.beta + self.beta_inc)

        def t(arr):
            return torch.from_numpy(arr).to(device)

        return (
            t(self._states[indices]),
            t(self._actions[indices]),
            t(self._rewards[indices]),
            t(self._next_states[indices]),
            t(self._dones[indices]),
            t(weights),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Recompute priorities from fresh TD errors."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.add(int(idx), float(p))
            self._max_priority = max(self._max_priority, float(p))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size