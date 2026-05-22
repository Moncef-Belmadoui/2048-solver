"""
dqn_agent.py
------------
Double Dueling DQN agent with Prioritized Experience Replay.

Design principles
-----------------
* Reward function is *external* – injected via the environment.
* The agent knows nothing about game rules; it only sees observations.
* Separate online and target networks (hard update every N steps).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.network import DuelingDQN
from agents.dqn.replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    Double Dueling DQN with PER.

    Parameters
    ----------
    obs_shape :       Observation shape, e.g. (1, 4, 4).
    n_actions :       Number of discrete actions.
    lr :              Learning rate.
    gamma :           Discount factor.
    batch_size :      Mini-batch size.
    buffer_capacity : Replay buffer size.
    eps_start :       Initial epsilon.
    eps_end :         Final epsilon.
    eps_decay_steps : Steps to decay epsilon from start to end.
    target_update :   Hard update interval (steps).
    grad_clip :       Max gradient norm.
    alpha, beta_* :   PER hyper-parameters.
    device :          'cpu' | 'cuda' | 'auto'.
    """

    def __init__(
        self,
        obs_shape:         Tuple[int, ...]  = (1, 4, 4),
        n_actions:         int              = 4,
        lr:                float            = 1e-4,
        gamma:             float            = 0.99,
        batch_size:        int              = 64,
        buffer_capacity:   int              = 100_000,
        eps_start:         float            = 1.0,
        eps_end:           float            = 0.05,
        eps_decay_steps:   int              = 200_000,
        target_update:     int              = 1_000,
        grad_clip:         float            = 10.0,
        alpha:             float            = 0.6,
        beta_start:        float            = 0.4,
        beta_end:          float            = 1.0,
        per_anneal_steps:  int              = 500_000,
        device:            str              = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_actions       = n_actions
        self.gamma           = gamma
        self.batch_size      = batch_size
        self.target_update   = target_update
        self.grad_clip       = grad_clip

        # Epsilon-greedy
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = (eps_start - eps_end) / max(eps_decay_steps, 1)

        # Networks
        in_ch = obs_shape[0]
        self.online_net = DuelingDQN(n_actions=n_actions, in_channels=in_ch).to(self.device)
        self.target_net = DuelingDQN(n_actions=n_actions, in_channels=in_ch).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # PER buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity      = buffer_capacity,
            alpha         = alpha,
            beta_start    = beta_start,
            beta_end      = beta_end,
            anneal_steps  = per_anneal_steps,
        )

        # Counters
        self.steps_done   = 0
        self.updates_done = 0

        # Running stats
        self.last_loss  : float = 0.0
        self.last_q_mean: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        obs :       (1, 4, 4) float32 numpy array.
        eval_mode : If True, always greedy (ε=0).
        """
        eps = 0.0 if eval_mode else self.eps

        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)

        state_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return self.online_net.get_action(state_t)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Store transition and optionally learn."""
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Periodic target update
        if self.steps_done % self.target_update == 0:
            self._update_target()

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the loss value, or None if the buffer isn't ready.
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        (states, actions, rewards, next_states,
         dones, weights, indices) = self.buffer.sample(self.batch_size, self.device)

        # --- Double DQN target ---
        with torch.no_grad():
            # Select actions with online network
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate with target network
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # --- Online Q values ---
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- TD errors for PER ---
        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # --- Weighted loss ---
        loss_per_sample = F.huber_loss(current_q, target_q, reduction="none")
        loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates_done += 1
        self.last_loss   = float(loss.item())
        self.last_q_mean = float(current_q.mean().item())

        return self.last_loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_net"   : self.online_net.state_dict(),
            "target_net"   : self.target_net.state_dict(),
            "optimizer"    : self.optimizer.state_dict(),
            "steps_done"   : self.steps_done,
            "updates_done" : self.updates_done,
            "eps"          : self.eps,
        }, path)

    def load(self, path: str, eval_mode: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done   = ckpt.get("steps_done",   0)
        self.updates_done = ckpt.get("updates_done", 0)
        self.eps          = ckpt.get("eps", self.eps_end)
        if eval_mode:
            self.eps = 0.0
            self.online_net.eval()

    def set_eval(self):
        self.eps = 0.0
        self.online_net.eval()


# Import F here to keep the top tidy
import torch.nn.functional as F
