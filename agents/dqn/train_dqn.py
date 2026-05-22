"""
train_dqn.py
------------
Full DQN training pipeline for 2048.

Usage
-----
    python -m agents.dqn.train_dqn
    python -m agents.dqn.train_dqn --episodes 5000 --render-every 500
    python -m agents.dqn.train_dqn --load saved_models/dqn_best.pt --eval
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

# Project imports
from game.environment import Game2048Env, OBS_LOG2
from game.rewards import reward_shaped_v1, make_shaped_reward
from agents.dqn.dqn_agent import DQNAgent
from utils.logger import EpisodeLogger
from utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    episodes          = 10_000,
    lr                = 1e-4,
    gamma             = 0.99,
    batch_size        = 64,
    buffer_capacity   = 100_000,
    eps_start         = 1.0,
    eps_end           = 0.05,
    eps_decay_steps   = 300_000,
    target_update     = 1_000,
    grad_clip         = 10.0,
    alpha             = 0.6,
    beta_start        = 0.4,
    per_anneal_steps  = 500_000,
    learn_every       = 4,        # gradient step every N env steps
    warmup_steps      = 5_000,    # fill buffer before learning
    render_every      = 0,        # 0 = never render
    save_every        = 500,      # save checkpoint every N episodes
    eval_every        = 200,      # run evaluation every N episodes
    eval_episodes     = 20,
    seed              = 42,
    device            = "auto",
    save_dir          = "saved_models",
    log_dir           = "logs",
    run_name          = "dqn_run",
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace):
    set_global_seed(cfg.seed)

    # -- Environment --
    reward_fn = reward_shaped_v1
    env = Game2048Env(
        reward_fn         = reward_fn,
        obs_mode          = OBS_LOG2,
        penalize_invalid  = True,
        max_invalid_moves = 100,
        seed              = cfg.seed,
    )

    # -- Agent --
    agent = DQNAgent(
        obs_shape        = env.observation_shape,
        n_actions        = 4,
        lr               = cfg.lr,
        gamma            = cfg.gamma,
        batch_size       = cfg.batch_size,
        buffer_capacity  = cfg.buffer_capacity,
        eps_start        = cfg.eps_start,
        eps_end          = cfg.eps_end,
        eps_decay_steps  = cfg.eps_decay_steps,
        target_update    = cfg.target_update,
        grad_clip        = cfg.grad_clip,
        alpha            = cfg.alpha,
        beta_start       = cfg.beta_start,
        per_anneal_steps = cfg.per_anneal_steps,
        device           = cfg.device,
    )

    if cfg.load:
        print(f"Loading checkpoint: {cfg.load}")
        agent.load(cfg.load, eval_mode=cfg.eval)

    if cfg.eval:
        _run_eval(agent, env, cfg.eval_episodes)
        return

    # -- Logger --
    logger = EpisodeLogger(
        log_dir  = cfg.log_dir,
        run_name = cfg.run_name,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)

    best_avg_score = -float("inf")
    global_steps   = 0

    # -- tqdm (optional) --
    try:
        from tqdm import tqdm
        episode_iter = tqdm(range(1, cfg.episodes + 1), desc="Training", unit="ep")
    except ImportError:
        episode_iter = range(1, cfg.episodes + 1)

    # -----------------------------------------------------------------------
    for episode in episode_iter:
        obs, info = env.reset()
        ep_reward       = 0.0
        ep_loss_sum     = 0.0
        ep_loss_count   = 0
        ep_q_sum        = 0.0
        ep_invalid      = 0
        ep_actions      = [0, 0, 0, 0]
        t0              = time.time()
        done            = False
        truncated       = False

        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.push(obs, action, reward, next_obs, done or truncated)
            global_steps += 1

            # Learn
            if (
                global_steps >= cfg.warmup_steps
                and global_steps % cfg.learn_every == 0
            ):
                loss = agent.learn()
                if loss is not None:
                    ep_loss_sum   += loss
                    ep_loss_count += 1
                    ep_q_sum      += agent.last_q_mean

            ep_reward  += reward
            ep_invalid += int(not info.get("valid_moves") or action not in info.get("valid_moves", [0,1,2,3]))
            ep_actions[action] += 1
            obs = next_obs

        # -- Episode stats --
        ep_time  = time.time() - t0
        avg_loss = ep_loss_sum  / max(ep_loss_count, 1)
        avg_q    = ep_q_sum     / max(ep_loss_count, 1)

        stats = {
            "episode"         : episode,
            "reward"          : ep_reward,
            "score"           : info["score"],
            "max_tile"        : info["max_tile"],
            "steps"           : info["moves"],
            "epsilon"         : agent.eps,
            "loss"            : avg_loss,
            "avg_q"           : avg_q,
            "invalid_moves"   : info["invalid_moves"],
            "won"             : int(info["won"]),
            "action_up"       : ep_actions[0],
            "action_down"     : ep_actions[1],
            "action_left"     : ep_actions[2],
            "action_right"    : ep_actions[3],
            "merges"          : info["merges"],
            "empty_cells"     : info["empty_cells"],
            "time_s"          : ep_time,
            "global_steps"    : global_steps,
        }
        logger.log(stats)

        # -- tqdm postfix --
        try:
            episode_iter.set_postfix(
                score=f"{info['score']}",
                max=f"{info['max_tile']}",
                eps=f"{agent.eps:.3f}",
                loss=f"{avg_loss:.4f}",
            )
        except AttributeError:
            if episode % 100 == 0:
                print(
                    f"Ep {episode:5d} | score={info['score']:6d} | "
                    f"max={info['max_tile']:4d} | eps={agent.eps:.3f} | "
                    f"loss={avg_loss:.4f}"
                )

        # -- Checkpoint --
        if episode % cfg.save_every == 0:
            path = os.path.join(cfg.save_dir, f"{cfg.run_name}_ep{episode}.pt")
            agent.save(path)

        # -- Periodic evaluation --
        if episode % cfg.eval_every == 0:
            avg_score = _run_eval(agent, env, cfg.eval_episodes, silent=True)
            print(f"\n[Eval ep {episode}] avg_score={avg_score:.1f}")
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt"))

        # -- Render --
        if cfg.render_every > 0 and episode % cfg.render_every == 0:
            _render_episode(agent, seed=cfg.seed + episode)

    # -- Final save --
    agent.save(os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pt"))
    logger.save()
    print(f"\nTraining complete. Best eval score: {best_avg_score:.1f}")
    print(f"Logs saved to: {cfg.log_dir}/{cfg.run_name}")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_eval(
    agent: DQNAgent,
    env:   Game2048Env,
    n_episodes: int,
    silent: bool = False,
) -> float:
    agent.online_net.eval()
    scores  = []
    tiles   = []
    wins    = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = agent.act(obs, eval_mode=True)
            obs, _, done, truncated, info = env.step(action)
        scores.append(info["score"])
        tiles.append(info["max_tile"])
        wins += int(info["won"])

    agent.online_net.train()
    avg_score = float(np.mean(scores))
    avg_max   = float(np.mean(tiles))
    win_rate  = wins / n_episodes

    if not silent:
        print(
            f"Eval ({n_episodes} eps): avg_score={avg_score:.1f} "
            f"avg_max_tile={avg_max:.1f} win_rate={win_rate:.2%}"
        )
    return avg_score


# ---------------------------------------------------------------------------
# Render one episode in the terminal (no pygame required)
# ---------------------------------------------------------------------------

def _render_episode(agent: DQNAgent, seed: Optional[int] = None):
    env = Game2048Env(reward_fn=reward_shaped_v1, seed=seed)
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = agent.act(obs, eval_mode=True)
        obs, _, done, truncated, info = env.step(action)
    print(env.render())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN agent on 2048")
    d = DEFAULTS
    p.add_argument("--episodes",         type=int,   default=d["episodes"])
    p.add_argument("--lr",               type=float, default=d["lr"])
    p.add_argument("--gamma",            type=float, default=d["gamma"])
    p.add_argument("--batch-size",       type=int,   default=d["batch_size"])
    p.add_argument("--buffer-capacity",  type=int,   default=d["buffer_capacity"])
    p.add_argument("--eps-start",        type=float, default=d["eps_start"])
    p.add_argument("--eps-end",          type=float, default=d["eps_end"])
    p.add_argument("--eps-decay-steps",  type=int,   default=d["eps_decay_steps"])
    p.add_argument("--target-update",    type=int,   default=d["target_update"])
    p.add_argument("--grad-clip",        type=float, default=d["grad_clip"])
    p.add_argument("--alpha",            type=float, default=d["alpha"])
    p.add_argument("--beta-start",       type=float, default=d["beta_start"])
    p.add_argument("--per-anneal-steps", type=int,   default=d["per_anneal_steps"])
    p.add_argument("--learn-every",      type=int,   default=d["learn_every"])
    p.add_argument("--warmup-steps",     type=int,   default=d["warmup_steps"])
    p.add_argument("--render-every",     type=int,   default=d["render_every"])
    p.add_argument("--save-every",       type=int,   default=d["save_every"])
    p.add_argument("--eval-every",       type=int,   default=d["eval_every"])
    p.add_argument("--eval-episodes",    type=int,   default=d["eval_episodes"])
    p.add_argument("--seed",             type=int,   default=d["seed"])
    p.add_argument("--device",           type=str,   default=d["device"])
    p.add_argument("--save-dir",         type=str,   default=d["save_dir"])
    p.add_argument("--log-dir",          type=str,   default=d["log_dir"])
    p.add_argument("--run-name",         type=str,   default=d["run_name"])
    p.add_argument("--load",             type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval",             action="store_true",
                   help="Evaluation only (requires --load)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
