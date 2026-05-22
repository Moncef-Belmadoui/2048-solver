"""
evaluate.py  (agents/expectiminimax/)
--------------------------------------
Evaluate the Expectiminimax agent over N games and print / save statistics.

Usage
-----
    python -m agents.expectiminimax.evaluate
    python -m agents.expectiminimax.evaluate --games 100 --depth 4
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from game.game2048 import Game2048
from agents.expectiminimax.expectiminimax_agent import ExpectiminimaxAgent
from utils.logger import EpisodeLogger
from utils.seed import set_global_seed


def evaluate(
    n_games:        int   = 50,
    depth:          int   = 3,
    prob_threshold: float = 0.005,
    seed:           int   = 42,
    log_dir:        str   = "logs",
    run_name:       str   = "expectiminimax_eval",
    verbose:        bool  = True,
) -> dict:
    """
    Run the Expectiminimax agent for *n_games* complete episodes.

    Returns a dict of aggregate statistics.
    """
    set_global_seed(seed)

    agent  = ExpectiminimaxAgent(depth=depth, prob_threshold=prob_threshold)
    logger = EpisodeLogger(log_dir=log_dir, run_name=run_name)

    scores:    list = []
    max_tiles: list = []
    steps_:    list = []
    times_:    list = []
    tile_freq: Counter = Counter()

    for game_idx in range(1, n_games + 1):
        game = Game2048(seed=seed + game_idx)
        t0   = time.time()

        while not game.is_over():
            obs = _game_to_obs(game)
            action = agent.act(obs)
            moved, _ = game.move(action)
            # Force a random valid move if expectiminimax returned an invalid one
            if not moved:
                valid = game.get_valid_moves()
                if valid:
                    game.move(valid[0])

        elapsed = time.time() - t0
        scores.append(game.score)
        max_tiles.append(game.max_tile())
        steps_.append(game.moves)
        times_.append(elapsed)
        tile_freq[game.max_tile()] += 1

        episode_stats = {
            "episode":   game_idx,
            "score":     game.score,
            "max_tile":  game.max_tile(),
            "steps":     game.moves,
            "time_s":    elapsed,
            "won":       int(game.has_won()),
        }
        logger.log(episode_stats)

        if verbose and game_idx % max(1, n_games // 10) == 0:
            print(
                f"Game {game_idx:4d}/{n_games} | "
                f"score={game.score:6d} | max_tile={game.max_tile():4d} | "
                f"steps={game.moves:4d} | time={elapsed:.2f}s"
            )

    logger.save()

    results = {
        "n_games":        n_games,
        "avg_score":      float(np.mean(scores)),
        "std_score":      float(np.std(scores)),
        "max_score":      int(max(scores)),
        "avg_max_tile":   float(np.mean(max_tiles)),
        "win_rate":       sum(t >= 2048 for t in max_tiles) / n_games,
        "avg_steps":      float(np.mean(steps_)),
        "avg_time_s":     float(np.mean(times_)),
        "tile_frequency": dict(tile_freq),
    }

    if verbose:
        _print_results(results)

    return results


def _game_to_obs(game: Game2048) -> np.ndarray:
    """Convert raw board to log2 observation (1, 4, 4)."""
    import numpy as np, math
    board = game.get_state().astype(np.float32)
    enc   = np.zeros((4, 4), dtype=np.float32)
    mask  = board > 0
    enc[mask] = np.log2(board[mask])
    return enc[np.newaxis]


def _print_results(r: dict):
    print("\n" + "=" * 55)
    print("  Expectiminimax Evaluation Results")
    print("=" * 55)
    print(f"  Games played   : {r['n_games']}")
    print(f"  Avg score      : {r['avg_score']:,.1f}  (±{r['std_score']:,.1f})")
    print(f"  Max score      : {r['max_score']:,}")
    print(f"  Avg max tile   : {r['avg_max_tile']:.1f}")
    print(f"  Win rate (2048): {r['win_rate']:.1%}")
    print(f"  Avg steps      : {r['avg_steps']:.1f}")
    print(f"  Avg time/game  : {r['avg_time_s']:.2f}s")
    print("\n  Max-tile distribution:")
    for tile in sorted(r["tile_frequency"]):
        pct = r["tile_frequency"][tile] / r["n_games"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:5d}: {bar:<25s} {pct:5.1f}%")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Evaluate Expectiminimax agent")
    p.add_argument("--games",           type=int,   default=50)
    p.add_argument("--depth",           type=int,   default=3)
    p.add_argument("--prob-threshold",  type=float, default=0.005)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--log-dir",         type=str,   default="logs")
    p.add_argument("--run-name",        type=str,   default="expectiminimax_eval")
    p.add_argument("--quiet",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    evaluate(
        n_games        = args.games,
        depth          = args.depth,
        prob_threshold = args.prob_threshold,
        seed           = args.seed,
        log_dir        = args.log_dir,
        run_name       = args.run_name,
        verbose        = not args.quiet,
    )
