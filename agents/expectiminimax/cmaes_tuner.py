"""
cmaes_tuner.py
--------------
CMA-ES optimisation of Expectiminimax heuristic weights and / or the
snake positional weight matrix.

What is CMA-ES?
---------------
Covariance Matrix Adaptation – Evolution Strategy.  A gradient-free
optimiser that adapts a multivariate Gaussian search distribution over
the parameter space.  It is state-of-the-art for smooth black-box
optimisation with up to ~100 parameters and noisy fitness evaluations.

What we optimise
----------------
Three modes, selectable via --mode:

  "weights" (7 params)
      Scalar combination weights:
      w_mono, w_smooth, w_free, w_max, w_corner, w_merge, w_snake

  "matrix" (16 params)
      The snake positional weight matrix (log-space parameterisation so
      CMA-ES works on a sensible scale; decoded back via exp).

  "full" (23 params)
      Both simultaneously.

Fitness function
----------------
Run K complete 2048 games using ExpectiminimaxAgent at the candidate
parameters.  Return the NEGATIVE average game score (CMA-ES minimises).

Depth 2 keeps each evaluation ~0.05-0.2 s/game so 200 CMA-ES iterations
with 5 games each completes in a few minutes on a laptop.

Usage
-----
    # Quick test – tune scalar weights only, fast depth
    python -m agents.expectiminimax.cmaes_tuner --mode weights --games 5 --depth 2 --iters 150

    # Tune snake matrix (more expressive)
    python -m agents.expectiminimax.cmaes_tuner --mode matrix --games 8 --depth 2 --iters 200

    # Full joint optimisation
    python -m agents.expectiminimax.cmaes_tuner --mode full --games 5 --depth 2 --iters 300

    # Evaluate saved result
    python -m agents.expectiminimax.cmaes_tuner --load saved_models/cmaes_best.json --eval --games 30 --depth 3
"""
from __future__ import annotations
from multiprocessing import Pool, cpu_count

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from game.game2048 import Game2048
from agents.expectiminimax.expectiminimax_agent import ExpectiminimaxAgent
from agents.expectiminimax.heuristics import (
    DEFAULT_WEIGHTS,
    SNAKE_MATRIX_BASE,
    evaluate as heuristic_evaluate,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message="'agents.expectiminimax.cmaes_tuner' found in sys.modules"
)

# ---------------------------------------------------------------------------
# Parameter codec
# ---------------------------------------------------------------------------

# Canonical weight key order (must stay fixed)
_WEIGHT_KEYS = ["w_mono", "w_smooth", "w_free", "w_max", "w_corner", "w_merge", "w_snake"]

# Initial scalar weight values (from DEFAULT_WEIGHTS)
_W0 = np.array([DEFAULT_WEIGHTS[k] for k in _WEIGHT_KEYS], dtype=np.float64)

# Log-space initial snake matrix
_LOG_SNAKE0 = np.log1p(SNAKE_MATRIX_BASE.flatten())   # log(x+1) keeps 0→0


class ParameterCodec:
    """
    Encodes / decodes between a flat CMA-ES θ vector and the
    (weights_dict, snake_matrix) pair used by the agent.

    Parameterisation
    ----------------
    weights : raw values (no transform – CMA-ES constrained to >= 0 after clip)
    matrix  : log1p space so all 16 values live in [0, ~11] regardless of
              the original scale (65536 → log(65536+1) ≈ 11.09).

    Modes
    -----
    "weights" : θ = weights (7 values)
    "matrix"  : θ = log1p(snake_matrix.flatten()) (16 values)
    "full"    : θ = [weights | log1p(snake_matrix.flatten())] (23 values)
    """

    def __init__(self, mode: str):
        assert mode in ("weights", "matrix", "full"), \
            "mode must be 'weights', 'matrix', or 'full'"
        self.mode = mode

    @property
    def dim(self) -> int:
        return {"weights": 7, "matrix": 16, "full": 23}[self.mode]

    def initial_theta(self) -> np.ndarray:
        if self.mode == "weights":
            return _W0.copy()
        if self.mode == "matrix":
            return _LOG_SNAKE0.copy()
        return np.concatenate([_W0, _LOG_SNAKE0])

    def decode(self, theta: np.ndarray) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Returns (weights_dict, snake_matrix_or_None).

        Scalar weights are clipped to [0, ∞) so the heuristic stays valid.
        The snake matrix is decoded from log1p space and reshaped to (4, 4).
        """
        theta = np.asarray(theta, dtype=np.float64)

        if self.mode == "weights":
            weights = dict(zip(_WEIGHT_KEYS, np.clip(theta[:7], 0, None)))
            return weights, None

        if self.mode == "matrix":
            matrix = np.expm1(theta[:16]).reshape(4, 4)   # inverse of log1p
            matrix = np.clip(matrix, 0, None)
            return dict(DEFAULT_WEIGHTS), matrix

        # full
        weights = dict(zip(_WEIGHT_KEYS, np.clip(theta[:7], 0, None)))
        matrix  = np.expm1(theta[7:23]).reshape(4, 4)
        matrix  = np.clip(matrix, 0, None)
        return weights, matrix

    def encode_result(self, weights: dict, snake_matrix: Optional[np.ndarray]) -> dict:
        """Serialise decoded params to a JSON-friendly dict."""
        out: dict = {"weights": weights}
        if snake_matrix is not None:
            out["snake_matrix"] = snake_matrix.tolist()
        return out

    @staticmethod
    def load(path: str) -> Tuple[dict, Optional[np.ndarray]]:
        """Load a previously saved best-params JSON."""
        with open(path) as f:
            data = json.load(f)
        weights = data["weights"]
        matrix  = np.array(data["snake_matrix"]) if "snake_matrix" in data else None
        return weights, matrix


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

# def fitness(
#     theta:    np.ndarray,
#     codec:    ParameterCodec,
#     n_games:  int,
#     depth:    int,
#     seed:     int,
# ) -> float:
#     """
#     Evaluate a candidate θ by playing n_games complete 2048 games.

#     Returns the NEGATIVE average score (CMA-ES minimises).
#     Returns a large positive penalty if decoding fails.
#     """
#     try:
#         weights, snake_matrix = codec.decode(theta)
#     except Exception:
#         return 1e9

#     agent = ExpectiminimaxAgent(
#         depth        = depth,
#         prob_threshold = 0.01,
#         weights      = weights,
#         snake_matrix = snake_matrix,
#     )

#     total_score = 0
#     for i in range(n_games):
#         game = Game2048(seed=seed + i)
#         while not game.is_over():
#             obs   = _game_to_obs(game)
#             action = agent.act(obs)
#             moved, _ = game.move(action)
#             if not moved:                     # fallback: first valid move
#                 valid = game.get_valid_moves()
#                 if valid:
#                     game.move(valid[0])
#         total_score += game.score

#     return -total_score / n_games            # minimise → maximise score

def fitness(
    theta:    np.ndarray,
    codec:    ParameterCodec,
    n_games:  int,
    depth:    int,
    seed:     int,
):
    """
    Composite fitness:
    - high average score
    - high average max tile
    - low score variance


    Returns:
        negative fitness (CMA-ES minimizes)
    """


    try:
        weights, snake_matrix = codec.decode(theta)
    except Exception:
        return 1e9, {}


    agent = ExpectiminimaxAgent(
        depth=depth,
        prob_threshold=0.01,
        weights=weights,
        snake_matrix=snake_matrix,
    )


    scores = []
    max_tiles = []
    wins_2048 = 0
    wins_4096 = 0
    max_tile_seen = 0


    rng = np.random.default_rng(seed)


    for _ in range(n_games):


        game_seed = int(rng.integers(0, 1_000_000_000))


        game = Game2048(seed=game_seed)


        while not game.is_over():


            obs = _game_to_obs(game)


            action = agent.act(obs)


            moved, _ = game.move(action)


            if not moved:


                valid = game.get_valid_moves()


                if valid:
                    game.move(valid[0])


        scores.append(game.score)
        max_tiles.append(game.max_tile())

        tile = game.max_tile()
        max_tile_seen = max(max_tile_seen, tile)

        if tile >= 2048:
            wins_2048 += 1

        if tile >= 4096:
            wins_4096 += 1


    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))


    mean_log_tile = float(
        np.mean(np.log2(np.maximum(max_tiles, 2)))
    )


    # --------------------------------------------------------
    # Composite fitness
    # --------------------------------------------------------
    final_score = (
        mean_score
        + 1000.0 * mean_log_tile
        - 0.3 * std_score
    )


    stats = {
        "mean_score": mean_score,
        "std_score": std_score,
        "mean_max_tile": float(np.mean(max_tiles)),
        "mean_log_tile": mean_log_tile,

        "max_tile_seen": int(max_tile_seen),

        "reached_2048": int(wins_2048),
        "reached_4096": int(wins_4096),
    }


    return -final_score, stats

def _game_to_obs(game: Game2048) -> np.ndarray:
    board = game.get_state().astype(np.float32)
    enc   = np.zeros((4, 4), dtype=np.float32)
    mask  = board > 0
    enc[mask] = np.log2(board[mask])
    return enc[np.newaxis]                   # (1, 4, 4)


# ---------------------------------------------------------------------------
# CMA-ES loop
# ---------------------------------------------------------------------------

# def run_cmaes(
#     mode:        str   = "full",
#     n_games:     int   = 5,
#     depth:       int   = 2,
#     sigma0:      float = 0.5,
#     max_iters:   int   = 200,
#     popsize:     int   = None,  # None → CMA-ES default (4 + floor(3*ln(dim)))
#     seed:        int   = 42,
#     save_dir:    str   = "saved_models",
#     plot_dir:    str   = "logs/cmaes_plots",
#     verbose:     bool  = True,
# ) -> dict:
#     """
#     Run CMA-ES optimisation and return the best decoded params.

#     Parameters
#     ----------
#     mode       : "weights" | "matrix" | "full"
#     n_games    : Games per fitness evaluation.
#     depth      : Expectiminimax search depth (2 = fast, 3 = better quality).
#     sigma0     : Initial step-size.  0.3-0.5 usually works well.
#     max_iters  : Number of CMA-ES generations.
#     popsize    : Population size per generation (None = auto).
#     seed       : Base random seed for game evaluation.
#     save_dir   : Where to write cmaes_best.json.
#     plot_dir   : Where to write the convergence plot.
#     verbose    : Print progress each generation.
#     """
#     try:
#         import cma                            # pip install cma
#     except ImportError:
#         raise ImportError(
#             "The 'cma' package is required for CMA-ES tuning.\n"
#             "Install it with:  pip install cma"
#         )

#     codec   = ParameterCodec(mode)
#     theta0  = codec.initial_theta()
#     dim     = codec.dim

#     # CMA-ES options
#     opts = cma.CMAOptions()
#     opts["maxiter"]   = max_iters
#     opts["seed"]      = seed
#     opts["verbose"]   = -9 if not verbose else 1   # -9 = silent internal logs
#     opts["tolx"]      = 1e-6
#     opts["tolfun"]    = 1e-6
#     if popsize is not None:
#         opts["popsize"] = popsize

#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"  CMA-ES Heuristic Tuner")
#         print(f"  Mode      : {mode}  ({dim} parameters)")
#         print(f"  Depth     : {depth}")
#         print(f"  Games/eval: {n_games}")
#         print(f"  Max iters : {max_iters}")
#         print(f"{'='*60}\n")

#     es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)

#     best_fitness   = float("inf")
#     best_theta     = theta0.copy()
#     history        = []
#     gen            = 0
#     t_start        = time.time()

#     while not es.stop():
#         solutions = es.ask()                  # sample population

#         # Evaluate fitness for each candidate
#         # fitnesses = []
#         # for theta in solutions:
#         #     f = fitness(theta, codec, n_games, depth, seed=seed + gen * 1000)
#         #     fitnesses.append(f)

#                 # Evaluate fitness for each candidate
#         fitnesses = []

#         total_candidates = len(solutions)

#         print(f"\nGeneration {gen + 1}/{max_iters}")
#         print("-" * 50)

#         gen_start = time.time()

#         for idx, theta in enumerate(solutions):
#             cand_start = time.time()

#             f = fitness(
#                 theta,
#                 codec,
#                 n_games,
#                 depth,
#                 seed=seed + gen * 1000 + idx
#             )

#             fitnesses.append(f)

#             cand_time = time.time() - cand_start
#             elapsed   = time.time() - gen_start

#             avg_time_per_candidate = elapsed / (idx + 1)
#             remaining = avg_time_per_candidate * (total_candidates - idx - 1)

#             progress = (idx + 1) / total_candidates
#             bar_len = 30
#             filled = int(bar_len * progress)

#             bar = "█" * filled + "-" * (bar_len - filled)

#             print(
#                 f"\r[{bar}] "
#                 f"{idx + 1}/{total_candidates} "
#                 f"| score={-f:8.1f} "
#                 f"| cand_time={cand_time:5.1f}s "
#                 f"| ETA={remaining:6.1f}s",
#                 end="",
#                 flush=True
#             )

#         print()

#         es.tell(solutions, fitnesses)         # update distribution

#         gen_best  = min(fitnesses)
#         gen_mean  = float(np.mean(fitnesses))
#         history.append({"gen": gen, "best": gen_best, "mean": gen_mean})

#         if gen_best < best_fitness:
#             best_fitness = gen_best
#             best_theta   = solutions[np.argmin(fitnesses)].copy()

#             print(
#                 f"\nNEW BEST FOUND -> avg_score={-best_fitness:.1f}"
#             )

#         gen += 1

#         if verbose:
#             elapsed = time.time() - t_start
#             print(
#                 f"  Gen {gen:4d} | best_score={-gen_best:8.1f} | "
#                 f"mean_score={-gen_mean:8.1f} | "
#                 f"sigma={es.sigma:.4f} | {elapsed:.0f}s"
#             )

#     # ── Decode and save ────────────────────────────────────────────────────
#     best_weights, best_matrix = codec.decode(best_theta)
#     result = codec.encode_result(best_weights, best_matrix)
#     result["fitness"]  = best_fitness
#     result["avg_score"] = -best_fitness
#     result["mode"]     = mode
#     result["depth"]    = depth
#     result["n_games"]  = n_games

#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "cmaes_best.json")
#     with open(save_path, "w") as f:
#         json.dump(result, f, indent=2)
#     print(f"\nBest params saved → {save_path}")
#     print(f"Best avg score    : {-best_fitness:.1f}")

#     # ── Convergence plot ───────────────────────────────────────────────────
#     _plot_convergence(history, plot_dir)

#     if verbose:
#         _print_best(best_weights, best_matrix)

#     return result

# def run_cmaes(
#     mode:        str   = "full",
#     n_games:     int   = 5,
#     depth:       int   = 2,
#     sigma0:      float = 0.5,
#     max_iters:   int   = 200,
#     popsize:     int   = None,
#     seed:        int   = 42,
#     save_dir:    str   = "saved_models",
#     plot_dir:    str   = "logs/cmaes_plots",
#     verbose:     bool  = True,
# ) -> dict:
#     """
#     Run CMA-ES optimisation and return the best decoded params.
#     """

#     try:
#         import cma
#     except ImportError:
#         raise ImportError(
#             "The 'cma' package is required for CMA-ES tuning.\n"
#             "Install it with: pip install cma"
#         )

#     codec   = ParameterCodec(mode)
#     theta0  = codec.initial_theta()
#     dim     = codec.dim

#     # ------------------------------------------------------------
#     # CMA-ES options
#     # ------------------------------------------------------------
#     opts = cma.CMAOptions()

#     opts["maxiter"] = max_iters
#     opts["seed"]    = seed
#     opts["verbose"] = -9 if not verbose else 1
#     opts["tolx"]    = 1e-6
#     opts["tolfun"]  = 1e-6

#     if popsize is not None:
#         opts["popsize"] = popsize

#     # ------------------------------------------------------------
#     # Header
#     # ------------------------------------------------------------
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"  CMA-ES Heuristic Tuner")
#         print(f"  Mode      : {mode} ({dim} parameters)")
#         print(f"  Depth     : {depth}")
#         print(f"  Games/eval: {n_games}")
#         print(f"  Max iters : {max_iters}")
#         print(f"{'='*60}\n")

#     es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)

#     best_fitness = float("inf")
#     best_theta   = theta0.copy()

#     history = []
#     gen     = 0

#     t_start = time.time()

#     # ------------------------------------------------------------
#     # Main CMA-ES loop
#     # ------------------------------------------------------------
#     while not es.stop():

#         solutions = es.ask()

#         total_candidates = len(solutions)

#         print(f"\nGeneration {gen + 1}/{max_iters}")
#         print("-" * 60)

#         # --------------------------------------------------------
#         # Parallel evaluation
#         # --------------------------------------------------------
#         args_list = [
#             (
#                 theta,
#                 codec,
#                 n_games,
#                 depth,
#                 seed + gen * 1000 + i
#             )
#             for i, theta in enumerate(solutions)
#         ]

#         num_workers = min(cpu_count(), total_candidates)

#         print(f"Using {num_workers} CPU workers")
#         print(f"Evaluating {total_candidates} candidates...")

#         gen_start = time.time()

#         with Pool(processes=num_workers) as pool:

#             async_result = pool.starmap_async(fitness, args_list)

#             # Progress display loop
#             while not async_result.ready():

#                 elapsed = time.time() - gen_start

#                 print(
#                     f"\rElapsed: {elapsed:6.1f}s",
#                     end="",
#                     flush=True
#                 )

#                 time.sleep(1)

#             # fitnesses = async_result.get()
#             results = async_result.get()

#             fitnesses = [r[0] for r in results]
#             stats_list = [r[1] for r in results]

#         gen_time = time.time() - gen_start

#         print(f"\nGeneration completed in {gen_time:.1f}s")

#         # --------------------------------------------------------
#         # Update CMA-ES
#         # --------------------------------------------------------
#         es.tell(solutions, fitnesses)

#         # gen_best = min(fitnesses)
#         # gen_mean = float(np.mean(fitnesses))

#         gen_best = min(fitnesses)
#         gen_mean = float(np.mean(fitnesses))

#         mean_scores = [
#             s["mean_score"]
#             for s in stats_list
#         ]

#         std_scores = [
#             s["std_score"]
#             for s in stats_list
#         ]

#         mean_tiles = [
#             s["mean_max_tile"]
#             for s in stats_list
#         ]

#         gen_avg_score = float(np.mean(mean_scores))
#         gen_avg_std   = float(np.mean(std_scores))
#         gen_avg_tile  = float(np.mean(mean_tiles))

#         # history.append({
#         #     "gen": gen,
#         #     "best": gen_best,
#         #     "mean": gen_mean
#         # })

#         history.append({
#     "generation": gen,
#     "best_fitness": float(gen_best),
#     "mean_fitness": float(gen_mean),
#     "avg_score": gen_avg_score,
#     "avg_std": gen_avg_std,
#     "avg_max_tile": gen_avg_tile,
#     "sigma": float(es.sigma),
#     "generation_time_sec": float(gen_time),
# })
#         os.makedirs("logs/cmaes_logs", exist_ok=True)

#         history_path = os.path.join(
#             "logs/cmaes_logs",
#             "cmaes_history.json"
#         )

#         with open(history_path, "w") as f:
#             json.dump(history, f, indent=2)

#         # --------------------------------------------------------
#         # Save new best checkpoint
#         # --------------------------------------------------------
#         if gen_best < best_fitness:

#             best_fitness = gen_best
#             best_theta   = solutions[np.argmin(fitnesses)].copy()

#             print(
#                 f"\nNEW BEST FOUND -> avg_score={-best_fitness:.1f}"
#             )

#             best_weights, best_matrix = codec.decode(best_theta)

#             checkpoint = codec.encode_result(
#                 best_weights,
#                 best_matrix
#             )

#             checkpoint["fitness"]   = best_fitness
#             checkpoint["avg_score"] = -best_fitness
#             checkpoint["generation"] = gen
#             checkpoint["mode"]      = mode
#             checkpoint["depth"]     = depth
#             checkpoint["n_games"]   = n_games

#             os.makedirs(save_dir, exist_ok=True)

#             checkpoint_path = os.path.join(
#                 save_dir,
#                 "cmaes_checkpoint.json"
#             )

#             with open(checkpoint_path, "w") as f:
#                 json.dump(checkpoint, f, indent=2)

#             print(f"Checkpoint saved -> {checkpoint_path}")

#         gen += 1

#         # --------------------------------------------------------
#         # Generation summary
#         # --------------------------------------------------------
#         if verbose:

#             elapsed_total = time.time() - t_start

#             print(
#             f"Gen {gen:4d} | "
#             f"fitness={-gen_best:8.1f} | "
#             f"avg_score={gen_avg_score:8.1f} | "
#             f"std={gen_avg_std:7.1f} | "
#             f"avg_tile={gen_avg_tile:7.1f} | "
#             f"sigma={es.sigma:.4f} | "
#             f"time={gen_time:6.1f}s"
#         )

#     # ------------------------------------------------------------
#     # Final decode + save
#     # ------------------------------------------------------------
#     best_weights, best_matrix = codec.decode(best_theta)

#     result = codec.encode_result(
#         best_weights,
#         best_matrix
#     )

#     result["fitness"]   = best_fitness
#     result["avg_score"] = -best_fitness
#     result["mode"]      = mode
#     result["depth"]     = depth
#     result["n_games"]   = n_games

#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(
#         save_dir,
#         "cmaes_best.json"
#     )

#     with open(save_path, "w") as f:
#         json.dump(result, f, indent=2)

#     print(f"\nBest params saved -> {save_path}")
#     print(f"Best avg score    : {-best_fitness:.1f}")

#     # ------------------------------------------------------------
#     # Plot convergence
#     # ------------------------------------------------------------
#     _plot_convergence(history, plot_dir)

#     if verbose:
#         _print_best(best_weights, best_matrix)

#     return result

def run_cmaes(
    mode:        str   = "full",
    n_games:     int   = 5,
    depth:       int   = 2,
    sigma0:      float = 0.5,
    max_iters:   int   = 200,
    popsize:     int   = None,
    seed:        int   = 42,
    save_dir:    str   = "saved_models",
    plot_dir:    str   = "logs/cmaes_plots",
    verbose:     bool  = True,
) -> dict:
    """
    Run CMA-ES optimisation and return the best decoded params.
    """


    try:
        import cma
    except ImportError:
        raise ImportError(
            "The 'cma' package is required for CMA-ES tuning.\n"
            "Install it with: pip install cma"
        )


    codec   = ParameterCodec(mode)
    theta0  = codec.initial_theta()
    dim     = codec.dim


    # ------------------------------------------------------------
    # CMA-ES options
    # ------------------------------------------------------------
    opts = cma.CMAOptions()


    opts["maxiter"] = max_iters
    opts["seed"]    = seed
    opts["verbose"] = -9 if not verbose else 1
    opts["tolx"]    = 1e-6
    opts["tolfun"]  = 1e-6


    if popsize is not None:
        opts["popsize"] = popsize


    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"  CMA-ES Heuristic Tuner")
        print(f"  Mode      : {mode} ({dim} parameters)")
        print(f"  Depth     : {depth}")
        print(f"  Games/eval: {n_games}")
        print(f"  Max iters : {max_iters}")
        print(f"{'='*60}\n")


    es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)


    best_fitness = float("inf")
    best_theta   = theta0.copy()


    history = []
    gen     = 0


    t_start = time.time()


    # ------------------------------------------------------------
    # Main CMA-ES loop
    # ------------------------------------------------------------
    while not es.stop():


        solutions = es.ask()


        total_candidates = len(solutions)


        print(f"\nGeneration {gen + 1}/{max_iters}")
        print("-" * 60)


        # --------------------------------------------------------
        # Parallel evaluation
        # --------------------------------------------------------
        args_list = [
            (
                theta,
                codec,
                n_games,
                depth,
                seed + gen * 1000 + i
            )
            for i, theta in enumerate(solutions)
        ]


        num_workers = min(cpu_count(), total_candidates)


        print(f"Using {num_workers} CPU workers")
        print(f"Evaluating {total_candidates} candidates...")


        gen_start = time.time()


        with Pool(processes=num_workers) as pool:


            async_result = pool.starmap_async(fitness, args_list)


            # Progress display loop
            while not async_result.ready():


                elapsed = time.time() - gen_start


                print(
                    f"\rElapsed: {elapsed:6.1f}s",
                    end="",
                    flush=True
                )


                time.sleep(1)


            # fitnesses = async_result.get()
            results = async_result.get()


            fitnesses = [r[0] for r in results]
            stats_list = [r[1] for r in results]


        gen_time = time.time() - gen_start


        print(f"\nGeneration completed in {gen_time:.1f}s")


        # --------------------------------------------------------
        # Update CMA-ES
        # --------------------------------------------------------
        es.tell(solutions, fitnesses)


        gen_best = min(fitnesses)
        gen_mean = float(np.mean(fitnesses))


        mean_scores = [
            s["mean_score"]
            for s in stats_list
        ]


        std_scores = [
            s["std_score"]
            for s in stats_list
        ]


        mean_tiles = [
            s["mean_max_tile"]
            for s in stats_list
        ]

        max_tiles_seen = [
            s["max_tile_seen"]
            for s in stats_list
        ]

        wins_2048_list = [
            s["reached_2048"]
            for s in stats_list
        ]

        wins_4096_list = [
            s["reached_4096"]
            for s in stats_list
        ]


        gen_avg_score = float(np.mean(mean_scores))
        gen_avg_std   = float(np.mean(std_scores))
        gen_avg_tile  = float(np.mean(mean_tiles))

        gen_best_tile = int(max(max_tiles_seen))

        gen_2048_rate = float(
            np.sum(wins_2048_list)
            / (len(wins_2048_list) * n_games)
        )

        gen_4096_rate = float(
            np.sum(wins_4096_list)
            / (len(wins_4096_list) * n_games)
        )


        history.append({
            "generation": gen,

            "best_fitness": float(gen_best),
            "mean_fitness": float(gen_mean),

            "avg_score": gen_avg_score,
            "avg_std": gen_avg_std,

            "avg_max_tile": gen_avg_tile,
            "best_tile_seen": gen_best_tile,

            "reached_2048_rate": gen_2048_rate,
            "reached_4096_rate": gen_4096_rate,

            "sigma": float(es.sigma),

            "generation_time_sec": float(gen_time),
        })

        os.makedirs("logs/cmaes_logs", exist_ok=True)


        history_path = os.path.join(
            "logs/cmaes_logs",
            "cmaes_history.json"
        )


        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)


        # --------------------------------------------------------
        # Save new best checkpoint
        # --------------------------------------------------------
        if gen_best < best_fitness:


            best_fitness = gen_best
            best_theta   = solutions[np.argmin(fitnesses)].copy()

            best_weights, best_matrix = codec.decode(best_theta)

            generation_result = codec.encode_result(
                best_weights,
                best_matrix
            )

            generation_result["fitness"] = float(best_fitness)
            generation_result["avg_score"] = float(-best_fitness)

            generation_result["generation"] = gen

            generation_result["mode"] = mode
            generation_result["depth"] = depth
            generation_result["n_games"] = n_games

            os.makedirs(
                "logs/cmaes_logs/generation_bests",
                exist_ok=True
            )

            gen_save_path = os.path.join(
                "logs/cmaes_logs/generation_bests",
                f"generation_{gen:03d}.json"
            )

            with open(gen_save_path, "w") as f:
                json.dump(generation_result, f, indent=2)


            print(
                f"\nNEW BEST FOUND -> avg_score={-best_fitness:.1f}"
            )


            # checkpoint = codec.encode_result(
            #     best_weights,
            #     best_matrix
            # )


            # checkpoint["fitness"]   = best_fitness
            # checkpoint["avg_score"] = -best_fitness
            # checkpoint["generation"] = gen
            # checkpoint["mode"]      = mode
            # checkpoint["depth"]     = depth
            # checkpoint["n_games"]   = n_games


            # os.makedirs(save_dir, exist_ok=True)


            # checkpoint_path = os.path.join(
            #     save_dir,
            #     "cmaes_checkpoint.json"
            # )


            # with open(checkpoint_path, "w") as f:
            #     json.dump(checkpoint, f, indent=2)


            # print(f"Checkpoint saved -> {checkpoint_path}")


        gen += 1


        # --------------------------------------------------------
        # Generation summary
        # --------------------------------------------------------
        if verbose:


            elapsed_total = time.time() - t_start

            print(
    f"Gen {gen:4d} | "
    f"fitness={-gen_best:8.1f} | "
    f"avg_score={gen_avg_score:8.1f} | "
    f"std={gen_avg_std:7.1f} | "
    f"avg_tile={gen_avg_tile:7.1f} | "
    f"best_tile={gen_best_tile:5d} | "
    f"2048={gen_2048_rate:.1%} | "
    f"4096={gen_4096_rate:.1%} | "
    f"sigma={es.sigma:.4f} | "
    f"time={gen_time:6.1f}s"
)

    # ------------------------------------------------------------
    # Final decode + save
    # ------------------------------------------------------------
    best_weights, best_matrix = codec.decode(best_theta)


    result = codec.encode_result(
        best_weights,
        best_matrix
    )


    result["fitness"]   = best_fitness
    result["avg_score"] = -best_fitness
    result["mode"]      = mode
    result["depth"]     = depth
    result["n_games"]   = n_games


    os.makedirs(save_dir, exist_ok=True)


    save_path = os.path.join(
        save_dir,
        "cmaes_best.json"
    )


    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)


    print(f"\nBest params saved -> {save_path}")
    print(f"Best avg score    : {-best_fitness:.1f}")


    # ------------------------------------------------------------
    # Plot convergence
    # ------------------------------------------------------------
    _plot_convergence(history, plot_dir)


    if verbose:
        _print_best(best_weights, best_matrix)


    return result


# ---------------------------------------------------------------------------
# Evaluation of saved params
# ---------------------------------------------------------------------------

def evaluate_saved(
    path:    str,
    n_games: int = 30,
    depth:   int = 3,
    seed:    int = 42,
):
    """Load saved CMA-ES result and evaluate it properly at higher depth."""
    weights, snake_matrix = ParameterCodec.load(path)

    print(f"\nLoaded params from: {path}")
    _print_best(weights, snake_matrix)

    agent = ExpectiminimaxAgent(
        depth        = depth,
        prob_threshold = 0.005,
        weights      = weights,
        snake_matrix = snake_matrix,
    )

    scores    = []
    max_tiles = []
    wins      = 0

    print(f"\nEvaluating {n_games} games at depth {depth}…\n")

    for i in range(n_games):
        game = Game2048(seed=seed + i)
        while not game.is_over():
            obs    = _game_to_obs(game)
            action = agent.act(obs)
            moved, _ = game.move(action)
            if not moved:
                valid = game.get_valid_moves()
                if valid:
                    game.move(valid[0])
        scores.append(game.score)
        max_tiles.append(game.max_tile())
        wins += int(game.has_won())
        print(f"  Game {i+1:3d}: score={game.score:6d}  max_tile={game.max_tile():4d}")

    print(f"\n  Avg score    : {np.mean(scores):.1f}")
    print(f"  Avg max tile : {np.mean(max_tiles):.1f}")
    print(f"  Win rate     : {wins/n_games:.1%}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_convergence(history: list, plot_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        # gens   = [h["gen"] for h in history]
        gens = [h["generation"] for h in history]
        bests  = [-h["best_fitness"] for h in history]
        means  = [-h["mean_fitness"] for h in history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gens, means, alpha=0.4, color="steelblue", label="Mean score")
        ax.plot(gens, bests, color="steelblue", linewidth=2, label="Best score")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Avg game score")
        ax.set_title("CMA-ES Convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(plot_dir, "cmaes_convergence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot → {path}")
    except ImportError:
        pass   # matplotlib optional


def _print_best(weights: dict, snake_matrix: Optional[np.ndarray]):
    print("\n  Best scalar weights:")
    for k, v in weights.items():
        print(f"    {k:<12s}: {v:.4f}")
    if snake_matrix is not None:
        print("\n  Best snake matrix:")
        for row in snake_matrix:
            print("    " + "  ".join(f"{v:10.1f}" for v in row))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="CMA-ES tuner for Expectiminimax")
    p.add_argument("--mode",      type=str,   default="full",
                   choices=["weights", "matrix", "full"])
    p.add_argument("--games",     type=int,   default=5,
                   help="Games per fitness evaluation")
    p.add_argument("--depth",     type=int,   default=2,
                   help="Search depth (2=fast, 3=better quality)")
    p.add_argument("--sigma0",    type=float, default=0.5,
                   help="Initial CMA-ES step-size")
    p.add_argument("--iters",     type=int,   default=200,
                   help="Max CMA-ES generations")
    p.add_argument("--popsize",   type=int,   default=None,
                   help="Population size (None = auto)")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--save-dir",  type=str,   default="saved_models")
    p.add_argument("--plot-dir",  type=str,   default="logs/cmaes_plots")
    p.add_argument("--load",      type=str,   default=None,
                   help="Load saved JSON and evaluate (skip training)")
    p.add_argument("--eval",      action="store_true")
    p.add_argument("--eval-depth",type=int,   default=3)
    p.add_argument("--quiet",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.load:
        evaluate_saved(
            path    = args.load,
            n_games = args.games,
            depth   = args.eval_depth,
            seed    = args.seed,
        )
    else:
        run_cmaes(
            mode      = args.mode,
            n_games   = args.games,
            depth     = args.depth,
            sigma0    = args.sigma0,
            max_iters = args.iters,
            popsize   = args.popsize,
            seed      = args.seed,
            save_dir  = args.save_dir,
            plot_dir  = args.plot_dir,
            verbose   = not args.quiet,
        )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()