import sys
sys.path.insert(0, ".")

from game.ui import run_ui

from agents.expectiminimax.expectiminimax_agent import (
    ExpectiminimaxAgent
)

from agents.expectiminimax.cmaes_tuner import (
    ParameterCodec
)

# ------------------------------------------------------------
# Load tuned CMA-ES parameters
# ------------------------------------------------------------
weights, snake_matrix = ParameterCodec.load(
    "saved_models/cmaes_best_1.json"
)

print("\nLoaded CMA-ES weights:\n")

for k, v in weights.items():
    print(f"{k:<12s}: {v:.4f}")

if snake_matrix is not None:
    print("\nLoaded snake matrix:")
    print(snake_matrix)

# ------------------------------------------------------------
# Create agent
# ------------------------------------------------------------
agent = ExpectiminimaxAgent(
    depth=2,
    prob_threshold=0.01,
    weights=weights,
    snake_matrix=snake_matrix,
)

# ------------------------------------------------------------
# Run UI
# ------------------------------------------------------------
run_ui(
    agent=agent,
    fps=2
)