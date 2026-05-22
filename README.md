# 2048 — Reinforcement Learning & Search Agents

A clean, modular, research-grade Python project implementing multiple agents
that learn to play the game 2048, with full training infrastructure,
logging, and visualization.

---

## Features

| Feature | Details |
|---|---|
| Game engine | Pure 4×4 logic, no external deps |
| Gymnasium env | Log2 / raw / one-hot obs, pluggable reward fns |
| Reward system | Sparse · Score delta · Shaped (mono + smooth + free + corner) |
| DQN agent | Double DQN + Dueling arch + Prioritized Replay (PER) |
| Search agent | Expectiminimax with chance nodes + prob-threshold pruning |
| Baseline | Random agent |
| Logging | Per-episode CSV + JSON summary, 14+ tracked metrics |
| Plotting | 8 individual plots + agent comparison charts |
| UI | Polished Pygame window – human play & AI autoplay |
| Reproducibility | Global seed utility covering Python / NumPy / PyTorch / CUDA |

---

## Project Structure

```
project/
│
├── game/
│   ├── game2048.py       # Pure 2048 logic (no RL, no rendering)
│   ├── environment.py    # Gymnasium-style wrapper
│   ├── rewards.py        # Pluggable reward functions
│   └── ui.py             # Pygame UI (human + agent mode)
│
├── agents/
│   ├── random_agent.py
│   ├── dqn/
│   │   ├── network.py          # Dueling DQN (CNN)
│   │   ├── replay_buffer.py    # Prioritized Experience Replay (SumTree)
│   │   ├── dqn_agent.py        # Double DQN agent
│   │   └── train_dqn.py        # Training pipeline
│   │
│   └── expectiminimax/
│       ├── heuristics.py              # Individual + combined heuristics
│       ├── expectiminimax_agent.py    # Search agent
│       └── evaluate.py               # Standalone EMM evaluator
│
├── utils/
│   ├── logger.py     # CSV + JSON episode logger
│   ├── plotting.py   # Matplotlib plot suite
│   └── seed.py       # Global seed utility
│
├── evaluate_agents.py   # Multi-agent comparison framework
├── saved_models/
├── logs/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / unzip the project
cd project2048

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Human play (Pygame)

```python
from game.ui import run_ui
run_ui()                  # Arrow keys to play, R to restart
```

### Train the DQN agent

```bash
# Default settings
python -m agents.dqn.train_dqn

# Custom hyperparameters
python -m agents.dqn.train_dqn \
    --episodes 20000 \
    --lr 3e-4 \
    --batch-size 128 \
    --render-every 1000 \
    --run-name my_experiment

# Resume training
python -m agents.dqn.train_dqn --load saved_models/dqn_run_best.pt

# Evaluation only
python -m agents.dqn.train_dqn --load saved_models/dqn_run_best.pt --eval
```

### Watch the DQN agent play (Pygame)

```python
from game.ui import run_ui
from agents.dqn.dqn_agent import DQNAgent

agent = DQNAgent()
agent.load("saved_models/dqn_run_best.pt", eval_mode=True)
run_ui(agent=agent, fps=10)
```

### Evaluate the Expectiminimax agent

```bash
python -m agents.expectiminimax.evaluate --games 100 --depth 4
```

### Compare all agents

```bash
python evaluate_agents.py \
    --games 100 \
    --dqn-checkpoint saved_models/dqn_run_best.pt \
    --emm-depth 3
```

---

## Reward Functions

The environment accepts any function with this signature:

```python
def my_reward(board_before, board_after, score_before, score_after,
              moved, done, invalid) -> float:
    ...
```

Built-in options in `game/rewards.py`:

| Name | Description |
|---|---|
| `reward_sparse` | +1 win / -1 lose / 0 otherwise |
| `reward_score_delta` | Raw score gain per step |
| `reward_log_score_delta` | log2(score gain + 1) |
| `reward_survival` | +0.1/step, -1 on death |
| `reward_shaped_v1` | Combo: score + mono + smooth + free + max + corner |
| `make_shaped_reward(**w)` | Factory with configurable weights |

```python
from game.environment import Game2048Env
from game.rewards import make_shaped_reward

reward_fn = make_shaped_reward(w_mono=2.0, w_corner=3.0)
env = Game2048Env(reward_fn=reward_fn)
```

---

## DQN Architecture

```
Input (1, 4, 4)  ← log2-encoded board
  Conv2d(1→64,   kernel=2, ReLU)
  Conv2d(64→128, kernel=2, ReLU)
  Flatten → FC(512, ReLU)
        ┌────────────────┬─────────────────────┐
   Value stream      Advantage stream
   FC(256) → FC(1)   FC(256) → FC(4)
        └────────────────┴─────────────────────┘
  Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

**Training features:** Double DQN target · Prioritized replay (proportional, SumTree) ·
Huber loss · Gradient clipping · Hard target updates · ε-greedy decay

---

## Expectiminimax

Alternates between:
- **MAX nodes** – player picks best move (α-β pruning)
- **CHANCE nodes** – weighted average over (tile=2, p=0.9) and (tile=4, p=0.1) spawns in all empty cells

Heuristic evaluation (configurable weights):

```
score = w_mono   × monotonicity
      + w_smooth × smoothness
      + w_free   × log(free_tiles + 1)
      + w_max    × log2(max_tile)
      + w_corner × corner_bonus
      + w_merge  × merge_potential
```

The raw game score is **not** included — delayed merges can build better long-term structures.

---

## Logging & Metrics

Every training run produces:

```
logs/<run_name>_<timestamp>/
  episodes.csv    ← one row per episode
  summary.json    ← aggregate min/mean/max/last for every metric
```

Tracked metrics (14+):

`reward · score · max_tile · steps · epsilon · loss · avg_q ·
invalid_moves · won · action_{up,down,left,right} · merges · empty_cells · time_s`

---

## Plotting

```python
from utils.plotting import plot_training, plot_agent_comparison

# Generate all plots for a training run
plot_training("logs/dqn_run_.../episodes.csv", save_dir="logs/plots")

# Cross-agent comparison
results = {"DQN": dqn_stats, "EMM": emm_stats, "Random": rnd_stats}
plot_agent_comparison(results, save_dir="logs/comparison_plots")
```

Available individual plots: `rewards · scores · loss · epsilon · max_tile_frequency ·
q_values · invalid_moves · action_distribution · agent_comparison · tile_freq_comparison`

---

## Reproducibility

```python
from utils.seed import set_global_seed
set_global_seed(42)   # sets Python, NumPy, PyTorch, CUDA seeds
```

---

## Extending the Project

**New reward function** → add to `game/rewards.py`, pass to `Game2048Env`.

**New agent** → implement `act(obs: np.ndarray) -> int`, drop into `agents/`.

**New heuristic** → add to `agents/expectiminimax/heuristics.py`, update `evaluate()`.

**New obs encoding** → add `OBS_*` constant and branch in `Game2048Env._get_obs()`.

---

## License

MIT
