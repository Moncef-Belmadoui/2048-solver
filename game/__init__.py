from game.game2048 import Game2048, UP, DOWN, LEFT, RIGHT
from game.environment import Game2048Env
from game.rewards import (
    reward_sparse,
    reward_score_delta,
    reward_log_score_delta,
    reward_survival,
    reward_shaped_v1,
    make_shaped_reward,
)