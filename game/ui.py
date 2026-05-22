"""
ui.py
-----
Polished Pygame interface for 2048.

Usage
-----
Human mode:
    from game.ui import run_ui
    run_ui()

Agent mode:
    from game.ui import run_ui
    run_ui(agent=my_agent, fps=10)

The agent object must implement:
    agent.act(obs)  ->  int   (action 0-3)
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Any

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore

from game.environment import Game2048Env
from game.rewards import reward_score_delta

# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------
BG_COLOR       = (187, 173, 160)
EMPTY_COLOR    = (205, 193, 180)
TEXT_DARK      = (119, 110, 101)
TEXT_LIGHT     = (249, 246, 242)
HEADER_BG      = (250, 248, 239)
SCORE_BOX_BG   = (187, 173, 160)
SCORE_BOX_FG   = (255, 255, 255)

TILE_COLORS: dict = {
    0:     (205, 193, 180),
    2:     (238, 228, 218),
    4:     (237, 224, 200),
    8:     (242, 177, 121),
    16:    (245, 149,  99),
    32:    (246, 124,  95),
    64:    (246,  94,  59),
    128:   (237, 207, 114),
    256:   (237, 204,  97),
    512:   (237, 200,  80),
    1024:  (237, 197,  63),
    2048:  (237, 194,  46),
}

WINDOW_W  = 500
WINDOW_H  = 600
GRID_SIZE = 400
PADDING   = 12
CELL_SIZE = (GRID_SIZE - PADDING * 5) // 4
GRID_X    = (WINDOW_W - GRID_SIZE) // 2
GRID_Y    = 150


def _tile_color(value: int):
    return TILE_COLORS.get(value, (60, 58, 50))


def _text_color(value: int):
    return TEXT_DARK if value in (2, 4) else TEXT_LIGHT


def _draw_rounded_rect(surface, color, rect, radius=8):
    pygame.draw.rect(surface, color, rect, border_radius=radius)


# ------------------------------------------------------------------
# Main UI class
# ------------------------------------------------------------------

class GameUI:
    def __init__(self, agent=None, fps: int = 30, env_kwargs: dict = None):
        assert pygame is not None, "pygame is required: pip install pygame"
        pygame.init()
        pygame.display.set_caption("2048 – RL Playground")

        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock  = pygame.time.Clock()
        self.fps    = fps
        self.agent  = agent

        env_kw = env_kwargs or {}
        self.env = Game2048Env(reward_fn=reward_score_delta, **env_kw)
        self.obs, self.info = self.env.reset()

        # Fonts
        self.font_tile_lg  = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_tile_sm  = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_tile_xs  = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ui       = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_label    = pygame.font.SysFont("Arial", 16)
        self.font_title    = pygame.font.SysFont("Arial", 52, bold=True)

        self.done = False
        self.total_reward = 0.0
        self.agent_steps  = 0

    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            running = self._handle_events()
            if self.agent and not self.done:
                self._agent_step()
            self._draw()
            self.clock.tick(self.fps)
        pygame.quit()

    # ------------------------------------------------------------------

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._restart()
                if not self.agent and not self.done:
                    action = {
                        pygame.K_UP:    0,
                        pygame.K_DOWN:  1,
                        pygame.K_LEFT:  2,
                        pygame.K_RIGHT: 3,
                    }.get(event.key)
                    if action is not None:
                        self._step(action)
        return True

    def _step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs   = obs
        self.info  = info
        self.done  = done or truncated
        self.total_reward += reward
        self.agent_steps  += 1

    def _agent_step(self):
        action = self.agent.act(self.obs)
        self._step(action)

    def _restart(self):
        self.obs, self.info = self.env.reset()
        self.done          = False
        self.total_reward  = 0.0
        self.agent_steps   = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(HEADER_BG)
        self._draw_header()
        self._draw_grid()
        if self.done:
            self._draw_overlay()
        pygame.display.flip()

    def _draw_header(self):
        # Title
        title = self.font_title.render("2048", True, TEXT_DARK)
        self.screen.blit(title, (GRID_X, 20))

        # Score box
        self._draw_score_box(GRID_X + 240, 20, "SCORE", self.info.get("score", 0))
        self._draw_score_box(GRID_X + 330, 20, "BEST",  self.info.get("max_tile", 0))

        # Agent info (if applicable)
        if self.agent:
            info_str = (
                f"Steps: {self.agent_steps}  "
                f"Reward: {self.total_reward:.1f}  "
                f"Max: {self.info.get('max_tile', 0)}"
            )
            surf = self.font_label.render(info_str, True, TEXT_DARK)
            self.screen.blit(surf, (GRID_X, 120))
        else:
            hint = self.font_label.render("Arrow keys to play | R to restart", True, TEXT_DARK)
            self.screen.blit(hint, (GRID_X, 120))

    def _draw_score_box(self, x: int, y: int, label: str, value: int):
        rect = pygame.Rect(x, y, 85, 60)
        _draw_rounded_rect(self.screen, SCORE_BOX_BG, rect)
        lbl = self.font_label.render(label, True, SCORE_BOX_FG)
        val = self.font_ui.render(str(value), True, SCORE_BOX_FG)
        self.screen.blit(lbl, (x + (85 - lbl.get_width()) // 2, y + 6))
        self.screen.blit(val, (x + (85 - val.get_width()) // 2, y + 28))

    def _draw_grid(self):
        grid_rect = pygame.Rect(GRID_X, GRID_Y, GRID_SIZE, GRID_SIZE)
        _draw_rounded_rect(self.screen, BG_COLOR, grid_rect)

        board = self.env.board
        for r in range(4):
            for c in range(4):
                self._draw_tile(r, c, int(board[r, c]))

    def _draw_tile(self, row: int, col: int, value: int):
        x = GRID_X + PADDING + col * (CELL_SIZE + PADDING)
        y = GRID_Y + PADDING + row * (CELL_SIZE + PADDING)
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        _draw_rounded_rect(self.screen, _tile_color(value), rect)

        if value > 0:
            color = _text_color(value)
            s = str(value)
            if len(s) <= 2:
                font = self.font_tile_lg
            elif len(s) == 3:
                font = self.font_tile_sm
            else:
                font = self.font_tile_xs
            surf = font.render(s, True, color)
            tx = x + (CELL_SIZE - surf.get_width())  // 2
            ty = y + (CELL_SIZE - surf.get_height()) // 2
            self.screen.blit(surf, (tx, ty))

    def _draw_overlay(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((238, 228, 218, 160))
        self.screen.blit(overlay, (0, 0))

        text = "You Win! 🎉" if self.info.get("won") else "Game Over!"
        msg = self.font_title.render(text, True, TEXT_DARK)
        self.screen.blit(msg, (
            (WINDOW_W - msg.get_width())  // 2,
            (WINDOW_H - msg.get_height()) // 2 - 20,
        ))
        hint = self.font_ui.render("Press R to restart", True, TEXT_DARK)
        self.screen.blit(hint, (
            (WINDOW_W - hint.get_width())  // 2,
            (WINDOW_H + msg.get_height()) // 2,
        ))


# ------------------------------------------------------------------
# Public entry-point
# ------------------------------------------------------------------

def run_ui(
    agent=None,
    fps: int = 30,
    env_kwargs: Optional[dict] = None,
):
    """
    Launch the Pygame window.

    Parameters
    ----------
    agent       : object with `.act(obs) -> int`, or None for human play.
    fps         : frames per second.
    env_kwargs  : extra keyword arguments forwarded to Game2048Env.
    """
    ui = GameUI(agent=agent, fps=fps, env_kwargs=env_kwargs)
    ui.run()


# if __name__ == "__main__":
#     run_ui()