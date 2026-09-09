import os
import warnings
from typing import Any

import numpy as np

from env.config import EnvironmentConfig
from env.demonstrations import evaluate_path


WINDOW_WIDTH = 720
WINDOW_HEIGHT = 520
MARGIN = 36


class PointReachRenderer:
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self._pygame: Any | None = None
        self._window: Any | None = None
        self._clock: Any | None = None
        self._owns_pygame_init = False
        self._owns_display_init = False
        self._owns_window = False
        self._restore_display_init = False

    def _load_pygame(self) -> Any:
        if self._pygame is None:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Deprecated call to `pkg_resources.declare_namespace.*",
                    category=DeprecationWarning,
                )
                import pygame

            self._pygame = pygame
        return self._pygame

    def _world_to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        x_low, x_high = self.config.visualization_x
        y_low, y_high = self.config.visualization_y
        scale = min(
            (WINDOW_WIDTH - 2 * MARGIN) / (x_high - x_low),
            (WINDOW_HEIGHT - 2 * MARGIN) / (y_high - y_low),
        )
        x = float(np.clip(point[0], x_low, x_high))
        y = float(np.clip(point[1], y_low, y_high))
        x_center = 0.5 * (x_low + x_high)
        y_center = 0.5 * (y_low + y_high)
        pixel_x = round(WINDOW_WIDTH / 2 + (x - x_center) * scale)
        pixel_y = round(WINDOW_HEIGHT / 2 - (y - y_center) * scale)
        return pixel_x, pixel_y

    def _draw_reference_paths(self, canvas: Any) -> None:
        pygame = self._load_pygame()
        times = np.linspace(
            0.0,
            1.0,
            self.config.horizon_steps + 1,
            dtype=np.float32,
        )
        paths = (
            (1.0, 0.27, (104, 157, 224)),
            (1.0, 0.53, (52, 101, 184)),
            (-1.0, 0.27, (244, 166, 103)),
            (-1.0, 0.53, (204, 91, 67)),
        )
        for sign, amplitude, color in paths:
            positions, _ = evaluate_path(sign, amplitude, times)
            pixels = [self._world_to_pixel(point) for point in positions]
            pygame.draw.aalines(canvas, color, False, pixels)

    def _draw_canvas(self, trajectory: np.ndarray) -> Any:
        pygame = self._load_pygame()
        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        canvas.fill((248, 249, 252))

        x_low, x_high = self.config.visualization_x
        y_low, y_high = self.config.visualization_y
        bottom_left = self._world_to_pixel(
            np.asarray((x_low, y_low), dtype=np.float32)
        )
        top_right = self._world_to_pixel(
            np.asarray((x_high, y_high), dtype=np.float32)
        )
        plot_rect = pygame.Rect(
            bottom_left[0],
            top_right[1],
            top_right[0] - bottom_left[0],
            bottom_left[1] - top_right[1],
        )
        pygame.draw.rect(canvas, (224, 228, 235), plot_rect, width=1)
        pygame.draw.aaline(
            canvas,
            (218, 222, 229),
            self._world_to_pixel(np.asarray((x_low, 0.0), dtype=np.float32)),
            self._world_to_pixel(np.asarray((x_high, 0.0), dtype=np.float32)),
        )

        self._draw_reference_paths(canvas)

        goal_pixel = self._world_to_pixel(self.config.goal_array())
        start_pixel = self._world_to_pixel(self.config.start_array())
        pixels_per_unit = plot_rect.width / (x_high - x_low)
        goal_radius = max(4, round(self.config.goal_tolerance * pixels_per_unit))
        pygame.draw.circle(canvas, (207, 235, 214), goal_pixel, goal_radius)
        pygame.draw.circle(canvas, (55, 145, 79), goal_pixel, goal_radius, width=2)
        pygame.draw.circle(canvas, (45, 49, 57), start_pixel, 6)

        trajectory_pixels = [
            self._world_to_pixel(point) for point in trajectory
        ]
        if len(trajectory_pixels) >= 2:
            pygame.draw.lines(
                canvas,
                (54, 57, 66),
                False,
                trajectory_pixels,
                width=3,
            )
        pygame.draw.circle(
            canvas,
            (126, 70, 190),
            trajectory_pixels[-1],
            7,
        )
        return canvas

    def render(self, trajectory: np.ndarray, mode: str) -> np.ndarray | None:
        pygame = self._load_pygame()
        canvas = self._draw_canvas(trajectory)
        if mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(canvas),
                axes=(1, 0, 2),
            ).astype(np.uint8, copy=False)

        if self._window is None:
            existing_surface = pygame.display.get_surface()
            if existing_surface is not None:
                raise RuntimeError(
                    "human rendering refuses to replace an existing "
                    "pygame display surface"
                )
            if not pygame.get_init():
                pygame.init()
                self._owns_pygame_init = True
            elif not pygame.display.get_init():
                pygame.display.init()
                self._owns_display_init = True
            else:
                self._restore_display_init = True
            self._window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            self._owns_window = True
            pygame.display.set_caption("PointReach2DPreference-v0")
            self._clock = pygame.time.Clock()
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(32)
        return None

    def close(self) -> None:
        if self._pygame is not None:
            if self._owns_pygame_init:
                self._pygame.quit()
            elif self._owns_display_init:
                self._pygame.display.quit()
            elif self._owns_window:
                self._pygame.display.quit()
                if self._restore_display_init:
                    self._pygame.display.init()
        self._window = None
        self._clock = None
        self._owns_pygame_init = False
        self._owns_display_init = False
        self._owns_window = False
        self._restore_display_init = False
