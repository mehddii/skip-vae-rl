from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class RGBImageObservation(ObservationWrapper):
    """Return rendered RGB frames as HWC uint8 observations."""

    def __init__(self, env: gym.Env, image_size: int = 64):
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8)

    def observation(self, observation: Any) -> np.ndarray:
        del observation
        frame = self.env.unwrapped.render()
        if frame.shape[0] == self.image_size and frame.shape[1] == self.image_size:
            return frame.astype(np.uint8)
        return _resize_nearest(frame, self.image_size).astype(np.uint8)


def _resize_nearest(frame: np.ndarray, image_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    y_idx = (np.linspace(0, h - 1, image_size)).astype(np.int64)
    x_idx = (np.linspace(0, w - 1, image_size)).astype(np.int64)
    return frame[y_idx][:, x_idx]


def make_env(env_id: str, image_size: int, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    env = RGBImageObservation(env, image_size=image_size)
    if seed is not None:
        env.reset(seed=seed)
    return env

