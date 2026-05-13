from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from skip_vae_rl.envs import make_env
from skip_vae_rl.utils import ensure_dir


class FrameDataset(Dataset):
    def __init__(self, frames: np.ndarray):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        x = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return x


def collect_random_frames(
    path: str | Path,
    env_id: str,
    image_size: int,
    num_frames: int,
    seed: int,
) -> np.ndarray:
    path = Path(path)
    ensure_dir(path.parent)
    env = make_env(env_id, image_size=image_size, seed=seed)
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(num_frames), desc="collecting frames"):
        frames.append(obs)
        if hasattr(env.action_space, "n"):
            action = int(rng.integers(env.action_space.n))
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    arr = np.stack(frames, axis=0).astype(np.uint8)
    np.savez_compressed(path, frames=arr)
    return arr


def load_or_collect_frames(
    path: str | Path,
    env_id: str,
    image_size: int,
    num_frames: int,
    seed: int,
    collect_if_missing: bool,
) -> np.ndarray:
    path = Path(path)
    if path.exists():
        data = np.load(path)
        return data["frames"]
    if not collect_if_missing:
        raise FileNotFoundError(f"Dataset not found: {path}")
    return collect_random_frames(path, env_id, image_size, num_frames, seed)
