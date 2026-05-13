from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from torch import nn

from skip_vae_rl.envs import make_env
from skip_vae_rl.models.vae import build_vae
from skip_vae_rl.utils import ensure_dir, load_config, save_json, set_seed


class VAEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        checkpoint: str,
        model_type: str,
        latent_dim: int,
        frozen: bool,
        skip_dropout: float = 0.0,
        skip_scale: float = 1.0,
    ):
        super().__init__(observation_space, features_dim=latent_dim)
        payload = torch.load(checkpoint, map_location="cpu")
        model_cfg = payload.get("config", {}).get("model", {})
        self.vae = build_vae(
            model_type=model_type,
            in_channels=3,
            latent_dim=latent_dim,
            skip_dropout=float(model_cfg.get("skip_dropout", skip_dropout)),
            skip_scale=float(model_cfg.get("skip_scale", skip_scale)),
        )
        self.vae.load_state_dict(payload["model"])
        self.vae.decoder = nn.Identity()
        if frozen:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        self.frozen = frozen

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        if self.frozen:
            with torch.no_grad():
                return self.vae.encode_features(x)
        return self.vae.encode_features(x)


def build_vec_env(env_id: str, image_size: int, seed: int) -> VecTransposeImage:
    def thunk():
        return Monitor(make_env(env_id, image_size=image_size, seed=seed))

    return VecTransposeImage(DummyVecEnv([thunk]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    run_dir = ensure_dir(cfg["run_dir"])

    env = build_vec_env(cfg["env"]["id"], int(cfg["env"]["image_size"]), seed)
    eval_env = build_vec_env(cfg["env"]["id"], int(cfg["env"]["image_size"]), seed + 1000)

    encoder_cfg = cfg["encoder"]
    policy_kwargs: dict[str, Any] = {}
    if encoder_cfg["mode"] == "pretrained":
        policy_kwargs = {
            "features_extractor_class": VAEFeatureExtractor,
            "features_extractor_kwargs": {
                "checkpoint": encoder_cfg["checkpoint"],
                "model_type": encoder_cfg["model_type"],
                "latent_dim": int(encoder_cfg["latent_dim"]),
                "frozen": bool(encoder_cfg.get("frozen", True)),
                "skip_dropout": float(encoder_cfg.get("skip_dropout", 0.0)),
                "skip_scale": float(encoder_cfg.get("skip_scale", 1.0)),
            },
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        }
        policy: str | type[ActorCriticPolicy] = "CnnPolicy"
    elif encoder_cfg["mode"] == "raw":
        policy = "CnnPolicy"
    else:
        raise ValueError(f"Unknown encoder mode: {encoder_cfg['mode']}")

    model = PPO(
        policy,
        env,
        learning_rate=float(cfg["rl"]["learning_rate"]),
        n_steps=int(cfg["rl"]["n_steps"]),
        batch_size=int(cfg["rl"]["batch_size"]),
        n_epochs=int(cfg["rl"]["n_epochs"]),
        gamma=float(cfg["rl"]["gamma"]),
        seed=seed,
        tensorboard_log=str(run_dir / "tb"),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    model.learn(total_timesteps=int(cfg["rl"]["total_timesteps"]), progress_bar=True)
    model.save(run_dir / "ppo_model")

    eval_seeds = cfg["rl"].get("eval_seeds")
    if eval_seeds is None:
        eval_seeds = [seed + 1000 + i for i in range(int(cfg["rl"].get("num_eval_seeds", 5)))]

    rewards = []
    lengths = []
    per_seed = []
    episodes_per_seed = int(cfg["rl"].get("eval_episodes_per_seed", cfg["rl"].get("eval_episodes", 5)))
    for eval_seed in eval_seeds:
        eval_env.close()
        eval_env = build_vec_env(cfg["env"]["id"], int(cfg["env"]["image_size"]), int(eval_seed))
        seed_rewards, seed_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=episodes_per_seed,
            return_episode_rewards=True,
        )
        rewards.extend(seed_rewards)
        lengths.extend(seed_lengths)
        per_seed.append(
            {
                "seed": int(eval_seed),
                "mean_reward": float(np.mean(seed_rewards)),
                "std_reward": float(np.std(seed_rewards)),
                "mean_length": float(np.mean(seed_lengths)),
                "rewards": [float(r) for r in seed_rewards],
                "lengths": [int(l) for l in seed_lengths],
            }
        )

    eval_payload = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "num_eval_seeds": len(eval_seeds),
        "episodes_per_seed": episodes_per_seed,
        "rewards": [float(r) for r in rewards],
        "lengths": [int(l) for l in lengths],
        "per_seed": per_seed,
    }
    save_json(Path(run_dir) / "eval.json", eval_payload)
    print(eval_payload)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
