from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from skip_vae_rl.datasets import FrameDataset, load_or_collect_frames
from skip_vae_rl.models.vae import build_vae
from skip_vae_rl.utils import ensure_dir, get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    fig_dir = ensure_dir(cfg.get("fig_dir", "reports/figures"))

    frames = load_or_collect_frames(
        path=cfg["dataset"]["path"],
        env_id=cfg["env"]["id"],
        image_size=int(cfg["env"]["image_size"]),
        num_frames=int(cfg["dataset"]["num_frames"]),
        seed=int(cfg["seed"]),
        collect_if_missing=bool(cfg["dataset"].get("collect_if_missing", False)),
    )
    dataset = FrameDataset(frames)
    num_points = min(int(cfg["visualization"]["num_points"]), len(dataset))
    subset = Subset(dataset, list(range(num_points)))
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=2)

    model = build_vae(
        model_type=cfg["model"]["type"],
        in_channels=int(cfg["model"]["in_channels"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        skip_dropout=float(cfg["model"].get("skip_dropout", 0.0)),
        skip_scale=float(cfg["model"].get("skip_scale", 1.0)),
    ).to(device)
    payload = torch.load(cfg["visualization"]["checkpoint"], map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    latents = []
    with torch.no_grad():
        for x in tqdm(loader, desc="encoding"):
            z = model.encode_features(x.to(device))
            latents.append(z.cpu().numpy())
    z_np = np.concatenate(latents, axis=0)

    method = cfg["visualization"].get("method", "pca")
    if method == "pca":
        points = PCA(n_components=2).fit_transform(z_np)
    elif method == "tsne":
        points = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(z_np)
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    color = np.linspace(0, 1, len(points))
    plt.figure(figsize=(7, 6))
    plt.scatter(points[:, 0], points[:, 1], c=color, s=8, cmap="viridis", alpha=0.8)
    plt.title(f"Latent space ({method.upper()})")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    out = fig_dir / f"latent_{Path(cfg['run_dir']).name}_{method}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
