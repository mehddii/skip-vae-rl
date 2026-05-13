from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from tqdm import tqdm

from skip_vae_rl.datasets import FrameDataset, load_or_collect_frames
from skip_vae_rl.models.vae import build_vae, vae_loss
from skip_vae_rl.utils import ensure_dir, get_device, load_config, save_json, set_seed


def save_reconstruction_grid(model, batch, path: Path, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        x = batch[:8].to(device)
        recon, _, _ = model(x)
    grid = make_grid(torch.cat([x.cpu(), recon.cpu()], dim=0), nrow=8)
    ensure_dir(path.parent)
    plt.figure(figsize=(12, 3))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    run_dir = ensure_dir(cfg["run_dir"])
    fig_dir = ensure_dir(cfg.get("fig_dir", "reports/figures"))

    frames = load_or_collect_frames(
        path=cfg["dataset"]["path"],
        env_id=cfg["env"]["id"],
        image_size=int(cfg["env"]["image_size"]),
        num_frames=int(cfg["dataset"]["num_frames"]),
        seed=int(cfg["seed"]),
        collect_if_missing=bool(cfg["dataset"].get("collect_if_missing", True)),
    )
    dataset = FrameDataset(frames)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_vae(
        model_type=cfg["model"]["type"],
        in_channels=int(cfg["model"]["in_channels"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        skip_dropout=float(cfg["model"].get("skip_dropout", 0.0)),
        skip_scale=float(cfg["model"].get("skip_scale", 1.0)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    beta = float(cfg["model"].get("beta", 1.0))

    best_val = float("inf")
    metrics: list[dict[str, float]] = []
    example_batch = next(iter(val_loader))

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        for x in tqdm(train_loader, desc=f"epoch {epoch} train"):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"epoch {epoch} val"):
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)
                val_loss += float(loss.item()) * x.shape[0]
                val_recon += float(recon_loss.item()) * x.shape[0]
                val_kl += float(kl.item()) * x.shape[0]

        val_loss /= len(val_ds)
        val_recon /= len(val_ds)
        val_kl /= len(val_ds)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
        }
        metrics.append(row)
        print(row)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, run_dir / "best.pt")

        if epoch % int(cfg["training"].get("save_every_epochs", 5)) == 0:
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, run_dir / f"epoch_{epoch}.pt")
            save_reconstruction_grid(
                model,
                example_batch,
                fig_dir / f"recon_{Path(cfg['run_dir']).name}_epoch_{epoch}.png",
                device,
            )

    save_json(run_dir / "metrics.json", {"metrics": metrics, "best_val": best_val})
    save_reconstruction_grid(model, example_batch, fig_dir / f"recon_{Path(cfg['run_dir']).name}_final.png", device)


if __name__ == "__main__":
    main()
