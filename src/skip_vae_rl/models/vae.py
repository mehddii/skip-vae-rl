from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        e1 = self.c1(x)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        flat = e4.flatten(1)
        return self.fc_mu(flat), self.fc_logvar(flat), [e1, e2, e3]


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar, _ = self.encoder(x)
        return mu, logvar

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        del skips
        h = self.fc_decode(z).view(z.shape[0], 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar


class SkipConvVAE(ConvVAE):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        skip_dropout: float = 0.0,
        skip_scale: float = 1.0,
    ):
        super().__init__(in_channels=in_channels, latent_dim=latent_dim)
        self.skip_dropout = skip_dropout
        self.skip_scale = skip_scale
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.fuse1 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.fuse2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.fuse3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1)

    def regularize_skip(self, skip: torch.Tensor) -> torch.Tensor:
        skip = skip * self.skip_scale
        if self.training and self.skip_dropout > 0:
            skip = F.dropout2d(skip, p=self.skip_dropout, training=True)
        return skip

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        if skips is None:
            raise ValueError("SkipConvVAE.decode requires encoder skip features")
        e1, e2, e3 = skips
        e1 = self.regularize_skip(e1)
        e2 = self.regularize_skip(e2)
        e3 = self.regularize_skip(e3)
        h = self.fc_decode(z).view(z.shape[0], 256, 4, 4)
        h = F.relu(self.up1(h), inplace=True)
        h = self.fuse1(torch.cat([h, e3], dim=1))
        h = F.relu(self.up2(h), inplace=True)
        h = self.fuse2(torch.cat([h, e2], dim=1))
        h = F.relu(self.up3(h), inplace=True)
        h = self.fuse3(torch.cat([h, e1], dim=1))
        return torch.sigmoid(self.up4(h))


def build_vae(
    model_type: str,
    in_channels: int,
    latent_dim: int,
    skip_dropout: float = 0.0,
    skip_scale: float = 1.0,
) -> ConvVAE:
    if model_type == "vae":
        return ConvVAE(in_channels=in_channels, latent_dim=latent_dim)
    if model_type == "skip_vae":
        return SkipConvVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            skip_dropout=skip_dropout,
            skip_scale=skip_scale,
        )
    raise ValueError(f"Unknown VAE model type: {model_type}")


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl
