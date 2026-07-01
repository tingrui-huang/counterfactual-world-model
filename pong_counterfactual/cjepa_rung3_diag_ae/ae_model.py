"""A deliberately small convolutional autoencoder with a SPATIAL latent.

Design choices (kept minimal on purpose — this is a diagnostic baseline, not an
architecture search):
  * input  : (B,1,84,84) grayscale in [-1,1] (same normalization the JEPA eye uses).
  * encoder: 4 stride-2 convs 84->42->21->11->6, channels 32,64,64,C_lat.
             latent = (B, C_lat, 6, 6) SPATIAL feature map (NOT a global vector) so
             paddle POSITION can survive spatially — the whole point of the probe.
  * decoder: mirror ConvTranspose stack back to 84x84, tanh output in [-1,1].
  * loss   : reconstruction only (MSE default; L1 selectable). No VQ, no perceptual /
             temporal / object terms.

    Objective note (evidence-based): the spec preferred "L1 initially", but a controlled
    tiny-overfit test (A0) showed L1 MEAN-COLLAPSES the sparse bright paddle — on 12 fixed
    frames L1 needs >4000 steps to reproduce the paddle (recon/target var ratio 0.78),
    whereas MSE reproduces it in <1500 steps (ratio 0.997). Squared error up-weights the
    high-contrast paddle pixels, which is exactly what a fair paddle-position probe needs.
    We therefore default to MSE so the reconstruction baseline gets its best fair shot;
    L1 remains available (loss="l1") and the collapse is documented, not hidden.

The encoder learns from pixels + reconstruction alone; ground-truth positions are used
only to EVALUATE the frozen latent, never to train it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class AEConfig:
    res: int = 84
    lat_ch: int = 16          # latent channels -> latent is (lat_ch, 6, 6) = 576 dims
    base: int = 32            # first conv width
    loss: str = "mse"         # "mse" (default, captures sparse paddle) or "l1"
    seed: int = 0
    # training
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch: int = 128
    steps: int = 4000
    val_every: int = 200
    name: str = "convAE_mse_lat16"


def _norm(frames_u8: torch.Tensor) -> torch.Tensor:
    """(B,res,res) uint8 -> (B,1,res,res) float in [-1,1] (JEPA-identical scaling)."""
    return frames_u8.float().div(255.0).mul(2).sub(1).unsqueeze(1)


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        assert cfg.res == 84, "encoder geometry is fixed for res84 (matches the JEPA eye)"
        self.cfg = cfg
        b = cfg.base
        self.enc = nn.Sequential(
            nn.Conv2d(1, b, 3, stride=2, padding=1),  nn.ReLU(inplace=True),   # 84->42
            nn.Conv2d(b, 2 * b, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 42->21
            nn.Conv2d(2 * b, 2 * b, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 21->11
            nn.Conv2d(2 * b, cfg.lat_ch, 3, stride=2, padding=1),                 # 11->6
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(cfg.lat_ch, 2 * b, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),                                                # 6->11
            nn.ConvTranspose2d(2 * b, 2 * b, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),                                                # 11->21
            nn.ConvTranspose2d(2 * b, b, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),                                                # 21->42
            nn.ConvTranspose2d(b, 1, 3, stride=2, padding=1, output_padding=1),   # 42->84
            nn.Tanh(),
        )

    def encode(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """(B,res,res) uint8 -> (B,lat_ch,6,6) float latent (the frozen representation)."""
        return self.enc(_norm(frames_u8))

    def forward(self, frames_u8: torch.Tensor):
        z = self.encode(frames_u8)
        recon = self.dec(z)                       # (B,1,84,84) in [-1,1]
        return recon, z

    def recon_loss(self, frames_u8: torch.Tensor):
        recon, z = self.forward(frames_u8)
        target = _norm(frames_u8)
        if self.cfg.loss == "l1":
            loss = (recon - target).abs().mean()
        else:
            loss = ((recon - target) ** 2).mean()   # MSE (default)
        return loss, recon, z

    @torch.no_grad()
    def encode_numpy(self, frames_u8, device="cpu", batch=256, flatten=True):
        """numpy (N,res,res) uint8 -> numpy latent. flatten=True -> (N, lat_ch*6*6)."""
        import numpy as np
        self.eval()
        out = []
        for i in range(0, len(frames_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch])).to(device)
            z = self.encode(x)
            out.append((z.reshape(z.shape[0], -1) if flatten else z).cpu().numpy())
        if not len(frames_u8):
            d = self.cfg.lat_ch * 6 * 6
            return np.zeros((0, d), np.float32)
        return np.concatenate(out)


def latent_dim(cfg: AEConfig) -> int:
    return cfg.lat_ch * 6 * 6


def config_to_dict(cfg: AEConfig) -> dict:
    return asdict(cfg)
