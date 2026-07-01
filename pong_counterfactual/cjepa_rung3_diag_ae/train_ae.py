"""Train (and gate) the small reconstruction autoencoder. CPU-friendly.

Gate A0 (tiny overfit): overfit a fixed tiny subset; recon loss must drop a lot and the
                        recon must TRACK different paddle positions (not collapse to a
                        constant frame). Saves tiny_overfit_metrics.json + gallery.
Gate A1 (small train) : a modest episode-split training run; saves checkpoint, config,
                        reconstruction curves, held-out galleries, paddle/ball crops.

Reconstruction quality alone is NOT declared as success — the probes (Steps 3-5) decide.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3_diag_ae.ae_model import (
    ConvAutoencoder, AEConfig, config_to_dict,
)
from pong_counterfactual.cjepa_rung3_diag_ae import common as CM


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(CM.REPO_ROOT)).decode().strip()
    except Exception:
        return "unknown"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------- paddle localisation
def player_paddle_column(frames_u8: np.ndarray) -> slice:
    """Heuristic column band containing the player paddle, for crop galleries only.
    In this preprocessed Pong the player paddle is the RIGHT vertical bar; we return a
    generous right-side column band (diagnostic visual aid, never used numerically)."""
    res = frames_u8.shape[-1]
    return slice(int(res * 0.72), res)


# ------------------------------------------------------------------- gallery helpers
def recon_gallery(model, frames_u8, py, path, device, title, n=8):
    plt = _plt()
    idx = np.linspace(0, len(frames_u8) - 1, min(n, len(frames_u8))).astype(int)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[idx])).to(device)
        recon, _ = model(x)
    recon = ((recon.squeeze(1).cpu().numpy() + 1) / 2 * 255).clip(0, 255)
    orig = frames_u8[idx].astype(np.float32)
    err = np.abs(orig - recon)
    rows, k = 3, len(idx)
    fig, ax = plt.subplots(rows, k, figsize=(1.6 * k, 5))
    for j in range(k):
        for r, (img, cmap, lab) in enumerate(
                [(orig[j], "gray", "orig"), (recon[j], "gray", "recon"),
                 (err[j], "magma", "|err|")]):
            a = ax[r, j] if k > 1 else ax[r]
            a.imshow(img, cmap=cmap, vmin=0, vmax=255 if r < 2 else err.max())
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"py={py[idx[j]]:.0f}", fontsize=8)
            if j == 0:
                a.set_ylabel(lab, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def crop_gallery(model, frames_u8, py, path, device, col: slice, title, n=8):
    plt = _plt()
    idx = np.linspace(0, len(frames_u8) - 1, min(n, len(frames_u8))).astype(int)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[idx])).to(device)
        recon, _ = model(x)
    recon = ((recon.squeeze(1).cpu().numpy() + 1) / 2 * 255).clip(0, 255)
    orig = frames_u8[idx].astype(np.float32)
    fig, ax = plt.subplots(2, len(idx), figsize=(1.4 * len(idx), 4))
    for j in range(len(idx)):
        for r, img in enumerate([orig[j][:, col], recon[j][:, col]]):
            a = ax[r, j] if len(idx) > 1 else ax[r]
            a.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"py={py[idx[j]]:.0f}", fontsize=8)
            if j == 0:
                a.set_ylabel(["orig", "recon"][r], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def curves_plot(hist, path):
    plt = _plt()
    fig, axx = plt.subplots(figsize=(6, 4))
    axx.plot(hist["step"], hist["train"], label="train L1")
    axx.plot(hist["val_step"], hist["val"], label="val L1", marker="o", ms=3)
    axx.set_xlabel("step"); axx.set_ylabel("recon loss (normalized [-1,1])")
    axx.legend(); axx.grid(alpha=0.3); axx.set_title("AE reconstruction curves")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------- Gate A0
def tiny_overfit(frames_u8, py, out_dir: Path, device="cpu", n_tiny=12, steps=600,
                 seed=0):
    """Overfit a fixed tiny subset chosen to SPAN paddle positions."""
    set_seed(seed)
    valid = ~np.isnan(py)
    fv, pv = frames_u8[valid], py[valid]
    order = np.argsort(pv)
    pick = order[np.linspace(0, len(order) - 1, n_tiny).astype(int)]  # spread over py
    tf, tp = fv[pick], pv[pick]
    cfg = AEConfig()
    model = ConvAutoencoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.from_numpy(np.ascontiguousarray(tf)).to(device)
    model.train()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss, _, _ = model.recon_loss(x)
        loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    # constant-frame check: how much does recon VARY across the tiny set vs the target?
    model.eval()
    with torch.no_grad():
        recon, _ = model(x)
    recon_np = recon.squeeze(1).cpu().numpy()
    tgt = (tf.astype(np.float32) / 255 * 2 - 1)
    recon_var = float(recon_np.var(axis=0).mean())      # spatial variance across samples
    tgt_var = float(tgt.var(axis=0).mean())
    metrics = {
        "n_tiny": int(n_tiny), "steps": int(steps),
        "loss_start": losses[0], "loss_end": losses[-1],
        "loss_drop_ratio": losses[0] / max(losses[-1], 1e-9),
        "recon_var_across_samples": recon_var,
        "target_var_across_samples": tgt_var,
        "recon_to_target_var_ratio": recon_var / max(tgt_var, 1e-9),
        "py_span": [float(tp.min()), float(tp.max())],
    }
    # PASS: loss dropped a lot AND recon is not a near-constant frame (tracks positions).
    metrics["pass"] = bool(metrics["loss_drop_ratio"] > 3.0
                           and metrics["recon_to_target_var_ratio"] > 0.3
                           and metrics["loss_end"] < 0.5 * metrics["loss_start"])
    out_dir.mkdir(parents=True, exist_ok=True)
    CM.write_json(out_dir / "tiny_overfit_metrics.json", metrics)
    recon_gallery(model, tf, tp, out_dir / "tiny_overfit_gallery.png", device,
                  "A0 tiny-overfit: orig / recon / |err| (py spans paddle range)",
                  n=n_tiny)
    return metrics


# ----------------------------------------------------------------------- Gate A1
def train(frames_u8, ep_seed, py, out_dir: Path, cfg: AEConfig, device="cpu",
          val_frac=0.15, log=print):
    set_seed(cfg.seed)
    tr_mask, val_mask = CM.group_train_test_split(ep_seed, test_frac=val_frac, seed=cfg.seed)
    Ftr = frames_u8[tr_mask]
    Fval = frames_u8[val_mask]
    log(f"[A1] AE-train frames {len(Ftr)} / AE-val frames {len(Fval)} "
        f"(episode-grouped, no overlap)")
    model = ConvAutoencoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    hist = {"step": [], "train": [], "val_step": [], "val": []}

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Fval), 256):
                x = torch.from_numpy(np.ascontiguousarray(Fval[i:i + 256])).to(device)
                l, _, _ = model.recon_loss(x)
                tot += float(l) * len(x)
        return tot / len(Fval)

    for step in range(1, cfg.steps + 1):
        model.train()
        idx = rng.integers(0, len(Ftr), cfg.batch)
        x = torch.from_numpy(np.ascontiguousarray(Ftr[idx])).to(device)
        opt.zero_grad()
        loss, _, _ = model.recon_loss(x)
        loss.backward(); opt.step()
        if step % 50 == 0 or step == 1:
            hist["step"].append(step); hist["train"].append(float(loss.detach()))
        if step % cfg.val_every == 0 or step == cfg.steps:
            vl = val_loss()
            hist["val_step"].append(step); hist["val"].append(vl)
            log(f"[A1] step {step:5d}  train {cfg.loss} {float(loss.detach()):.4f}  "
                f"val {cfg.loss} {vl:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "config": config_to_dict(cfg),
        "git_commit": git_commit(),
        "seed": cfg.seed,
        "data_dir": str(CM.DEFAULT_DATA_DIR),
        "n_ae_train_frames": int(len(Ftr)),
        "val_frac": val_frac,
        "final_val_l1": hist["val"][-1],
    }
    torch.save(ckpt, out_dir / "ae_checkpoint.pt")
    (out_dir / "ae_config.yaml").write_text(_yaml(config_to_dict(cfg), ckpt))
    curves_plot(hist, out_dir / "reconstruction_curves.png")
    # held-out galleries (AE-val frames, spanning paddle positions)
    pv = py[val_mask]
    order = np.argsort(np.nan_to_num(pv, nan=-1))
    span = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    recon_gallery(model, Fval[span], pv[span],
                  out_dir / "heldout_reconstruction_gallery.png", device,
                  "A1 held-out: orig / recon / |err|")
    col = player_paddle_column(Fval)
    crop_gallery(model, Fval[span], pv[span], out_dir / "paddle_crop_gallery.png",
                 device, col, "A1 held-out player-paddle crop (right band)")
    crop_gallery(model, Fval[span], pv[span], out_dir / "ball_crop_gallery.png",
                 device, slice(0, int(Fval.shape[-1] * 0.6)),
                 "A1 held-out ball region crop (left/centre band)")
    return model, hist, ckpt


def _yaml(cfg_d: dict, ckpt: dict) -> str:
    lines = ["# AE config (recorded alongside checkpoint)"]
    for k, v in cfg_d.items():
        lines.append(f"{k}: {v}")
    lines += [f"git_commit: {ckpt['git_commit']}", f"seed: {ckpt['seed']}",
              f"data_dir: {ckpt['data_dir']}",
              f"n_ae_train_frames: {ckpt['n_ae_train_frames']}",
              f"final_val_l1: {ckpt['final_val_l1']:.5f}"]
    return "\n".join(lines) + "\n"
