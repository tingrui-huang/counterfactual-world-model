"""Stage 1 — pretrain the eye (GPU; the only heavy step). Checkpoint-resumable.

Trains the Discrete-JEPA tokenizer on the Stage-0 TRAIN-split frames at s=S_TRAIN
(the eye never sees the eval episodes — the Stage-2 probes stay honest). Pixels only;
the stored ground-truth 4-vectors are not even loaded here.

Colab contract: checkpoints (model + opt + step + RNG + config) to Drive every
ckpt_every steps AND at the end, keep last k; on startup RESUME from the newest
readable checkpoint. A fresh runtime loses at most ckpt_every steps.

Run:  python -m pong_counterfactual.cjepa_rung3.stage1_pretrain [--smoke]
      [--name M4_K64_res84] [--M 4] [--K 64] [--steps 30000]
Every (M, K, res) config gets its own tokenizer_<name>/ directory — the Stage-2 gate
log accumulates one row per config tried (the Problem-B table).
"""
import argparse
import dataclasses
import json
import math
import time

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3 import ckpt as CK
from pong_counterfactual.cjepa_rung3.data import load_split
from pong_counterfactual.cjepa_rung3.tokenizer import DiscreteJEPA


def load_train_frames(n_eps):
    eps = load_split(C.data_dir(C.S_TRAIN), "train", C.TRAIN_BASE_SEED, n_eps)
    frames = np.concatenate([ep.frames for ep in eps])      # (N,res,res) uint8
    return frames


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * step / max(1, cfg.warmup)
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def train(cfg: C.TokenizerConfig, n_train_eps: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = C.tok_dir(cfg.name)
    ckpt_dir = out_dir / "ckpts"
    log_path = out_dir / "train_log.jsonl"
    print(f"[stage1] config={cfg.name} device={device} -> {out_dir}")

    frames = load_train_frames(n_train_eps)
    print(f"[stage1] {len(frames)} train frames @ {frames.shape[-1]}px")
    frames_t = torch.from_numpy(frames)            # stays uint8 in RAM

    torch.manual_seed(cfg.seed)
    model = DiscreteJEPA(cfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    gen = torch.Generator().manual_seed(cfg.seed)   # CPU gen for mask sampling
    samp = np.random.default_rng(cfg.seed)

    start = 0
    payload, path = CK.load_latest(ckpt_dir, map_location=device)
    if payload is not None:
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        scaler.load_state_dict(payload["scaler"])
        gen.set_state(payload["gen"])
        samp.bit_generator.state = payload["samp"]
        CK.restore_rng(payload["rng"])
        start = payload["step"]
        print(f"[stage1] RESUMED from {path.name} at step {start}")

    def save(step):
        CK.save(ckpt_dir, step, {
            "model": model.state_dict(), "opt": opt.state_dict(),
            "scaler": scaler.state_dict(), "gen": gen.get_state(),
            "samp": samp.bit_generator.state,
            "config": dataclasses.asdict(cfg)}, keep=cfg.keep_ckpts)

    model.train()
    t0 = time.time()
    for step in range(start, cfg.steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = samp.integers(0, len(frames_t), cfg.batch)
        img = (frames_t[idx].to(device).float().div(255.0).mul(2).sub(1)
               .unsqueeze(1))
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            loss, parts = model.loss(img, gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        m = cfg.ema_lo + (cfg.ema_hi - cfg.ema_lo) * (step + 1) / cfg.steps
        model.ema_update(m)

        if (step + 1) % 50 == 0 or step == start:
            row = {"step": step + 1, "loss": float(loss.detach()), **parts, "lr": lr,
                   "sec": round(time.time() - t0, 1)}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[stage1] {row}")
        if (step + 1) % cfg.ckpt_every == 0:
            save(step + 1)

    save(cfg.steps)
    final = out_dir / "final.pt"
    tmp = out_dir / ".tmp_final.pt"
    torch.save({"model": model.state_dict(),
                "config": dataclasses.asdict(cfg)}, tmp)
    tmp.replace(final)
    print(f"[stage1] done -> {final}")


def parse_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    n_eps = C.N_TRAIN_EPS
    if args.smoke:
        o = C.smoke_overrides()
        cfg, n_eps = o["tok"], o["n_train_eps"]
    else:
        cfg = C.TokenizerConfig()
    for k in ("M", "K", "steps"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.name:
        cfg.name = args.name
    elif not args.smoke:
        cfg.name = f"M{cfg.M}_K{cfg.K}_res{cfg.res}"
    return cfg, n_eps


if __name__ == "__main__":
    train(*parse_cfg())
