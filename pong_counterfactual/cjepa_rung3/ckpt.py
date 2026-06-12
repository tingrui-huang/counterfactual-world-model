"""Resume-safe checkpointing to Drive (the Colab contract).

Rules implemented here (skill, non-negotiable):
  - save every N steps AND at the end; payload = model + optimizer + step + RNG
    states + config;
  - keep the last k checkpoints (Drive writes can be interrupted mid-write — never
    overwrite the only copy); writes go to a temp file then rename;
  - on startup, load the NEWEST checkpoint that unpickles cleanly (a torn write of
    the latest falls back to the previous one).
"""
import re
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state()}


def restore_rng(st):
    torch.set_rng_state(st["torch"].cpu() if torch.is_tensor(st["torch"]) else st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError:
            pass                       # device count changed across runtimes — fine
    np.random.set_state(st["numpy"])


def save(ckpt_dir: Path, step: int, payload: dict, keep: int = 3):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, step=step, rng=rng_state())
    tmp = ckpt_dir / f".tmp_step{step:08d}.pt"
    torch.save(payload, tmp)
    tmp.replace(ckpt_dir / f"ckpt_step{step:08d}.pt")
    olds = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    for p in olds[:-keep]:
        p.unlink(missing_ok=True)


def load_latest(ckpt_dir: Path, map_location="cpu"):
    """Newest checkpoint that loads cleanly, or None. Returns (payload, path)."""
    if not ckpt_dir.exists():
        return None, None
    for p in sorted(ckpt_dir.glob("ckpt_step*.pt"), reverse=True):
        try:
            return torch.load(p, map_location=map_location, weights_only=False), p
        except Exception as e:                      # torn Drive write -> try older
            print(f"[ckpt] {p.name} unreadable ({e}); falling back")
    return None, None


def step_of(path: Path) -> int:
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1
