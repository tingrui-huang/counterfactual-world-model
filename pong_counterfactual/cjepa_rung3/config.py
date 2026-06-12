"""Rung 3 configuration — the six locked decisions + Colab/Drive path resolution.

Decisions (skill defaults, all "sane default, cheap to change"):
  ① frame format : grayscale, crop rows [34,194) (the 160x160 play area), resize 84x84.
                   ⚠ the resize is itself a coarsening: the paddle's 4.59 px/step
                   becomes ~2.41 px. If the Stage-2 collision gate FAILS, this is the
                   FIRST suspect -> set RES=128 (or 160 = no resize) and retrain.
  ② code base    : self-contained Discrete-JEPA assembly in tokenizer.py (I-JEPA-style
                   context/EMA-target/predictor + VQ; Discrete-JEPA has no public
                   code). DEVIATION from the skill's "pull eb_jepa" default, reasoned:
                   a pinned external repo inside a Colab notebook is a fragile moving
                   dependency, and at this scale (a few-M-param ViT on 84x84 frames)
                   the recipe is small enough to own outright — every knob the gate
                   may need to turn (M, K, mask ratio, loss weights) is then in ONE
                   file. The training recipe follows eb_jepa's structure (context
                   encoder, EMA target, predictor, latent L2, no reconstruction).
  ③ slots/global : v1 = GLOBAL semantic tokens (frame -> M discrete tokens).
                   Object-centric slots stay in reserve as the positive control if
                   the global gate cannot be made to pass.
  ④ shape        : M = 4 tokens/frame, codebook K = 64. PRIMARY gate knobs
                   (then mask ratio, loss weights, then ① resolution).
  ⑤ transition   : FACTORIZED — M independent K-way categoricals (one softmax per
                   slot), mirroring Rung 1/2's "4 dims, each its own categorical",
                   so gumbel_abduction.py applies per slot with zero change.
  ⑥ scoring      : DUAL. Primary = token space (encode the oracle CF frame through
                   the frozen eye, compare tokens). Secondary = physical px via the
                   Stage-2 linear probe (comparable to Rungs 1-2.5).

Environment / data settings are inherited from Rung 2 (fs=2, sticky s, same seeds)
so the cached oracle quantities line up with Rung 2.5's.

Path resolution (Colab-first):
  CJEPA_RUNG3_ROOT env var          -> use it (local runs / tests)
  /content/drive exists (Colab)     -> /content/drive/MyDrive/cjepa_rung3
  otherwise                         -> <this package>/artifacts   (local fallback)
Every stage LOADS its inputs from this root and WRITES its outputs back to it, so
stages survive runtime disconnects independently.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths (Colab/Drive)
def root() -> Path:
    env = os.environ.get("CJEPA_RUNG3_ROOT")
    if env:
        p = Path(env)
    elif Path("/content/drive").exists():
        p = Path("/content/drive/MyDrive/cjepa_rung3")
    else:
        p = Path(__file__).resolve().parent / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(s: float) -> Path:
    # resolution-tagged: a dataset preprocessed at a different FrameSpec.res is a
    # DIFFERENT dataset (different frames + oracle frame-cache). Tagging lets the
    # 84px and 128px datasets coexist on Drive, each resume-safe, without clobbering.
    d = root() / f"data_s{s:g}_res{FRAME.res}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tok_dir(name: str) -> Path:
    d = root() / f"tokenizer_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trans_dir(name: str, s: float) -> Path:
    d = root() / f"transition_{name}_s{s:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- env / collection (R2)
FRAMESKIP = 2          # Rung 2 value (fs=1 has a 1-frame paddle latency -> degenerate)
S_TRAIN = 0.5          # the sticky level the eye + headline run at (Rung 2's clearest)
SS_EVAL = (0.0, 0.5)   # 0.0 = collapse control (check A/C); 0.5 = headline
N_TRAIN_EPS = 40
N_EVAL_EPS = 14
T = 250
TRAIN_BASE_SEED, TRAIN_POLICY_SEED = 0, 1       # Rung 2's exact seeds
EVAL_BASE_SEED, EVAL_POLICY_SEED = 1000, 2      # disjoint
N_HIST = 8             # history window (load-bearing under sticky; Rung 2 value)
N_SAMPLES = 250        # fixed eval tuples per s (rng(7) subsample, as Rung 2/2.5)

# ------------------------------------------------------------------- ① frame format
@dataclass(frozen=True)
class FrameSpec:
    crop_top: int = 34       # ALE Pong: rows [34,194) = the 160x160 play area
    crop_bottom: int = 194
    res: int = 128           # output H = W. Bumped 84->128 (decision ①, the first
                             # suspect): at 84px the paddle's 4.59 px/step shrank to
                             # 2.41 resized px and the M4/K64 eye localized the paddle
                             # only to ~4 px (linear-probe gate fail); 128px gives the
                             # eye more pixels ON the paddle. NOTE the gate is
                             # scale-invariant (both probe MAE and the threshold scale
                             # with res), so this is no free pass — it helps only if
                             # more input pixels genuinely sharpen localization.

    @property
    def scale(self) -> float:        # exact resized-px per screen-px
        return self.res / (self.crop_bottom - self.crop_top)

    def to_resized(self, vec4):
        """(ball_x, ball_y, player_y, enemy_y) screen px -> resized px (exact)."""
        import numpy as np
        v = np.asarray(vec4, dtype=np.float64).copy()
        v[0] = v[0] * self.scale                          # x: full 160 width kept
        for i in (1, 2, 3):                               # y dims: crop then scale
            v[i] = (v[i] - self.crop_top) * self.scale
        return v

FRAME = FrameSpec()
# paddle step: 4.586 px/screen (measured at Rung 2.5) -> in resized units:
PADDLE_STEP_SCREEN_PX = 4.586
def paddle_step_resized() -> float:
    return PADDLE_STEP_SCREEN_PX * FRAME.scale

# ---------------------------------------------------------------- ④ tokenizer shape
@dataclass
class TokenizerConfig:
    name: str = "M4_K64_res128"
    M: int = 4               # semantic tokens per frame        (primary knob 1)
    K: int = 64              # codebook size                    (primary knob 2)
    res: int = 128           # must match FrameSpec.res
    patch: int = 16          # 128/16 -> 8x8 = 64 patches (84/12 didn't divide 128)
    dim: int = 128
    depth: int = 4
    heads: int = 4
    pred_depth: int = 2
    code_dim: int = 32       # VQ in a projected low-dim space (better usage)
    vis_lo: float = 0.5      # visible-patch ratio range during training; hi=1.0 so
    vis_hi: float = 1.0      # full-frame inference is in-distribution
    w_s2p: float = 1.0       # semantic->patch loss (forces frame content into tokens)
    w_p2p: float = 1.0       # standard I-JEPA masked patch->patch loss
    beta_commit: float = 0.25
    vq_decay: float = 0.99
    dead_code_steps: int = 200   # reinit codes unused this many steps
    # training
    steps: int = 30_000
    batch: int = 256
    lr: float = 1.5e-4
    weight_decay: float = 0.04
    warmup: int = 1_000
    ema_lo: float = 0.996    # target-encoder momentum schedule ema_lo -> ema_hi
    ema_hi: float = 1.0
    ckpt_every: int = 500
    keep_ckpts: int = 3
    seed: int = 0

# ------------------------------------------------------------- ⑤ transition model
@dataclass
class TransitionConfig:
    name: str = "mlp512"
    hidden: int = 512
    steps: int = 4_000
    batch: int = 256
    lr: float = 1e-3
    ckpt_every: int = 500
    keep_ckpts: int = 3
    seed: int = 0

# ----------------------------------------------------------------- Stage-2 gate rule
COLLISION_MAX = 0.25     # Rung 2.5 knee yardstick: c=4px had collision 0.246 and the
                         # CF advantage already degenerating; the eye must sit on the
                         # GOOD side of the knee.
PROBE_PY_MAX_FACTOR = 1.0  # probe player_y MAE must be < 1.0 * paddle step (resized)

# --------------------------------------------------------------------- smoke mode
def smoke_overrides():
    """Tiny end-to-end settings to validate plumbing on CPU (not science)."""
    return dict(n_train_eps=4, n_eval_eps=2, T=60, n_samples=12,
                tok=TokenizerConfig(name="smoke_M4_K16", M=4, K=16, dim=64, depth=2,
                                    heads=2, pred_depth=1, code_dim=16, steps=40,
                                    batch=32, warmup=5, ckpt_every=20, keep_ckpts=2),
                trans=TransitionConfig(name="smoke", hidden=128, steps=60, batch=64,
                                       ckpt_every=30))
