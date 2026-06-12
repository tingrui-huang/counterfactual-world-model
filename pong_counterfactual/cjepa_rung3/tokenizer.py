"""Decision ② — the Discrete-JEPA tokenizer, assembled (no public code exists):
I-JEPA-style latent predictive coding + a VQ codebook on the semantic representation.

Architecture (decisions ③④: GLOBAL semantic tokens, M per frame, codebook K):
  context encoder  : small ViT over a random VISIBLE subset of patches, with M
                     learnable [SEM] summary tokens appended; the SEM outputs are the
                     frame's semantic representation.
  VQ codebook      : the M SEM vectors are projected to code_dim and quantized
                     against a shared K-entry EMA codebook -> M discrete tokens.
                     Straight-through estimator + commitment loss; dead codes are
                     re-seeded from batch vectors (collapse mitigation).
  target encoder   : EMA copy of the context encoder, sees the FULL frame; its patch
                     outputs (LayerNorm'd, no grad) are the prediction targets.
  S2P predictor    : predicts ALL target patch latents from the M QUANTIZED semantic
                     tokens alone — the loss that forces frame content THROUGH the
                     discrete bottleneck (Discrete-JEPA's semantics->patch objective).
  P2P predictor    : standard I-JEPA — predicts target latents at MASKED positions
                     from the visible context patches (representation shaping).

Loss = w_s2p * L2(S2P, targets) + w_p2p * L2(P2P, masked targets) + beta * commit.
NO reconstruction loss anywhere — the eye is predict-only (the whole Problem-B point:
does a predictability-driven abstraction keep the unpredictable noise signal?).

encode(frames) -> (B, M) int64 code indices: full visibility, deterministic. This is
the ONLY interface downstream stages may use (the 4 numbers never enter here).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pong_counterfactual.cjepa_rung3.config import TokenizerConfig


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    """ViT over (a visible subset of) patches + M [SEM] summary tokens."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.P = (cfg.res // cfg.patch) ** 2
        self.patch_embed = nn.Conv2d(1, cfg.dim, cfg.patch, stride=cfg.patch)
        self.pos = nn.Parameter(torch.randn(1, self.P, cfg.dim) * 0.02)
        self.sem = nn.Parameter(torch.randn(1, cfg.M, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads)
                                    for _ in range(cfg.depth))
        self.ln = nn.LayerNorm(cfg.dim)

    def forward(self, img, vis_idx=None):
        """img (B,1,res,res) in [-1,1]; vis_idx (B,n_vis) or None (= full frame).
        Returns (patch_out (B,n_vis|P,dim), sem_out (B,M,dim))."""
        B = img.shape[0]
        x = self.patch_embed(img).flatten(2).transpose(1, 2) + self.pos  # (B,P,dim)
        if vis_idx is not None:
            x = torch.gather(x, 1, vis_idx[..., None].expand(-1, -1, x.shape[-1]))
        x = torch.cat([x, self.sem.expand(B, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return x[:, :-self.cfg.M], x[:, -self.cfg.M:]


class VectorQuantizerEMA(nn.Module):
    """Shared K-entry EMA codebook in a projected code_dim space, with dead-code
    re-seeding. Buffers travel inside state_dict -> checkpoint/resume safe."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.dim, cfg.code_dim)
        self.proj_out = nn.Linear(cfg.code_dim, cfg.dim)
        self.register_buffer("codebook", torch.randn(cfg.K, cfg.code_dim))
        self.register_buffer("ema_count", torch.ones(cfg.K))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.register_buffer("last_used", torch.zeros(cfg.K, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def quantize_indices(self, z):
        d = (z.pow(2).sum(-1, keepdim=True) - 2 * z @ self.codebook.T
             + self.codebook.pow(2).sum(-1))
        return d.argmin(-1)

    def forward(self, sem):
        """sem (B,M,dim) -> (quantized (B,M,dim), indices (B,M), commit loss,
        perplexity). EMA codebook update happens here when training."""
        z = self.proj_in(sem)                                  # (B,M,cd)
        flat = z.reshape(-1, self.cfg.code_dim)
        idx = self.quantize_indices(flat)                      # (B*M,)
        e = self.codebook[idx]
        if self.training:
            with torch.no_grad():
                self.step += 1
                onehot = F.one_hot(idx, self.cfg.K).float()
                cnt = onehot.sum(0)
                self.ema_count.mul_(self.cfg.vq_decay).add_(cnt, alpha=1 - self.cfg.vq_decay)
                self.ema_sum.mul_(self.cfg.vq_decay).add_(onehot.T @ flat,
                                                          alpha=1 - self.cfg.vq_decay)
                # Laplace-smoothed centroid update
                ncnt = (self.ema_count + 1e-5) / (self.ema_count.sum() + self.cfg.K * 1e-5) \
                       * self.ema_count.sum()
                self.codebook.copy_(self.ema_sum / ncnt[:, None])
                self.last_used[idx.unique()] = self.step
                # dead-code re-seed: codes unused for dead_code_steps -> random batch z
                dead = (self.step - self.last_used) > self.cfg.dead_code_steps
                if dead.any():
                    pick = torch.randint(0, flat.shape[0], (int(dead.sum()),),
                                         device=flat.device)
                    self.codebook[dead] = flat[pick]
                    self.ema_sum[dead] = flat[pick]
                    self.ema_count[dead] = 1.0
                    self.last_used[dead] = self.step
        commit = F.mse_loss(flat, e.detach())
        zq = flat + (e - flat).detach()                        # straight-through
        with torch.no_grad():
            p = F.one_hot(idx, self.cfg.K).float().mean(0)
            perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
        B = sem.shape[0]
        return (self.proj_out(zq).view(B, self.cfg.M, self.cfg.dim),
                idx.view(B, self.cfg.M), commit, perplexity)


class Predictor(nn.Module):
    """Small transformer: [conditioning tokens] + [pos-embedded queries] -> latents
    at the query positions. Used twice (S2P: cond = quantized SEM; P2P: cond =
    visible context patches)."""

    def __init__(self, cfg: TokenizerConfig, P):
        super().__init__()
        self.mask_tok = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        self.qpos = nn.Parameter(torch.randn(1, P, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads)
                                    for _ in range(cfg.pred_depth))
        self.head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))

    def forward(self, cond, q_idx):
        """cond (B,n_cond,dim); q_idx (B,n_q) patch positions to predict."""
        B, n_q = q_idx.shape
        q = self.mask_tok.expand(B, n_q, -1) + torch.gather(
            self.qpos.expand(B, -1, -1), 1,
            q_idx[..., None].expand(-1, -1, cond.shape[-1]))
        x = torch.cat([cond, q], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, -n_q:])


class DiscreteJEPA(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.context = Encoder(cfg)
        self.target = Encoder(cfg)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.vq = VectorQuantizerEMA(cfg)
        self.P = self.context.P
        self.pred_s2p = Predictor(cfg, self.P)
        self.pred_p2p = Predictor(cfg, self.P)
        # context-patch pos for the P2P conditioning side
        self.ctx_pos = nn.Parameter(torch.randn(1, self.P, cfg.dim) * 0.02)

    @torch.no_grad()
    def ema_update(self, m):
        for pt, pc in zip(self.target.parameters(), self.context.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)

    def loss(self, img, gen: torch.Generator):
        """One training step's losses on a batch of frames (B,1,res,res)."""
        cfg, B, P = self.cfg, img.shape[0], self.P
        dev = img.device
        # per-batch visible ratio; per-sample random subsets of that size
        u = cfg.vis_lo + (cfg.vis_hi - cfg.vis_lo) * torch.rand((), generator=gen).item()
        n_vis = max(1, min(P, round(u * P)))
        perm = torch.argsort(torch.rand(B, P, generator=gen), dim=1).to(dev)
        vis_idx, mask_idx = perm[:, :n_vis], perm[:, n_vis:]

        with torch.no_grad():
            tgt_all, _ = self.target(img)                       # (B,P,dim)
            tgt_all = F.layer_norm(tgt_all, (cfg.dim,))

        ctx_patch, sem = self.context(img, vis_idx)
        sem_q, idx, commit, perplexity = self.vq(sem)

        # S2P: every patch latent, from the discrete semantic tokens ALONE
        all_idx = torch.arange(P, device=dev).expand(B, -1)
        s2p = self.pred_s2p(sem_q, all_idx)
        loss_s2p = F.mse_loss(s2p, tgt_all)

        # P2P: masked patch latents from visible context patches (I-JEPA)
        if mask_idx.shape[1] > 0:
            cond = ctx_patch + torch.gather(
                self.ctx_pos.expand(B, -1, -1), 1,
                vis_idx[..., None].expand(-1, -1, cfg.dim))
            p2p = self.pred_p2p(cond, mask_idx)
            tgt_mask = torch.gather(tgt_all, 1,
                                    mask_idx[..., None].expand(-1, -1, cfg.dim))
            loss_p2p = F.mse_loss(p2p, tgt_mask)
        else:
            loss_p2p = torch.zeros((), device=dev)

        total = (cfg.w_s2p * loss_s2p + cfg.w_p2p * loss_p2p
                 + cfg.beta_commit * commit)
        return total, {"s2p": float(loss_s2p.detach()), "p2p": float(loss_p2p.detach()),
                       "commit": float(commit.detach()), "perplexity": float(perplexity)}

    @torch.no_grad()
    def encode(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """(B,res,res) uint8 -> (B,M) int64 code indices. Deterministic, full frame.
        THE downstream interface — the eye's entire output."""
        self.eval()
        img = frames_u8.float().div(255.0).mul(2).sub(1).unsqueeze(1)
        _, sem = self.context(img)
        z = self.vq.proj_in(sem).reshape(-1, self.cfg.code_dim)
        return self.vq.quantize_indices(z).view(frames_u8.shape[0], self.cfg.M)


def load_eye(name: str, device="cpu"):
    """Load a FROZEN tokenizer by config name: final.pt if the run finished, else
    the newest readable checkpoint (so the gate can be probed mid-training)."""
    from pong_counterfactual.cjepa_rung3 import config as C
    from pong_counterfactual.cjepa_rung3 import ckpt as CK
    out_dir = C.tok_dir(name)
    final = out_dir / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location=device, weights_only=False)
        src = "final.pt"
    else:
        payload, path = CK.load_latest(out_dir / "ckpts", map_location=device)
        if payload is None:
            raise FileNotFoundError(f"no tokenizer checkpoint under {out_dir}")
        src = f"{path.name} (training incomplete)"
    cfg = TokenizerConfig(**payload["config"])
    model = DiscreteJEPA(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[eye] loaded {name} from {src}")
    return model, cfg


def encode_frames(model: DiscreteJEPA, frames_u8, device, batch=512):
    """numpy (N,res,res) uint8 -> numpy (N,M) int64, batched."""
    import numpy as np
    out = []
    for i in range(0, len(frames_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch])).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cfg.M), dtype=np.int64)
