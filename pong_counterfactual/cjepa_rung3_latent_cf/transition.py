"""Discrete-noise latent transition model + hard-EM training + argmin abduction.

The faithful continuous-target analog of Rung 2's Gumbel-Max SCM (see design_audit.md §C):
  mechanism   z_t1 = f_theta(z_t, [a_int, a_prev], u),  u in {0..K-1} discrete exogenous
  abduction   u_hat = argmin_u || f_theta(z_t, a, u) - z_t1_obs ||^2   (normalized latent)
The noise u is a latent mixture index trained by hard-EM (E-step = the same argmin; the
oracle stuck label is NEVER used in training). Forward prediction NEVER sees z_t1 (only
abduction does). Small MLP, comparable in scale to Rung 2's per-dim MLPClassifier((64,)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LatentTransition(nn.Module):
    def __init__(self, z_dim=192, a_dim=8, K=2, hidden=256, depth=2):
        super().__init__()
        self.z_dim, self.a_dim, self.K = z_dim, a_dim, K
        din = z_dim + a_dim + K
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, a, u_idx):
        """z_t (B,z_dim) norm, a (B,a_dim), u_idx (B,) long -> z_t1 (B,z_dim) norm."""
        u = torch.zeros(z_t.shape[0], self.K, device=z_t.device)
        u[torch.arange(z_t.shape[0]), u_idx] = 1.0
        return self.net(torch.cat([z_t, a, u], 1))

    @torch.no_grad()
    def predict_all_u(self, z_t, a):
        """-> (B, K, z_dim): prediction under every discrete noise value."""
        B = z_t.shape[0]
        outs = []
        for k in range(self.K):
            uk = torch.full((B,), k, dtype=torch.long, device=z_t.device)
            outs.append(self.forward(z_t, a, uk))
        return torch.stack(outs, 1)

    @torch.no_grad()
    def abduct(self, z_t, a, z_t1_obs):
        """u_hat = argmin_u ||f(z_t,a,u) - z_t1_obs||^2 (per-sample). Returns (u_hat, err)."""
        preds = self.predict_all_u(z_t, a)                  # (B,K,z)
        err = ((preds - z_t1_obs[:, None, :]) ** 2).mean(-1)  # (B,K)
        u_hat = err.argmin(1)
        return u_hat, err


def _to_t(x, device="cpu", long=False):
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.long().to(device) if long else t.float().to(device)


def train_hard_em(Zt, A, Zt1, K=2, hidden=256, depth=2, lr=1e-3, weight_decay=0.0,
                  batch=256, base_steps=1500, em_rounds=8, m_steps_per_round=400,
                  seed=0, device="cpu", log=lambda *a: None):
    """Zt,Zt1 (N,192) NORMALIZED; A (N,8). Returns (model, u_assignments)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Zt)
    Zt_t, A_t, Zt1_t = _to_t(Zt, device), _to_t(A, device), _to_t(Zt1, device)
    model = LatentTransition(Zt.shape[1], A.shape[1], K, hidden, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    # label-free init: split by latent-displacement magnitude (small~stuck / large~moved)
    disp = np.linalg.norm(Zt1 - Zt, axis=1)
    u = (disp > np.median(disp)).astype(np.int64)           # 0=small, 1=large (K=2)
    if K > 2:                                                # generic: quantile bins
        u = np.digitize(disp, np.quantile(disp, np.linspace(0, 1, K + 1)[1:-1]))

    def m_step(u_arr, steps):
        u_t = _to_t(u_arr, device, long=True)
        model.train()
        for _ in range(steps):
            idx = rng.integers(0, n, min(batch, n))
            it = torch.from_numpy(idx).to(device)
            opt.zero_grad()
            pred = model(Zt_t[it], A_t[it], u_t[it])
            loss = ((pred - Zt1_t[it]) ** 2).mean()
            loss.backward(); opt.step()
        return float(loss.detach())

    def e_step():
        model.eval()
        u_hat, _ = model.abduct(Zt_t, A_t, Zt1_t)
        u_new = u_hat.cpu().numpy()
        # reseed empty clusters to keep K modes alive
        for k in range(K):
            if (u_new == k).sum() == 0:
                u_new[rng.integers(0, n, max(1, n // (4 * K)))] = k
        return u_new

    l = m_step(u, base_steps)                                # warm-up on the init split
    log(f"  [em] warm-up loss {l:.4f}, init sizes {np.bincount(u, minlength=K).tolist()}")
    for r in range(em_rounds):
        u = e_step()
        l = m_step(u, m_steps_per_round)
        log(f"  [em] round {r+1}: loss {l:.4f}, u sizes {np.bincount(u, minlength=K).tolist()}")
    return model, u


@torch.no_grad()
def factual_error(model, Zt, A, Zt1, device="cpu"):
    """Per-sample abduction-then-predict latent MSE to the observed next latent (norm)."""
    Zt_t, A_t, Zt1_t = _to_t(Zt, device), _to_t(A, device), _to_t(Zt1, device)
    u_hat, err = model.abduct(Zt_t, A_t, Zt1_t)
    per = err[torch.arange(len(Zt)), u_hat]
    return u_hat.cpu().numpy(), per.cpu().numpy()
