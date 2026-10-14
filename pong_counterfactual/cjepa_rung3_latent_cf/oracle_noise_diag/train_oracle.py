"""Train the oracle-noise-conditioned transition model (supervised u = true stuck).

Same latent I/O + same small architecture as the parent (`transition.LatentTransition`);
hard-EM is REPLACED by direct supervision with the trusted `sticky_fired` label (the whole
point: measure the oracle-noise upper bound). Also trains a no-noise twin (u≡0) as the
O1 baseline. Run: python -m ...oracle_noise_diag.train_oracle
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX
from pong_counterfactual.cjepa_rung3_latent_cf.oracle_noise_diag import oracle_noise as ON

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "main"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(LA.REPO)).decode().strip()
    except Exception:
        return "unknown"


def train_supervised(Zt, A, Zt1, u, K=2, hidden=256, depth=2, lr=1e-3, batch=256,
                     steps=3000, seed=0, device="cpu"):
    """Direct MSE fit of f(z_t, a, u)->z_t1 with GIVEN u (no EM)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Zt)
    Zt_t, A_t = TX._to_t(Zt, device), TX._to_t(A, device)
    Zt1_t, u_t = TX._to_t(Zt1, device), TX._to_t(u, device, long=True)
    m = TX.LatentTransition(Zt.shape[1], A.shape[1], K, hidden, depth).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    m.train()
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, min(batch, n))).to(device)
        opt.zero_grad()
        loss = ((m(Zt_t[idx], A_t[idx], u_t[idx]) - Zt1_t[idx]) ** 2).mean()
        loss.backward(); opt.step()
    return m, float(loss.detach())


def tiny_check(Zt, A, Zt1, u, cfg):
    """Memorize a tiny fixed batch; verify action & noise change output and alignment."""
    n = 32
    m, _ = train_supervised(Zt[:n], A[:n], Zt1[:n], u[:n], K=cfg["K"], hidden=cfg["hidden"],
                            depth=cfg["depth"], lr=2e-3, batch=32, steps=1500, seed=0)
    with torch.no_grad():
        zt, a = TX._to_t(Zt[:n]), TX._to_t(A[:n])
        u_t = TX._to_t(u[:n], long=True)
        mse = float(((m(zt, a, u_t) - TX._to_t(Zt1[:n])) ** 2).mean())
        a_sw = A[:n].copy(); a_sw[:, [2, 3]] = a_sw[:, [3, 2]]
        d_act = float((m(zt, a, u_t) - m(zt, TX._to_t(a_sw), u_t)).abs().mean())
        d_noise = float((m(zt, a, torch.zeros(n, dtype=torch.long)) -
                         m(zt, a, torch.ones(n, dtype=torch.long))).abs().mean())
    res = {"memorize_mse": mse, "action_delta": d_act, "noise_delta": d_noise,
           "pass": bool(mse < 0.15 and d_act > 1e-3 and d_noise > 1e-3)}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ON.load_config()
    print("[oracle-noise] D0 gate ...")
    d0 = ON.gate_d0(cfg)
    (OUT / "d0.json").write_text(json.dumps(d0, indent=2, default=str))
    print(f"[oracle-noise] D0 pass={d0['pass']} (train stuck rate {d0['train_stuck_rate']:.3f})")
    if not d0["pass"]:
        print("D0 FAILED — stop."); return

    train = LA.build_train_transitions(cfg)
    stuck, key = ON.build_train_stuck(cfg)
    ON.assert_alignment(train, key)
    keep = stuck >= 0                                        # drop ill-posed (rare)
    norm = LA.fit_normalizer(train)
    probe = LA.fit_position_probe(train)
    Zt, Zt1 = norm.fwd(train.z_t)[keep], norm.fwd(train.z_t1)[keep]
    A = LA.action_feats(train.a_int, train.a_prev)[keep]
    u = stuck[keep].astype(np.int64)
    print(f"[oracle-noise] {keep.sum()} train transitions, stuck rate {u.mean():.3f}")

    print("[oracle-noise] tiny known-answer check ...")
    tiny = tiny_check(Zt, A, Zt1, u, cfg)
    (OUT / "tiny_known_answer.json").write_text(json.dumps(tiny, indent=2))
    print(f"  memorize {tiny['memorize_mse']:.4f} act {tiny['action_delta']:.3f} "
          f"noise {tiny['noise_delta']:.3f} pass={tiny['pass']}")
    if not tiny["pass"]:
        print("Tiny check FAILED — stop."); return

    models, no_noise = [], []
    for s in cfg["seeds"]:
        m, l = train_supervised(Zt, A, Zt1, u, K=cfg["K"], hidden=cfg["hidden"],
                                depth=cfg["depth"], lr=cfg["lr"], batch=cfg["batch"],
                                steps=cfg["steps"], seed=s)
        # no-noise twin: identical arch, u forced to 0 (ignores the mode)
        mn, _ = train_supervised(Zt, A, Zt1, np.zeros_like(u), K=cfg["K"],
                                 hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["lr"],
                                 batch=cfg["batch"], steps=cfg["steps"], seed=s)
        print(f"  seed {s}: oracle-noise train MSE {l:.4f}")
        models.append({"seed": s, "state_dict": m.state_dict()})
        no_noise.append({"seed": s, "state_dict": mn.state_dict()})

    prior = np.bincount(u, minlength=cfg["K"]).astype(float); prior /= prior.sum()
    torch.save({"models": models, "no_noise": no_noise, "config": cfg,
                "norm_mu": norm.mu, "norm_sd": norm.sd, "prior": prior,
                "probe": {"coef": probe.coef, "intercept": probe.intercept,
                          "mu": probe.mu, "sd": probe.sd},
                "git_commit": git_commit(),
                "ae_train_git_commit": d0["ae_train_git_commit"],
                "n_train": int(keep.sum())}, OUT / "oracle_ckpt.pt")
    print(f"[oracle-noise] saved -> {OUT/'oracle_ckpt.pt'}")


if __name__ == "__main__":
    main()
