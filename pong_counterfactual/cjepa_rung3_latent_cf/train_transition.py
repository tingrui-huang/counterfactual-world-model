"""Train the discrete-noise latent transition model (+ tiny known-answer check).

Trains one model per seed via hard-EM, fits the frozen position-probe readout, and saves
everything to outputs/<run>/transition_ckpt.pt. Run:
  python -m pong_counterfactual.cjepa_rung3_latent_cf.train_transition
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "main"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(LA.REPO)).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- tiny known-answer
def tiny_known_answer(Zt, A, Zt1, cfg, device="cpu"):
    """Memorize a tiny fixed set; verify action & noise change the output and targets align."""
    n = 32
    idx = np.arange(n)
    zt, a, zt1 = Zt[idx], A[idx], Zt1[idx]
    model, u = TX.train_hard_em(zt, a, zt1, K=cfg["K"], hidden=cfg["hidden"],
                                depth=cfg["depth"], lr=2e-3, batch=32,
                                base_steps=1200, em_rounds=4, m_steps_per_round=300,
                                seed=0, device=device)
    _, err = TX.factual_error(model, zt, a, zt1, device)
    zt_t, a_t = TX._to_t(zt, device), TX._to_t(a, device)
    # action sensitivity: swap a_int one-hot (dims 0..3) UP<->DOWN
    a_swap = a.copy(); a_swap[:, [2, 3]] = a_swap[:, [3, 2]]
    with torch.no_grad():
        u0 = torch.zeros(n, dtype=torch.long)
        d_act = float((model(zt_t, a_t, u0) -
                       model(zt_t, TX._to_t(a_swap, device), u0)).abs().mean())
        d_noise = float((model(zt_t, a_t, torch.zeros(n, dtype=torch.long)) -
                         model(zt_t, a_t, torch.ones(n, dtype=torch.long))).abs().mean())
    m = {"memorize_mse": float(err.mean()), "action_delta": d_act, "noise_delta": d_noise,
         "u_sizes": np.bincount(u, minlength=cfg["K"]).tolist()}
    m["pass"] = bool(m["memorize_mse"] < 0.15 and d_act > 1e-3 and d_noise > 1e-3)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LA.load_config()
    print("[latent-cf] D0 adapter gate ...")
    d0 = LA.gate_d0(cfg)
    (OUT / "d0.json").write_text(json.dumps(d0, indent=2, default=str))
    print(f"[latent-cf] D0 pass = {d0['pass']}")
    if not d0["pass"]:
        print("D0 FAILED — stop."); return

    train = LA.build_train_transitions(cfg)
    norm = LA.fit_normalizer(train)
    A = LA.action_feats(train.a_int, train.a_prev)
    Zt, Zt1 = norm.fwd(train.z_t), norm.fwd(train.z_t1)
    probe = LA.fit_position_probe(train)

    print("[latent-cf] tiny known-answer check ...")
    tiny = tiny_known_answer(Zt, A, Zt1, cfg)
    (OUT / "tiny_known_answer.json").write_text(json.dumps(tiny, indent=2, default=str))
    print(f"  memorize_mse {tiny['memorize_mse']:.4f}  action_delta {tiny['action_delta']:.4f} "
          f"noise_delta {tiny['noise_delta']:.4f}  pass={tiny['pass']}")
    if not tiny["pass"]:
        print("Tiny known-answer FAILED — stop."); return

    print(f"[latent-cf] training {len(cfg['seeds'])} models via hard-EM "
          f"(n={len(Zt)} transitions) ...")
    models = []
    for s in cfg["seeds"]:
        model, u = TX.train_hard_em(
            Zt, A, Zt1, K=cfg["K"], hidden=cfg["hidden"], depth=cfg["depth"],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch=cfg["batch"],
            base_steps=cfg["base_steps"], em_rounds=cfg["em_rounds"],
            m_steps_per_round=cfg["m_steps_per_round"], seed=s, log=print)
        _, err = TX.factual_error(model, Zt, A, Zt1)
        print(f"  seed {s}: train factual MSE {err.mean():.4f}, "
              f"u sizes {np.bincount(u, minlength=cfg['K']).tolist()}")
        models.append({"seed": s, "state_dict": model.state_dict(),
                       "u_train": u, "train_factual_mse": float(err.mean())})

    torch.save({
        "models": models, "config": cfg,
        "norm_mu": norm.mu, "norm_sd": norm.sd,
        "probe": {"coef": probe.coef, "intercept": probe.intercept,
                  "mu": probe.mu, "sd": probe.sd},
        "git_commit": git_commit(), "ae_train_git_commit": d0["ae_train_git_commit"],
        "n_train_transitions": len(Zt),
    }, OUT / "transition_ckpt.pt")
    print(f"[latent-cf] saved -> {OUT/'transition_ckpt.pt'}")


if __name__ == "__main__":
    main()
