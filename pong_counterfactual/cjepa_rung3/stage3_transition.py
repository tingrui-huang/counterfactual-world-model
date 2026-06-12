"""Stage 3 — the forecaster on learned tokens (light GPU/CPU). GATE-CHECKED.

p(next frame's M tokens | last N frames' tokens + intended-action history), as
decision ⑤: FACTORIZED — M independent K-way categoricals, one softmax per token
slot, mirroring Rung 1/2's "4 dims, each its own categorical" so gumbel_abduction.py
applies per slot with zero change. Inter-slot correlation is deliberately ignored
(structural isomorphism with the validated rungs, not the best possible model).

The model is Rung 2's HistModel translated to tokens: an MLP on the one-hot history
window (N frames x M codes + N intended actions; the CURRENT action is the one
swapped for the counterfactual — past history is shared, the intervention is at k).
History stays load-bearing under sticky (the override target is the previous action).

Inputs: the FROZEN eye's token caches (built here once per s and stored to Drive) —
pixels and ground-truth numbers never enter this model.

REFUSES to train unless the Stage-2 gate PASSED (--force to override — that burns
compute on a blind eye, the exact hard pitfall the gate exists to prevent).

Run:  python -m pong_counterfactual.cjepa_rung3.stage3_transition --name M4_K64_res84
      [--smoke] [--force]
"""
import argparse
import dataclasses
import json
import time

import numpy as np
import torch
import torch.nn as nn

from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3 import ckpt as CK
from pong_counterfactual.cjepa_rung3.data import load_split, valid_ks, ACTIONS_H
from pong_counterfactual.cjepa_rung3.tokenizer import load_eye, encode_frames

A_IDX = {a: i for i, a in enumerate(ACTIONS_H)}      # NOOP/UP/DOWN -> 0/1/2


# ------------------------------------------------------------------ token caches
def token_cache(s, split, base_seed, n_eps, tok_name, device):
    """Encode every frame of a split through the frozen eye ONCE; cache to Drive.
    Returns list of (T+1, M) int64 arrays aligned with the episode shards."""
    d = C.data_dir(s)
    path = d / f"tokens_{tok_name}_{split}.npz"
    eps = load_split(d, split, base_seed, n_eps)
    if path.exists():
        z = np.load(path)
        if all(f"ep{i}" in z and len(z[f"ep{i}"]) == len(eps[i].frames)
               for i in range(n_eps)):
            return eps, [z[f"ep{i}"] for i in range(n_eps)]
    model, _ = load_eye(tok_name, device)
    toks = [encode_frames(model, ep.frames, device) for ep in eps]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"ep{i}": t for i, t in enumerate(toks)})
    tmp.replace(path)
    print(f"[tokens] cached {split} s={s} -> {path.name}")
    return eps, toks


# ------------------------------------------------------------------ features
def feat(tokens, intended, k, N, M, K, override_action=None):
    """One-hot history window for the transition at k: frames k-N+1..k tokens +
    intended actions k-N+1..k (current action swappable for the CF input)."""
    x = np.zeros(N * M * K + N * len(ACTIONS_H), dtype=np.float32)
    for j in range(N):
        codes = tokens[k - N + 1 + j]
        for m in range(M):
            x[(j * M + m) * K + codes[m]] = 1.0
    acts = list(intended[k - N + 1:k + 1])
    if override_action is not None:
        acts[-1] = override_action
    base = N * M * K
    for j, a in enumerate(acts):
        x[base + j * len(ACTIONS_H) + A_IDX[int(a)]] = 1.0
    return x


class TransitionMLP(nn.Module):
    """Shared trunk + M independent K-way heads (decision ⑤)."""

    def __init__(self, in_dim, M, K, hidden):
        super().__init__()
        self.M, self.K = M, K
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, K) for _ in range(M))

    def forward(self, x):
        h = self.trunk(x)
        return torch.stack([head(h) for head in self.heads], dim=1)  # (B,M,K)


def check_gate(tok_name, force):
    rep = C.tok_dir(tok_name) / "gate_report.json"
    if not rep.exists():
        raise SystemExit(f"[stage3] no gate report for {tok_name} — run stage2 first")
    passed = json.loads(rep.read_text())["gate"]["PASSED"]
    if not passed and not force:
        raise SystemExit(f"[stage3] GATE FAILED for {tok_name}: refusing to train on "
                         "a blind eye (skill hard pitfall). Turn the Stage-1 knobs, "
                         "re-gate, or pass --force if you know what you are doing.")
    if not passed:
        print("[stage3] WARNING: gate failed but --force given")


def train_transition(s, tok_name, tcfg: C.TransitionConfig, n_train_eps, device):
    eps, toks = token_cache(s, "train", C.TRAIN_BASE_SEED, n_train_eps,
                            tok_name, device)
    # M, K from the eye's config, not the data (unused codes keep their slots)
    eye_cfg = json.loads((C.tok_dir(tok_name) / "gate_report.json").read_text())["config"]
    M, K = eye_cfg["M"], eye_cfg["K"]
    N = C.N_HIST

    X, Y = [], []
    for ep, tk in zip(eps, toks):
        for k in valid_ks(ep, N):
            X.append(feat(tk, ep.intended, k, N, M, K))
            Y.append(tk[k + 1])
    X = torch.from_numpy(np.stack(X))
    Y = torch.from_numpy(np.stack(Y).astype(np.int64))
    print(f"[stage3] s={s}: {len(X)} transitions, in_dim={X.shape[1]}, M={M}, K={K}")

    out_dir = C.trans_dir(f"{tok_name}_{tcfg.name}", s)
    ckpt_dir = out_dir / "ckpts"
    torch.manual_seed(tcfg.seed)
    model = TransitionMLP(X.shape[1], M, K, tcfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    samp = np.random.default_rng(tcfg.seed)
    start = 0
    payload, path = CK.load_latest(ckpt_dir, map_location=device)
    if payload is not None:
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        samp.bit_generator.state = payload["samp"]
        CK.restore_rng(payload["rng"])
        start = payload["step"]
        print(f"[stage3] RESUMED from {path.name} at step {start}")

    ce = nn.CrossEntropyLoss()
    Xd, Yd = X.to(device), Y.to(device)
    model.train()
    for step in range(start, tcfg.steps):
        idx = torch.from_numpy(samp.integers(0, len(Xd), tcfg.batch)).to(device)
        logits = model(Xd[idx])                                  # (B,M,K)
        loss = ce(logits.reshape(-1, K), Yd[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step == start:
            with torch.no_grad():
                acc = (logits.argmax(-1) == Yd[idx]).float().mean()
            print(f"[stage3] s={s} step {step+1}/{tcfg.steps} "
                  f"ce={float(loss):.3f} top1={float(acc)*100:.1f}%")
        if (step + 1) % tcfg.ckpt_every == 0:
            CK.save(ckpt_dir, step + 1,
                    {"model": model.state_dict(), "opt": opt.state_dict(),
                     "samp": samp.bit_generator.state, "in_dim": X.shape[1],
                     "M": M, "K": K, "config": dataclasses.asdict(tcfg)},
                    keep=tcfg.keep_ckpts)
    CK.save(ckpt_dir, tcfg.steps,
            {"model": model.state_dict(), "opt": opt.state_dict(),
             "samp": samp.bit_generator.state, "in_dim": X.shape[1],
             "M": M, "K": K, "config": dataclasses.asdict(tcfg)},
            keep=tcfg.keep_ckpts)
    tmp = out_dir / ".tmp_final.pt"
    torch.save({"model": model.state_dict(), "in_dim": X.shape[1], "M": M, "K": K,
                "hidden": tcfg.hidden}, tmp)
    tmp.replace(out_dir / "final.pt")
    print(f"[stage3] s={s} done -> {out_dir / 'final.pt'}")


def load_transition(s, tok_name, trans_name, device):
    p = C.trans_dir(f"{tok_name}_{trans_name}", s) / "final.pt"
    z = torch.load(p, map_location=device, weights_only=False)
    model = TransitionMLP(z["in_dim"], z["M"], z["K"], z["hidden"]).to(device)
    model.load_state_dict(z["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="tokenizer config name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    n_tr = C.N_TRAIN_EPS
    tcfg = C.TransitionConfig()
    if args.smoke:
        o = C.smoke_overrides()
        n_tr, tcfg = o["n_train_eps"], o["trans"]
        tok_name = args.name or o["tok"].name
    else:
        tok_name = args.name or C.TokenizerConfig().name
    check_gate(tok_name, args.force)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for s in C.SS_EVAL:
        train_transition(s, tok_name, tcfg, n_tr, device)


if __name__ == "__main__":
    main()
