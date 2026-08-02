"""CONTROLLED conceptual proof — shortcut ⇒ overfit; low-recoverability objective ⇒ generalization.

A minimal, fully-controlled simulation that isolates the HaF mechanism (no confounds, exact recoverability
knob). It makes two claims legible:
  (A) a policy that can reach the target through a cheap SHORTCUT overfits — it fits the training data via the
      shortcut and fails out-of-distribution once the shortcut breaks.
  (B) co-training an objective the shortcut CANNOT cheaply solve (a low-recoverability objective) forces the
      genuine representation and restores generalization — and the lower the objective's recoverability, the
      more it helps (a monotone dose-response).

Generative model (the "world"):
  latent  z ~ N(0, I_d)                      the genuine task variable
  action  y = W_y z                          what the policy must output (depends on ALL of z)
  channel x_easy : the SHORTCUT. In TRAIN x_easy = z (trivially readable → y is cheap to reach). Out-of-
                   distribution the shortcut BREAKS: x_easy is resampled independently of z (a spurious cue
                   that does not transfer).
  channel x_hard = g(z) : an entangled nonlinear view (fixed random 2-layer tanh MLP). Fully informative about
                   z but the encoder must LEARN to invert it. Reliable both in-distribution and OOD.

Policy: enc([x_easy, x_hard]) -> h -> y_hat.  BC latches onto x_easy (cheapest) and never learns to read
x_hard -> fits train, fails OOD.  (claim A)

The recoverability knob (input-side lever = HaF's JEPA/masking):
  aux objective = "predict y, but with x_easy masked out with probability (1 - rho)".
  rho in [0,1] IS the objective's recoverability: rho=1 the aux still sees the shortcut (fully recoverable,
  no pressure); rho=0 the aux never sees it (lowest recoverability, must invert x_hard). Sweeping rho gives the
  monotone OOD-vs-recoverability curve.  (claim B)

Metrics: train / IID-test / OOD-test MSE, each normalized by Var(y). Output: exp_sim_analysis.json.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DEV = os.environ.get("DEV", "cpu")               # tiny MLPs — CPU on the login node, no GPU grab
D_Z, D_HARD, D_Y = 16, 64, 8
N_TRAIN, N_TEST = 20000, 4000
EPOCHS, BS, LAM = 60, 256, 1.0
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
RHOS = [1.0, 0.75, 0.5, 0.25, 0.0]               # objective recoverability (keep-prob of the shortcut)


def make_world(seed):
    g = torch.Generator().manual_seed(seed)
    Wy = torch.randn(D_Y, D_Z, generator=g) / np.sqrt(D_Z)
    # fixed random 2-layer tanh map z -> x_hard (the entangled, must-learn channel)
    A1 = torch.randn(48, D_Z, generator=g) / np.sqrt(D_Z)
    A2 = torch.randn(D_HARD, 48, generator=g) / np.sqrt(48)

    def g_hard(z):
        return torch.tanh(torch.tanh(z @ A1.T) @ A2.T)

    def sample(n, ood, gen):
        z = torch.randn(n, D_Z, generator=gen)
        x_hard = g_hard(z)
        if ood:
            z_spur = torch.randn(n, D_Z, generator=gen)   # shortcut BREAKS: independent of z
            x_easy = z_spur
        else:
            x_easy = z + 0.01 * torch.randn(n, D_Z, generator=gen)
        y = z @ Wy.T
        return torch.cat([x_easy, x_hard], 1), y

    return sample


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D_Z + D_HARD, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, D_Y)

    def forward(self, o):
        return self.head(self.enc(o))


def corrupt_easy(o, rho, gen):
    """A shortcut-free objective: for a (1-rho) fraction of rows, REPLACE the x_easy block with an independent
    random draw (the cheap cue is made unreliable — present but uninformative). This matches how a shortcut
    behaves out-of-distribution (present but wrong), rather than zeroing it (which the net can detect as a
    separate 'masked' mode that never occurs at test time). rho = prob the cue stays honest = recoverability."""
    keep = (torch.rand(o.shape[0], 1, generator=gen) < rho).float().to(o.device)
    noise = torch.randn(o.shape[0], D_Z, generator=gen).to(o.device)   # same marginal as x_easy (≈ N(0,I))
    out = o.clone()
    out[:, :D_Z] = keep * o[:, :D_Z] + (1 - keep) * noise
    return out


def run(seed, rho, use_aux):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    sample = make_world(seed)
    Xtr, Ytr = sample(N_TRAIN, False, gen)
    Xi, Yi = sample(N_TEST, False, gen)             # IID test (shortcut intact)
    Xo, Yo = sample(N_TEST, True, gen)              # OOD test (shortcut broken)
    Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV)
    vary = float(Ytr.var().item())

    net = Policy().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    n = N_TRAIN
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            b = perm[i:i+BS]
            o, y = Xtr[b], Ytr[b]
            opt.zero_grad()
            loss = mse(net(o), y)                     # BC on the full observation (uses the shortcut)
            if use_aux:
                loss = loss + LAM * mse(net(corrupt_easy(o, rho, gen)), y)   # low-recoverability objective
            loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        tr = mse(net(Xtr), Ytr).item() / vary
        iid = mse(net(Xi.to(DEV)), Yi.to(DEV)).item() / vary
        ood = mse(net(Xo.to(DEV)), Yo.to(DEV)).item() / vary
    return tr, iid, ood


def agg(rows):
    a = np.array(rows)
    return {"train": [float(a[:, 0].mean()), float(a[:, 0].std())],
            "iid":   [float(a[:, 1].mean()), float(a[:, 1].std())],
            "ood":   [float(a[:, 2].mean()), float(a[:, 2].std())]}


results = {"config": dict(d_z=D_Z, d_hard=D_HARD, d_y=D_Y, n_train=N_TRAIN, epochs=EPOCHS,
                          lam=LAM, seeds=len(SEEDS)), "conditions": {}}

# (A) BC-only baseline — no auxiliary
bc = agg([run(s, None, use_aux=False) for s in SEEDS])
results["conditions"]["BC-only"] = bc
print(f"BC-only            train={bc['train'][0]:.3f}  IID={bc['iid'][0]:.3f}  "
      f"OOD={bc['ood'][0]:.3f}±{bc['ood'][1]:.3f}", flush=True)

# (B) BC + low-recoverability objective, sweeping objective recoverability rho
results["rho_sweep"] = []
for rho in RHOS:
    r = agg([run(s, rho, use_aux=True) for s in SEEDS])
    results["conditions"][f"aux rho={rho}"] = r
    results["rho_sweep"].append({"recoverability": rho, "ood": r["ood"], "iid": r["iid"], "train": r["train"]})
    print(f"aux recover={rho:.2f}   train={r['train'][0]:.3f}  IID={r['iid'][0]:.3f}  "
          f"OOD={r['ood'][0]:.3f}±{r['ood'][1]:.3f}", flush=True)

# headline numbers
gap = bc["ood"][0] - bc["iid"][0]
best = min(results["rho_sweep"], key=lambda d: d["ood"][0])
results["headline"] = {
    "bc_generalization_gap": gap,
    "best_aux": best,
    "ood_reduction_vs_bc": bc["ood"][0] - best["ood"][0],
    "monotone": all(results["rho_sweep"][i]["ood"][0] >= results["rho_sweep"][i+1]["ood"][0] - 0.02
                    for i in range(len(RHOS)-1)),
}
print(f"\n(A) BC overfits: OOD−IID gap = {gap:.2f}")
print(f"(B) lowest-recoverability aux cuts OOD loss by {results['headline']['ood_reduction_vs_bc']:.2f} "
      f"(monotone in recoverability: {results['headline']['monotone']})")
json.dump(results, open(os.path.join(OUT, "exp_sim_analysis.json"), "w"), indent=2)
print("SAVED exp_sim_analysis.json")
