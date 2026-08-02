"""Exp 1 — Stage 2: encode frames (DINOv2) + instruction (MiniLM), compute R_triv / G_obs.

R_triv = 1 - L_trivial/L_marginal            (shortcut availability)
G_obs  = 1 - min(L_trivial,L_probe)/L_marginal (total learnable-from-o_t signal)

GroupKFold by episode prevents leakage (R1/R2/R3 targets are per-episode constant).
Set DEV=cuda on an L40S slurm node (node01/node100); torch cu126 fails on B200/node200.
"""
import os, json
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "droid")
NPZ = os.path.join(OUT, f"cache/transitions_{TAG}.npz")
CACHE = os.path.join(OUT, f"cache/dino_latents_{TAG}.npz")
RESULTS = os.path.join(OUT, f"results_{TAG}.json")
DEV = os.environ.get("DEV", "cpu")
if DEV == "cpu":
    torch.set_num_threads(min(32, os.cpu_count() or 8))

FRAME_KEYS = ["Fpl", "Fps", "Ft", "Ffs", "Ffl", "F0", "Flast"]
Z_KEYS = ["z_pl", "z_ps", "zt", "z_fs", "z_fl", "z0", "z_last"]


def encode_images(d):
    if os.path.exists(CACHE):
        print("loading cached latents", flush=True)
        z = np.load(CACHE); return {k: z[k] for k in z.files}
    print(f"DINOv2 encoding on {DEV} ...", flush=True)
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def enc(frames):
        out = []
        bs = 512 if DEV == "cuda" else 256
        for i in range(0, len(frames), bs):
            b = torch.from_numpy(frames[i:i + bs]).to(DEV).float().permute(0, 3, 1, 2) / 255.0
            b = (b - mean) / std
            out.append(model(pixel_values=b).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)

    lat = {zk: enc(d[fk]) for fk, zk in zip(FRAME_KEYS, Z_KEYS)}
    np.savez_compressed(CACHE, **lat)
    return lat


def embed_text(strings):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            enc = tok(list(strings[i:i + 128]), padding=True, truncation=True, return_tensors="pt").to(DEV)
            out = m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs)


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def gmarginal(Y, g):
    p = np.zeros_like(Y, float)
    for tr, te in GroupKFold(5).split(Y, groups=g):
        p[te] = Y[tr].mean(0)
    return mse(Y, p)


def gridge(X, Y, g):
    p = np.zeros_like(Y, float)
    for tr, te in GroupKFold(5).split(X, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=100.0).fit(sc.transform(X[tr]), Y[tr])
        p[te] = np.asarray(m.predict(sc.transform(X[te]))).reshape(len(te), -1)
    return mse(Y, p)


def gmlp(X, Y, g, epochs=150):
    p = np.zeros_like(Y, float)
    for tr, te in GroupKFold(5).split(X, groups=g):
        sc = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=DEV)
        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=DEV)
        Ytr = torch.tensor(Y[tr], dtype=torch.float32, device=DEV)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(),
                                  torch.nn.Dropout(0.1), torch.nn.Linear(256, Y.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lf = torch.nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad(); lf(net(Xtr), Ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p[te] = net(Xte).cpu().numpy()
    return mse(Y, p)


def main():
    d = np.load(NPZ, allow_pickle=True)
    lat = encode_images(d)
    Z = {k: l2n(lat[k]) for k in Z_KEYS}
    zt = Z["zt"]
    g = d["ep_id"]

    def zs(a):
        a = np.asarray(a, float).reshape(len(a), -1)
        return (a - a.mean(0)) / (a.std(0) + 1e-8)

    cart0, cartt = zs(d["cart0"]), zs(d["cartt"])
    cart_last = zs(d["cart_last"])
    grip, grip_f = zs(d["gript"]), zs(d["grip_fut"])
    act, act_f = zs(d["actt"]), zs(d["act_fut"])
    act_p, grip_p = zs(d["act_prev"]), zs(d["grip_prev"])
    act_ch = zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1))  # 15-step action chunk (the policy output)
    prog = zs(d["progress"])
    both = np.concatenate([zt, Z["z_fs"]], axis=1)  # inverse-dynamics feature: current + future frame

    # instruction text embedding (non-empty subset)
    instr = d["instr"]
    has = np.array([bool(str(s).strip()) for s in instr])
    txt = None
    if has.any():
        try:
            txt = l2n(embed_text(instr))
        except Exception as e:
            print("R3 text embed skipped:", e, flush=True)

    # The inputs a VLA actually sees: image (DINOv2 o_t) + state (pose+gripper) + language.
    # Trivial rules and probes may only use these; the ACTION is an output, never an input.
    state_feat = zs(np.concatenate([d["cartt"], d["gript"]], axis=1))
    lang_feat = txt if txt is not None else np.zeros((len(zt), 1), np.float32)
    Xf = np.concatenate([zt, state_feat, lang_feat], axis=1)      # {image, state, language}
    Xf_nl = np.concatenate([zt, state_feat], axis=1)              # instruction target: exclude language
    Xinv = np.concatenate([zt, Z["z_fs"], state_feat], axis=1)    # inverse dynamics: both frames + state

    def meanpred(Y):
        return np.broadcast_to(Y.mean(0), Y.shape)

    # name: (Y, trivial, family, feature, needs_mlp, mask)
    # trivial rule may ONLY copy an actual input: image (obs targets) or a state dim (pose/gripper).
    # action targets get a MARGINAL trivial (the action is an output, not an input — nothing to copy).
    # probe feature = DINOv2(o_t) image (conservative; state/language would need per-target leakage handling).
    T = {
        "Mfar past-obs k~45":  (Z["z_pl"], zt, "retrospective", zt, True, None),
        "Mnear past-obs k~5":  (Z["z_ps"], zt, "prospective_sym", zt, True, None),
        "P1s future-obs k~5":  (Z["z_fs"], zt, "prospective", zt, True, None),
        "P1l future-obs k~45": (Z["z_fl"], zt, "prospective", zt, True, None),
        "R1 initial-obs":      (Z["z0"],  zt, "retrospective", zt, True, None),
        "R2 initial-pose":     (cart0, cartt, "retrospective", zt, False, None),
        # symmetric prospective anchors (episode END) — the fair control for initial-obs / initial-pose
        "P0 final-obs":        (Z["z_last"], zt, "prospective", zt, True, None),
        "P0p final-pose":      (cart_last, cartt, "prospective", zt, False, None),
        "R3 instruction":      (txt, None, "retrospective", zt, True, has) if txt is not None else None,
        "P2 future-action":    (act_f, None, "prospective", zt, False, None),
        "P3 future-gripper":   (grip_f, grip, "prospective", zt, False, None),
        "Mact prev-action":    (act_p, None, "retrospective", zt, False, None),
        "Mgrip prev-gripper":  (grip_p, grip, "retrospective", zt, False, None),
        "Iinv act|both-frames":(act, None, "introspective", both, False, None),
        "I-gripper now":       (grip, None, "introspective", zt, False, None),
        "I-progress t/T":      (prog, None, "introspective", zt, False, None),
        "BC action|o_t":       (act_ch, None, "policy", zt, False, None),  # target = 15-step chunk
    }

    results = {}
    for name, spec in T.items():
        if spec is None:
            continue
        Y, triv, fam, feat, needs_mlp, mask = spec
        Y = np.asarray(Y, float).reshape(len(Y), -1)
        gg = g
        X = feat
        if mask is not None:
            Y, X, gg = Y[mask], feat[mask], g[mask]
        if triv is None:
            triv = meanpred(Y)              # trivial == marginal (no cheap copy rule)
        triv = np.asarray(triv, float).reshape(len(Y), -1)
        if mask is not None:
            triv = triv  # already sized (built from masked Y) — for meanpred; for copy rules mask N/A here
        L_marg = gmarginal(Y, gg)
        L_triv = mse(Y, triv if triv.shape[0] == len(Y) else triv[mask])
        L_lin = gridge(X, Y, gg)
        probes = [L_triv, L_lin]
        row = dict(family=fam, n=int(len(Y)), L_marginal=L_marg, L_trivial=L_triv, L_probe_lin=L_lin)
        if needs_mlp:
            L_mlp = gmlp(X, Y, gg)
            row["L_probe_mlp"] = L_mlp; probes.append(L_mlp)
        L_obs = min(probes)
        row["R_triv"] = 1 - L_triv / L_marg
        row["G_obs"] = 1 - L_obs / L_marg
        row["probe_beyond_trivial"] = (L_triv - L_obs) / L_marg
        results[name] = row
        print(f"{name:24s} n={row['n']:5d} R_triv={row['R_triv']:+.3f} "
              f"G_obs={row['G_obs']:+.3f} probe>triv={row['probe_beyond_trivial']:+.3f}", flush=True)

    json.dump(results, open(RESULTS, "w"), indent=2)
    print("SAVED", RESULTS, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
    os._exit(0)  # torch/transformers can hang non-daemon threads on exit; force clean exit for slurm
