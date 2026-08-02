"""Exp 1 — Stage 3: plot the R_triv x G_obs plane for the DROID run + R1 length gate."""
import os, json
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "droid")
RESULTS = os.path.join(OUT, f"results_{TAG}.json")
PNG = os.path.join(OUT, f"exp1_plane_{TAG}.png")
CACHE = os.path.join(OUT, f"dino_latents_{TAG}.npz")
NPZ = os.path.join(OUT, f"transitions_{TAG}.npz")


def color_for(name):
    p = name.split()[0]
    if p[0] in ("R", "M"):      # retrospective / relative-past
        return "#9E2A4F"
    if p[0] == "P":             # prospective / future
        return "#1F4E6B"
    if p.startswith("I"):       # introspective (answer in obs)
        return "#4A85A6"
    return "#2F6B4F"            # policy (BC)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    res = json.load(open(RESULTS))
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.axvspan(0.6, 1.05, color="#EDEFF2", alpha=0.6, zorder=0)
    ax.axhspan(0.02, 1.05, xmin=0.0, xmax=0.52, color="#FBEDF1", alpha=0.45, zorder=0)
    ax.text(0.02, 1.02, "HARD BUT LEARNABLE — USEFUL", color="#9E2A4F", fontsize=9, va="top")
    ax.text(1.03, -0.02, "TRIVIALLY SATISFIABLE (a cheap copy solves it)", color="#8E9AAB", fontsize=8.5, ha="right")
    ax.plot([-0.8, 1.05], [-0.8, 1.05], ls=":", color="#C2C9D3", lw=1, zorder=1)

    for name, r in res.items():
        x, y = r["R_triv"], r["G_obs"]
        c = color_for(name)
        ax.scatter([x], [y], s=95, color=c, zorder=3, edgecolor="white", linewidth=1.1)
        dx, dy = (8, 5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.3, color=c)

    ax.axhline(0, color="#CDD4DE", lw=1); ax.axvline(0, color="#CDD4DE", lw=1)
    ax.set_xlabel("R_triv  —  shortcut availability  (how much a trivial copy already gets)  →")
    ax.set_ylabel("G_obs  —  learnable-from-o_t signal  →")
    n_ep = "?"
    try:
        n_ep = len(set(np.load(NPZ, allow_pickle=True)["ep_id"].tolist()))
    except Exception:
        pass
    ax.set_title(f"Exp 1 — DROID subset ({n_ep} episodes): target redundancy vs learnability\n"
                 "future action/gripper = most trivial; retrospective initial-obs = least; BC action barely linearly recoverable")
    ax.set_xlim(-0.8, 1.05); ax.set_ylim(-0.7, 1.08)
    handles = [
        mlines.Line2D([], [], marker="o", ls="", color="#9E2A4F", label="retrospective / past (R*, M*)"),
        mlines.Line2D([], [], marker="o", ls="", color="#1F4E6B", label="prospective / future (P*)"),
        mlines.Line2D([], [], marker="o", ls="", color="#4A85A6", label="introspective (I*)"),
        mlines.Line2D([], [], marker="o", ls="", color="#2F6B4F", label="policy (BC)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(PNG, dpi=150)
    print("SAVED", PNG)


def stratify_R1():
    if not (os.path.exists(CACHE) and os.path.exists(NPZ)):
        return
    lat = np.load(CACHE); d = np.load(NPZ, allow_pickle=True)

    def l2n(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    z0, zt = l2n(lat["z0"]), l2n(lat["zt"])
    T = d["T"].astype(float)
    q1, q2 = np.quantile(T, [1 / 3, 2 / 3])
    print("\nR1 (initial-obs) stratified by episode length:")
    rows = {}
    for label, m in [("short", T <= q1), ("mid", (T > q1) & (T <= q2)), ("long", T > q2)]:
        Y, X = z0[m], zt[m]
        Lm = float(np.mean(np.sum((Y - Y.mean(0)) ** 2, 1)))
        Lt = float(np.mean(np.sum((Y - X) ** 2, 1)))
        rows[label] = dict(n=int(m.sum()), R_triv=1 - Lt / Lm, cos=float(np.mean(np.sum(Y * X, 1))))
        print(f"  {label:5s} n={int(m.sum()):5d} R_triv={rows[label]['R_triv']:+.3f} cos={rows[label]['cos']:+.3f}")
    json.dump(rows, open(os.path.join(OUT, "R1_stratified_droid.json"), "w"), indent=2)


if __name__ == "__main__":
    stratify_R1()
    plot()
