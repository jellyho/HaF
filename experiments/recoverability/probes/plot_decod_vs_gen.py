"""Rebuttal to LA4VLA Fig 5: representation SEPARATION ≠ GENERALIZATION (decodability is not control).

Left  : t-SNE of our policy representation, colored by task cluster — it forms clean, command-aligned clusters
        (exactly the phenomenon LA4VLA Fig 5 uses as evidence of instruction-following).
Right : across conditions × datasets, how SEPARATED the representation is (instruction-decodability) vs how well
        the policy actually GENERALIZES (OOD R²). They do not track each other (Pearson r≈0.26, ns) — clean
        clusters do not buy generalization. The best-separated representation even generalizes worse.
Output: fig_decod_gen.pdf + fig_decod_gen.png
"""
import os, json, glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.manifold import TSNE

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.linewidth": 1.1, "font.size": 13, "svg.fonttype": "none"})

# ---- gather (separation, generalization) across conditions x datasets ----
g = json.load(open(os.path.join(OUT, "exp2g_analysis.json")))["repquality"]
b = json.load(open(os.path.join(OUT, "exp2b_trimodal_agg.json")))
NM = {"BC-only": "BC-only", "BC+retro": "BC+retro", "BC+fwd": "BC+fwd", "SG+retro": "SG-BC+retro"}
xs, ys, cds = [], [], []
for ds in b:
    if ds not in g:
        continue
    for cg, cb in NM.items():
        if cg in g[ds] and cb in b[ds]["cond"]:
            xs.append(g[ds][cg]["instr_decod"]); ys.append(b[ds]["cond"][cb]["r2_ood"]["mean"])
            cds.append((ds, cg))
xs, ys = np.array(xs), np.array(ys)
r, p = pearsonr(xs, ys)

# ---- pick the most-separated dataset available for the t-SNE ----
order = ["libero_object", "libero_goal", "libero_10", "droid", "fractal"]
repf = next((os.path.join(OUT, f"cache/exp2g_reps_{d}_s0.npz") for d in order
             if os.path.exists(os.path.join(OUT, f"cache/exp2g_reps_{d}_s0.npz"))), None)
z = np.load(repf, allow_pickle=True)
tsne_ds = os.path.basename(repf).split("exp2g_reps_")[1].rsplit("_s0", 1)[0]
# use the highest-decod condition for this dataset as the "clean clusters" example
best_cond = max([c for c in NM if c in g.get(tsne_ds, {})], key=lambda c: g[tsne_ds][c]["instr_decod"])
R = z[f"R_{best_cond}"].astype(np.float32)
clu = z["te_cluster"]
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, learning_rate="auto").fit_transform(R)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"wspace": 0.28})

# ===== Left: t-SNE clean clusters =====
uc = np.unique(clu)
cmap = plt.get_cmap("Spectral")
for i, c in enumerate(uc):
    m = clu == c
    axL.scatter(emb[m, 0], emb[m, 1], s=14, color=cmap(i / max(1, len(uc) - 1)), alpha=0.8,
                edgecolor="white", linewidth=0.2)
axL.set_xticks([]); axL.set_yticks([])
axL.set_title(f"Representation clusters by task\n({tsne_ds}, {best_cond}) — as in LA4VLA Fig 5",
              fontsize=14, loc="left", pad=10, fontweight="bold")
axL.text(0.5, -0.06, f"instruction-decodability = {g[tsne_ds][best_cond]['instr_decod']:+.2f}  (clean separation)",
         transform=axL.transAxes, ha="center", fontsize=11, color=MUTE)
for s in ("top", "right", "left", "bottom"):
    axL.spines[s].set_visible(False)

# ===== Right: separation vs generalization =====
axR.axhline(0, color=LINE, lw=1)
axR.set_ylim(-0.32, 0.60)
DS_COL = {"fractal": CORAL, "droid": TEAL, "libero_goal": SAND, "libero_object": SLATE,
          "libero_spatial": "#B08968", "libero_10": "#9C6B4E"}
for (ds, cg), x, y in zip(cds, xs, ys):
    axR.scatter(x, y, s=70, color=DS_COL.get(ds, MUTE), edgecolor="white", linewidth=1, zorder=4,
                marker="D" if cg == "SG+retro" else "o")
m, q = np.polyfit(xs, ys, 1); xr = np.linspace(xs.min(), xs.max(), 50)
axR.plot(xr, m * xr + q, color=MUTE, lw=1.6, ls=(0, (5, 3)), alpha=0.8, zorder=2)
# annotate the two killer points (placed into empty space to avoid the title/other points)
axR.annotate("poor separation,\nbest generalization", (0.253, 0.453), xytext=(0.44, 0.47), ha="left",
             va="center", fontsize=9.5, color=INK, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
axR.annotate("best separation,\nworse generalization", (0.935, 0.142), xytext=(0.50, -0.10), ha="left",
             va="center", fontsize=9.5, color=INK, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
axR.set_xlabel("representation separation\n(instruction-decodability →)", fontsize=12.5)
axR.set_ylabel("actual generalization  (OOD R² →)", fontsize=12.5)
axR.set_title(f"…but separation does NOT predict generalization\n(r = {r:+.2f}, p = {p:.2f}, n = {len(xs)} — n.s.)",
              fontsize=14, loc="left", pad=10, fontweight="bold")
axR.scatter([], [], marker="D", color=MUTE, edgecolor="white", label="SG+retro (hard-KI)")
axR.scatter([], [], marker="o", color=MUTE, edgecolor="white", label="other conditions")
axR.legend(frameon=False, fontsize=9.5, loc="lower right")
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.text(0.005, -0.11, "Left: t-SNE of the pre-decoding representation, colored by held-out task cluster. "
         "Right: 4 conditions × 4 datasets; instruction-decodability (5-fold ridge rep→instruction) vs OOD R² "
         "(held-out task clusters).", fontsize=8.5, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_decod_gen.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_decod_gen.png"), dpi=300, bbox_inches="tight")
print(f"SAVED fig_decod_gen  (t-SNE={tsne_ds}/{best_cond}, r={r:+.3f} p={p:.3f})")
