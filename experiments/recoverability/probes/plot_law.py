"""The LAW: objective deep-recoverability vs OOD generalization (mini V-L-A, flow-matching BC).

Each dot = one auxiliary objective co-trained with a flow-matching BC head. x = deep recoverability
(1 - L_val/L_marg, end-to-end), y = OOD generalization (action R^2 on held-out task clusters), aggregated
over 3 seeds. The law: the more recoverable the objective (bigger shortcut), the WORSE the policy generalizes.
Output: fig_law.pdf + fig_law.png
"""
import os, json
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#C0A06B"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.linewidth": 1.1, "font.size": 12.5, "svg.fonttype": "none"})

S = [json.load(open(f"{OUT}/exp2h_fractal_s{s}.json")) for s in range(3)]
names = list(S[0]["objectives"])
bc = float(np.mean([s["BC_only_ood"] for s in S]))
TYP = {"prosp": ("prospective", CORAL), "retro": ("retrospective", SLATE), "intro": ("introspective", TEAL)}


def agg(n, f):
    v = [s["objectives"][n][f] for s in S if s["objectives"][n][f] is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


D = []
for n in names:
    t = S[0]["objectives"][n]["type"]
    dr, _ = agg(n, "deep_recoverability"); g, gs = agg(n, "generalization")
    D.append((n, t, dr, g, gs))
x = np.array([d[2] for d in D]); y = np.array([d[3] for d in D])
r, p = pearsonr(x, y)

fig, ax = plt.subplots(figsize=(8.8, 6.2))
b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
ax.plot(xs, b0 + b1 * xs, color=CORAL, lw=2.2, zorder=2, alpha=0.9)
ax.axhline(bc, color=MUTE, lw=1.1, ls=(0, (5, 4)), zorder=1)
ax.text(x.max(), bc + 0.004, "BC-only (no aux)", ha="right", va="bottom", fontsize=9, color=MUTE, style="italic")

for n, t, dr, g, gs in D:
    _, col = TYP[t]
    ax.errorbar(dr, g, yerr=gs, fmt="none", ecolor=col, elinewidth=1, alpha=0.35, zorder=3)
    ax.scatter([dr], [g], s=95, color=col, edgecolor="white", linewidth=1.3, zorder=5)


def lab(n, dx, dy, ha):
    d = next(d for d in D if d[0] == n)
    ax.annotate(n, (d[2], d[3]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9.2,
                color=INK, fontweight="bold")
lab("final-pose", -8, -14, "right")
lab("final-obs", 6, 8, "left")
lab("mae-mask25", 6, -14, "left")
lab("initial-pose", -8, 8, "right")

ax.text(x.min(), y.max() + 0.008, "hard to recover  ->  grounds the policy  ->  generalizes",
        ha="left", fontsize=10, color=TEAL, fontweight="bold")
ax.text(x.max(), y.min() - 0.012, "easy to recover  ->  shortcut  ->  overfits",
        ha="right", fontsize=10, color=CORAL, fontweight="bold")

ax.set_xlabel("deep recoverability of the auxiliary objective   (1 - L_val / L_marg,  end-to-end ->)", fontsize=12)
ax.set_ylabel("OOD generalization   (action R2, held-out tasks ->)", fontsize=12)
ax.set_title(f"The LAW: recoverability down => generalization up        Pearson r = {r:+.2f}  (p = {p:.3f})",
             fontsize=13.5, loc="left", pad=12, fontweight="bold")
for k, (lab_, col) in TYP.items():
    ax.scatter([], [], s=95, color=col, edgecolor="white", linewidth=1.3, label=lab_)
ax.legend(loc="lower left", frameon=False, fontsize=10.5, handletextpad=0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.text(0.005, -0.02,
         "Mini V-L-A (DINOv2-small + MiniLM + state -> transformer), flow-matching action head. RT-1/fractal, "
         "N=4000, 3 seeds. Each dot = one auxiliary objective co-trained with BC; deep recoverability = how "
         "cheaply the model learns to produce the target end-to-end. The most recoverable target (final-pose) "
         "generalizes worst - below BC-only.", fontsize=8.3, color=MUTE, wrap=True)
fig.savefig(f"{OUT}/fig_law.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_law.png", dpi=300, bbox_inches="tight")
print(f"SAVED fig_law  (r={r:+.3f}, p={p:.3f}, n={len(D)})")
