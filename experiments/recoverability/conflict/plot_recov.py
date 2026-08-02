"""Figure: RECOVERABILITY governs which modality a policy exploits (the winner rotates).

Three regimes, each makes a DIFFERENT modality the most recoverable (highest SNR); all three always carry the
true direction. The exploited modality (reliance = DAR paired − DAR conflict) rotates vision→language→state,
following recoverability — not modality identity. Output: fig_recov.pdf + fig_recov.png.
"""
import os, glob, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
MODS = ["vision", "language", "state"]
REGS = ["V", "L", "S"]
REGLAB = {"V": "vision\nmost recoverable", "L": "language\nmost recoverable", "S": "state\nmost recoverable"}
MODCOL = {"vision": "#D2795E", "language": "#5AA099", "state": "#B0A597"}
INK = "#1F1F1F"; MUTE = "#6E6A66"; LINE = "#DAD6D0"; BG = "#FFFFFF"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LINE,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.linewidth": 1.0,
    "font.size": 11, "svg.fonttype": "none",
})


def rel(R):
    fs = sorted(glob.glob(os.path.join(OUT, f"exp_recov_{R}_s*.json")))
    return ({m: np.mean([json.load(open(f))["reliance"][m] for f in fs]) for m in MODS},
            {m: np.std([json.load(open(f))["reliance"][m] for f in fs]) for m in MODS})


fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(len(REGS)); bw = 0.24
means = {R: rel(R)[0] for R in REGS}; stds = {R: rel(R)[1] for R in REGS}
for j, m in enumerate(MODS):
    vals = [means[R][m] for R in REGS]; errs = [stds[R][m] for R in REGS]
    ax.bar(x + (j - 1) * bw, vals, bw, yerr=errs, capsize=2.5, color=MODCOL[m], label=m,
           edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 1, "ecolor": MUTE, "alpha": 0.6})
# mark the exploited (winning) modality per regime
for i, R in enumerate(REGS):
    win = max(MODS, key=lambda m: means[R][m]); j = MODS.index(win)
    ax.annotate("exploited", (x[i] + (j - 1) * bw, means[R][win] + 0.03), ha="center", va="bottom",
                fontsize=8.6, color=MODCOL[win], fontweight="bold")
ax.axhline(0, color=INK, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels([REGLAB[R] for R in REGS], fontsize=10)
ax.set_ylim(-0.12, 1.18)
ax.set_ylabel("reliance   (DAR paired − DAR conflict)", fontsize=10.5)
ax.set_title("Recoverability governs which modality is exploited — the winner rotates",
             fontsize=13, loc="left", pad=18, fontweight="bold")
ax.text(0, 1.055, "all three inputs always carry the true command; only their recoverability (SNR) differs",
        fontsize=9, color=MUTE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=10.5,
          handlelength=1.1)
fig.text(0.005, -0.14, "Paired-trained DINOv2-small policy · 3 seeds · single-modality conflict diagnostic. "
         "In each regime one modality is high-SNR, the other two low-SNR (but still carry θ).",
         fontsize=8, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_recov.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_recov.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_recov.pdf + fig_recov.png")
