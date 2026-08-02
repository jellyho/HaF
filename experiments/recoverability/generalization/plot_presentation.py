"""Presentation figures — one message per figure, clean (for the research-proposal talk).

fig_p2_overfit.png : shortcut ⇒ overfit (BC fits train & IID, fails OOD).
fig_p3_cure.png    : the cure — a low-recoverability objective restores generalization (monotone) + real data.
(fig_p1 = the winner-rotation figure fig_recov.png, built by conflict/plot_recov.py.)
"""
import os, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#D2795E"; TEAL = "#5AA099"; SAND = "#B0A597"; SLATE = "#4A5A63"
INK = "#1F1F1F"; MUTE = "#6E6A66"; LINE = "#DAD6D0"; BG = "#FFFFFF"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LINE,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.linewidth": 1.1,
    "font.size": 15, "svg.fonttype": "none",
})
S = json.load(open(os.path.join(OUT, "exp_sim_analysis.json")))
bc = S["conditions"]["BC-only"]
sweep = sorted(S["rho_sweep"], key=lambda d: d["recoverability"])

# ============ fig_p2 : shortcut ⇒ overfit ============
fig, ax = plt.subplots(figsize=(6.6, 5.2))
labs = ["train", "in-distribution\ntest", "NEW situations\n(out-of-dist.)"]
vals = [bc["train"][0], bc["iid"][0], bc["ood"][0]]
cols = [SLATE, SLATE, CORAL]
b = ax.bar(range(3), vals, 0.62, color=cols, edgecolor="white", linewidth=1)
ax.bar_label(b, labels=[f"{v:.2f}" for v in vals], padding=6, fontsize=15, fontweight="bold",
             color=INK)
ax.annotate("the shortcut breaks,\nso the policy fails", (1.68, 1.9), xytext=(0.30, 1.35),
            ha="left", va="center", fontsize=14, color=CORAL, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.8))
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=13.5)
ax.set_ylabel("prediction error  (↓ better)", fontsize=14)
ax.set_ylim(0, 2.15)
ax.set_title("A policy that takes a shortcut\noverfits: perfect on training, fails on new data",
             fontsize=16.5, loc="left", pad=14, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT, "fig_p2_overfit.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_p2_overfit.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ============ fig_p3 : the cure (monotone + real data) ============
fig, (axB, axC) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"wspace": 0.34})
rec = [d["recoverability"] for d in sweep]
ood = [d["ood"][0] for d in sweep]; oode = [d["ood"][1] for d in sweep]
axB.set_yscale("log")
axB.errorbar(rec, ood, yerr=oode, marker="o", ms=9, lw=3, color=CORAL, capsize=3,
             mfc="white", mec=CORAL, mew=2.4, zorder=5)
axB.axhline(bc["ood"][0], color=SAND, lw=1.8, ls=(0, (5, 3)))
axB.text(0.98, bc["ood"][0] * 1.06, "no auxiliary (behavior cloning)", fontsize=12, color=MUTE, ha="right", va="bottom")
axB.annotate("shortcut-free\nobjective", (0.0, ood[0]), xytext=(0.16, 0.20), fontsize=13, color=CORAL,
             fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.5))
axB.set_xlabel("how recoverable the objective is\nfrom the shortcut  (→ more recoverable)", fontsize=13.5)
axB.set_ylabel("error on NEW situations  (log, ↓ better)", fontsize=13.5)
axB.set_title("Adding a shortcut-free objective\nrestores generalization", fontsize=16, loc="left", pad=14, fontweight="bold")
axB.set_xlim(-0.05, 1.05)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

c = json.load(open(os.path.join(OUT, "exp2c_analysis.json")))["axisB"]
vl = np.array(c["vision_legit"]); ben = np.array(c["mean_benefit"]); ds = c["datasets"]
r = c.get("vislegit_vs_benefit_pearson", float("nan"))
axC.scatter(vl, ben, s=90, color=TEAL, edgecolor="white", linewidth=1.2, zorder=5)
m, q = np.polyfit(vl, ben, 1); xs = np.linspace(vl.min(), vl.max(), 50)
axC.plot(xs, m * xs + q, color=TEAL, lw=2, ls=(0, (5, 3)), alpha=0.85)
axC.axhline(0, color=LINE, lw=1)
for x, y, name in zip(vl, ben, ds):
    axC.annotate(name.replace("libero_", "lib-").replace("fractal", "RT-1"), (x, y), fontsize=10.5,
                 color=MUTE, xytext=(5, 4), textcoords="offset points")
axC.set_xlabel("← vision is a shortcut     vision truly needed →", fontsize=13.5)
axC.set_ylabel("benefit of the auxiliary", fontsize=13.5)
axC.set_title(f"Same law on 6 real robot datasets\n(r = {r:+.2f})", fontsize=16, loc="left", pad=14, fontweight="bold")
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT, "fig_p3_cure.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_p3_cure.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("SAVED fig_p2_overfit + fig_p3_cure")
