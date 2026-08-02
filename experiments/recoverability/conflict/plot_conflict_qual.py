"""Qualitative figure — how the action is actually generated (LA4VLA Fig 5 left, our version).

Four compasses: the paired-trained policy's predicted 2-D action, under the aligned baseline and a single-modality
conflict (flip vision / language / state to the opposite direction). Each arrow is a predicted action, colored by
the TRUE commanded direction; the thick arrow is the per-command mean; the faint grey tick is the command itself.
Flip vision → the action reverses; flip language / state → the action is unchanged (ignored).
Output: fig_conflict_qual.pdf + fig_conflict_qual.png
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "font.size": 12, "svg.fonttype": "none"})

d = np.load(os.path.join(OUT, "exp_conflict_qual.npz"))
theta = d["theta_true"]
DIRS = np.arange(8) * (np.pi / 4)
cmap = plt.get_cmap("hsv")

PANELS = [("aligned", "Aligned baseline", "action follows the command"),
          ("vision_flip", "Flip vision", "action reverses → vision exploited"),
          ("language_flip", "Flip language", "action unchanged → language ignored"),
          ("state_flip", "Flip state", "action unchanged → state ignored")]

fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.6), subplot_kw=dict(aspect="equal"))
rng = np.random.default_rng(0)
for ax, (key, title, note) in zip(axes, PANELS):
    P = d[key]
    # faint unit circle
    tt = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(tt), np.sin(tt), color=LINE, lw=1, zorder=1)
    ax.axhline(0, color=LINE, lw=0.8, zorder=1); ax.axvline(0, color=LINE, lw=0.8, zorder=1)
    for j, th in enumerate(DIRS):
        col = cmap(j / 8.0)
        mask = np.isclose(theta, th)
        pts = P[mask]
        # faint reference tick for the command direction
        ax.plot([0, 0.92*np.cos(th)], [0, 0.92*np.sin(th)], color=col, lw=1, ls=(0, (2, 3)), alpha=0.45, zorder=2)
        # subsample of individual predicted actions
        sub = pts[rng.choice(len(pts), size=min(22, len(pts)), replace=False)]
        ax.quiver(np.zeros(len(sub)), np.zeros(len(sub)), sub[:, 0], sub[:, 1], color=col, alpha=0.16,
                  angles="xy", scale_units="xy", scale=1, width=0.006, headwidth=3, zorder=3)
        # mean predicted action (thick)
        mu = pts.mean(0)
        ax.quiver(0, 0, mu[0], mu[1], color=col, angles="xy", scale_units="xy", scale=1, width=0.014,
                  headwidth=4, zorder=5, edgecolor="white", linewidth=0.5)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=8)
    ax.text(0, -1.60, note, ha="center", fontsize=10, color=MUTE)

fig.suptitle("How the action is actually generated — paired-trained policy, arrows colored by the commanded direction",
             fontsize=14.5, fontweight="bold", x=0.02, ha="left", y=1.02)
fig.text(0.5, -0.10, "Each thin arrow is one predicted 2-D action; the thick arrow is the per-command mean; the "
         "dashed tick is the command. Flip vision and every command's action rotates 180°; flip language or state "
         "and the action does not move.", ha="center", fontsize=8.8, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_conflict_qual.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_conflict_qual.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_conflict_qual")
