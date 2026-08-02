"""Concept figure: why past ≠ future prediction — goal-directed demos are irreversible (the entropy funnel).

Top: demonstration trajectories start spread (many origins) and CONVERGE to the goal. Forward = many-to-one
(recoverable); backward = one-to-many (origin washed out).
Bottom: recoverability of a frame at time t given the present — symmetric near the present (both copies), low at
the far past (erased), higher toward the goal. Real-data R_triv anchors overlaid.
Layout uses separate horizontal bands so trajectories / arrows / labels never overlap.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
try:
    from scipy.interpolate import PchipInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
LOWBG = "#FADFD4"; HIBG = "#E1EEEB"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "font.size": 12, "svg.fonttype": "none"})

x = np.linspace(0, 1, 300)
present_x = 0.45
goal_y = 0.66
starts = np.linspace(0.40, 0.94, 11)   # trajectories live in the UPPER band (y>0.36); bottom band stays clear

fig, (axT, axB) = plt.subplots(2, 1, figsize=(11.6, 8.0),
                               gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.42})

# ================= TOP: the funnel =================
axT.axvspan(0, present_x, color=LOWBG, alpha=0.45, zorder=0)
axT.axvspan(present_x, 1, color=HIBG, alpha=0.45, zorder=0)
for s in starts:
    conv = (1 - x) ** 1.5
    wig = 0.022 * np.sin(3.1 * np.pi * x + s * 8) * (1 - x)
    y = goal_y + (s - goal_y) * conv + wig
    axT.plot(x, y, color=TEAL, lw=1.3, alpha=0.55, zorder=2)
    axT.scatter([0], [y[0]], s=15, color=SAND, zorder=3, edgecolor="white", linewidth=0.4)
axT.scatter([1], [goal_y], s=150, color=CORAL, zorder=5, edgecolor="white", linewidth=1.2)
axT.text(1.012, goal_y, "goal", va="center", ha="left", fontsize=12, color=CORAL, fontweight="bold")
# origin label — clear top-left, above the highest trajectory
axT.text(0.0, 1.16, "many possible origins", ha="left", va="bottom", fontsize=10.5, color=SAND, fontweight="bold")
# present line — only across the trajectory band, not into the arrow band
axT.plot([present_x, present_x], [0.34, 1.10], color=INK, lw=1.6, ls=(0, (4, 3)), zorder=4)
axT.text(present_x, 1.16, "now (observed)", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
# arrows + labels live in the CLEAR bottom band (y < 0.30)
axT.add_patch(FancyArrowPatch((present_x - 0.02, 0.20), (0.05, 0.20), arrowstyle="-|>", mutation_scale=15,
                              color=SLATE, lw=1.8))
axT.text((present_x + 0.05) / 2 - 0.02, 0.05, "backward: one-to-many\n(diverges — origin erased)", ha="center",
         va="bottom", fontsize=9.8, color=SLATE, fontweight="bold")
axT.add_patch(FancyArrowPatch((present_x + 0.02, 0.20), (0.95, 0.20), arrowstyle="-|>", mutation_scale=15,
                              color=TEAL, lw=1.8))
axT.text((present_x + 0.95) / 2 + 0.02, 0.05, "forward: many-to-one\n(converges to the goal)", ha="center",
         va="bottom", fontsize=9.8, color=TEAL, fontweight="bold")
axT.set_xlim(-0.02, 1.12); axT.set_ylim(0, 1.34)
axT.set_xticks([]); axT.set_yticks([])
for sp in axT.spines.values():
    sp.set_visible(False)
axT.set_title("Why past ≠ future prediction: goal-directed demonstrations are irreversible",
              fontsize=14.5, fontweight="bold", loc="left", pad=12)

# ================= BOTTOM: recoverability-from-present curve =================
anch = [(0.0, 0.07), (0.15, 0.33), (0.30, 0.70), (present_x, 1.0), (0.60, 0.72), (0.80, 0.52), (1.0, 0.42)]
ax_x = np.array([a for a, _ in anch]); ax_y = np.array([b for _, b in anch])
curve = PchipInterpolator(ax_x, ax_y)(x) if HAVE_SCIPY else np.interp(x, ax_x, ax_y)
axB.axvspan(0, present_x, color=LOWBG, alpha=0.45, zorder=0)
axB.axvspan(present_x, 1, color=HIBG, alpha=0.45, zorder=0)
axB.plot(x, curve, color=CORAL, lw=3, zorder=4)
axB.plot([present_x, present_x], [0, 1.0], color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3)
# real-data anchors (Exp1 R_triv) — points with labels placed clear of the curve
axB.scatter([0.05], [0.10], s=70, color=CORAL, zorder=6, edgecolor="white", linewidth=1)
axB.scatter([0.30], [0.70], s=70, color=SLATE, zorder=6, edgecolor="white", linewidth=1)
axB.scatter([0.60], [0.70], s=70, color=SLATE, zorder=6, edgecolor="white", linewidth=1)
axB.annotate("initial-obs (far past)", (0.05, 0.10), xytext=(0.12, 0.095), fontsize=9, color=CORAL,
             fontweight="bold", ha="left", va="center")
axB.text(0.30, 0.60, "near-past", ha="center", va="top", fontsize=9, color=SLATE, fontweight="bold")
axB.text(0.60, 0.60, "near-future", ha="center", va="top", fontsize=9, color=SLATE, fontweight="bold")
# zone labels ABOVE the curve peak (curve max = 1.0), in the headroom
axB.text(0.02, 1.34, "far past = washed out\n→ shortcut-FREE (hindsight)", fontsize=9.4, color=CORAL,
         va="top", ha="left", fontweight="bold", linespacing=1.25)
axB.text(0.985, 1.34, "converges + smooth\n→ shortcut (prospective)", fontsize=9.4, color=TEAL,
         va="top", ha="right", fontweight="bold", linespacing=1.25)
axB.set_xlim(-0.02, 1.12); axB.set_ylim(0, 1.48)
axB.set_yticks([0, 0.5, 1.0]); axB.set_yticklabels(["0", ".5", "1"])
axB.set_xticks([0.0, present_x, 1.0]); axB.set_xticklabels(["origin", "now", "goal"], fontsize=11)
axB.set_ylabel("recoverability\nfrom the present", fontsize=10.5)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)

fig.text(0.5, -0.01, "Near-term is symmetric — both directions are ≈ a copy of the present, so predicting an "
         "adjacent frame is a shortcut either way. Only the far ends differ; the asymmetry is entropy, not the "
         "arrow of time.", ha="center", va="top", fontsize=9.2, color=MUTE, style="italic")
fig.savefig(os.path.join(OUT, "fig_funnel.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_funnel.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_funnel")
