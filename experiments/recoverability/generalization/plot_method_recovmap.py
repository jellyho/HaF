"""Method schematic for Evidence 1 — how we measure recoverability (Exp 1), with the fair matched-pair logic.

Sample frames at symmetric offsets around the present, encode each with DINOv2, then ask how well COPYING the
present latent predicts each target (R_triv). Each retrospective target is compared with its mirror prospective
target (same horizon / endpoint) so the comparison is fair.
Output: fig_m_recovmap.pdf + fig_m_recovmap.png
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"; PANEL = "#F7F3EC"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "svg.fonttype": "none"})

fig, ax = plt.subplots(figsize=(12.6, 6.4)); ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.4); ax.axis("off")
ax.text(0.15, 6.05, "How we measure recoverability (Evidence 1)", fontsize=17, fontweight="bold", color=INK)
ax.text(0.15, 5.62, "sample frames at symmetric offsets · encode each · ask how well COPYING the present predicts "
        "the target", fontsize=11.5, color=MUTE)

# ---- timeline ----
yT = 3.7
ax.plot([0.7, 11.9], [yT, yT], color=LINE, lw=2, zorder=1)
# marks: (x, label, offset-label, color, kind)
marks = [(0.9, "o₀", "origin", SLATE), (2.6, "o₋₄₅", "far-past", SLATE), (4.4, "o₋₅", "near-past", SLATE),
         (6.3, "oₜ", "NOW", CORAL), (8.2, "o₊₅", "near-future", TEAL), (10.0, "o₊₄₅", "far-future", TEAL),
         (11.7, "o_T", "goal", TEAL)]
for x, lab, off, col in marks:
    big = lab == "oₜ"
    ax.scatter([x], [yT], s=260 if big else 150, color=col, zorder=4, edgecolor="white", linewidth=1.4)
    ax.text(x, yT + 0.34, lab, ha="center", fontsize=11.5 if big else 10.5, fontweight="bold", color=col)
    ax.text(x, yT - 0.42, off, ha="center", fontsize=9, color=MUTE)
ax.text(5.35, yT - 0.42, "input:", ha="right", fontsize=8.5, color=CORAL, style="italic")

# ---- matched-pair arcs above the timeline ----
def arc(x1, x2, h, label, col):
    cx = (x1 + x2) / 2; w = (x2 - x1)
    ax.add_patch(Arc((cx, yT), w, h, theta1=0, theta2=180, color=col, lw=1.6, ls=(0, (4, 2))))
    ax.text(cx, yT + h/2 + 0.06, label, ha="center", fontsize=8.8, color=col, fontweight="bold")


arc(4.4, 8.2, 1.0, "near-past ↔ near-future  (±5)", INK)
arc(2.6, 10.0, 1.7, "far-past ↔ far-future  (±45)", INK)
arc(0.9, 11.7, 2.5, "origin ↔ goal  (endpoints)", CORAL)

# ---- encode step ----
box = FancyBboxPatch((3.4, 2.05), 5.8, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12", fc=PANEL, ec=LINE, lw=1.4)
ax.add_patch(box)
ax.text(6.3, 2.41, "each frame → DINOv2-base → 768-d latent z  (L2-normalized)", ha="center", va="center",
        fontsize=11, color=INK, fontweight="bold")
ax.add_patch(FancyArrowPatch((6.3, 3.4), (6.3, 2.79), arrowstyle="-|>", mutation_scale=14, color=MUTE, lw=1.5))

# ---- R_triv formula + interpretation ----
ax.add_patch(FancyBboxPatch((0.5, 0.25), 11.6, 1.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#FBEDE6", ec=CORAL, lw=1.5))
ax.text(0.85, 1.28, "recoverability of a target", fontsize=10.5, color=CORAL, fontweight="bold")
ax.text(0.85, 0.82, r"$R_{triv} = 1 - \dfrac{\mathrm{MSE}(\,\mathrm{copy}\ z_t \rightarrow z_{target}\,)}"
        r"{\mathrm{MSE}(\,\mathrm{mean} \rightarrow z_{target}\,)}$", fontsize=15, color=INK, va="center")
ax.text(6.7, 1.02, "high (→1): the target is ≈ the present", fontsize=10.5, color=INK)
ax.text(6.7, 1.02 - 0.02, "", fontsize=1)
ax.text(6.95, 0.72, "→ a copy shortcut", fontsize=10, color=CORAL, fontweight="bold")
ax.text(6.7, 0.42, "low (→0): copying the present fails", fontsize=10.5, color=INK)
ax.text(6.95, 0.12 + 0.0, "", fontsize=1)
ax.text(10.55, 0.42, "→ must be modeled", fontsize=10, color=TEAL, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig_m_recovmap.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_m_recovmap.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_m_recovmap")
