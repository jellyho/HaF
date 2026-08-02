"""Method schematics — 'how the experiment works' diagrams (clean, PI-style).

fig_m_sim.png   : the controlled generalization simulation (hidden z → shortcut vs entangled channel → policy).
fig_m_probe.png : the direction probe (three inputs all carry the command → policy → conflict diagnostic).
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; PANEL = "#F7F3EC"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "svg.fonttype": "none"})


def box(ax, x, y, w, h, title, sub=None, fc=PANEL, ec=LINE, tc=INK, lw=1.4, title_size=13, sub_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    ax.text(x + w/2, y + h/2 + (0.11 if sub else 0), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.16, sub, ha="center", va="center", fontsize=10,
                color=sub_color or MUTE)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, ls=ls, shrinkA=2, shrinkB=2))


# ================= fig_m_sim =================
fig, ax = plt.subplots(figsize=(12.4, 5.0)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 5); ax.axis("off")
ax.text(0.1, 4.72, "How the controlled proof works", fontsize=17, fontweight="bold", color=INK)
ax.text(0.1, 4.3, "one hidden cause, two ways to read it — a cheap shortcut and an honest-but-hard channel",
        fontsize=11.5, color=MUTE)

box(ax, 0.25, 2.05, 2.35, 1.0, "hidden cause  z", "the true task", fc="#EFEBE3")
# two channels
box(ax, 3.4, 3.15, 3.5, 1.0, "shortcut  x_easy", "trivially readable", fc="#FBEDE6", ec=CORAL, lw=1.6)
ax.text(5.15, 2.98, "= the answer in training,  but noise on new data", ha="center", fontsize=9.6,
        color=CORAL, fontstyle="italic")
box(ax, 3.4, 1.05, 3.5, 1.0, "entangled  x_hard = g(z)", "must be learned · always reliable", fc="#EAF0EE", ec=TEAL, lw=1.6)
# policy + action
box(ax, 7.9, 2.05, 2.0, 1.0, "policy", "neural net", fc="#EFEBE3")
box(ax, 10.3, 2.05, 1.8, 1.0, "action  y", None, fc="#F7F3EC")

arrow(ax, 2.6, 2.75, 3.4, 3.55, color=MUTE)
arrow(ax, 2.6, 2.35, 3.4, 1.65, color=MUTE)
arrow(ax, 6.9, 3.5, 7.9, 2.75, color=MUTE)
arrow(ax, 6.9, 1.5, 7.9, 2.35, color=MUTE)
arrow(ax, 9.9, 2.55, 10.3, 2.55, color=INK)

# aux callout
ax.add_patch(FancyBboxPatch((3.4, 0.12), 6.5, 0.62, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="none", ec=CORAL, lw=1.5, ls=(0, (5, 3))))
ax.text(3.62, 0.43, "auxiliary objective:", fontsize=10.5, color=CORAL, fontweight="bold", va="center")
ax.text(6.05, 0.43, "predict y with the shortcut hidden (fraction 1−ρ)  →  can't use x_easy",
        fontsize=10.2, color=INK, va="center")
fig.savefig(os.path.join(OUT, "fig_m_sim.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_m_sim.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ================= fig_m_probe =================
fig, ax = plt.subplots(figsize=(12.4, 5.0)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 5); ax.axis("off")
ax.text(0.1, 4.72, "How the ignoring probe works", fontsize=17, fontweight="bold", color=INK)
ax.text(0.1, 4.3, "all three inputs carry the same command — then we flip one and watch the action",
        fontsize=11.5, color=MUTE)

ax.text(1.5, 3.75, 'command:  "move in direction θ"', fontsize=12, color=INK, fontweight="bold", ha="left")
box(ax, 0.3, 2.35, 2.7, 0.95, "vision", "a blob at angle θ", fc="#FBEDE6", ec=CORAL, lw=1.6)
box(ax, 0.3, 1.25, 2.7, 0.95, "language", '"move north…"', fc="#EAF0EE", ec=TEAL, lw=1.6)
box(ax, 0.3, 0.15, 2.7, 0.95, "state", "a direction vector", fc="#F0EDE7", ec=SAND, lw=1.6)
ax.text(1.65, 3.42, "each independently encodes θ", fontsize=9.5, color=MUTE, ha="center")

box(ax, 4.4, 1.35, 2.0, 1.4, "policy", "trained\n(paired)", fc="#EFEBE3")
box(ax, 7.0, 1.55, 1.9, 1.0, "action", None, fc="#F7F3EC")
for y in (2.82, 1.72, 0.62):
    arrow(ax, 3.0, y, 4.4, 2.05, color=MUTE)
arrow(ax, 6.4, 2.05, 7.0, 2.05, color=INK)

# diagnostic
ax.add_patch(FancyBboxPatch((9.2, 0.3), 3.0, 3.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#FBEDE6", ec=CORAL, lw=1.5))
ax.text(10.7, 3.4, "the diagnostic", fontsize=12, color=CORAL, fontweight="bold", ha="center")
ax.text(10.7, 2.75, "flip ONE input\nto the opposite\ndirection —", fontsize=10.8, color=INK, ha="center", va="center")
ax.text(10.7, 1.75, "does the action\nfollow it?", fontsize=10.8, color=INK, ha="center", va="center", fontweight="bold")
ax.text(10.7, 0.85, "follows ⇒ exploited\nunmoved ⇒ ignored", fontsize=9.8, color=MUTE, ha="center", va="center")
arrow(ax, 8.9, 2.05, 9.2, 2.05, color=CORAL)
fig.savefig(os.path.join(OUT, "fig_m_probe.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_m_probe.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("SAVED fig_m_sim + fig_m_probe")
