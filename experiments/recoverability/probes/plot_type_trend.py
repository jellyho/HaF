"""Do the 3 objective TYPES separate on the recoverability axis? — Honest answer: no.

Each target is placed on the recoverability axis (G_obs from o_t), in a lane by type (retrospective /
prospective / introspective). Retrospective and prospective overlap almost entirely; the real gradient is what
the target REQUIRES — copy the present (high), extrapolate (medium), infer an abstract/erased quantity (low).
Output: fig_type_trend.pdf + fig_type_trend.png
"""
import os, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; SLATE = "#4A5A63"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.linewidth": 1.1, "font.size": 12.5, "svg.fonttype": "none"})

DIV = ["fractal", "droid", "bridge"]
data = {d: json.load(open(os.path.join(OUT, f"results_{d}.json"))) for d in DIV}
TYPES = {
    "retrospective": (SLATE, [("Mnear past-obs k~5", "near-past"), ("Mfar past-obs k~45", "far-past"),
                              ("R1 initial-obs", "initial-obs"), ("Mgrip prev-gripper", "prev-grip"),
                              ("Mact prev-action", "prev-act"), ("R2 initial-pose", "init-pose")]),
    "prospective": (CORAL, [("P1s future-obs k~5", "near-fut"), ("P1l future-obs k~45", "far-fut"),
                            ("P0 final-obs", "final-obs"), ("P3 future-gripper", "fut-grip"),
                            ("P2 future-action", "fut-act"), ("P0p final-pose", "final-pose")]),
    "introspective": (TEAL, [("I-gripper now", "gripper"), ("I-progress t/T", "progress"),
                             ("R3 instruction", "instruction"), ("Iinv act|both-frames", "inv-dyn"),
                             ("BC action|o_t", "action")]),
}


def gobs(k):
    v = [data[d][k]["G_obs"] for d in DIV if k in data[d] and data[d][k].get("G_obs") is not None]
    return float(np.mean(v)) if v else None


fig, ax = plt.subplots(figsize=(12.4, 5.6))
# requirement zones (by recoverability)
ax.axvspan(0.55, 0.95, color="#FBEDE6", alpha=0.5, zorder=0)
ax.axvspan(0.28, 0.55, color="#F4EEE4", alpha=0.6, zorder=0)
ax.axvspan(-0.62, 0.28, color="#EAF3F1", alpha=0.5, zorder=0)
ax.text(0.75, 3.62, "COPY the present", ha="center", fontsize=10.5, color=CORAL, fontweight="bold")
ax.text(0.415, 3.62, "extrapolate", ha="center", fontsize=10.5, color=SAND, fontweight="bold")
ax.text(-0.08, 3.62, "infer / abstract", ha="center", fontsize=10.5, color=TEAL, fontweight="bold")

lanes = {"retrospective": 3, "prospective": 2, "introspective": 1}
for tname, (col, items) in TYPES.items():
    y = lanes[tname]
    xs = []
    for k, lab in items:
        g = gobs(k)
        if g is None:
            continue
        xs.append(g)
        ax.scatter([g], [y], s=90, color=col, edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(lab, (g, y), xytext=(0, 11 if items.index((k, lab)) % 2 == 0 else -17),
                    textcoords="offset points", ha="center", fontsize=8.2, color=INK)
    # type mean marker
    m = np.mean(xs)
    ax.plot([m, m], [y - 0.22, y + 0.22], color=col, lw=3, zorder=6)
    ax.text(-0.60, y, tname, ha="left", va="center", fontsize=12, color=col, fontweight="bold")
    ax.text(0.98, y, f"mean {m:+.2f}", ha="right", va="center", fontsize=10, color=col, style="italic")

ax.axvline(0, color=LINE, lw=1)
ax.set_xlim(-0.62, 1.0); ax.set_ylim(0.4, 3.9)
ax.set_yticks([])
ax.set_xlabel("recoverability from the current observation  (G_obs →)", fontsize=12.5)
ax.set_title("The 3 objective types do NOT separate on recoverability — retro ≈ prospective; "
             "the gradient is what the target requires",
             fontsize=13.5, loc="left", pad=12, fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.text(0.005, -0.02, "Each dot = one target, mean G_obs over RT-1 · DROID · Bridge; thick bar = type mean. "
         "Retrospective and prospective overlap (both span copy→infer); introspective sits low but is confounded "
         "(non-obs targets + near-zero probe on DROID).", fontsize=8.3, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_type_trend.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_type_trend.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_type_trend")
