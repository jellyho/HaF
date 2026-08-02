"""Compute-free: R_triv (shortcut) vs probe_beyond_trivial (learnable beyond the cheat), 3 datasets."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DS = [("droid", "DROID (Franka)"), ("bridge", "Bridge (WidowX)"), ("fractal", "RT-1 (Google)")]


def col(name):
    p = name.split()[0]
    if p[0] in ("R", "M"):
        return "#9E2A4F"
    if p[0] == "P":
        return "#1F4E6B"
    if p.startswith("I"):
        return "#4A85A6"
    return "#2F6B4F"


fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, (tag, title) in zip(axes, DS):
    res = json.load(open(os.path.join(OUT, f"results_{tag}.json")))
    ax.axvspan(-1.0, 0.25, color="#FBEDF1", alpha=0.4, zorder=0)  # low shortcut
    ax.axhspan(0.1, 0.8, color="#E7F1EB", alpha=0.35, zorder=0)   # learnable
    for name, v in res.items():
        x, y = v["R_triv"], v["probe_beyond_trivial"]
        c = col(name)
        star = name.startswith("R1")
        ax.scatter([x], [y], s=180 if star else 80, marker="*" if star else "o",
                   color=c, edgecolor="white", linewidth=1.1, zorder=4 if star else 3)
        if star or y > 0.15 or (x > 0.7):
            ax.annotate(name.split(" k")[0], (x, y), textcoords="offset points",
                        xytext=(7, 4), fontsize=8, color=c)
    ax.axhline(0, color="#CDD4DE", lw=1); ax.axvline(0, color="#CDD4DE", lw=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("R_triv — shortcut availability →")
    ax.set_xlim(-1.0, 1.0)
axes[0].set_ylabel("probe beyond copy — learnable past the shortcut →")
axes[0].set_ylim(-0.15, 0.7)
axes[1].text(0.5, 0.62, "★ = initial-obs (retrospective anchor)", fontsize=9, color="#9E2A4F", ha="center")
fig.suptitle("The useful corner (low shortcut, high learnability) is where the retrospective initial-obs lands — more so as scenes get structured",
             fontsize=12.5, y=1.02)
fig.tight_layout()
p = os.path.join(OUT, "exp1_learnable.png")
fig.savefig(p, dpi=140, bbox_inches="tight")
print("SAVED", p)
