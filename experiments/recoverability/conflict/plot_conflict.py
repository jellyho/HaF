"""Publication figure for the LA4VLA preliminary probe (Physical-Intelligence-style aesthetic).

Message (LA4VLA §3, generalized): a policy trained on fully-paired inputs commits to ONE modality. Flip that
modality and the action collapses (DAR → chance / below); flip a modality it ignored and nothing moves. The
'ignoring' axis is general over {vision, language, state}, not language-specific.

Panel A: DAR across the four LA4VLA conditions (paired / removed / unaligned / conflict), grouped by modality.
Panel B: reliance = DAR(paired) − DAR(conflict) per modality — who the policy committed to.
Outputs: fig_conflict.pdf + fig_conflict.png (300 DPI).
"""
import os, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
MODS = ["vision", "language", "state"]
CONDS = ["paired", "removed", "unaligned", "conflict"]
CLABEL = {"paired": "paired  +", "removed": "removed  ∅", "unaligned": "unaligned  ~", "conflict": "conflict  −"}
# softened PI palette
MODCOL = {"vision": "#D2795E", "language": "#5AA099", "state": "#B0A597"}
# condition ramp: intact slate → coral break
CCOL = {"paired": "#4A5A63", "removed": "#9AA7AD", "unaligned": "#CBB9A6", "conflict": "#D2795E"}
INK = "#1F1F1F"; MUTE = "#6E6A66"; LINE = "#DAD6D0"; BG = "#FFFFFF"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LINE,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.linewidth": 1.0,
    "font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none",
})

agg = json.load(open(os.path.join(OUT, "exp_conflict_analysis.json")))
M = agg["modalities"]


def dar(mod, cond):
    return M[mod]["conditions"][cond]["DAR"]  # [mean, std]


fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                               gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.30})

# ============ Panel A: DAR across 4 conditions, grouped by modality ============
x = np.arange(len(MODS)); bw = 0.19
for k, cond in enumerate(CONDS):
    vals = [dar(m, cond)[0] for m in MODS]
    errs = [dar(m, cond)[1] for m in MODS]
    axA.bar(x + (k - 1.5) * bw, vals, bw, yerr=errs, capsize=2, color=CCOL[cond], label=CLABEL[cond],
            edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 1, "ecolor": MUTE, "alpha": 0.6})
axA.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)))
axA.text(len(MODS) - 0.5, 0.515, "chance", fontsize=8.2, color=MUTE, ha="right")
# mark the modality whose conflict bar collapses (the exploited one)
for i, m in enumerate(MODS):
    if M[m]["reliance_DAR"] > 0.15:
        axA.annotate("action flips\nto the opposite", (x[i] + 1.5 * bw, dar(m, "conflict")[0] + 0.01),
                     xytext=(x[i] + 1.5 * bw + 0.02, 0.30), ha="center", va="bottom",
                     fontsize=8.3, color=MODCOL[m], fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=MODCOL[m], lw=1.2))
axA.set_xticks(x); axA.set_xticklabels([m for m in MODS], fontsize=11)
axA.set_ylim(0, 1.08); axA.set_ylabel("DAR  (action aligned with the\ntrue commanded direction)", fontsize=10)
axA.set_title("Flip one modality — only the exploited one breaks the action",
              fontsize=12.5, loc="left", pad=10, color=INK, fontweight="bold")
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=9.2, handlelength=1.0)

# ============ Panel B: reliance per modality ============
rank = agg["exploited_ranking"]
rvals = [M[m]["reliance_DAR"] for m in rank]
y = np.arange(len(rank))[::-1]
axB.barh(y, rvals, color=[MODCOL[m] for m in rank], edgecolor="white", height=0.62)
for yi, m, v in zip(y, rank, rvals):
    axB.text(v + 0.02, yi, f"{v:+.2f}", va="center", fontsize=10, color=INK, fontweight="bold")
axB.set_yticks(y); axB.set_yticklabels(rank, fontsize=11)
axB.set_xlim(0, max(0.3, max(rvals) * 1.35))
axB.set_title("reliance = DAR(paired) − DAR(conflict)", fontsize=11, loc="left", pad=10, color=INK, fontweight="bold")
axB.text(0, len(rank) - 0.35, "how much flipping each modality\nbreaks the action", fontsize=8.4, color=MUTE, va="bottom")
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(length=0)

fig.text(0.005, -0.03, f"LA4VLA §3 preliminary probe, generalized to all modalities · paired-trained "
         f"DINOv2-small policy · {agg['n_seeds']} seeds · four conditions per modality {{paired, removed, "
         f"unaligned, conflict}} · metrics DAR/DCS/SR/SS.", fontsize=8, color=MUTE)

fig.savefig(os.path.join(OUT, "fig_conflict.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_conflict.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_conflict.pdf + fig_conflict.png")
