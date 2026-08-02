"""Conceptual figure: the lesson from LLMs — a low-recoverability objective forces general computation.

Three columns: LLM next-token prediction (low recoverability → meta-learning), robot imitation today
(high recoverability → memorize), robot + HaF (low recoverability → generalize). Same principle.
Output: fig_llm_lesson.pdf + fig_llm_lesson.png
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SAND = "#B4A896"; INK = "#26231F"; MUTE = "#8A8378"
LINE = "#D9D2C7"; BG = "#FFFFFF"
GOOD = "#4F958B"; BAD = "#B9532F"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "svg.fonttype": "none"})

fig, ax = plt.subplots(figsize=(13.6, 7.5)); ax.set_xlim(0, 13.6); ax.set_ylim(0, 7.5); ax.axis("off")
ax.text(0.15, 7.18, "The lesson from LLMs", fontsize=20, fontweight="bold", color=INK)
ax.text(0.15, 6.76, "a training objective with LOW shortcut-recoverability forces a general, "
        "context-integrating representation", fontsize=12.5, color=MUTE)

cols = [
    dict(x=0.3, accent=TEAL, tag="LANGUAGE", title="LLM · next-token prediction",
         obj='"predict the next token"', rec="LOW", recnote="no cheap shortcut — high entropy",
         force="must attend to the whole past\n→ infer the task from context",
         result="in-context meta-learning", ok=True),
    dict(x=4.75, accent=SAND, tag="ROBOT · TODAY", title="Imitation learning (BC)",
         obj='"predict action from the\ncurrent frame"', rec="HIGH", recnote="the scene ≈ the answer",
         force="forced to do nothing —\nread the scene, ignore the rest",
         result="memorizes; fails to generalize", ok=False),
    dict(x=9.2, accent=CORAL, tag="ROBOT · HaF (ours)", title="+ hindsight / foresight objectives",
         obj='"predict the past, the goal,\nmasked structure"', rec="LOW",
         recnote="can't be faked from the current frame",
         force="must attend to\nlanguage + history",
         result="generalization", ok=True),
]
W, Y0, H = 4.1, 1.55, 4.85
for c in cols:
    x = c["x"]
    ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0.02,rounding_size=0.16",
                                fc="#FCFAF6", ec=LINE, lw=1.4))
    # header band
    ax.add_patch(FancyBboxPatch((x, Y0 + H - 0.95), W, 0.95, boxstyle="round,pad=0.02,rounding_size=0.16",
                                fc=c["accent"], ec="none"))
    ax.text(x + W/2, Y0 + H - 0.40, c["tag"], ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")
    ax.text(x + W/2, Y0 + H - 0.72, c["title"], ha="center", va="center", fontsize=11.5,
            color="white", fontweight="bold")
    # objective chip
    ax.text(x + W/2, Y0 + H - 1.45, "objective", ha="center", fontsize=8.5, color=MUTE,
            fontfamily="monospace")
    ax.text(x + W/2, Y0 + H - 1.95, c["obj"], ha="center", va="center", fontsize=11, color=INK,
            fontstyle="italic")
    # recoverability badge
    rc = GOOD if c["rec"] == "LOW" else BAD
    ax.text(x + W/2, Y0 + H - 2.62, "shortcut-recoverability", ha="center", fontsize=8.5, color=MUTE,
            fontfamily="monospace")
    ax.text(x + W/2, Y0 + H - 3.12, c["rec"], ha="center", va="center", fontsize=22, color=rc,
            fontweight="bold")
    ax.text(x + W/2, Y0 + H - 3.55, c["recnote"], ha="center", va="center", fontsize=8.6, color=MUTE)
    # mechanism
    ax.text(x + W/2, Y0 + 0.95, c["force"], ha="center", va="center", fontsize=9.8, color=INK)
    # result
    mark = "✓" if c["ok"] else "✗"
    mc = GOOD if c["ok"] else BAD
    ax.text(x + W/2, Y0 + 0.32, f"{mark}  {c['result']}", ha="center", va="center", fontsize=10.5,
            color=mc, fontweight="bold")
    # divider above the result line
    ax.plot([x + 0.5, x + W - 0.5], [Y0 + 0.66, Y0 + 0.66], color=LINE, lw=1)

# ---- bottom: the two levers to lower recoverability (NTP, JEPA, HaF are all instances) ----
ax.text(6.8, 1.30, "same principle across paradigms — engineer the objective's recoverability, and you "
        "control whether the model memorizes or generalizes", ha="center", fontsize=10, color=INK,
        fontstyle="italic")
ax.add_patch(FancyBboxPatch((0.3, 0.18), 13.0, 0.86, boxstyle="round,pad=0.02,rounding_size=0.14",
                            fc="#FBEDE6", ec=CORAL, lw=1.4))
ax.text(0.62, 0.61, "TWO LEVERS TO\nLOWER RECOVERABILITY", ha="left", va="center", fontsize=9.5,
        color=CORAL, fontweight="bold", linespacing=1.25)
ax.text(3.55, 0.79, "target-side", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.text(3.55, 0.44, "predict the past / goal  —  hindsight", ha="left", va="center", fontsize=10, color=INK)
ax.plot([8.15, 8.15], [0.34, 0.88], color=LINE, lw=1.2)
ax.text(8.4, 0.79, "input-side", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.text(8.4, 0.44, "mask the input, predict in latent space  —  JEPA / MAE", ha="left", va="center",
        fontsize=10, color=INK)
fig.savefig(os.path.join(OUT, "fig_llm_lesson.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_llm_lesson.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_llm_lesson")
