"""Real-data evidence (toy replacement) — FAIR matched-pair version.

(A) On real robot data, compare each retrospective target with its SYMMETRIC prospective counterpart at the
    SAME condition: near-past↔near-future (±5), far-past↔far-future (±45), and episode ORIGIN↔END
    (initial-obs↔final-obs). At matched horizon past ≈ future (both copy shortcuts); the one shortcut-free target
    is the episode ORIGIN (initial-obs < final-obs).
(B) The trained BC policy's action follows the image (sensitivity ≈ 2) but is unmoved by the instruction (≈ 0).
Uses the diverse OXE datasets that have the final-obs anchor computed (auto-expands as more are re-run).
Output: fig_realdata_recov.pdf + fig_realdata_recov.png
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
                     "axes.linewidth": 1.1, "font.size": 13, "svg.fonttype": "none"})


def load(d):
    return json.load(open(os.path.join(OUT, f"results_{d}.json")))


# diverse OXE datasets (latent R_triv is stable on varied scenes) that HAVE the final-obs anchor
DIVERSE = ["fractal", "droid", "bridge"]
AVAIL = [d for d in DIVERSE if "P0 final-obs" in load(d)]
# fair matched pairs: (retrospective key, prospective key, group label)
PAIRS = [("Mnear past-obs k~5", "P1s future-obs k~5", "near-term\n(±5 steps)"),
         ("Mfar past-obs k~45", "P1l future-obs k~45", "far\n(±45 steps)"),
         ("R1 initial-obs", "P0 final-obs", "episode\nendpoints")]


def meanstd(key):
    vals = [load(d)[key]["R_triv"] for d in AVAIL]
    return np.mean(vals), (np.std(vals) if len(vals) > 1 else 0.0)


retro_m = [meanstd(r)[0] for r, _, _ in PAIRS]; retro_s = [meanstd(r)[1] for r, _, _ in PAIRS]
prosp_m = [meanstd(p)[0] for _, p, _ in PAIRS]; prosp_s = [meanstd(p)[1] for _, p, _ in PAIRS]

# Panel B: exp2b BC-only vision vs language sensitivity
b = json.load(open(os.path.join(OUT, "exp2b_trimodal_agg.json")))
sv = [b[ds]["cond"]["BC-only"]["vision_sensitivity"]["mean"] for ds in b]
sl = [b[ds]["cond"]["BC-only"]["lang_sensitivity"]["mean"] for ds in b]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.3), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.30})

# ===== Panel A: matched pairs =====
xa = np.arange(len(PAIRS)); bw = 0.36
axA.bar(xa - bw/2, retro_m, bw, yerr=retro_s, capsize=3, color=SLATE, label="retrospective (look back)",
        edgecolor="white", error_kw={"elinewidth": 1.1, "ecolor": MUTE, "alpha": 0.7})
axA.bar(xa + bw/2, prosp_m, bw, yerr=prosp_s, capsize=3, color=CORAL, label="prospective (look forward)",
        edgecolor="white", error_kw={"elinewidth": 1.1, "ecolor": MUTE, "alpha": 0.7})
axA.axhline(0, color=INK, lw=1)
for i in range(len(PAIRS)):
    axA.text(xa[i] - bw/2, retro_m[i] + (0.02 if retro_m[i] >= 0 else -0.05), f"{retro_m[i]:+.2f}",
             ha="center", va="bottom" if retro_m[i] >= 0 else "top", fontsize=10.5, fontweight="bold", color=SLATE)
    axA.text(xa[i] + bw/2, prosp_m[i] + 0.02, f"{prosp_m[i]:+.2f}", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color=CORAL)
axA.set_xticks(xa); axA.set_xticklabels([g for _, _, g in PAIRS], fontsize=11)
axA.set_ylabel("copy-shortcut availability\nR_triv  (higher = more of a shortcut)", fontsize=11)
axA.set_ylim(min(0, min(retro_m + prosp_m) - 0.1), 0.92)
axA.set_title("(A) Fair matched pairs: at the same condition,\npast ≈ future — the origin is the exception",
              fontsize=13.5, loc="left", pad=12, fontweight="bold")
# annotate the shortcut-free origin
axA.annotate("shortcut-free\n(the episode origin)", (xa[2] - bw/2, retro_m[2] + 0.10), ha="center",
             va="bottom", fontsize=9.3, color=SLATE, fontweight="bold")
axA.annotate("near obs = copy\n(either direction)", (0, 0.80), ha="center", fontsize=9.3, color=MUTE)
axA.legend(frameon=False, fontsize=9.5, loc="upper right", handlelength=1.1)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

# ===== Panel B: BC vision vs language =====
axB.bar([0], [np.mean(sv)], 0.6, yerr=[np.std(sv)], capsize=4, color=CORAL, edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": MUTE, "alpha": 0.7})
axB.bar([1], [np.mean(sl)], 0.6, yerr=[np.std(sl)], capsize=4, color=SLATE, edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": MUTE, "alpha": 0.7})
axB.text(0, np.mean(sv) + 0.05, f"{np.mean(sv):.2f}", ha="center", fontsize=13, fontweight="bold")
axB.text(1, np.mean(sl) + 0.05, f"{np.mean(sl):.3f}", ha="center", fontsize=13, fontweight="bold", color=SLATE)
axB.set_xticks([0, 1]); axB.set_xticklabels(["change the\nimage", "change the\ninstruction"], fontsize=11.5)
axB.set_ylabel("how much the action moves\n(sensitivity)", fontsize=11)
axB.set_ylim(0, 2.3)
axB.set_title("(B) The trained policy reads\nvision — not language", fontsize=13.5, loc="left", pad=12, fontweight="bold")
axB.annotate("action follows\nthe scene", (0, np.mean(sv) - 0.5), fontsize=10, color=CORAL, ha="center", fontweight="bold")
axB.annotate("instruction\nignored", (1, 0.35), fontsize=10, color=SLATE, ha="center", fontweight="bold")
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

fig.suptitle("On real robot data, the objectives BC and foresight already optimize are shortcuts",
             fontsize=15, fontweight="bold", x=0.02, ha="left", y=1.04)
fig.text(0.005, -0.03, f"(A) R_triv = MSE(copy current frame)/MSE(marginal) on {' · '.join(d.upper() for d in AVAIL)} "
         f"(diverse scenes; each retrospective target vs its horizon-/endpoint-matched prospective counterpart). "
         f"(B) behavior cloning, 6 datasets × 5 seeds.", fontsize=8.3, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_realdata_recov.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_realdata_recov.png"), dpi=300, bbox_inches="tight")
print("SAVED  AVAIL=", AVAIL, "| retro", [f"{v:+.2f}" for v in retro_m], "prosp", [f"{v:+.2f}" for v in prosp_m])
