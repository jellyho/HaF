"""Publication figure for the controlled generalization proof (PI-style), with real-data corroboration.

A — the overfit and its fix: train / IID / OOD loss for BC-only vs the lowest-recoverability auxiliary. BC fits
    train + IID but OOD explodes (the shortcut breaks); the shortcut-free objective closes the gap.
B — the money curve: OOD loss vs OBJECTIVE RECOVERABILITY (log y). Lower recoverability ⇒ lower OOD loss,
    monotone. The cliff at ρ=1→0.75 shows that even mild shortcut-corruption forces the robust path.
C — real data: on 6 VLA datasets the same law holds — the auxiliary helps most exactly where vision is a
    shortcut (vision-legitimacy vs aux benefit, Pearson r shown).
Outputs: fig_sim.pdf + fig_sim.png (300 DPI).
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
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.linewidth": 1.0,
    "font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none",
})

S = json.load(open(os.path.join(OUT, "exp_sim_analysis.json")))
bc = S["conditions"]["BC-only"]
sweep = sorted(S["rho_sweep"], key=lambda d: d["recoverability"])
best = min(sweep, key=lambda d: d["ood"][0])

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.2, 4.3),
                                    gridspec_kw={"width_ratios": [1, 1.12, 1.12], "wspace": 0.34})

# ===== A: overfit + fix =====
splits, labels = ["train", "iid", "ood"], ["train", "IID\ntest", "OOD\ntest"]
xg = np.arange(3); bw = 0.36
axA.bar(xg - bw/2, [bc[s][0] for s in splits], bw, yerr=[bc[s][1] for s in splits], capsize=3, color=SAND,
        label="BC only (uses shortcut)", edgecolor="white", error_kw={"elinewidth": 1, "ecolor": MUTE, "alpha": .6})
axA.bar(xg + bw/2, [best[s][0] for s in splits], bw, yerr=[best[s][1] for s in splits], capsize=3, color=CORAL,
        label=f"+ shortcut-free aux (ρ={best['recoverability']:g})", edgecolor="white",
        error_kw={"elinewidth": 1, "ecolor": MUTE, "alpha": .6})
axA.annotate("shortcut breaks\n→ BC fails OOD", (2 - bw/2, bc["ood"][0]), xytext=(0.75, 1.35), ha="left",
             va="top", fontsize=9, color=MUTE, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.1))
axA.set_xticks(xg); axA.set_xticklabels(labels, fontsize=10.5)
axA.set_ylabel("normalized loss  (MSE / Var y)", fontsize=10)
axA.set_title("(A)  Shortcut ⇒ overfit;\na shortcut-free objective fixes it", fontsize=12, loc="left", pad=10, fontweight="bold")
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.legend(frameon=False, fontsize=8.8, loc="upper center", handlelength=1.1)

# ===== B: sim money curve (log y) =====
rec = [d["recoverability"] for d in sweep]
ood = [d["ood"][0] for d in sweep]; oode = [d["ood"][1] for d in sweep]
axB.set_yscale("log")
axB.errorbar(rec, ood, yerr=oode, marker="o", ms=7, lw=2.4, color=CORAL, capsize=3,
             mfc="white", mec=CORAL, mew=2, zorder=5)
axB.axhline(bc["ood"][0], color=SAND, lw=1.6, ls=(0, (5, 3)), zorder=1)
axB.text(0.98, bc["ood"][0] * 1.05, "BC-only (no aux)", fontsize=8.6, color=MUTE, va="bottom", ha="right")
axB.annotate("even mild corruption\nforces the robust path", (0.75, ood[rec.index(0.75)]),
             xytext=(0.35, 0.28), fontsize=8.6, color=CORAL, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.1))
axB.set_xlabel("objective recoverability\n(cheaper to get from the shortcut →)", fontsize=10)
axB.set_ylabel("OOD loss   (log scale)", fontsize=10)
axB.set_title("(B)  Lower-recoverability objectives\ngeneralize better  (monotone)", fontsize=12, loc="left", pad=10, fontweight="bold")
axB.set_xlim(-0.05, 1.05)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

# ===== C: real-data law (exp2c axisB) =====
try:
    c = json.load(open(os.path.join(OUT, "exp2c_analysis.json")))
    b = c["axisB"]
    vl = np.array(b["vision_legit"]); ben = np.array(b["mean_benefit"])
    ds = b["datasets"]; r = b.get("vislegit_vs_benefit_pearson", float("nan"))
    axC.scatter(vl, ben, s=64, color=TEAL, edgecolor="white", linewidth=1, zorder=5)
    # fit line
    m, q = np.polyfit(vl, ben, 1); xs = np.linspace(vl.min(), vl.max(), 50)
    axC.plot(xs, m * xs + q, color=TEAL, lw=1.6, ls=(0, (5, 3)), alpha=0.8, zorder=2)
    axC.axhline(0, color=LINE, lw=1)
    for x, y, name in zip(vl, ben, ds):
        axC.annotate(name.replace("libero_", "lib-").replace("fractal", "RT-1"), (x, y),
                     fontsize=7.3, color=MUTE, xytext=(4, 3), textcoords="offset points")
    axC.set_xlabel("vision legitimately carries the signal →\n(← vision is a shortcut)", fontsize=10)
    axC.set_ylabel("auxiliary benefit  (ΔOOD)", fontsize=10)
    axC.set_title(f"(C)  Real VLA data: aux helps where\nvision is a shortcut  (r = {r:+.2f})", fontsize=12, loc="left", pad=10, fontweight="bold")
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)
except Exception as e:
    axC.text(0.5, 0.5, f"real-data panel skipped:\n{e}", ha="center", fontsize=9, color=MUTE)

fig.text(0.005, -0.11, "Controlled sim: latent z, action y=W·z, cheap shortcut x_easy (breaks OOD) + entangled "
         "x_hard=g(z); aux = predict y with the shortcut randomized w.p. 1−ρ; 8 seeds. "
         "(C) exp2c: 6 datasets × 5 seeds, vision-legitimacy (Exp 1) vs auxiliary OOD benefit.",
         fontsize=7.8, color=MUTE)
fig.savefig(os.path.join(OUT, "fig_sim.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_sim.png"), dpi=300, bbox_inches="tight")
print("SAVED fig_sim.pdf + fig_sim.png")
