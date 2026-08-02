"""Money figure: how you MEASURE recoverability decides whether you recover the law — or its opposite.
Panel A: correlation of each recoverability measure with OOD generalization (wrong-sign vs right-sign).
Panel B: the sign flip — linear-probe recoverability (WRONG, +) vs policy-dynamics recoverability (law, −).
Reads exp2h_{TAG}_s*.json. Output: fig_measures.pdf/.png
"""
import os, json, sys
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "frac4k"
CORAL = "#CF6F53"; TEAL = "#4F958B"; SLATE = "#4A5A63"; SAND = "#C0904B"
INK = "#26231F"; MUTE = "#8A8378"; LINE = "#D9D2C7"; BG = "#FFFFFF"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.linewidth": 1.1, "font.size": 12, "svg.fonttype": "none"})

S = [json.load(open(f"{OUT}/exp2h_{TAG}_s{s}.json")) for s in range(3) if os.path.exists(f"{OUT}/exp2h_{TAG}_s{s}.json")]
names = list(S[0]["objectives"])
TYP = {"prosp": CORAL, "retro": SLATE, "intro": TEAL}
def agg(n, f):
    v = [s["objectives"][n][f] for s in S if s["objectives"][n].get(f) is not None]
    return float(np.mean(v)) if v else None
gen = {n: agg(n, "generalization") for n in names}

MEAS = [("R_linear","linear probe","frozen · linear","asym"),
        ("R_mlp","MLP probe","frozen · MLP","asym"),
        ("R_mdl","prequential MDL","frozen · MLP","dyn"),
        ("R_deep","end-to-end","policy class","asym"),
        ("R_deep_aulc","AULC (learning curve)","policy class","dyn"),
        ("R_deep_early","@¼ budget (speed)","policy class","dyn")]
def corr(key):
    xs=[agg(n,key) for n in names]; pts=[(x,gen[n]) for n,x in zip(names,xs) if x is not None and gen[n] is not None]
    X=np.array([p[0] for p in pts]); Y=np.array([p[1] for p in pts])
    return pearsonr(X,Y)[0], len(pts)

fig = plt.figure(figsize=(9.6, 10.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)

# ---- Panel A: correlation bars ----
axA = fig.add_subplot(gs[0, 0])
labels=[]; vals=[]; cols=[]; ns=[]
for key,lab,cls,kind in MEAS:
    r,n=corr(key); labels.append(f"{lab}\n{cls} · {'dynamics' if kind=='dyn' else 'asymptotic'}")
    vals.append(r); ns.append(n); cols.append(TEAL if r<0 else CORAL)
y=np.arange(len(vals))[::-1]
axA.barh(y, vals, color=cols, edgecolor="white", height=0.66, zorder=3)
axA.axvline(0, color=INK, lw=1.2)
for yi,v,n in zip(y,vals,ns):
    axA.text(v+(0.03 if v>=0 else -0.03), yi, f"{v:+.2f}", va="center",
             ha="left" if v>=0 else "right", fontsize=11, fontweight="bold",
             color=TEAL if v<0 else CORAL)
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=10.5)
axA.set_xlim(-0.95, 0.95)
axA.set_xlabel("correlation with OOD generalization  (Pearson r)", fontsize=11.5)
axA.text(-0.9, len(vals)-0.4, "predicts the law  ✓", color=TEAL, fontweight="bold", fontsize=10.5, ha="left")
axA.text(0.9, len(vals)-0.4, "✗  gets it backwards", color=CORAL, fontweight="bold", fontsize=10.5, ha="right")
axA.set_title("A · How you measure recoverability decides the answer", loc="left", fontsize=12.5, fontweight="bold", pad=8)
for s in ("top","right","left"): axA.spines[s].set_visible(False)
axA.tick_params(axis="y", length=0)

# ---- Panel B: the sign flip (two scatters) ----
axB = fig.add_subplot(gs[1, 0])
def scat(key, mk, off):
    xs=[agg(n,key) for n in names]
    pts=[(x,gen[n],n) for n,x in zip(names,xs) if x is not None and gen[n] is not None]
    X=np.array([p[0] for p in pts]); Y=np.array([p[1] for p in pts])
    # normalize X to [0,1] within-measure for overlay
    Xn=(X-X.min())/(X.max()-X.min()+1e-9)
    for (x,y,n),xn in zip(pts,Xn):
        axB.scatter(xn, y, s=64, color=TYP[S[0]["objectives"][n]["type"]], marker=mk,
                    edgecolor="white", linewidth=1.0, zorder=4, alpha=0.9)
    b1,b0=np.polyfit(Xn,Y,1); xx=np.linspace(0,1,50)
    r=pearsonr(X,Y)[0]
    axB.plot(xx,b0+b1*xx, color=off, lw=2.4, zorder=3)
    return r
r_lin=scat("R_linear","o",CORAL)
r_dyn=scat("R_deep_early","D",TEAL)
axB.text(0.02, 0.03, f"linear-probe recoverability  →  r = {r_lin:+.2f}  (wrong way)",
         transform=axB.transAxes, color=CORAL, fontsize=10.5, fontweight="bold")
axB.text(0.98, 0.94, f"policy-dynamics recoverability  →  r = {r_dyn:+.2f}  (the law)",
         transform=axB.transAxes, color=TEAL, fontsize=10.5, fontweight="bold", ha="right")
axB.set_xlabel("recoverability of the objective  (min→max, per measure)", fontsize=11.5)
axB.set_ylabel("OOD generalization  (action R²)", fontsize=11.5)
axB.set_title("B · Same objectives, opposite slopes", loc="left", fontsize=12.5, fontweight="bold", pad=8)
axB.scatter([],[],marker="o",color=MUTE,label="linear probe (frozen)")
axB.scatter([],[],marker="D",color=MUTE,label="policy dynamics (speed)")
axB.legend(loc="center right", frameon=False, fontsize=9.5)
for s in ("top","right"): axB.spines[s].set_visible(False)

fig.text(0.005,-0.02, f"Mini V-L-A, RT-1/fractal, {TAG}, 3 seeds, 18 objectives. Recoverability = normalized V-information "
         "(Xu 2020); measured under a frozen linear probe it correlates the WRONG way with generalization; measured "
         "under the policy's own function class — and as learning dynamics (how cheaply the target is grabbed early) — "
         "it recovers the law. Selectivity control (shuffled target) ≈ 0.", fontsize=8.2, color=MUTE, wrap=True)
fig.savefig(f"{OUT}/fig_measures_{TAG}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_measures_{TAG}.png", dpi=300, bbox_inches="tight")
print(f"SAVED fig_measures_{TAG}  (linear r={r_lin:+.2f}, dynamics r={r_dyn:+.2f})")
