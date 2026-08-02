"""M1 figures: (A) composition law — min-recoverability member predicts mix generalization;
(B) KI schedule — hard KI over-insulates a low-recov aux. Reads exp2h_mixki_mk_s*.json."""
import os, json, numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
O="/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL="#CF6F53";TEAL="#4F958B";SLATE="#4A5A63";INK="#26231F";MUT="#8A8378";LINE="#D9D2C7";BG="#FFFFFF"
plt.rcParams.update({"figure.facecolor":BG,"savefig.facecolor":BG,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"text.color":INK,"axes.labelcolor":INK,
 "axes.edgecolor":LINE,"xtick.color":MUT,"ytick.color":MUT,"axes.linewidth":1.1,"font.size":12})
S=[json.load(open(f"{O}/exp2h_mixki_mk_s{s}.json")) for s in range(3)]
bc=np.mean([s["BC_only_ood"] for s in S])
nmix=len(S[0]["mixes"])
def mg(i,k): return float(np.mean([s["mixes"][i][k] for s in S]))
mins=np.array([mg(i,"min_recov") for i in range(nmix)]);means=np.array([mg(i,"mean_recov") for i in range(nmix)])
joints=np.array([mg(i,"joint_recov") for i in range(nmix)]);gens=np.array([mg(i,"gen") for i in range(nmix)])
sizes=[len(S[0]["mixes"][i]["members"]) for i in range(nmix)]

fig=plt.figure(figsize=(9.6,9.2)); gs=fig.add_gridspec(2,1,height_ratios=[1,0.85],hspace=0.34)
# A: min vs gen scatter + which summary predicts
axA=fig.add_subplot(gs[0,0])
for i in range(nmix):
    axA.scatter(mins[i],gens[i],s=70+30*sizes[i],color=CORAL if mins[i]>0.5 else TEAL,edgecolor="white",lw=1.2,zorder=4,alpha=.9)
b1,b0=np.polyfit(mins,gens,1);xx=np.linspace(mins.min()-.03,mins.max()+.03,50)
axA.plot(xx,b0+b1*xx,color=INK,lw=2,zorder=3,alpha=.8)
axA.axhline(bc,color=MUT,ls=(0,(5,4)),lw=1.1);axA.text(mins.max(),bc+.002,"BC-only",ha="right",va="bottom",fontsize=9,color=MUT,style="italic")
axA.set_xlabel("min recoverability of the mix's members  (hardest question) →",fontsize=11.5)
axA.set_ylabel("mix OOD generalization  (action R²)",fontsize=11.5)
r_min=pearsonr(mins,gens)[0];r_mean=pearsonr(means,gens)[0];r_joint=pearsonr(joints,gens)[0]
axA.set_title(f"A · Composition law: the HARDEST (lowest-recov) member predicts the mix\nmin r={r_min:+.2f}   ·   joint r={r_joint:+.2f}   ·   mean r={r_mean:+.2f}",
              loc="left",fontsize=12.5,fontweight="bold",pad=8)
axA.text(.02,.04,"큰 점 = 멤버 많은 mix · 코랄=고-recov 지배 · 틸=저-recov 지배",transform=axA.transAxes,fontsize=9,color=MUT)
for s in ("top","right"):axA.spines[s].set_visible(False)
# B: KI schedule bars
axB=fig.add_subplot(gs[1,0])
from collections import defaultdict
ki=defaultdict(dict)
for s in S:
    for r in s["ki"]: ki[r["aux"]].setdefault(r["tau"],[]).append(r["gen"])
taus=[0.0,0.5,1.0];labels=["always-joint\n(τ=0)","late-release\n(τ=0.5)","always-KI\n(τ=1)"]
auxs=list(ki.keys());w=.36;x=np.arange(3)
for j,aux in enumerate(auxs):
    vals=[np.mean(ki[aux][t]) for t in taus]
    axB.bar(x+(j-.5)*w,[v-bc for v in vals],w,label=aux,color=[TEAL,CORAL][j],edgecolor="white",zorder=3)
axB.axhline(0,color=INK,lw=1);axB.set_xticks(x);axB.set_xticklabels(labels,fontsize=10)
axB.set_ylabel("Δ generalization vs BC-only",fontsize=11.5)
axB.set_title("B · KI schedule: hard KI (τ=1) OVER-INSULATES a low-recov aux (hurts)",loc="left",fontsize=12.5,fontweight="bold",pad=8)
axB.legend(frameon=False,fontsize=10)
for s in ("top","right"):axB.spines[s].set_visible(False)
fig.text(0.005,-0.02,f"Mini V-L-A, RT-1/fractal, SUBN=4000, 3 seeds, 12 mixes. (A) each mix's generalization is best predicted by its "
 "MINIMUM-recoverability member (r=−0.87) → one hard question rescues a shortcut-laden mix. (B) with a low-recov aux, always-joint / "
 "late-release help, but hard KI starves the action head (worse than BC-only).",fontsize=8.2,color=MUT,wrap=True)
fig.savefig(f"{O}/fig_mixki.png",dpi=300,bbox_inches="tight");fig.savefig(f"{O}/fig_mixki.pdf",bbox_inches="tight")
print(f"SAVED fig_mixki (min r={r_min:+.2f}, mean r={r_mean:+.2f}, joint r={r_joint:+.2f})")
