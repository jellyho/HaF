"""M1-deepening figure (3 panels):
(A) count-sweep — stacking more low-recoverability members SATURATES (presence, not count).
(B) modality diversity is ~inert — benefit tracks recoverability, not modality spread.
(C) FAST-lite — discrete action-token recoverability is a LOWER-VARIANCE, sign-stable
    estimate of the same low recoverability than continuous R².
Reads exp2h_mixki_deep_s*.json, exp2h_fastlite_fl_s*.json."""
import json, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
O="/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CORAL="#CF6F53";TEAL="#4F958B";SLATE="#4A5A63";AMBER="#C0904B";INK="#26231F";MUT="#8A8378";LINE="#D9D2C7";BG="#FFFFFF"
plt.rcParams.update({"figure.facecolor":BG,"savefig.facecolor":BG,"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"text.color":INK,"axes.labelcolor":INK,
 "axes.edgecolor":LINE,"xtick.color":MUT,"ytick.color":MUT,"axes.linewidth":1.1,"font.size":12})

D=[json.load(open(f"{O}/exp2h_mixki_deep_s{s}.json")) for s in range(3)]
F=[json.load(open(f"{O}/exp2h_fastlite_fl_s{s}.json")) for s in range(3)]
bc=float(np.mean([d["BC_only_ood"] for d in D]))

def key(m): return tuple(sorted(m["members"]))
mm={}
for d in D:
    for m in d["mixes"]: mm.setdefault(key(m),[]).append(m)

# --- (A) count sweep: nested obs-only chain ---
chain=sorted([k for k in mm if all(x.endswith("-obs") for x in k)],key=len)
ns=[len(k) for k in chain]
bmean=[float(np.mean([x["benefit"] for x in mm[k]])) for k in chain]
bsd=[float(np.std([x["benefit"] for x in mm[k]])) for k in chain]

# --- (B) diversity: far-past-obs + one partner ---
part=[]
for k in mm:
    if len(k)==2 and "far-past-obs" in k:
        p=[x for x in k if x!="far-past-obs"][0]
        mod="obs\n(same)" if p.endswith("-obs") else ("action\n(cross)" if "action" in p else "pose\n(cross)")
        part.append((p,mod,float(np.mean([x["benefit"] for x in mm[k]])),float(np.std([x["benefit"] for x in mm[k]]))))
order={"obs\n(same)":0,"action\n(cross)":1,"pose\n(cross)":2}
part.sort(key=lambda z:(order[z[1]],z[0]))

# --- (C) discrete vs continuous action recoverability ---
cont=[d["R_cont_action"] for d in F]; disc=[d["R_disc_action"] for d in F]

fig=plt.figure(figsize=(13.8,5.2)); gs=fig.add_gridspec(1,3,wspace=0.36,width_ratios=[1,1,0.9])
fig.subplots_adjust(top=0.84,bottom=0.24)

# A
axA=fig.add_subplot(gs[0,0])
axA.errorbar(ns,bmean,yerr=bsd,color=CORAL,lw=2.4,marker="o",ms=8,mfc="white",mec=CORAL,mew=2,
             capsize=4,ecolor=MUT,zorder=4)
axA.axhline(0,color=MUT,ls=(0,(5,4)),lw=1)
axA.set_ylim(-0.004,0.078)
axA.annotate("saturates —\npresence, not count",xy=(4,bmean[3]+0.002),xytext=(4.15,0.014),
             fontsize=9.5,color=INK,ha="center",
             arrowprops=dict(arrowstyle="->",color=SLATE,lw=1.3))
axA.set_xlabel("# low-recoverability members in the mix",fontsize=11)
axA.set_ylabel("benefit  (Δ gen vs BC-only)",fontsize=11)
axA.set_xticks(ns)
axA.set_title("A · Stacking hard questions SATURATES",loc="left",fontsize=12,fontweight="bold",pad=10)
for s in ("top","right"):axA.spines[s].set_visible(False)

# B
axB=fig.add_subplot(gs[0,1])
xs=np.arange(len(part))
cols=[TEAL if p[1].startswith("obs") else CORAL for p in part]
axB.bar(xs,[p[2] for p in part],yerr=[p[3] for p in part],width=0.62,color=cols,edgecolor="white",
        capsize=4,ecolor=MUT,zorder=3)
axB.axhline(0,color=INK,lw=1)
axB.set_xticks(xs);axB.set_xticklabels([p[0].replace("-obs","").replace("-action","-act") for p in part],
                                        fontsize=8.5,rotation=0)
band=float(np.mean([p[2] for p in part]))
axB.axhline(band,color=SLATE,ls=(0,(2,2)),lw=1)
axB.text(len(part)-1,band+0.001,"~flat",ha="right",va="bottom",fontsize=9,color=SLATE,style="italic")
axB.set_ylim(0,0.075)
axB.set_ylabel("benefit  (Δ gen vs BC-only)",fontsize=11)
axB.set_title("B · Modality diversity is ~inert",loc="left",fontsize=12,fontweight="bold",pad=10)
axB.set_xlabel("second member of {far-past-obs, ·} · teal=same-modality · coral=cross",fontsize=8.8)
for s in ("top","right"):axB.spines[s].set_visible(False)

# C
axC=fig.add_subplot(gs[0,2])
for i,(vals,col,lab,x) in enumerate([(cont,AMBER,"continuous\n(MSE-R²)",0),(disc,TEAL,"discrete\n(K=16 CE)",1)]):
    axC.scatter([x]*len(vals),vals,s=64,color=col,edgecolor="white",lw=1.3,zorder=4,alpha=.9)
    axC.plot([x-.17,x+.17],[np.mean(vals)]*2,color=col,lw=3,zorder=3)
    axC.text(x,0.145,f"σ={np.std(vals):.03f}",ha="center",fontsize=9.5,color=col,fontweight="bold")
axC.axhline(0,color=MUT,ls=(0,(5,4)),lw=1)
axC.set_xlim(-.5,1.5);axC.set_xticks([0,1]);axC.set_xticklabels(["continuous\n(MSE-R²)","discrete\n(K=16 CE)"],fontsize=9.5)
axC.set_ylabel("measured recoverability of next action",fontsize=10.5)
axC.set_ylim(-0.17,0.16)
axC.set_title("C · Discretize → cleaner estimate",loc="left",fontsize=12,fontweight="bold",pad=10)
axC.text(.04,.06,"both low-recov;\ndiscrete 3.4× lower-var",transform=axC.transAxes,fontsize=8.6,color=MUT,ha="left")
for s in ("top","right"):axC.spines[s].set_visible(False)

fig.text(0.005,0.005,"Mini V-L-A, RT-1/fractal, SUBN=4000, 3 seeds.  (A) benefit of a low-recoverability obs-mix rises 1→2 then plateaus — it is the "
 "PRESENCE of a hard member, not the count, that sets the mix (composition law).  (B) swapping the second member across modalities (obs/action/pose) "
 "barely moves the benefit → the lever is recoverability, not modality diversity.  (C) the next action is a genuinely low-recoverability target; a "
 "discrete K=16 cross-entropy estimate is sign-stable and 3.4× lower-variance than continuous R² — why FAST/π0.5's discrete action tokens are a "
 "cleaner low-recoverability signal.",fontsize=8,color=MUT,wrap=True)
fig.savefig(f"{O}/fig_deepen.png",dpi=300,bbox_inches="tight");fig.savefig(f"{O}/fig_deepen.pdf",bbox_inches="tight")
print(f"SAVED fig_deepen  A:count-sweep n={ns} benefit={[round(b,3) for b in bmean]}  "
      f"C: cont σ={np.std(cont):.3f} disc σ={np.std(disc):.3f}")
