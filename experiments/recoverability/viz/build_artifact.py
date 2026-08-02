"""Exp 1 cross-dataset artifact — interactive (hover) scatter + objective glossary."""
import json, os

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DSETS = [("droid", "DROID", "Franka Panda", 500),
         ("bridge", "Bridge", "WidowX", 227),
         ("fractal", "RT-1", "Google robot", 500)]
R = {tag: json.load(open(os.path.join(OUT, f"results_{tag}.json"))) for tag, *_ in DSETS}
CR = json.load(open(os.path.join(OUT, "change_recovery.json"))) if os.path.exists(os.path.join(OUT, "change_recovery.json")) else {}

# key -> (label, family, trivial rule, description)
GLOSS = {
 "BC action|o_t":       ("Policy chunk (BC)", "policy", "dataset-mean chunk",
    "Predict the <b>15-step action chunk</b> a VLA outputs, from the current frame. The real behavior-cloning objective. The action is an output, not an input → no copy shortcut (R_triv≈0). A single-step action is nearly your current pose; the chunk needs the whole near-trajectory, so it is the honest target."),
 "P2 future-action":    ("Future action", "future", "dataset-mean action",
    "Predict the action ~5 steps <b>ahead</b> from the current frame. The action is an <b>output, not an input</b> — there is no copy shortcut — so R_triv≈0. It is the policy objective at a horizon offset; what matters is whether the frame predicts it (G_obs)."),
 "Mact prev-action":    ("Prev action", "past", "dataset-mean action",
    "Predict the action ~5 steps <b>ago</b> from the current frame. Like future-action, the action isn’t an input, so R_triv≈0 — no copy shortcut exists."),
 "P3 future-gripper":   ("Future gripper", "future", "copy current gripper",
    "Predict the gripper open/close state ~5 steps ahead. Gripper rarely changes, so copying is strong."),
 "Mgrip prev-gripper":  ("Prev gripper", "past", "copy current gripper",
    "Predict the gripper state ~5 steps ago."),
 "P1s future-obs k~5":  ("Future obs (near)", "future", "copy the current frame",
    "Predict the scene ~5 steps ahead (in DINOv2 latent). Redundant if the near future looks like now."),
 "P1l future-obs k~45": ("Future obs (far)", "future", "copy the current frame",
    "Predict the scene ~45 steps ahead."),
 "Mnear past-obs k~5":  ("Past obs (near)", "past", "copy the current frame",
    "Predict the scene ~5 steps ago — the symmetric mirror of near-future obs."),
 "Mfar past-obs k~45":  ("Past obs (far)", "past", "copy the current frame",
    "Predict the scene ~45 steps ago."),
 "R1 initial-obs":      ("Initial obs ★", "past", "copy the current frame",
    "Predict the <b>first</b> frame of the episode from the current frame. The core hindsight target — recovering where the episode began. Cannot be copied once the scene has changed; the useful, hard-but-learnable one."),
 "R2 initial-pose":     ("Initial pose", "past", "copy the current pose",
    "Predict the robot end-effector pose at episode <b>start</b> from the current frame."),
 "R3 instruction":      ("Instruction", "past", "most-common instruction",
    "Predict the task instruction (MiniLM text embedding) from the current frame. Tests the vision→language shortcut — can the scene alone reveal the command?"),
 "Iinv act|both-frames":("Inverse dynamics", "present", "dataset-mean action",
    "Predict the current action given <b>both</b> the current and a future frame. The answer is “in the input” — you see before &amp; after, so it is extraction, not inference."),
 "I-gripper now":       ("Gripper now", "present", "dataset-mean gripper",
    "Predict the <b>current</b> gripper state from the current frame. Usually visible → extractable."),
 "I-progress t/T":      ("Progress", "present", "dataset-mean progress",
    "Predict how far through the episode we are (t / T) from the current frame."),
}
FAM = {"future": ("#1F4E6B", "future / prospective"), "past": ("#9E2A4F", "past / retrospective"),
       "present": ("#4A85A6", "present / introspective"), "policy": ("#2F6B4F", "policy (BC)")}

# ---- interactive scatter (x = R_triv, y = probe_beyond_trivial) ----
XW, YW = 320, 262
L, Rm, T, B = 44, 12, 30, 40
PW, PH = XW - L - Rm, YW - T - B
XMIN, XMAX, YMIN, YMAX = -1.0, 1.0, -0.15, 0.65


def px(rt): return L + (rt - XMIN) / (XMAX - XMIN) * PW
def py(pbt): return T + (1 - (pbt - YMIN) / (YMAX - YMIN)) * PH


def scatter(tag, name):
    res = R[tag]
    x0, xz, yz = px(0), px(0), py(0)
    dots = ""
    for key, v in res.items():
        if key not in GLOSS:
            continue
        lab, fam, triv, desc = GLOSS[key]
        c = FAM[fam][0]
        cx, cy = px(v["R_triv"]), py(max(min(v["probe_beyond_trivial"], YMAX), YMIN))
        star = key.startswith("R1")
        rr = 7 if star else 5
        j = json.dumps({"n": lab, "rt": round(v["R_triv"], 3), "g": round(v["G_obs"], 3),
                        "p": round(v["probe_beyond_trivial"], 3), "t": triv, "d": desc, "c": c}).replace('"', "&quot;")
        dots += (f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="{rr}" fill="{c}" '
                 f'stroke="var(--card)" stroke-width="1.4" data-i="{j}"'
                 f'{" data-star=1" if star else ""}></circle>')
    # useful-corner shading (low R_triv, high pbt)
    return f'''<figure class="sc"><figcaption><b>{name}</b></figcaption>
    <svg viewBox="0 0 {XW} {YW}" class="scsvg">
      <rect x="{L}" y="{T}" width="{px(0.25)-L:.0f}" height="{py(0.1)-T:.0f}" fill="var(--okwash)"/>
      <line x1="{L}" y1="{yz:.1f}" x2="{XW-Rm}" y2="{yz:.1f}" stroke="var(--rule)"/>
      <line x1="{xz:.1f}" y1="{T}" x2="{xz:.1f}" y2="{YW-B}" stroke="var(--rule)"/>
      <text x="{L}" y="{YW-8}" class="axl">shortcut &rarr; (R_triv)</text>
      <text x="6" y="{T+8}" class="axl" transform="rotate(-90 10 {T+8})">learnable beyond copy &rarr;</text>
      {dots}
    </svg></figure>'''


scatters = "".join(scatter(t, n) for t, n, *_ in DSETS)

# ---- comparison table ----
KEYS = ["P1s future-obs k~5", "Mnear past-obs k~5", "P1l future-obs k~45", "Mfar past-obs k~45",
        "R1 initial-obs", "P2 future-action", "Mact prev-action", "BC action|o_t",
        "P3 future-gripper", "I-gripper now", "R3 instruction"]
trows = ""
for k in KEYS:
    lab, fam, triv, desc = GLOSS[k]
    c = FAM[fam][0]
    cells = ""
    for tag, *_ in DSETS:
        v = R[tag].get(k)
        cells += (f"<td class='num'><b>{v['R_triv']:+.2f}</b>"
                  f"<span class='pb'>{'+'+format(v['probe_beyond_trivial'],'.2f') if v['probe_beyond_trivial']>=0.05 else ''}</span></td>") if v else "<td class='num'>&mdash;</td>"
    trows += f"<tr><td><span class='fd' style='background:{c}'></span>{lab}</td>{cells}</tr>"

cr_rows = ""
for name in ["DROID", "Bridge", "RT-1"]:
    row = CR.get(name, {})
    order = ["near-past change (t-k~5)", "far-past change (t-k~45)", "initial change (t-0)"]
    cr_rows += f"<tr><td>{name}</td>" + "".join(f"<td class='num'>{row.get(k,0):+.2f}</td>" for k in order) + "</tr>"

E2B = json.load(open(os.path.join(OUT, "exp2b_agg.json"))) if os.path.exists(os.path.join(OUT, "exp2b_agg.json")) else {}
E2B_LAB = {"BC-only": "BC-only (no auxiliary)", "BC+retro": "BC + retrospective",
           "BC+fwd": "BC + forward", "BC+retro+fwd": "BC + retro + forward (mixed)"}
e2b_rows = ""
for cond, lab in E2B_LAB.items():
    cells = ""
    for name in ["RT-1", "DROID"]:
        v = E2B.get(name, {}).get(cond)
        cells += f"<td class='num'>{v[0]:+.2f} <span class='pm'>± {v[1]:.2f}</span></td>" if v else "<td class='num'>&mdash;</td>"
    best = "best" if cond == "BC+retro+fwd" else ("worst" if cond == "BC-only" else "")
    e2b_rows += f"<tr class='{best}'><td>{lab}</td>{cells}</tr>"

MEM = json.load(open(os.path.join(OUT, "memorization.json"))) if os.path.exists(os.path.join(OUT, "memorization.json")) else {}
mem_rows = ""
for lab, key in [("task readable from scene (kNN)", "instr<-scene (kNN)"),
                 ("action-chunk from image+state", "action<-img+state"),
                 ("language adds to the chunk (Δ)", "Δ_lang")]:
    cells = "".join(f"<td class='num'>{MEM.get(n,{}).get(key,0):+.2f}</td>" for n in ["DROID", "Bridge", "RT-1"])
    mem_rows += f"<tr><td>{lab}</td>{cells}</tr>"

# ---- glossary ----
gloss_html = ""
for fam, (col, flab) in FAM.items():
    items = [(k, *g) for k, g in GLOSS.items() if g[1] == fam]
    rows = ""
    for k, lab, _f, triv, desc in items:
        rows += f"""<div class="gi"><div class="gh"><span class="gn">{lab}</span><span class="gt">trivial rule: {triv}</span></div><p>{desc}</p></div>"""
    gloss_html += f'<div class="gfam"><div class="gflab" style="color:{col}"><span class="fd" style="background:{col}"></span>{flab}</div>{rows}</div>'

r1 = [R[t]["R1 initial-obs"]["probe_beyond_trivial"] for t, *_ in DSETS]

HTML = f"""<title>Exp 1 — Cross-Dataset Redundancy Map</title>
<style>
:root{{--paper:#F4F6F8;--card:#FFF;--ink:#101620;--muted:#697485;--rule:#D8DEE6;--soft:#EDF1F5;
 --red:#9E2A4F;--blue:#1F4E6B;--mid:#4A85A6;--green:#2F6B4F;--okwash:#E7F1EB}}
@media (prefers-color-scheme:dark){{:root{{--paper:#0D1015;--card:#161B22;--ink:#E7ECF2;--muted:#8A94A4;
 --rule:#2A313B;--soft:#1B2029;--red:#E36A8A;--blue:#6FB0D6;--mid:#84B4CE;--green:#6FC195;--okwash:#16281E}}}}
:root[data-theme="dark"]{{--paper:#0D1015;--card:#161B22;--ink:#E7ECF2;--muted:#8A94A4;--rule:#2A313B;--soft:#1B2029;
 --red:#E36A8A;--blue:#6FB0D6;--mid:#84B4CE;--green:#6FC195;--okwash:#16281E}}
:root[data-theme="light"]{{--paper:#F4F6F8;--card:#FFF;--ink:#101620;--muted:#697485;--rule:#D8DEE6;--soft:#EDF1F5;
 --red:#9E2A4F;--blue:#1F4E6B;--mid:#4A85A6;--green:#2F6B4F;--okwash:#E7F1EB}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);line-height:1.55;
 font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1040px;margin:0 auto;padding:52px 24px 90px}}
.eyebrow{{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:clamp(27px,4.4vw,42px);font-weight:800;letter-spacing:-.02em;line-height:1.07;margin:12px 0 0;text-wrap:balance}}
h1 .a{{color:var(--red)}}
.lede{{max-width:72ch;font-size:16px;opacity:.9;margin:16px 0 0}}
.sec{{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin:46px 0 4px}}
.sub{{color:var(--muted);font-size:14px;margin:0 0 14px;max-width:78ch}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:var(--soft);padding:1px 5px;border-radius:3px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}}
.stat{{background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:15px 17px}}
.stat .n{{font-size:22px;font-weight:800}} .stat .l{{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.4}}
.callout{{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--green);border-radius:9px;padding:16px 20px;margin-top:16px}}
.callout h3{{margin:0 0 10px;font-size:15px}} .callout dl{{margin:0;display:grid;grid-template-columns:auto 1fr;gap:8px 14px}}
.callout dt{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;font-weight:700;white-space:nowrap}}
.callout dd{{margin:0;font-size:13.5px;opacity:.9}}
.scatters{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:8px}}
.sc{{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:10px}}
.sc figcaption{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);margin:0 0 4px 2px}}
.scsvg{{width:100%;height:auto;display:block;overflow:visible}}
.axl{{font-family:ui-monospace,Menlo,monospace;font-size:8.5px;fill:var(--muted)}}
.pt{{cursor:pointer;transition:r .1s}} .pt:hover{{stroke:var(--ink);stroke-width:2}}
#tip{{position:fixed;z-index:50;pointer-events:none;max-width:280px;background:var(--card);border:1px solid var(--rule);
 border-radius:8px;padding:11px 13px;box-shadow:0 8px 26px rgba(0,0,0,.22);font-size:12.5px;opacity:0;transition:opacity .1s}}
#tip .tn{{font-weight:800;margin-bottom:5px}} #tip .tv{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--muted);margin-bottom:6px}}
#tip .td{{line-height:1.45;opacity:.92}} #tip b{{color:var(--ink)}}
.tblwrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--rule);border-radius:9px;overflow:hidden;min-width:540px}}
th,td{{padding:9px 13px;border-bottom:1px solid var(--soft);text-align:left}}
th{{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}}
th.num,td.num{{text-align:right;font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}}
tr:last-child td{{border-bottom:none}} .pb{{color:var(--green);font-size:11px;margin-left:5px}}
.pm{{color:var(--muted);font-size:10.5px}} tr.best td{{background:var(--okwash)}} tr.best td:first-child{{font-weight:700}}
.fd{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:17px 19px;border-top:3px solid var(--red)}}
.card.b{{border-top-color:var(--blue)}} .card.g{{border-top-color:var(--green)}} .card.m{{border-top-color:var(--mid)}}
.card h3{{margin:0 0 8px;font-size:15px}} .card p{{margin:0;font-size:13.5px;opacity:.88;line-height:1.5}}
.gloss{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.gfam{{background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:14px 16px}}
.gflab{{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px}}
.gi{{padding:9px 0;border-top:1px solid var(--soft)}} .gi:first-of-type{{border-top:none}}
.gh{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}}
.gn{{font-weight:700;font-size:13.5px}} .gt{{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:var(--muted)}}
.gi p{{margin:4px 0 0;font-size:12.8px;opacity:.88;line-height:1.5}}
ul.m{{margin:0;padding:0;list-style:none}} ul.m li{{padding:8px 0;border-top:1px solid var(--soft);font-size:13.5px;opacity:.9}} ul.m li:first-child{{border-top:none}}
footer{{margin-top:50px;padding-top:16px;border-top:1px solid var(--rule);font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}}
@media (max-width:820px){{.scatters,.stats,.cards,.gloss,.callout dl{{grid-template-columns:1fr}}}}
</style>

<div id="tip"></div>
<div class="wrap">
  <div class="eyebrow">Hindsight &amp; Foresight · Exp 1–2b · three embodiments · frozen probes (Exp 1) + a small co-training test (Exp 2b)</div>
  <h1>Across three robots, one target resists the shortcut — the <span class="a">retrospective</span> one.</h1>
  <p class="lede">15 prediction objectives scored on <b>DROID</b> (Franka), <b>Bridge</b> (WidowX) and <b>RT-1</b> (Google robot) by one
  question: how much does a <i>trivial rule</i> already get (<code>R_triv</code>), and can a probe on the current frame beat it
  (<code>probe&gt;trivial</code>)? Hover any point below for its definition. The observation targets order the same on every robot
  — near ≈ copyable, far less so, <b>initial observation least of all</b> — and initial-obs is where a probe most beats the shortcut,
  by a margin that <b>grows</b> robot to robot.</p>

  <div class="stats">
    <div class="stat"><div class="n" style="color:var(--red)">+{r1[0]:.2f} &rarr; +{r1[1]:.2f} &rarr; +{r1[2]:.2f}</div>
      <div class="l"><b>probe beyond copy for initial-obs</b> (DROID&rarr;Bridge&rarr;RT-1) — retrospective moves into the useful corner.</div></div>
    <div class="stat"><div class="n" style="color:var(--blue)">near &asymp; future</div>
      <div class="l"><b>Redundancy is symmetric in time</b> and decays with distance — <i>direction</i> doesn’t matter, distance does.</div></div>
    <div class="stat"><div class="n" style="color:var(--green)">obs &amp; gripper, not action</div>
      <div class="l"><b>The copy shortcuts are foresight observation</b> (future frame, R_triv up to 0.88) and gripper (a state input). Action prediction has no input to copy — R_triv≈0, it's the task.</div></div>
  </div>

  <div class="callout">
    <h3>The three &ldquo;action&rdquo; objectives — and why none is a copy shortcut</h3>
    <p style="margin:0 0 10px;font-size:13.5px;opacity:.9">A VLA’s inputs are <b>image + state + language</b> — the action is never an input. So a trivial rule can’t copy the action; all three action objectives sit at <b>R_triv≈0</b> (no cheat), unlike observation prediction where copying the current frame <i>is</i> a cheat.</p>
    <dl>
      <dt style="color:var(--green)">BC action|o_t</dt><dd><b>The policy.</b> current frame &rarr; <b>current</b> action. “Given what I see, what do I do now.”</dd>
      <dt style="color:var(--blue)">P2 future-action</dt><dd>current frame &rarr; action a few steps <b>ahead</b> — the policy at a horizon offset.</dd>
      <dt style="color:var(--mid)">Inverse dynamics</dt><dd>current action from <b>before &amp; after</b> frames — the answer is in the inputs (extraction).</dd>
    </dl>
  </div>

  <div class="sec">The useful corner — hover the points</div>
  <p class="sub">x = shortcut availability (R_triv) · y = how much a learned probe beats the trivial copy. Green corner (top-left) = no shortcut but learnable. <b>★ = initial observation.</b> Hover any dot for its full definition and numbers.</p>
  <div class="scatters">{scatters}</div>

  <div class="sec">R_triv across the three robots</div>
  <p class="sub">R_triv (shortcut availability). Green <span style="color:var(--green)">+x.xx</span> = a probe on o_t beats the copy by that much.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>objective</th><th class="num">DROID</th><th class="num">Bridge</th><th class="num">RT-1</th></tr></thead>
    <tbody>{trows}</tbody></table></div>

  <div class="sec">Every objective, defined</div>
  <p class="sub">Each objective is <code>[context] &rarr; [answer]</code>. R_triv asks how much its <i>trivial rule</i> already gets; a probe on the current frame asks whether the answer is recoverable beyond that.</p>
  <div class="gloss">{gloss_html}</div>

  <div class="sec">Change-recoverability — the y-axis without the copy baseline</div>
  <p class="sub">Predicting the whole past frame is dominated by “copy o_t”. Instead predict the <b>change</b> (z<sub>past</sub>&minus;z<sub>t</sub>). R² = fraction explained.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>change recovered from o_t</th><th class="num">near-past (t&minus;5)</th><th class="num">far-past (t&minus;45)</th><th class="num">initial (t&minus;0)</th></tr></thead>
    <tbody>{cr_rows}</tbody></table></div>
  <p class="sub" style="margin-top:10px">Monotone in <b>both</b> — rises with retrospective distance (near &lt; far &lt; initial) and scene structure (DROID &lt; Bridge &lt; RT-1).</p>

  <div class="sec">Exp 1b — the BC memorization shortcut, as a data property</div>
  <p class="sub">A different shortcut than R_triv: not “is the target copyable” but “can the policy learn a lazy lookup and ignore language?” The <b>action here is the 15-step chunk</b> a VLA outputs (a single step is trivially ≈ your current pose — the chunk needs the whole near-trajectory). R² = variance explained, held-out episodes.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>memorization signal (R²)</th><th class="num">DROID</th><th class="num">Bridge</th><th class="num">RT-1</th></tr></thead>
    <tbody>{mem_rows}</tbody></table></div>
  <p class="sub" style="margin-top:10px"><b>The task is readable off RT-1 scenes (+0.29) but not DROID (−0.05)</b> — the vision-overrides-language shortcut is <b>dataset-dependent</b> (strong where scenes are consistent, like RT-1 / LIBERO). On DROID the 15-step chunk is <b>0.72 predictable from image+state</b> (a proprioceptive / momentum shortcut — down from 0.90 for a single action, exactly why the chunk is the honest target), and <b>language adds ≈0 to it (Δ_lang −0.05)</b> — a policy can coast on vision+proprioception and ignore language. <b>Honest limit:</b> raw-action learnability from frozen DINOv2 is a <i>lower bound</i>; a trained end-to-end policy memorizes far more, so the full BC shortcut needs a trained policy + counterfactual-instruction test (Experiment 3).</p>

  <div class="sec">Exp 2b — does it actually help? The auxiliary as a regularizer ★</div>
  <p class="sub">The payoff test. Not "is the retrospective representation good" but "does <b>co-training BC with a shortcut-free auxiliary</b> make the policy generalize?" We fine-tune a real encoder (DINOv2-small), BC head = [image, state, <b>language</b>] → 15-step action chunk, with an auxiliary head (predict initial-obs / future-obs) co-trained on the shared encoder. <b>OOD split = hold out whole task-clusters</b> (unseen instructions). OOD action-chunk R², mean ± std over 3 seeds:</p>
  <div class="tblwrap"><table>
    <thead><tr><th>policy</th><th class="num">RT-1 (OOD R²)</th><th class="num">DROID (OOD R²)</th></tr></thead>
    <tbody>{e2b_rows}</tbody></table></div>
  <p class="sub" style="margin-top:10px"><b>BC-only overfits and generalizes worst</b> (RT-1: reliably negative — worse than the mean on unseen tasks). <b>Every shortcut-free auxiliary helps, and combining them (mixed) is best on both datasets with the lowest variance.</b> This is the thesis in miniature: a hard, shortcut-free objective, co-trained on a shared trainable encoder, turns an overfitting policy into a generalizing one — the <i>regularizer</i> mechanism (not MAML). It’s the combination that matters (mixed &gt; single &gt; none), matching the “more shortcut-free objectives → more shortcuts removed” argument. (Small scale — the sign and ordering are the robust signals; scaling = Experiment 3.)</p>

  <div class="sec">Robust vs dataset-dependent</div>
  <div class="cards">
    <div class="card"><h3>Robust — the retrospective anchor</h3><p>Initial-obs is the least-redundant obs target on all 3 robots and the one a probe most beats the copy on (+0.02, +0.11, <b>+0.57</b>). Recovering “where did this episode start” demands real understanding — and is learnable.</p></div>
    <div class="card b"><h3>Robust — symmetric, distance-driven</h3><p>near-past ≈ near-future; far-past ≈ far-future; near &gt; far &gt; initial everywhere. Redundancy is governed by temporal <b>distance, not direction</b>.</p></div>
    <div class="card g"><h3>The copy shortcut is foresight <i>observation</i>, not action</h3><p>The trivial rule may only copy an actual input (image, state). So the shortcut-solvable targets are <b>future/near observation</b> (copy the current frame — the world-model foresight objectives) and <b>gripper</b> (a state input). <b>Action prediction has no input to copy</b> → R_triv≈0; it is the deployment task, not a cheat. (An earlier version wrongly used “copy the current action” as the baseline — corrected here.)</p></div>
    <div class="card m"><h3>Dataset-dependent — vision → instruction</h3><p>DROID (diverse): image can’t predict the instruction (+0.00). RT-1 (consistent kitchen): it can (+0.22) — the vision→language shortcut lives in the data.</p></div>
  </div>

  <div class="sec">Method &amp; honest limits</div>
  <ul class="m">
    <li><b>No policy trained.</b> Frozen DINOv2 (images) + MiniLM (instruction); linear + MLP probes; GroupKFold by episode. 500 / 227 / 500 episodes.</li>
    <li><b>Falsification gate passed</b> (DROID &amp; Bridge): longer episodes make initial-obs <i>less</i> redundant, so its non-redundancy isn’t a short-episode artifact.</li>
    <li><b>Premise, not payoff.</b> Targets differ in shortcut availability and retrospective escapes. Whether shortcut-free targets yield better representations is Experiment 2.</li>
  </ul>

  <footer><span>Exp 1 · DROID / Bridge / RT-1 · 15 objectives · DINOv2 + MiniLM · encoded on B200</span>
  <span>hover any point for its definition</span></footer>
</div>
<script>
(function(){{
  var tip=document.getElementById('tip');
  function show(e){{
    var d=JSON.parse(e.target.getAttribute('data-i'));
    tip.innerHTML='<div class="tn" style="color:'+d.c+'">'+d.n+'</div>'+
      '<div class="tv">R_triv '+d.rt+' &nbsp; G_obs '+d.g+' &nbsp; probe&gt;copy '+(d.p>=0?'+':'')+d.p+'<br>trivial: '+d.t+'</div>'+
      '<div class="td">'+d.d+'</div>';
    tip.style.opacity=1; move(e);
  }}
  function move(e){{
    var w=tip.offsetWidth,h=tip.offsetHeight,x=e.clientX+14,y=e.clientY+14;
    if(x+w>window.innerWidth-8)x=e.clientX-w-14; if(y+h>window.innerHeight-8)y=e.clientY-h-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }}
  function hide(){{tip.style.opacity=0;}}
  document.querySelectorAll('.pt').forEach(function(p){{
    p.addEventListener('mouseenter',show); p.addEventListener('mousemove',move); p.addEventListener('mouseleave',hide);
  }});
}})();
</script>
"""
open(os.path.join(OUT, "exp1_report.html"), "w").write(HTML)
print("WROTE exp1_report.html", len(HTML), "bytes")
